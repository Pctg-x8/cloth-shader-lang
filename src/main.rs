use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    io::Write,
};

use typed_arena::Arena;

mod spirv;
use spirv as spv;

fn main() {
    let src = std::fs::read_to_string("./sample_bloom_extract.csh").expect("Failed to load source");
    let mut tokenizer = Tokenizer {
        source: &src,
        line: 0,
        col: 0,
        current_line_indent: 0,
    };
    // populate line indent for first line
    tokenizer.populate_line_indent();
    let mut tokens = Vec::new();
    while let Some(t) = tokenizer.next_token().unwrap() {
        tokens.push(t);
    }

    let mut parse_state = ParseState {
        token_list: tokens,
        token_ptr: 0,
        indent_context_stack: Vec::new(),
    };
    let mut tlds = Vec::new();
    while parse_state.current_token().is_some() {
        let tld = parse_toplevel_declaration(&mut parse_state).unwrap();
        // println!("tld: {tld:#?}");
        tlds.push(tld);
    }

    let symbol_scope_arena = Arena::new();
    let global_symbol_scope = symbol_scope_arena.alloc(SymbolScope2::new_intrinsics());

    let mut partially_typed_types = HashMap::new();
    let mut user_defined_function_nodes = Vec::new();
    for tld in tlds.iter() {
        match tld {
            ToplevelDeclaration::Struct(s) => {
                partially_typed_types.insert(
                    s.name_token.slice,
                    (
                        SourceRef::from(&s.name_token),
                        UserDefinedTypePartiallyTyped::Struct(
                            s.member_list
                                .iter()
                                .map(|x| UserDefinedStructMemberPartiallyTyped {
                                    name: SourceRef::from(&x.name_token),
                                    ty: x.ty.clone(),
                                    attributes: x
                                        .attribute_lists
                                        .iter()
                                        .flat_map(|xs| {
                                            xs.attribute_list.iter().map(|x| x.0.clone())
                                        })
                                        .collect(),
                                })
                                .collect(),
                        ),
                    ),
                );
            }
            ToplevelDeclaration::Function(f) => user_defined_function_nodes.push(f),
        }
    }

    let top_scope_opaque_types = partially_typed_types.keys().copied().collect();
    let top_scope = symbol_scope_arena.alloc(SymbolScope2::new(Some(global_symbol_scope), false));
    top_scope
        .user_defined_type_symbols
        .extend(partially_typed_types.into_iter().map(|(k, (org, v))| {
            (
                k,
                (
                    org,
                    match v {
                        UserDefinedTypePartiallyTyped::Struct(s) => UserDefinedType::Struct(
                            s.into_iter()
                                .map(|x| UserDefinedStructMember {
                                    attribute: x
                                        .attributes
                                        .into_iter()
                                        .fold(SymbolAttribute::default(), |attrs, a| {
                                            eval_symbol_attributes(attrs, a)
                                        }),
                                    name: SourceRefSliceEq(x.name),
                                    ty: ConcreteType::build(
                                        global_symbol_scope,
                                        &top_scope_opaque_types,
                                        x.ty,
                                    ),
                                })
                                .collect(),
                        ),
                    },
                ),
            )
        }));
    for f in user_defined_function_nodes {
        top_scope.declare_function(UserDefinedFunctionSymbol {
            occurence: SourceRef::from(&f.fname_token),
            attribute: f
                .attribute_lists
                .iter()
                .flat_map(|xs| xs.attribute_list.iter().map(|(a, _)| a.clone()))
                .fold(SymbolAttribute::default(), |attrs, a| {
                    eval_symbol_attributes(attrs, a)
                }),
            inputs: match &f.input_args {
                FunctionDeclarationInputArguments::Single {
                    attribute_lists,
                    varname_token,
                    ty,
                    ..
                } => vec![(
                    attribute_lists
                        .iter()
                        .flat_map(|xs| xs.attribute_list.iter().map(|(a, _)| a.clone()))
                        .fold(SymbolAttribute::default(), |attrs, a| {
                            eval_symbol_attributes(attrs, a)
                        }),
                    SourceRef::from(varname_token),
                    ConcreteType::build(global_symbol_scope, &top_scope_opaque_types, ty.clone())
                        .instantiate(&top_scope),
                )],
                FunctionDeclarationInputArguments::Multiple { args, .. } => args
                    .iter()
                    .map(|(attribute_lists, varname_token, _, ty, _)| {
                        (
                            attribute_lists
                                .iter()
                                .flat_map(|xs| xs.attribute_list.iter().map(|(a, _)| a.clone()))
                                .fold(SymbolAttribute::default(), |attrs, a| {
                                    eval_symbol_attributes(attrs, a)
                                }),
                            SourceRef::from(varname_token),
                            ConcreteType::build(
                                global_symbol_scope,
                                &top_scope_opaque_types,
                                ty.clone(),
                            )
                            .instantiate(&top_scope),
                        )
                    })
                    .collect(),
            },
            output: match &f.output {
                Some(FunctionDeclarationOutput::Single {
                    attribute_lists,
                    ty,
                }) => vec![(
                    attribute_lists
                        .iter()
                        .flat_map(|xs| xs.attribute_list.iter().map(|(a, _)| a.clone()))
                        .fold(SymbolAttribute::default(), |attrs, a| {
                            eval_symbol_attributes(attrs, a)
                        }),
                    ConcreteType::build(global_symbol_scope, &top_scope_opaque_types, ty.clone())
                        .instantiate(&top_scope),
                )],
                Some(FunctionDeclarationOutput::Tupled { elements, .. }) => elements
                    .iter()
                    .map(|(attribute_lists, ty, _)| {
                        (
                            attribute_lists
                                .iter()
                                .flat_map(|xs| xs.attribute_list.iter().map(|(a, _)| a.clone()))
                                .fold(SymbolAttribute::default(), |attrs, a| {
                                    eval_symbol_attributes(attrs, a)
                                }),
                            ConcreteType::build(
                                global_symbol_scope,
                                &top_scope_opaque_types,
                                ty.clone(),
                            )
                            .instantiate(&top_scope),
                        )
                    })
                    .collect(),
                None => Vec::new(),
            },
        })
    }

    for d in tlds {
        match d {
            ToplevelDeclaration::Function(f) => {
                let function_symbol_scope =
                    symbol_scope_arena.alloc(SymbolScope2::new(Some(top_scope), true));
                match f.input_args {
                    FunctionDeclarationInputArguments::Single {
                        varname_token, ty, ..
                    } => {
                        function_symbol_scope.declare_function_input(
                            SourceRef::from(&varname_token),
                            ConcreteType::build(function_symbol_scope, &HashSet::new(), ty)
                                .instantiate(&function_symbol_scope),
                        );
                    }
                    FunctionDeclarationInputArguments::Multiple { args, .. } => {
                        for (_, n, _, ty, _) in args {
                            function_symbol_scope.declare_function_input(
                                SourceRef::from(&n),
                                ConcreteType::build(function_symbol_scope, &HashSet::new(), ty)
                                    .instantiate(&function_symbol_scope),
                            );
                        }
                    }
                }
                let mut simplify_context = SimplificationContext {
                    symbol_scope_arena: &symbol_scope_arena,
                    vars: Vec::new(),
                };
                let (mut last_var_id, mut last_var_type) =
                    simplify_expression(f.body, &mut simplify_context, function_symbol_scope);
                if top_scope.user_defined_function_symbols[f.fname_token.slice]
                    .attribute
                    .module_entry_point
                {
                    match f.output {
                        None => panic!("module entry point must output at least one value"),
                        Some(FunctionDeclarationOutput::Single { ty, .. }) => {
                            if last_var_type
                                != ConcreteType::build(function_symbol_scope, &HashSet::new(), ty)
                                    .instantiate(function_symbol_scope)
                            {
                                panic!("Error: output type mismatching");
                            }

                            last_var_id = simplify_context.add(
                                SimplifiedExpression::StoreOutput(last_var_id, 0),
                                IntrinsicType::Unit.into(),
                            );
                            last_var_type = IntrinsicType::Unit.into();
                        }
                        Some(FunctionDeclarationOutput::Tupled { elements, .. }) => {
                            let element_count = elements.len();
                            if last_var_type
                                != ConcreteType::Tuple(
                                    elements
                                        .into_iter()
                                        .map(|x| {
                                            ConcreteType::build(
                                                function_symbol_scope,
                                                &HashSet::new(),
                                                x.1,
                                            )
                                            .instantiate(function_symbol_scope)
                                        })
                                        .collect(),
                                )
                            {
                                panic!("Error: output type mismatching");
                            }

                            last_var_id = simplify_context.add(
                                SimplifiedExpression::DistributeOutputTuple(
                                    last_var_id,
                                    (0..element_count).collect(),
                                ),
                                IntrinsicType::Unit.into(),
                            );
                            last_var_type = IntrinsicType::Unit.into();
                        }
                    }
                }
                optimize_pure_expr(
                    &mut simplify_context.vars,
                    function_symbol_scope,
                    Some(&mut last_var_id),
                );

                top_scope.attach_function_body(
                    f.fname_token.slice,
                    FunctionBody {
                        symbol_scope: function_symbol_scope,
                        expressions: simplify_context.vars,
                        returning: last_var_id,
                        returning_type: last_var_type,
                    },
                );
            }
            ToplevelDeclaration::Struct(_) => (),
        }
    }

    for f in top_scope.user_defined_function_symbols.values() {
        let fb = top_scope.user_defined_function_body.0.borrow();
        let body = fb.get(f.occurence.slice);

        if body.is_some() {
            println!("toplevel function '{}':", f.occurence.slice);
        } else {
            println!("toplevel function prototype '{}':", f.occurence.slice);
        }
        println!("SymbolMeta = {f:#?}");
        if let Some(b) = body {
            println!("Function Scope = {:#?}", b.symbol_scope);
            println!("Body:");
            for (n, (x, t)) in b.expressions.iter().enumerate() {
                print_simp_expr(x, t, n, 0);
            }
            println!("returning: {:?}(ty = {:?})", b.returning, b.returning_type);
        }
    }

    let entry_points = top_scope
        .user_defined_function_symbols
        .values()
        .filter_map(|f| {
            if !f.attribute.module_entry_point {
                return None;
            }

            Some((
                f.occurence.slice,
                extract_shader_entry_point_description(f, &top_scope),
            ))
        })
        .collect::<HashMap<_, _>>();

    let mut spv_context = SpvModuleEmissionContext::new();
    spv_context.header_ops.push(spv::Instruction::Capability {
        capability: spv::Capability::Shader,
    });
    spv_context.header_ops.push(spv::Instruction::Capability {
        capability: spv::Capability::InputAttachment,
    });
    spv_context.header_ops.push(spv::Instruction::MemoryModel {
        addressing_model: spv::AddressingModel::Logical,
        memory_model: spv::MemoryModel::GLSL450,
    });
    for (n, f) in top_scope.user_defined_function_body.0.borrow().iter() {
        let entry_point_maps = emit_entry_point_spv_ops(&entry_points[n], &mut spv_context);
        let mut body_context = SpvFunctionBodyEmissionContext::new(entry_point_maps);
        let main_label_id = body_context.new_id();
        body_context.ops.push(spv::Instruction::Label {
            result: main_label_id,
        });
        emit_function_body_spv_ops(
            &f.expressions,
            f.returning,
            &mut spv_context,
            &mut body_context,
        );
        body_context.ops.push(spv::Instruction::Return);

        let fn_result_ty = spv_context.request_type_id(spv::Type::Void);
        let fnty = spv_context.request_type_id(spv::Type::Function {
            return_type: Box::new(spv::Type::Void),
            parameter_types: Vec::new(),
        });
        let fnid = spv_context.new_function_id();
        spv_context.function_ops.push(spv::Instruction::Function {
            result_type: fn_result_ty,
            result: fnid,
            function_control: spv::FunctionControl::empty(),
            function_type: fnty,
        });
        let fnid_offset = spv_context.latest_function_id;
        spv_context.latest_function_id += body_context.latest_id;
        spv_context
            .function_ops
            .extend(body_context.ops.into_iter().map(|x| {
                x.relocate(|id| match id {
                    SpvSectionLocalId::CurrentFunction(x) => {
                        SpvSectionLocalId::Function(x + fnid_offset)
                    }
                    x => x,
                })
            }));
        spv_context.function_ops.push(spv::Instruction::FunctionEnd);
        spv_context
            .entry_point_ops
            .push(spv::Instruction::EntryPoint {
                execution_model: entry_points[n].execution_model,
                entry_point: fnid,
                name: (*n).into(),
                interface: body_context
                    .entry_point_maps
                    .interface_global_vars
                    .iter()
                    .copied()
                    .collect(),
            });
        spv_context
            .execution_mode_ops
            .extend(
                entry_points[n]
                    .execution_mode_modifiers
                    .iter()
                    .map(|m| match m {
                        spv::ExecutionModeModifier::OriginUpperLeft => {
                            spv::Instruction::ExecutionMode {
                                entry_point: fnid,
                                mode: spv::ExecutionMode::OriginUpperLeft,
                                args: Vec::new(),
                            }
                        }
                    }),
            );
    }

    let (module_ops, max_id) = spv_context.serialize_ops();
    println!("module ops:");
    for x in module_ops.iter() {
        println!("  {x:?}");
    }

    let outfile = std::fs::File::options()
        .create(true)
        .write(true)
        .truncate(true)
        .open("out.spv")
        .expect("Failed to open outfile");
    let mut writer = std::io::BufWriter::new(outfile);
    spv::BinaryModuleHeader {
        magic_number: spv::BinaryModuleHeader::MAGIC_NUMBER,
        major_version: 1,
        minor_version: 4,
        generator_magic_number: 0,
        bound: max_id + 1,
    }
    .serialize(&mut writer)
    .expect("Failed to serialize module header");
    for x in module_ops.iter() {
        x.serialize_binary(&mut writer)
            .expect("Failed to serialize op");
    }
    writer.flush().expect("Failed to flush bufwriter");
}

fn print_simp_expr(x: &SimplifiedExpression, ty: &ConcreteType, vid: usize, nested: usize) {
    match x {
        SimplifiedExpression::ScopedBlock {
            expressions,
            returning,
            symbol_scope,
        } => {
            println!("  {}%{vid}: {ty:?} = Scope {{", "  ".repeat(nested));
            println!("  {}Function Inputs:", "  ".repeat(nested + 1));
            for (n, a) in symbol_scope.0.function_input_vars.iter().enumerate() {
                println!(
                    "  {}  {n} = {}: {:?}",
                    "  ".repeat(nested + 1),
                    a.occurence.slice,
                    a.ty
                );
            }
            println!("  {}Local Vars:", "  ".repeat(nested + 1));
            for (n, a) in symbol_scope.0.local_vars.borrow().iter().enumerate() {
                println!(
                    "  {}  {n} = {}: {:?}",
                    "  ".repeat(nested + 1),
                    a.occurence.slice,
                    a.ty
                );
            }
            for (n, (x, t)) in expressions.iter().enumerate() {
                print_simp_expr(x, t, n, nested + 1);
            }
            println!("  {}returning {returning:?}", "  ".repeat(nested + 1));
            println!("  {}}}", "  ".repeat(nested));
        }
        _ => println!("  {}%{vid}: {ty:?} = {x:?}", "  ".repeat(nested)),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RefPath<'s> {
    FunctionInput(usize),
    Member(Box<RefPath<'s>>, &'s str),
}

#[derive(Debug)]
pub struct ShaderInterfaceInputVariable<'s> {
    pub ty: spv::Type,
    pub original_refpath: RefPath<'s>,
    pub decorations: Vec<spv::Decorate>,
}

#[derive(Debug)]
pub struct ShaderInterfaceOutputVariable {
    pub ty: spv::Type,
    pub decorations: Vec<spv::Decorate>,
}

#[derive(Debug)]
pub struct ShaderInterfaceUniformVariable<'s> {
    pub ty: spv::Type,
    pub original_refpath: RefPath<'s>,
    pub decorations: Vec<spv::Decorate>,
}

#[derive(Debug)]
pub struct ShaderInterfacePushConstantVariable<'s> {
    pub ty: spv::Type,
    pub original_refpath: RefPath<'s>,
    pub offset: u32,
}

#[derive(Debug)]
pub struct ShaderEntryPointDescription<'s> {
    pub name: &'s str,
    pub execution_model: spv::ExecutionModel,
    pub execution_mode_modifiers: Vec<spv::ExecutionModeModifier>,
    pub input_variables: Vec<ShaderInterfaceInputVariable<'s>>,
    pub output_variables: Vec<ShaderInterfaceOutputVariable>,
    pub uniform_variables: Vec<ShaderInterfaceUniformVariable<'s>>,
    pub push_constant_variables: Vec<ShaderInterfacePushConstantVariable<'s>>,
}
fn extract_shader_entry_point_description<'s>(
    func: &UserDefinedFunctionSymbol<'s>,
    scope: &SymbolScope2<'_, 's>,
) -> ShaderEntryPointDescription<'s> {
    let execution_model = match func.attribute.shader_model {
        Some(ShaderModel::VertexShader) => spv::ExecutionModel::Vertex,
        Some(ShaderModel::TessellationControlShader) => spv::ExecutionModel::TessellationControl,
        Some(ShaderModel::TessellationEvaluationShader) => {
            spv::ExecutionModel::TessellationEvaluation
        }
        Some(ShaderModel::GeometryShader) => spv::ExecutionModel::Geometry,
        Some(ShaderModel::FragmentShader) => spv::ExecutionModel::Fragment,
        Some(ShaderModel::ComputeShader) => spv::ExecutionModel::GLCompute,
        None => unreachable!("not a entry point function"),
    };
    let execution_mode_modifiers = if execution_model == spv::ExecutionModel::Fragment {
        vec![spv::ExecutionModeModifier::OriginUpperLeft]
    } else {
        vec![]
    };

    let (
        mut input_variables,
        mut output_variables,
        mut uniform_variables,
        mut push_constant_variables,
    ) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (n, (attr, _, ty)) in func.inputs.iter().enumerate() {
        process_entry_point_inputs(
            attr,
            &RefPath::FunctionInput(n),
            ty,
            scope,
            &mut input_variables,
            &mut uniform_variables,
            &mut push_constant_variables,
        );
    }
    for (attr, ty) in func.output.iter() {
        process_entry_point_outputs(attr, ty, scope, &mut output_variables);
    }

    ShaderEntryPointDescription {
        name: func.occurence.slice,
        execution_model,
        execution_mode_modifiers,
        input_variables,
        output_variables,
        uniform_variables,
        push_constant_variables,
    }
}

pub enum ProcessedEntryPointInput<'s> {
    InputVariable(ShaderInterfaceInputVariable<'s>),
}

#[inline(always)]
const fn roundup2(x: usize, a: usize) -> usize {
    (x + (a - 1)) & !(a - 1)
}

fn process_entry_point_inputs<'s>(
    attr: &SymbolAttribute,
    refpath: &RefPath<'s>,
    ty: &ConcreteType<'s>,
    scope: &SymbolScope2<'_, 's>,
    input_variables: &mut Vec<ShaderInterfaceInputVariable<'s>>,
    uniform_variables: &mut Vec<ShaderInterfaceUniformVariable<'s>>,
    push_constant_variables: &mut Vec<ShaderInterfacePushConstantVariable<'s>>,
) {
    match attr {
        SymbolAttribute {
            module_entry_point: true,
            ..
        } => panic!("Error: entry point input cannot be a module entry point"),
        SymbolAttribute {
            shader_model: Some(_),
            ..
        } => panic!("Error: entry point input cannot have any shader model attributes"),
        SymbolAttribute {
            descriptor_set_location: Some(set),
            descriptor_set_binding: Some(binding),
            input_attachment_index: None,
            push_constant_offset: None,
            bound_location: None,
            bound_builtin_io: None,
            ..
        } => match ty {
            ConcreteType::Struct(members) => {
                let spv_ty = spv::Type::Struct {
                    member_types: members
                        .iter()
                        .scan(0, |top, x| {
                            if !x.attribute.is_empty() {
                                panic!("a member of an uniform struct cannot have any attributes");
                            }

                            let offset = roundup2(
                                *top,
                                x.ty.std140_alignment()
                                    .expect("this type cannot be a member of an uniform struct"),
                            );
                            *top = offset
                                + x.ty
                                    .std140_size()
                                    .expect("this type cannot be a member of an uniform struct");

                            Some(spv::TypeStructMember {
                                offset: offset as _,
                                ty: x.ty.make_spv_type(scope),
                                decorations: Vec::new(),
                            })
                        })
                        .collect(),
                };

                uniform_variables.push(ShaderInterfaceUniformVariable {
                    ty: spv_ty,
                    original_refpath: refpath.clone(),
                    decorations: vec![
                        spv::Decorate::DescriptorSet(*set),
                        spv::Decorate::Binding(*binding),
                    ],
                });
            }
            _ => panic!("Error: descriptor buffer binding can be done only for struct types"),
        },
        SymbolAttribute {
            descriptor_set_location: Some(set),
            descriptor_set_binding: Some(binding),
            input_attachment_index: Some(aix),
            push_constant_offset: None,
            bound_location: None,
            bound_builtin_io: None,
            ..
        } => match ty {
            ConcreteType::Intrinsic(IntrinsicType::SubpassInput) => {
                uniform_variables.push(ShaderInterfaceUniformVariable {
                    ty: spv::Type::subpass_data_image_type(),
                    original_refpath: refpath.clone(),
                    decorations: vec![
                        spv::Decorate::DescriptorSet(*set),
                        spv::Decorate::Binding(*binding),
                        spv::Decorate::InputAttachmentIndex(*aix),
                    ],
                })
            }
            _ => panic!(
                "Error: descriptor subpass input binding canbe done only for the SubpassInput type"
            ),
        },
        SymbolAttribute {
            push_constant_offset: Some(pc),
            bound_location: None,
            bound_builtin_io: None,
            ..
        } => {
            if !ty.can_uniform_struct_member() {
                panic!("Error: this type cannot be used as push constant value")
            }

            push_constant_variables.push(ShaderInterfacePushConstantVariable {
                ty: ty.make_spv_type(scope),
                original_refpath: refpath.clone(),
                offset: *pc,
            });
        }
        SymbolAttribute {
            bound_location: Some(loc),
            bound_builtin_io: None,
            ..
        } => input_variables.push(ShaderInterfaceInputVariable {
            ty: ty.make_spv_type(scope),
            original_refpath: refpath.clone(),
            decorations: vec![spv::Decorate::Location(*loc)],
        }),
        SymbolAttribute {
            bound_builtin_io: Some(b),
            ..
        } => input_variables.push(ShaderInterfaceInputVariable {
            ty: ty.make_spv_type(scope),
            original_refpath: refpath.clone(),
            decorations: vec![spv::Decorate::Builtin(match b {
                BuiltinInputOutput::Position => spv::Builtin::Position,
                BuiltinInputOutput::VertexID => spv::Builtin::VertexId,
                BuiltinInputOutput::InstanceID => spv::Builtin::InstanceId,
            })],
        }),
        _ => match ty {
            ConcreteType::Struct(members) => {
                for m in members {
                    process_entry_point_inputs(
                        &m.attribute,
                        &RefPath::Member(Box::new(refpath.clone()), m.name.0.slice),
                        &m.ty,
                        scope,
                        input_variables,
                        uniform_variables,
                        push_constant_variables,
                    );
                }
            }
            _ => panic!("Error: non-decorated primitive input found"),
        },
    }
}

fn process_entry_point_outputs<'s>(
    attr: &SymbolAttribute,
    ty: &ConcreteType<'s>,
    scope: &SymbolScope2<'_, 's>,
    output_variables: &mut Vec<ShaderInterfaceOutputVariable>,
) {
    match attr {
        SymbolAttribute {
            module_entry_point: true,
            ..
        } => panic!("Error: entry point output cannot be a module entry point"),
        SymbolAttribute {
            shader_model: Some(_),
            ..
        } => panic!("Error: entry point output cannot have any shader model attributes"),
        SymbolAttribute {
            descriptor_set_location: Some(_),
            descriptor_set_binding: Some(_),
            push_constant_offset: None,
            bound_location: None,
            bound_builtin_io: None,
            ..
        } => panic!("Error: entry point output cannot have any descriptor set bindings"),
        SymbolAttribute {
            push_constant_offset: Some(_),
            bound_location: None,
            bound_builtin_io: None,
            ..
        } => panic!("Error: entry point output cannot have any push constant binding"),
        SymbolAttribute {
            bound_location: Some(loc),
            bound_builtin_io: None,
            ..
        } => output_variables.push(ShaderInterfaceOutputVariable {
            ty: ty.make_spv_type(scope),
            decorations: vec![spv::Decorate::Location(*loc)],
        }),
        SymbolAttribute {
            bound_builtin_io: Some(b),
            ..
        } => output_variables.push(ShaderInterfaceOutputVariable {
            ty: ty.make_spv_type(scope),
            decorations: vec![spv::Decorate::Builtin(match b {
                BuiltinInputOutput::Position => spv::Builtin::Position,
                BuiltinInputOutput::VertexID => spv::Builtin::VertexId,
                BuiltinInputOutput::InstanceID => spv::Builtin::InstanceId,
            })],
        }),
        _ => match ty {
            ConcreteType::Struct(members) => {
                for m in members {
                    process_entry_point_outputs(&m.attribute, &m.ty, scope, output_variables);
                }
            }
            _ => panic!("Error: non-decorated primitive output found"),
        },
    }
}

#[derive(Clone, Copy, Debug)]
pub enum SpvSectionLocalId {
    TypeConst(spv::Id),
    GlobalVariable(spv::Id),
    Function(spv::Id),
    CurrentFunction(spv::Id),
}

pub struct SpvFunctionBodyEmissionContext<'s> {
    pub entry_point_maps: ShaderEntryPointMaps<'s>,
    pub ops: Vec<spv::Instruction<SpvSectionLocalId>>,
    pub latest_id: spv::Id,
    pub emitted_expression_id: HashMap<usize, Option<(SpvSectionLocalId, spv::Type)>>,
}
impl<'s> SpvFunctionBodyEmissionContext<'s> {
    pub fn new(maps: ShaderEntryPointMaps<'s>) -> Self {
        Self {
            entry_point_maps: maps,
            ops: Vec::new(),
            latest_id: 0,
            emitted_expression_id: HashMap::new(),
        }
    }

    pub fn new_id(&mut self) -> SpvSectionLocalId {
        self.latest_id += 1;

        SpvSectionLocalId::CurrentFunction(self.latest_id - 1)
    }
}

pub enum GlobalAccessType {
    Direct(SpvSectionLocalId, spv::Type),
    PushConstantStruct {
        struct_var: SpvSectionLocalId,
        member_index: u32,
        member_ty: spv::Type,
    },
}

pub struct ShaderEntryPointMaps<'s> {
    refpath_to_global_var: HashMap<RefPath<'s>, GlobalAccessType>,
    output_global_vars: Vec<SpvSectionLocalId>,
    interface_global_vars: Vec<SpvSectionLocalId>,
}
fn emit_entry_point_spv_ops<'s>(
    ep: &ShaderEntryPointDescription<'s>,
    ctx: &mut SpvModuleEmissionContext,
) -> ShaderEntryPointMaps<'s> {
    let mut entry_point_maps = ShaderEntryPointMaps {
        refpath_to_global_var: HashMap::new(),
        output_global_vars: Vec::new(),
        interface_global_vars: Vec::new(),
    };

    for a in ep.output_variables.iter() {
        let type_id = ctx.request_type_id(spv::Type::Pointer {
            storage_class: spv::StorageClass::Output,
            base_type: Box::new(a.ty.clone()),
        });
        let gvid = ctx.new_global_variable_id();
        ctx.global_variable_ops.push(spv::Instruction::Variable {
            result_type: type_id,
            result: gvid,
            storage_class: spv::StorageClass::Output,
            initializer: None,
        });
        ctx.annotation_ops
            .extend(a.decorations.iter().map(|d| match d {
                spv::Decorate::Location(loc) => spv::Instruction::Decorate {
                    target: gvid,
                    decoration: spv::Decoration::Location,
                    args: vec![*loc],
                },
                spv::Decorate::Builtin(b) => spv::Instruction::Decorate {
                    target: gvid,
                    decoration: spv::Decoration::Builtin,
                    args: vec![*b as _],
                },
                _ => unimplemented!(),
            }));
        entry_point_maps.output_global_vars.push(gvid);
        entry_point_maps.interface_global_vars.push(gvid);
    }

    for a in ep.input_variables.iter() {
        let type_id = ctx.request_type_id(spv::Type::Pointer {
            storage_class: spv::StorageClass::Input,
            base_type: Box::new(a.ty.clone()),
        });
        let result_id = ctx.new_global_variable_id();

        ctx.global_variable_ops.push(spv::Instruction::Variable {
            result_type: type_id,
            result: result_id,
            storage_class: spv::StorageClass::Input,
            initializer: None,
        });
        entry_point_maps.refpath_to_global_var.insert(
            a.original_refpath.clone(),
            GlobalAccessType::Direct(result_id, a.ty.clone()),
        );
        entry_point_maps.interface_global_vars.push(result_id);
        ctx.annotation_ops
            .extend(a.decorations.iter().map(|d| match d {
                spv::Decorate::Location(loc) => spv::Instruction::Decorate {
                    target: result_id,
                    decoration: spv::Decoration::Location,
                    args: vec![*loc],
                },
                spv::Decorate::Builtin(b) => spv::Instruction::Decorate {
                    target: result_id,
                    decoration: spv::Decoration::Builtin,
                    args: vec![*b as _],
                },
                _ => unimplemented!(),
            }));
    }

    for a in ep.uniform_variables.iter() {
        let type_id = ctx.request_type_id(spv::Type::Pointer {
            storage_class: spv::StorageClass::Uniform,
            base_type: Box::new(a.ty.clone()),
        });
        let result_id = ctx.new_global_variable_id();

        ctx.global_variable_ops.push(spv::Instruction::Variable {
            result_type: type_id,
            result: result_id,
            storage_class: spv::StorageClass::Uniform,
            initializer: None,
        });
        entry_point_maps.refpath_to_global_var.insert(
            a.original_refpath.clone(),
            GlobalAccessType::Direct(result_id, a.ty.clone()),
        );
        entry_point_maps.interface_global_vars.push(result_id);
        ctx.annotation_ops
            .extend(a.decorations.iter().map(|d| match d {
                spv::Decorate::DescriptorSet(loc) => spv::Instruction::Decorate {
                    target: result_id,
                    decoration: spv::Decoration::DescriptorSet,
                    args: vec![*loc],
                },
                spv::Decorate::Binding(b) => spv::Instruction::Decorate {
                    target: result_id,
                    decoration: spv::Decoration::Binding,
                    args: vec![*b],
                },
                spv::Decorate::InputAttachmentIndex(b) => spv::Instruction::Decorate {
                    target: result_id,
                    decoration: spv::Decoration::InputAttachmentIndex,
                    args: vec![*b],
                },
                _ => unimplemented!(),
            }));
    }

    if !ep.push_constant_variables.is_empty() {
        let push_constant_uniform_block = spv::Type::Struct {
            member_types: ep
                .push_constant_variables
                .iter()
                .map(|x| spv::TypeStructMember {
                    offset: x.offset,
                    ty: x.ty.clone(),
                    decorations: vec![spv::Decorate::Offset(x.offset)],
                })
                .collect(),
        };
        let push_constant_uniform_block_tid =
            ctx.request_type_id(push_constant_uniform_block.clone());
        let push_constant_uniform_block_var_type = spv::Type::Pointer {
            storage_class: spv::StorageClass::PushConstant,
            base_type: Box::new(push_constant_uniform_block.clone()),
        };
        let push_constant_uniform_block_var_tid =
            ctx.request_type_id(push_constant_uniform_block_var_type);
        let push_constant_uniform_block_var_id = ctx.new_global_variable_id();
        ctx.global_variable_ops.push(spv::Instruction::Variable {
            result_type: push_constant_uniform_block_var_tid,
            result: push_constant_uniform_block_var_id,
            storage_class: spv::StorageClass::PushConstant,
            initializer: None,
        });
        ctx.annotation_ops.push(spv::Instruction::Decorate {
            target: push_constant_uniform_block_tid,
            decoration: spv::Decoration::Block,
            args: vec![],
        });
        for (n, a) in ep.push_constant_variables.iter().enumerate() {
            entry_point_maps.refpath_to_global_var.insert(
                a.original_refpath.clone(),
                GlobalAccessType::PushConstantStruct {
                    struct_var: push_constant_uniform_block_var_id,
                    member_index: n as _,
                    member_ty: a.ty.clone(),
                },
            );
        }
        entry_point_maps
            .interface_global_vars
            .push(push_constant_uniform_block_var_id);
    }

    entry_point_maps
}

fn emit_function_body_spv_ops(
    body: &[(SimplifiedExpression, ConcreteType)],
    returning: ExprRef,
    module_ctx: &mut SpvModuleEmissionContext,
    ctx: &mut SpvFunctionBodyEmissionContext,
) {
    let _ = emit_expr_spv_ops(body, returning.0, module_ctx, ctx);
}

fn emit_expr_spv_ops(
    body: &[(SimplifiedExpression, ConcreteType)],
    expr_id: usize,
    module_ctx: &mut SpvModuleEmissionContext,
    ctx: &mut SpvFunctionBodyEmissionContext,
) -> Option<(SpvSectionLocalId, spv::Type)> {
    if let Some(r) = ctx.emitted_expression_id.get(&expr_id) {
        return r.clone();
    }

    let result = match &body[expr_id].0 {
        SimplifiedExpression::Select(c, t, e) => {
            let (c, ct) = emit_expr_spv_ops(body, c.0, module_ctx, ctx).unwrap();

            let requires_control_flow_branching = ct != spv::Type::Bool
                || matches!(
                    (&body[t.0].0, &body[e.0].0),
                    (SimplifiedExpression::ScopedBlock { .. }, _)
                        | (_, SimplifiedExpression::ScopedBlock { .. })
                );

            if requires_control_flow_branching {
                // impure select
                unimplemented!("Control Flow Branching strategy");
            } else {
                let (t, tt) = emit_expr_spv_ops(body, t.0, module_ctx, ctx).unwrap();
                let (e, et) = emit_expr_spv_ops(body, e.0, module_ctx, ctx).unwrap();
                assert_eq!(tt, et);

                let result_type = module_ctx.request_type_id(tt.clone());
                let result_id = ctx.new_id();
                ctx.ops.push(spv::Instruction::Select {
                    result_type,
                    result: result_id,
                    condition: c,
                    object1: t,
                    object2: e,
                });
                Some((result_id, tt.clone()))
            }
        }
        SimplifiedExpression::ConstructIntrinsicComposite(it, args) => {
            let (args, arg_ty): (Vec<_>, Vec<_>) = args
                .iter()
                .map(|&x| emit_expr_spv_ops(body, x.0, module_ctx, ctx).unwrap())
                .unzip();

            match it {
                IntrinsicType::Float4 => {
                    assert_eq!(args.len(), 4);
                    assert!(arg_ty.iter().all(|x| *x == spv::Type::Float { width: 32 }));

                    let result_ty = spv::Type::Vector {
                        component_type: Box::new(spv::Type::Float { width: 32 }),
                        component_count: 4,
                    };
                    let result_type = module_ctx.request_type_id(result_ty.clone());
                    let is_constant = args
                        .iter()
                        .all(|x| matches!(x, SpvSectionLocalId::TypeConst(_)));

                    let result_id;
                    if is_constant {
                        result_id = module_ctx.new_type_const_id();
                        module_ctx
                            .type_const_ops
                            .push(spv::Instruction::ConstantComposite {
                                result_type,
                                result: result_id,
                                constituents: args,
                            });
                    } else {
                        result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::CompositeConstruct {
                            result_type,
                            result: result_id,
                            constituents: args,
                        });
                    };

                    Some((result_id, result_ty))
                }
                _ => unimplemented!("Composite construction for {it:?}"),
            }
        }
        SimplifiedExpression::LogAnd(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, module_ctx, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, module_ctx, ctx).unwrap();
            assert_eq!(lt, rt);
            assert_eq!(lt, spv::Type::Bool);

            let result_type = module_ctx.request_type_id(spv::Type::Bool);
            let result_id = ctx.new_id();
            ctx.ops.push(spv::Instruction::LogicalAnd {
                result_type,
                result: result_id,
                operand1: l,
                operand2: r,
            });
            Some((result_id, spv::Type::Bool))
        }
        SimplifiedExpression::LogOr(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, module_ctx, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, module_ctx, ctx).unwrap();
            assert_eq!(lt, rt);
            assert_eq!(lt, spv::Type::Bool);

            let result_type = module_ctx.request_type_id(spv::Type::Bool);
            let result_id = ctx.new_id();
            ctx.ops.push(spv::Instruction::LogicalAnd {
                result_type,
                result: result_id,
                operand1: l,
                operand2: r,
            });
            Some((result_id, spv::Type::Bool))
        }
        SimplifiedExpression::Eq(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, module_ctx, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, module_ctx, ctx).unwrap();
            assert_eq!(lt, rt);

            match lt {
                spv::Type::Bool => {
                    let result_ty = spv::Type::Bool;
                    let result_type = module_ctx.request_type_id(result_ty.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::LogicalEqual {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, result_ty))
                }
                spv::Type::Int { .. } => {
                    let result_ty = spv::Type::Bool;
                    let result_type = module_ctx.request_type_id(result_ty.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::IEqual {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, result_ty))
                }
                spv::Type::Float { .. } => {
                    let result_ty = spv::Type::Bool;
                    let result_type = module_ctx.request_type_id(result_ty.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::FOrdEqual {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, result_ty))
                }
                spv::Type::Vector {
                    component_type,
                    component_count,
                } => match *component_type {
                    spv::Type::Bool => {
                        let result_ty = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Bool),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(result_ty.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::LogicalEqual {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, result_ty))
                    }
                    spv::Type::Int { .. } => {
                        let result_ty = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Bool),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(result_ty.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::IEqual {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, result_ty))
                    }
                    spv::Type::Float { .. } => {
                        let result_ty = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Bool),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(result_ty.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::FOrdEqual {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, result_ty))
                    }
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        }
        SimplifiedExpression::Ne(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, module_ctx, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, module_ctx, ctx).unwrap();
            assert_eq!(lt, rt);

            match lt {
                spv::Type::Bool => {
                    let result_ty = spv::Type::Bool;
                    let result_type = module_ctx.request_type_id(result_ty.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::LogicalNotEqual {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, result_ty))
                }
                spv::Type::Int { .. } => {
                    let result_ty = spv::Type::Bool;
                    let result_type = module_ctx.request_type_id(result_ty.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::INotEqual {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, result_ty))
                }
                spv::Type::Float { .. } => {
                    let result_ty = spv::Type::Bool;
                    let result_type = module_ctx.request_type_id(result_ty.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::FOrdNotEqual {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, result_ty))
                }
                spv::Type::Vector {
                    component_type,
                    component_count,
                } => match *component_type {
                    spv::Type::Bool => {
                        let result_ty = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Bool),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(result_ty.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::LogicalNotEqual {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, result_ty))
                    }
                    spv::Type::Int { .. } => {
                        let result_ty = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Bool),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(result_ty.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::INotEqual {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, result_ty))
                    }
                    spv::Type::Float { .. } => {
                        let result_ty = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Bool),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(result_ty.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::FOrdNotEqual {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, result_ty))
                    }
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        }
        SimplifiedExpression::Lt(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, module_ctx, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, module_ctx, ctx).unwrap();
            assert_eq!(lt, rt);

            match lt {
                spv::Type::Int {
                    signedness: true, ..
                } => {
                    let result_ty = spv::Type::Bool;
                    let result_type = module_ctx.request_type_id(result_ty.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::SLessThan {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, result_ty))
                }
                spv::Type::Int {
                    signedness: false, ..
                } => {
                    let result_ty = spv::Type::Bool;
                    let result_type = module_ctx.request_type_id(result_ty.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::ULessThan {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, result_ty))
                }
                spv::Type::Float { .. } => {
                    let result_ty = spv::Type::Bool;
                    let result_type = module_ctx.request_type_id(result_ty.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::FOrdLessThan {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, result_ty))
                }
                spv::Type::Vector {
                    component_type,
                    component_count,
                } => match *component_type {
                    spv::Type::Int {
                        signedness: true, ..
                    } => {
                        let result_ty = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Bool),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(result_ty.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::SLessThan {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, result_ty))
                    }
                    spv::Type::Int {
                        signedness: false, ..
                    } => {
                        let result_ty = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Bool),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(result_ty.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::ULessThan {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, result_ty))
                    }
                    spv::Type::Float { .. } => {
                        let result_ty = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Bool),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(result_ty.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::FOrdLessThan {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, result_ty))
                    }
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        }
        SimplifiedExpression::Le(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, module_ctx, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, module_ctx, ctx).unwrap();
            assert_eq!(lt, rt);

            match lt {
                spv::Type::Int {
                    signedness: true, ..
                } => {
                    let result_ty = spv::Type::Bool;
                    let result_type = module_ctx.request_type_id(result_ty.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::SLessThanEqual {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, result_ty))
                }
                spv::Type::Int {
                    signedness: false, ..
                } => {
                    let result_ty = spv::Type::Bool;
                    let result_type = module_ctx.request_type_id(result_ty.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::ULessThanEqual {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, result_ty))
                }
                spv::Type::Float { .. } => {
                    let result_ty = spv::Type::Bool;
                    let result_type = module_ctx.request_type_id(result_ty.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::FOrdLessThanEqual {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, result_ty))
                }
                spv::Type::Vector {
                    component_type,
                    component_count,
                } => match *component_type {
                    spv::Type::Int {
                        signedness: true, ..
                    } => {
                        let result_ty = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Bool),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(result_ty.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::SLessThanEqual {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, result_ty))
                    }
                    spv::Type::Int {
                        signedness: false, ..
                    } => {
                        let result_ty = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Bool),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(result_ty.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::ULessThanEqual {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, result_ty))
                    }
                    spv::Type::Float { .. } => {
                        let result_ty = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Bool),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(result_ty.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::FOrdLessThanEqual {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, result_ty))
                    }
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        }
        SimplifiedExpression::Gt(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, module_ctx, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, module_ctx, ctx).unwrap();
            assert_eq!(lt, rt);

            match lt {
                spv::Type::Int {
                    signedness: true, ..
                } => {
                    let result_ty = spv::Type::Bool;
                    let result_type = module_ctx.request_type_id(result_ty.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::SGreaterThan {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, result_ty))
                }
                spv::Type::Int {
                    signedness: false, ..
                } => {
                    let result_ty = spv::Type::Bool;
                    let result_type = module_ctx.request_type_id(result_ty.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::UGreaterThan {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, result_ty))
                }
                spv::Type::Float { .. } => {
                    let result_ty = spv::Type::Bool;
                    let result_type = module_ctx.request_type_id(result_ty.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::FOrdGreaterThan {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, result_ty))
                }
                spv::Type::Vector {
                    component_type,
                    component_count,
                } => match *component_type {
                    spv::Type::Int {
                        signedness: true, ..
                    } => {
                        let result_ty = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Bool),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(result_ty.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::SGreaterThan {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, result_ty))
                    }
                    spv::Type::Int {
                        signedness: false, ..
                    } => {
                        let result_ty = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Bool),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(result_ty.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::UGreaterThan {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, result_ty))
                    }
                    spv::Type::Float { .. } => {
                        let result_ty = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Bool),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(result_ty.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::FOrdGreaterThan {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, result_ty))
                    }
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        }
        SimplifiedExpression::Ge(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, module_ctx, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, module_ctx, ctx).unwrap();
            assert_eq!(lt, rt);

            match lt {
                spv::Type::Int {
                    signedness: true, ..
                } => {
                    let result_ty = spv::Type::Bool;
                    let result_type = module_ctx.request_type_id(result_ty.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::SGreaterThanEqual {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, result_ty))
                }
                spv::Type::Int {
                    signedness: false, ..
                } => {
                    let result_ty = spv::Type::Bool;
                    let result_type = module_ctx.request_type_id(result_ty.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::UGreaterThanEqual {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, result_ty))
                }
                spv::Type::Float { .. } => {
                    let result_ty = spv::Type::Bool;
                    let result_type = module_ctx.request_type_id(result_ty.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::FOrdGreaterThanEqual {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, result_ty))
                }
                spv::Type::Vector {
                    component_type,
                    component_count,
                } => match *component_type {
                    spv::Type::Int {
                        signedness: true, ..
                    } => {
                        let result_ty = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Bool),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(result_ty.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::SGreaterThanEqual {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, result_ty))
                    }
                    spv::Type::Int {
                        signedness: false, ..
                    } => {
                        let result_ty = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Bool),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(result_ty.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::UGreaterThanEqual {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, result_ty))
                    }
                    spv::Type::Float { .. } => {
                        let result_ty = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Bool),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(result_ty.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::FOrdGreaterThanEqual {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, result_ty))
                    }
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        }
        SimplifiedExpression::BitAnd(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, module_ctx, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, module_ctx, ctx).unwrap();
            assert_eq!(lt, rt);

            match lt {
                spv::Type::Int { width, signedness } => {
                    let rt = spv::Type::Int { width, signedness };
                    let result_type = module_ctx.request_type_id(rt.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::BitwiseAnd {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, rt))
                }
                spv::Type::Vector {
                    component_type,
                    component_count,
                } => match *component_type {
                    spv::Type::Int { width, signedness } => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Int { width, signedness }),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::BitwiseAnd {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, rt))
                    }
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        }
        SimplifiedExpression::BitOr(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, module_ctx, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, module_ctx, ctx).unwrap();
            assert_eq!(lt, rt);

            match lt {
                spv::Type::Int { width, signedness } => {
                    let rt = spv::Type::Int { width, signedness };
                    let result_type = module_ctx.request_type_id(rt.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::BitwiseOr {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, rt))
                }
                spv::Type::Vector {
                    component_type,
                    component_count,
                } => match *component_type {
                    spv::Type::Int { width, signedness } => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Int { width, signedness }),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::BitwiseOr {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, rt))
                    }
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        }
        SimplifiedExpression::BitXor(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, module_ctx, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, module_ctx, ctx).unwrap();
            assert_eq!(lt, rt);

            match lt {
                spv::Type::Int { width, signedness } => {
                    let rt = spv::Type::Int { width, signedness };
                    let result_type = module_ctx.request_type_id(rt.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::BitwiseXor {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, rt))
                }
                spv::Type::Vector {
                    component_type,
                    component_count,
                } => match *component_type {
                    spv::Type::Int { width, signedness } => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Int { width, signedness }),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::BitwiseXor {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, rt))
                    }
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        }
        SimplifiedExpression::Add(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, module_ctx, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, module_ctx, ctx).unwrap();
            assert_eq!(lt, rt);

            match lt {
                spv::Type::Int { width, signedness } => {
                    let rt = spv::Type::Int { width, signedness };
                    let result_type = module_ctx.request_type_id(rt.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::IAdd {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, rt))
                }
                spv::Type::Float { width } => {
                    let rt = spv::Type::Float { width };
                    let result_type = module_ctx.request_type_id(rt.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::FAdd {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, rt))
                }
                spv::Type::Vector {
                    component_type,
                    component_count,
                } => match *component_type {
                    spv::Type::Int { width, signedness } => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Int { width, signedness }),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::IAdd {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, rt))
                    }
                    spv::Type::Float { width } => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Float { width }),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::FAdd {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, rt))
                    }
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        }
        SimplifiedExpression::Sub(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, module_ctx, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, module_ctx, ctx).unwrap();
            assert_eq!(lt, rt);

            match lt {
                spv::Type::Int { width, signedness } => {
                    let rt = spv::Type::Int { width, signedness };
                    let result_type = module_ctx.request_type_id(rt.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::ISub {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, rt))
                }
                spv::Type::Float { width } => {
                    let rt = spv::Type::Float { width };
                    let result_type = module_ctx.request_type_id(rt.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::FSub {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, rt))
                }
                spv::Type::Vector {
                    component_type,
                    component_count,
                } => match *component_type {
                    spv::Type::Int { width, signedness } => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Int { width, signedness }),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::ISub {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, rt))
                    }
                    spv::Type::Float { width } => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Float { width }),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::FSub {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, rt))
                    }
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        }
        SimplifiedExpression::Mul(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, module_ctx, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, module_ctx, ctx).unwrap();
            assert_eq!(lt, rt);

            match lt {
                spv::Type::Int { width, signedness } => {
                    let rt = spv::Type::Int { width, signedness };
                    let result_type = module_ctx.request_type_id(rt.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::IMul {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, rt))
                }
                spv::Type::Float { width } => {
                    let rt = spv::Type::Float { width };
                    let result_type = module_ctx.request_type_id(rt.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::FMul {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, rt))
                }
                spv::Type::Vector {
                    component_type,
                    component_count,
                } => match *component_type {
                    spv::Type::Int { width, signedness } => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Int { width, signedness }),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::IMul {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, rt))
                    }
                    spv::Type::Float { width } => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Float { width }),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::FMul {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, rt))
                    }
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        }
        SimplifiedExpression::Div(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, module_ctx, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, module_ctx, ctx).unwrap();
            assert_eq!(lt, rt);

            match lt {
                spv::Type::Int {
                    width,
                    signedness: true,
                } => {
                    let rt = spv::Type::Int {
                        width,
                        signedness: true,
                    };
                    let result_type = module_ctx.request_type_id(rt.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::SDiv {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, rt))
                }
                spv::Type::Int {
                    width,
                    signedness: false,
                } => {
                    let rt = spv::Type::Int {
                        width,
                        signedness: false,
                    };
                    let result_type = module_ctx.request_type_id(rt.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::UDiv {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, rt))
                }
                spv::Type::Float { width } => {
                    let rt = spv::Type::Float { width };
                    let result_type = module_ctx.request_type_id(rt.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::FDiv {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, rt))
                }
                spv::Type::Vector {
                    component_type,
                    component_count,
                } => match *component_type {
                    spv::Type::Int {
                        width,
                        signedness: true,
                    } => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Int {
                                width,
                                signedness: true,
                            }),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::SDiv {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, rt))
                    }
                    spv::Type::Int {
                        width,
                        signedness: false,
                    } => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Int {
                                width,
                                signedness: false,
                            }),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::UDiv {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, rt))
                    }
                    spv::Type::Float { width } => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Float { width }),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::FDiv {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, rt))
                    }
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        }
        SimplifiedExpression::Rem(l, r) => {
            let (l, lt) = emit_expr_spv_ops(body, l.0, module_ctx, ctx).unwrap();
            let (r, rt) = emit_expr_spv_ops(body, r.0, module_ctx, ctx).unwrap();
            assert_eq!(lt, rt);

            match lt {
                spv::Type::Int {
                    width,
                    signedness: true,
                } => {
                    let rt = spv::Type::Int {
                        width,
                        signedness: true,
                    };
                    let result_type = module_ctx.request_type_id(rt.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::SRem {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, rt))
                }
                spv::Type::Int {
                    width,
                    signedness: false,
                } => {
                    let rt = spv::Type::Int {
                        width,
                        signedness: false,
                    };
                    let result_type = module_ctx.request_type_id(rt.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::UMod {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, rt))
                }
                spv::Type::Float { width } => {
                    let rt = spv::Type::Float { width };
                    let result_type = module_ctx.request_type_id(rt.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::FRem {
                        result_type,
                        result: result_id,
                        operand1: l,
                        operand2: r,
                    });
                    Some((result_id, rt))
                }
                spv::Type::Vector {
                    component_type,
                    component_count,
                } => match *component_type {
                    spv::Type::Int {
                        width,
                        signedness: true,
                    } => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Int {
                                width,
                                signedness: true,
                            }),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::SRem {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, rt))
                    }
                    spv::Type::Int {
                        width,
                        signedness: false,
                    } => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Int {
                                width,
                                signedness: false,
                            }),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::UMod {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, rt))
                    }
                    spv::Type::Float { width } => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Float { width }),
                            component_count,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::FRem {
                            result_type,
                            result: result_id,
                            operand1: l,
                            operand2: r,
                        });
                        Some((result_id, rt))
                    }
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        }
        SimplifiedExpression::Cast(x, t) => {
            let (x, xt) = emit_expr_spv_ops(body, x.0, module_ctx, ctx).unwrap();

            match (xt, t) {
                (
                    spv::Type::Int {
                        width: 32,
                        signedness: true,
                    },
                    &ConcreteType::Intrinsic(IntrinsicType::Float),
                ) => {
                    let rt = spv::Type::Float { width: 32 };
                    let result_type = module_ctx.request_type_id(rt.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::ConvertSToF {
                        result_type,
                        result: result_id,
                        signed_value: x,
                    });
                    Some((result_id, rt))
                }
                (
                    spv::Type::Int {
                        width: 32,
                        signedness: false,
                    },
                    &ConcreteType::Intrinsic(IntrinsicType::Float),
                ) => {
                    let rt = spv::Type::Float { width: 32 };
                    let result_type = module_ctx.request_type_id(rt.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::ConvertUToF {
                        result_type,
                        result: result_id,
                        unsigned_value: x,
                    });
                    Some((result_id, rt))
                }
                (spv::Type::Float { width: 32 }, &ConcreteType::Intrinsic(IntrinsicType::SInt)) => {
                    let rt = spv::Type::Int {
                        width: 32,
                        signedness: true,
                    };
                    let result_type = module_ctx.request_type_id(rt.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::ConvertFToS {
                        result_type,
                        result: result_id,
                        float_value: x,
                    });
                    Some((result_id, rt))
                }
                (spv::Type::Float { width: 32 }, &ConcreteType::Intrinsic(IntrinsicType::UInt)) => {
                    let rt = spv::Type::Int {
                        width: 32,
                        signedness: false,
                    };
                    let result_type = module_ctx.request_type_id(rt.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::ConvertFToU {
                        result_type,
                        result: result_id,
                        float_value: x,
                    });
                    Some((result_id, rt))
                }
                (
                    spv::Type::Vector {
                        component_type,
                        component_count,
                    },
                    to,
                ) => match (*component_type, component_count, to) {
                    (
                        spv::Type::Int {
                            width: 32,
                            signedness: true,
                        },
                        2,
                        &ConcreteType::Intrinsic(IntrinsicType::Float2),
                    ) => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Float { width: 32 }),
                            component_count: 2,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::ConvertSToF {
                            result_type,
                            result: result_id,
                            signed_value: x,
                        });
                        Some((result_id, rt))
                    }
                    (
                        spv::Type::Int {
                            width: 32,
                            signedness: true,
                        },
                        3,
                        &ConcreteType::Intrinsic(IntrinsicType::Float3),
                    ) => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Float { width: 32 }),
                            component_count: 3,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::ConvertSToF {
                            result_type,
                            result: result_id,
                            signed_value: x,
                        });
                        Some((result_id, rt))
                    }
                    (
                        spv::Type::Int {
                            width: 32,
                            signedness: true,
                        },
                        4,
                        &ConcreteType::Intrinsic(IntrinsicType::Float4),
                    ) => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Float { width: 32 }),
                            component_count: 4,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::ConvertSToF {
                            result_type,
                            result: result_id,
                            signed_value: x,
                        });
                        Some((result_id, rt))
                    }
                    (
                        spv::Type::Int {
                            width: 32,
                            signedness: false,
                        },
                        2,
                        &ConcreteType::Intrinsic(IntrinsicType::Float2),
                    ) => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Float { width: 32 }),
                            component_count: 2,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::ConvertUToF {
                            result_type,
                            result: result_id,
                            unsigned_value: x,
                        });
                        Some((result_id, rt))
                    }
                    (
                        spv::Type::Int {
                            width: 32,
                            signedness: false,
                        },
                        3,
                        &ConcreteType::Intrinsic(IntrinsicType::Float3),
                    ) => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Float { width: 32 }),
                            component_count: 3,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::ConvertUToF {
                            result_type,
                            result: result_id,
                            unsigned_value: x,
                        });
                        Some((result_id, rt))
                    }
                    (
                        spv::Type::Int {
                            width: 32,
                            signedness: false,
                        },
                        4,
                        &ConcreteType::Intrinsic(IntrinsicType::Float4),
                    ) => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Float { width: 32 }),
                            component_count: 4,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::ConvertUToF {
                            result_type,
                            result: result_id,
                            unsigned_value: x,
                        });
                        Some((result_id, rt))
                    }
                    (
                        spv::Type::Float { width: 32 },
                        2,
                        &ConcreteType::Intrinsic(IntrinsicType::SInt2),
                    ) => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Int {
                                width: 32,
                                signedness: true,
                            }),
                            component_count: 2,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::ConvertFToS {
                            result_type,
                            result: result_id,
                            float_value: x,
                        });
                        Some((result_id, rt))
                    }
                    (
                        spv::Type::Float { width: 32 },
                        3,
                        &ConcreteType::Intrinsic(IntrinsicType::SInt3),
                    ) => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Int {
                                width: 32,
                                signedness: true,
                            }),
                            component_count: 3,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::ConvertFToS {
                            result_type,
                            result: result_id,
                            float_value: x,
                        });
                        Some((result_id, rt))
                    }
                    (
                        spv::Type::Float { width: 32 },
                        4,
                        &ConcreteType::Intrinsic(IntrinsicType::SInt4),
                    ) => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Int {
                                width: 32,
                                signedness: true,
                            }),
                            component_count: 4,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::ConvertFToS {
                            result_type,
                            result: result_id,
                            float_value: x,
                        });
                        Some((result_id, rt))
                    }
                    (
                        spv::Type::Float { width: 32 },
                        2,
                        &ConcreteType::Intrinsic(IntrinsicType::UInt2),
                    ) => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Int {
                                width: 32,
                                signedness: false,
                            }),
                            component_count: 2,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::ConvertFToU {
                            result_type,
                            result: result_id,
                            float_value: x,
                        });
                        Some((result_id, rt))
                    }
                    (
                        spv::Type::Float { width: 32 },
                        3,
                        &ConcreteType::Intrinsic(IntrinsicType::UInt3),
                    ) => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Int {
                                width: 32,
                                signedness: false,
                            }),
                            component_count: 3,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::ConvertFToU {
                            result_type,
                            result: result_id,
                            float_value: x,
                        });
                        Some((result_id, rt))
                    }
                    (
                        spv::Type::Float { width: 32 },
                        4,
                        &ConcreteType::Intrinsic(IntrinsicType::UInt4),
                    ) => {
                        let rt = spv::Type::Vector {
                            component_type: Box::new(spv::Type::Int {
                                width: 32,
                                signedness: false,
                            }),
                            component_count: 4,
                        };
                        let result_type = module_ctx.request_type_id(rt.clone());
                        let result_id = ctx.new_id();
                        ctx.ops.push(spv::Instruction::ConvertFToU {
                            result_type,
                            result: result_id,
                            float_value: x,
                        });
                        Some((result_id, rt))
                    }
                    _ => unreachable!(),
                },
                (xt, t) => unreachable!("undefined cast: {xt:?} -> {t:?}"),
            }
        }
        SimplifiedExpression::Swizzle1(x, a) => {
            let (x, xt) = emit_expr_spv_ops(body, x.0, module_ctx, ctx).unwrap();
            let spv::Type::Vector { component_type, .. } = xt else {
                unreachable!()
            };

            let rt = *component_type;
            let result_type = module_ctx.request_type_id(rt.clone());
            let result_id = ctx.new_id();
            ctx.ops.push(spv::Instruction::CompositeExtract {
                result_type,
                result: result_id,
                composite: x,
                indexes: vec![*a as _],
            });
            Some((result_id, rt))
        }
        SimplifiedExpression::Swizzle2(x, a, b) => {
            let (x, xt) = emit_expr_spv_ops(body, x.0, module_ctx, ctx).unwrap();
            let spv::Type::Vector { component_type, .. } = xt else {
                unreachable!()
            };

            let rt = spv::Type::Vector {
                component_type,
                component_count: 2,
            };
            let result_type = module_ctx.request_type_id(rt.clone());
            let result_id = ctx.new_id();
            ctx.ops.push(spv::Instruction::VectorShuffle {
                result_type,
                result: result_id,
                vector1: x,
                vector2: x,
                components: vec![*a as _, *b as _],
            });
            Some((result_id, rt))
        }
        SimplifiedExpression::Swizzle3(x, a, b, c) => {
            let (x, xt) = emit_expr_spv_ops(body, x.0, module_ctx, ctx).unwrap();
            let spv::Type::Vector { component_type, .. } = xt else {
                unreachable!()
            };

            let rt = spv::Type::Vector {
                component_type,
                component_count: 3,
            };
            let result_type = module_ctx.request_type_id(rt.clone());
            let result_id = ctx.new_id();
            ctx.ops.push(spv::Instruction::VectorShuffle {
                result_type,
                result: result_id,
                vector1: x,
                vector2: x,
                components: vec![*a as _, *b as _, *c as _],
            });
            Some((result_id, rt))
        }
        SimplifiedExpression::Swizzle4(x, a, b, c, d) => {
            let (x, xt) = emit_expr_spv_ops(body, x.0, module_ctx, ctx).unwrap();
            let spv::Type::Vector { component_type, .. } = xt else {
                unreachable!()
            };

            let rt = spv::Type::Vector {
                component_type,
                component_count: 4,
            };
            let result_type = module_ctx.request_type_id(rt.clone());
            let result_id = ctx.new_id();
            ctx.ops.push(spv::Instruction::VectorShuffle {
                result_type,
                result: result_id,
                vector1: x,
                vector2: x,
                components: vec![*a as _, *b as _, *c as _, *d as _],
            });
            Some((result_id, rt))
        }
        SimplifiedExpression::LoadByCanonicalRefPath(rp) => {
            match ctx.entry_point_maps.refpath_to_global_var.get(rp) {
                Some(GlobalAccessType::Direct(gv, vty)) => {
                    let (gv, vty) = (gv.clone(), vty.clone());

                    let result_type = module_ctx.request_type_id(vty.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::Load {
                        result_type,
                        result: result_id,
                        pointer: gv,
                    });
                    Some((result_id, vty))
                }
                Some(&GlobalAccessType::PushConstantStruct {
                    struct_var,
                    member_index,
                    ref member_ty,
                }) => {
                    let member_ty = member_ty.clone();

                    let ac_result_type = module_ctx.request_type_id(spv::Type::Pointer {
                        storage_class: spv::StorageClass::PushConstant,
                        base_type: Box::new(member_ty.clone()),
                    });
                    let ac_result_id = ctx.new_id();
                    let ac_index_id = module_ctx.request_const_id(spv::Constant::Constant {
                        result_type: spv::Type::Int {
                            width: 32,
                            signedness: false,
                        },
                        value_bits: member_index,
                    });
                    ctx.ops.push(spv::Instruction::AccessChain {
                        result_type: ac_result_type,
                        result: ac_result_id,
                        base: struct_var,
                        indexes: vec![ac_index_id],
                    });

                    let result_type = module_ctx.request_type_id(member_ty.clone());
                    let result_id = ctx.new_id();
                    ctx.ops.push(spv::Instruction::Load {
                        result_type,
                        result: result_id,
                        pointer: ac_result_id,
                    });
                    Some((result_id, member_ty))
                }
                None => panic!("no corresponding canonical refpath found? {rp:?}"),
            }
        }
        SimplifiedExpression::ConstSInt(s, mods) => {
            let mut x: i32 = match s
                .0
                .slice
                .strip_prefix("0x")
                .or_else(|| s.0.slice.strip_prefix("0X"))
            {
                Some(left) => i32::from_str_radix(left, 16).unwrap(),
                None => s.0.slice.parse().unwrap(),
            };
            if mods.contains(ConstModifiers::NEGATE) {
                x = -x;
            }
            if mods.contains(ConstModifiers::BIT_NOT) {
                x = !x;
            }
            if mods.contains(ConstModifiers::LOGICAL_NOT) {
                x = if x == 0 { 1 } else { 0 };
            }

            let result_id = module_ctx.request_const_id(spv::Constant::Constant {
                result_type: spv::Type::Int {
                    width: 32,
                    signedness: true,
                },
                value_bits: unsafe { core::mem::transmute(x) },
            });
            Some((
                result_id,
                spv::Type::Int {
                    width: 32,
                    signedness: true,
                },
            ))
        }
        SimplifiedExpression::ConstUInt(s, mods) => {
            let mut x: u32 = match s
                .0
                .slice
                .strip_prefix("0x")
                .or_else(|| s.0.slice.strip_prefix("0X"))
            {
                Some(left) => u32::from_str_radix(left, 16).unwrap(),
                None => s.0.slice.parse().unwrap(),
            };
            if mods.contains(ConstModifiers::NEGATE) {
                panic!("applying negate to unsigned number?");
            }
            if mods.contains(ConstModifiers::BIT_NOT) {
                x = !x;
            }
            if mods.contains(ConstModifiers::LOGICAL_NOT) {
                x = if x == 0 { 1 } else { 0 };
            }

            let result_id = module_ctx.request_const_id(spv::Constant::Constant {
                result_type: spv::Type::Int {
                    width: 32,
                    signedness: false,
                },
                value_bits: unsafe { core::mem::transmute(x) },
            });
            Some((
                result_id,
                spv::Type::Int {
                    width: 32,
                    signedness: false,
                },
            ))
        }
        SimplifiedExpression::ConstFloat(s, mods) => {
            let mut x: f32 =
                s.0.slice
                    .strip_suffix(['f', 'F'])
                    .unwrap_or(s.0.slice)
                    .parse()
                    .unwrap();
            if mods.contains(ConstModifiers::NEGATE) {
                x = -x;
            }
            if mods.contains(ConstModifiers::BIT_NOT) {
                x = unsafe { core::mem::transmute(!core::mem::transmute::<_, u32>(x)) };
            }
            if mods.contains(ConstModifiers::LOGICAL_NOT) {
                x = if x != 0.0 { 0.0 } else { 1.0 };
            }

            let result_id = module_ctx.request_const_id(spv::Constant::Constant {
                result_type: spv::Type::Float { width: 32 },
                value_bits: unsafe { core::mem::transmute(x) },
            });
            Some((result_id, spv::Type::Float { width: 32 }))
        }
        SimplifiedExpression::IntrinsicFuncall("Cloth.Intrinsic.SubpassLoad", _, args) => {
            let (args, arg_ty): (Vec<_>, Vec<_>) = args
                .iter()
                .map(|x| emit_expr_spv_ops(body, x.0, module_ctx, ctx).unwrap())
                .unzip();
            assert!(arg_ty.len() == 1 && arg_ty[0] == spv::Type::subpass_data_image_type());

            let rt = spv::Type::Vector {
                component_type: Box::new(spv::Type::Float { width: 32 }),
                component_count: 4,
            };
            let result_type = module_ctx.request_type_id(rt.clone());
            let result_id = ctx.new_id();
            let coordinate_const = module_ctx.request_const_id(spv::Constant::Composite {
                result_type: spv::Type::Vector {
                    component_type: Box::new(spv::Type::Int {
                        width: 32,
                        signedness: true,
                    }),
                    component_count: 2,
                },
                constituents: vec![
                    spv::Constant::Constant {
                        result_type: spv::Type::Int {
                            width: 32,
                            signedness: true,
                        },
                        value_bits: 0,
                    },
                    spv::Constant::Constant {
                        result_type: spv::Type::Int {
                            width: 32,
                            signedness: true,
                        },
                        value_bits: 0,
                    },
                ],
            });
            ctx.ops.push(spv::Instruction::ImageRead {
                result_type,
                result: result_id,
                image: args[0],
                coordinate: coordinate_const,
            });
            Some((result_id, rt))
        }
        SimplifiedExpression::StoreOutput(x, o) => {
            let (x, _xt) = emit_expr_spv_ops(body, x.0, module_ctx, ctx).unwrap();
            ctx.ops.push(spv::Instruction::Store {
                pointer: ctx.entry_point_maps.output_global_vars[*o],
                object: x,
            });

            None
        }
        SimplifiedExpression::ScopedBlock {
            expressions,
            returning,
            ..
        } => emit_expr_spv_ops(expressions, returning.0, module_ctx, ctx),
        x => unimplemented!("{x:?}"),
    };

    ctx.emitted_expression_id.insert(expr_id, result.clone());
    result
}

pub struct SpvModuleEmissionContext {
    pub header_ops: Vec<spv::Instruction<SpvSectionLocalId>>,
    pub entry_point_ops: Vec<spv::Instruction<SpvSectionLocalId>>,
    pub execution_mode_ops: Vec<spv::Instruction<SpvSectionLocalId>>,
    pub annotation_ops: Vec<spv::Instruction<SpvSectionLocalId>>,
    pub type_const_ops: Vec<spv::Instruction<SpvSectionLocalId>>,
    pub global_variable_ops: Vec<spv::Instruction<SpvSectionLocalId>>,
    pub function_ops: Vec<spv::Instruction<SpvSectionLocalId>>,
    pub latest_type_const_id: spv::Id,
    pub latest_global_variable_id: spv::Id,
    pub latest_function_id: spv::Id,
    pub defined_const_map: HashMap<spv::Constant, SpvSectionLocalId>,
    pub defined_type_map: HashMap<spv::Type, SpvSectionLocalId>,
}
impl SpvModuleEmissionContext {
    pub fn new() -> Self {
        Self {
            header_ops: Vec::new(),
            entry_point_ops: Vec::new(),
            execution_mode_ops: Vec::new(),
            annotation_ops: Vec::new(),
            type_const_ops: Vec::new(),
            global_variable_ops: Vec::new(),
            function_ops: Vec::new(),
            latest_type_const_id: 0,
            latest_global_variable_id: 0,
            latest_function_id: 0,
            defined_const_map: HashMap::new(),
            defined_type_map: HashMap::new(),
        }
    }

    pub fn serialize_ops(self) -> (Vec<spv::Instruction>, spv::Id) {
        let type_const_offset = 1;
        let global_variable_offset = type_const_offset + self.latest_type_const_id;
        let function_offset = global_variable_offset + self.latest_global_variable_id;

        let mut max_id = 1;
        let ops = self
            .header_ops
            .into_iter()
            .chain(self.entry_point_ops.into_iter())
            .chain(self.execution_mode_ops.into_iter())
            .chain(self.annotation_ops.into_iter())
            .chain(self.type_const_ops.into_iter())
            .chain(self.global_variable_ops.into_iter())
            .chain(self.function_ops.into_iter())
            .map(|x| {
                x.relocate(|id| {
                    let nid = match id {
                        SpvSectionLocalId::TypeConst(x) => x + type_const_offset,
                        SpvSectionLocalId::GlobalVariable(x) => x + global_variable_offset,
                        SpvSectionLocalId::Function(x) => x + function_offset,
                        SpvSectionLocalId::CurrentFunction(_) => unreachable!(),
                    };
                    max_id = max_id.max(nid);
                    nid
                })
            })
            .collect();
        (ops, max_id)
    }

    pub fn new_type_const_id(&mut self) -> SpvSectionLocalId {
        self.latest_type_const_id += 1;

        SpvSectionLocalId::TypeConst(self.latest_type_const_id - 1)
    }

    pub fn new_global_variable_id(&mut self) -> SpvSectionLocalId {
        self.latest_global_variable_id += 1;

        SpvSectionLocalId::GlobalVariable(self.latest_global_variable_id - 1)
    }

    pub fn new_function_id(&mut self) -> SpvSectionLocalId {
        self.latest_function_id += 1;

        SpvSectionLocalId::Function(self.latest_function_id - 1)
    }

    pub fn request_const_id(&mut self, c: spv::Constant) -> SpvSectionLocalId {
        if let Some(id) = self.defined_const_map.get(&c) {
            return *id;
        }

        let id = match c {
            spv::Constant::True { ref result_type } => {
                let result_type = self.request_type_id(result_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::ConstantTrue {
                    result_type,
                    result: id,
                });
                id
            }
            spv::Constant::False { ref result_type } => {
                let result_type = self.request_type_id(result_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::ConstantFalse {
                    result_type,
                    result: id,
                });
                id
            }
            spv::Constant::Constant {
                ref result_type,
                value_bits,
            } => {
                let result_type = self.request_type_id(result_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::Constant {
                    result_type,
                    result: id,
                    value_bits,
                });
                id
            }
            spv::Constant::Composite {
                ref result_type,
                ref constituents,
            } => {
                let result_type = self.request_type_id(result_type.clone());
                let constituents = constituents
                    .iter()
                    .map(|x| self.request_const_id(x.clone()))
                    .collect();
                let id = self.new_type_const_id();
                self.type_const_ops
                    .push(spv::Instruction::ConstantComposite {
                        result_type,
                        result: id,
                        constituents,
                    });
                id
            }
            spv::Constant::Sampler {
                ref result_type,
                sampler_addressing_mode,
                normalized,
                sampler_filter_mode,
            } => {
                let result_type = self.request_type_id(result_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::ConstantSampler {
                    result_type,
                    result: id,
                    sampler_addressing_mode,
                    normalized,
                    sampler_filter_mode,
                });
                id
            }
            spv::Constant::Null { ref result_type } => {
                let result_type = self.request_type_id(result_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::ConstantNull {
                    result_type,
                    result: id,
                });
                id
            }
            spv::Constant::Undef { ref result_type } => {
                let result_type = self.request_type_id(result_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::Undef {
                    result_type,
                    result: id,
                });
                id
            }
        };

        self.defined_const_map.insert(c, id);
        id
    }

    pub fn request_type_id(&mut self, t: spv::Type) -> SpvSectionLocalId {
        if let Some(id) = self.defined_type_map.get(&t) {
            return *id;
        }

        match t {
            spv::Type::Void => {
                let id = self.new_type_const_id();
                self.type_const_ops
                    .push(spv::Instruction::TypeVoid { result: id });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Bool => {
                let id = self.new_type_const_id();
                self.type_const_ops
                    .push(spv::Instruction::TypeBool { result: id });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Int { width, signedness } => {
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::TypeInt {
                    result: id,
                    width,
                    signedness,
                });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Float { width } => {
                let id = self.new_type_const_id();
                self.type_const_ops
                    .push(spv::Instruction::TypeFloat { result: id, width });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Vector {
                ref component_type,
                component_count,
            } => {
                let component_type = self.request_type_id(*component_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::TypeVector {
                    result: id,
                    component_type,
                    component_count,
                });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Matrix {
                ref column_type,
                column_count,
            } => {
                let column_type = self.request_type_id(*column_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::TypeMatrix {
                    result: id,
                    column_type,
                    column_count,
                });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Image {
                ref sampled_type,
                dim,
                depth,
                arrayed,
                multisampled,
                sampled,
                image_format,
                access_qualifier,
            } => {
                let sampled_type = self.request_type_id(*sampled_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::TypeImage {
                    result: id,
                    sampled_type,
                    dim,
                    depth,
                    arrayed,
                    multisampled,
                    sampled,
                    image_format,
                    access_qualifier,
                });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Sampler => {
                let id = self.new_type_const_id();
                self.type_const_ops
                    .push(spv::Instruction::TypeSampler { result: id });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::SampledImage { ref image_type } => {
                let image_type = self.request_type_id(*image_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops
                    .push(spv::Instruction::TypeSampledImage {
                        result: id,
                        image_type,
                    });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Array {
                ref element_type,
                ref length,
            } => {
                let element_type = self.request_type_id(*element_type.clone());
                let id = self.new_type_const_id();
                let length = match **length {
                    spv::TypeArrayLength::SpecConstantID(x) => SpvSectionLocalId::TypeConst(x),
                    spv::TypeArrayLength::ConstExpr(ref x) => self.request_const_id(x.clone()),
                };

                self.type_const_ops.push(spv::Instruction::TypeArray {
                    result: id,
                    element_type,
                    length,
                });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::RuntimeArray { ref element_type } => {
                let element_type = self.request_type_id(*element_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops
                    .push(spv::Instruction::TypeRuntimeArray {
                        result: id,
                        element_type,
                    });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Struct { ref member_types } => {
                let member_type_ids = member_types
                    .iter()
                    .map(|x| self.request_type_id(x.ty.clone()))
                    .collect();
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::TypeStruct {
                    result: id,
                    member_types: member_type_ids,
                });

                for (n, x) in member_types.iter().enumerate() {
                    self.annotation_ops
                        .extend(x.decorations.iter().map(|d| match d {
                            spv::Decorate::Offset(x) => spv::Instruction::MemberDecorate {
                                struct_type: id,
                                member: n as _,
                                decoration: spv::Decoration::Offset,
                                args: vec![*x],
                            },
                            _ => unimplemented!(),
                        }));
                }

                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Opaque { ref name } => {
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::TypeOpaque {
                    result: id,
                    name: name.clone(),
                });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Pointer {
                storage_class,
                ref base_type,
            } => {
                let base_type = self.request_type_id(*base_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::TypePointer {
                    result: id,
                    storage_class,
                    base_type,
                });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::Function {
                ref return_type,
                ref parameter_types,
            } => {
                let return_type = self.request_type_id(*return_type.clone());
                let parameter_types = parameter_types
                    .iter()
                    .map(|x| self.request_type_id(x.clone()))
                    .collect();
                let id = self.new_type_const_id();
                self.type_const_ops.push(spv::Instruction::TypeFunction {
                    result: id,
                    return_type,
                    parameter_types,
                });
                self.defined_type_map.insert(t, id);
                id
            }
            spv::Type::ForwardPointer {
                ref pointer_type,
                storage_class,
            } => {
                let pointer_type = self.request_type_id(*pointer_type.clone());
                let id = self.new_type_const_id();
                self.type_const_ops
                    .push(spv::Instruction::TypeForwardPointer {
                        result: id,
                        pointer_type,
                        storage_class,
                    });
                self.defined_type_map.insert(t, id);
                id
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct IntrinsicFunctionSymbol {
    pub name: &'static str,
    pub ty: ConcreteType<'static>,
    pub is_pure: bool,
}

#[derive(Debug, Clone)]
pub struct FunctionInputVariable<'s> {
    pub occurence: SourceRef<'s>,
    pub ty: ConcreteType<'s>,
}

#[derive(Debug, Clone)]
pub struct LocalVariable<'s> {
    pub occurence: SourceRef<'s>,
    pub ty: ConcreteType<'s>,
    pub init_expr_id: ExprRef,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VarId {
    FunctionInput(usize),
    ScopeLocal(usize),
    IntrinsicTypeConstructor(IntrinsicType),
}

#[derive(Debug, Clone)]
pub enum VarLookupResult<'x, 's> {
    FunctionInputVar(usize, &'x ConcreteType<'s>),
    ScopeLocalVar(usize, ConcreteType<'s>),
    IntrinsicFunction(&'x IntrinsicFunctionSymbol),
    IntrinsicTypeConstructor(IntrinsicType),
}

#[derive(Debug, Clone)]
pub struct UserDefinedFunctionSymbol<'s> {
    pub occurence: SourceRef<'s>,
    pub attribute: SymbolAttribute,
    pub inputs: Vec<(SymbolAttribute, SourceRef<'s>, ConcreteType<'s>)>,
    pub output: Vec<(SymbolAttribute, ConcreteType<'s>)>,
}

#[derive(Debug, Clone)]
pub struct FunctionBody<'a, 's> {
    pub symbol_scope: &'a SymbolScope2<'a, 's>,
    pub expressions: Vec<(SimplifiedExpression<'a, 's>, ConcreteType<'s>)>,
    pub returning: ExprRef,
    pub returning_type: ConcreteType<'s>,
}

#[derive(Debug, Clone)]
pub struct SymbolScope2<'a, 's> {
    parent: Option<&'a SymbolScope2<'a, 's>>,
    is_toplevel_function: bool,
    intrinsic_symbols: HashMap<&'s str, IntrinsicFunctionSymbol>,
    user_defined_type_symbols: HashMap<&'s str, (SourceRef<'s>, UserDefinedType<'s>)>,
    user_defined_function_symbols: HashMap<&'s str, UserDefinedFunctionSymbol<'s>>,
    user_defined_function_body: DebugPrintGuard<RefCell<HashMap<&'s str, FunctionBody<'a, 's>>>>,
    function_input_vars: Vec<FunctionInputVariable<'s>>,
    local_vars: RefCell<Vec<LocalVariable<'s>>>,
    var_id_by_name: RefCell<HashMap<&'s str, VarId>>,
}
impl<'a, 's> SymbolScope2<'a, 's> {
    pub fn new(parent: Option<&'a SymbolScope2<'a, 's>>, is_toplevel_function: bool) -> Self {
        Self {
            parent,
            is_toplevel_function,
            intrinsic_symbols: HashMap::new(),
            user_defined_type_symbols: HashMap::new(),
            user_defined_function_symbols: HashMap::new(),
            user_defined_function_body: DebugPrintGuard(RefCell::new(HashMap::new())),
            function_input_vars: Vec::new(),
            local_vars: RefCell::new(Vec::new()),
            var_id_by_name: RefCell::new(HashMap::new()),
        }
    }

    pub fn new_intrinsics() -> Self {
        let mut intrinsic_symbols = HashMap::new();
        let mut var_id_by_name = HashMap::new();

        intrinsic_symbols.insert(
            "subpassLoad",
            IntrinsicFunctionSymbol {
                name: "Cloth.Intrinsic.SubpassLoad",
                ty: ConcreteType::Function {
                    args: vec![ConcreteType::Intrinsic(IntrinsicType::SubpassInput)],
                    output: Some(Box::new(ConcreteType::Intrinsic(IntrinsicType::Float4))),
                },
                is_pure: true,
            },
        );
        var_id_by_name.insert(
            "Float4",
            VarId::IntrinsicTypeConstructor(IntrinsicType::Float4),
        );

        Self {
            parent: None,
            is_toplevel_function: false,
            intrinsic_symbols,
            user_defined_type_symbols: HashMap::new(),
            user_defined_function_symbols: HashMap::new(),
            user_defined_function_body: DebugPrintGuard(RefCell::new(HashMap::new())),
            function_input_vars: Vec::new(),
            local_vars: RefCell::new(Vec::new()),
            var_id_by_name: RefCell::new(var_id_by_name),
        }
    }

    pub fn merge_local_vars(&self, from: &'a Self) -> usize {
        let offset = self.local_vars.borrow().len();
        self.local_vars
            .borrow_mut()
            .extend(from.local_vars.borrow_mut().drain(..));
        self.var_id_by_name
            .borrow_mut()
            .extend(from.var_id_by_name.borrow_mut().drain().map(|(k, v)| {
                (
                    k,
                    match v {
                        VarId::ScopeLocal(x) => VarId::ScopeLocal(x + offset),
                        x => x,
                    },
                )
            }));

        offset
    }

    pub fn declare_function(&mut self, details: UserDefinedFunctionSymbol<'s>) {
        match self
            .user_defined_function_symbols
            .entry(details.occurence.slice)
        {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(details);
            }
            std::collections::hash_map::Entry::Occupied(v) => {
                panic!(
                    "Error: same name function was already declared at {}:{}",
                    v.get().occurence.line + 1,
                    v.get().occurence.col + 1
                );
            }
        }
    }

    pub fn attach_function_body(&self, fname: &'s str, body: FunctionBody<'a, 's>) {
        match self.user_defined_function_body.0.borrow_mut().entry(fname) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(body);
            }
            std::collections::hash_map::Entry::Occupied(_) => {
                panic!("Error: same name function body was already declared");
            }
        }
    }

    pub fn declare_function_input(&mut self, name: SourceRef<'s>, ty: ConcreteType<'s>) -> VarId {
        match self.var_id_by_name.borrow_mut().entry(name.slice) {
            std::collections::hash_map::Entry::Vacant(v) => {
                self.function_input_vars.push(FunctionInputVariable {
                    occurence: name.clone(),
                    ty,
                });
                let vid = VarId::FunctionInput(self.function_input_vars.len() - 1);
                v.insert(vid);
                vid
            }
            std::collections::hash_map::Entry::Occupied(_) => {
                panic!("Function Input {} is already declared", name.slice);
            }
        }
    }

    pub fn declare_local_var(
        &self,
        name: SourceRef<'s>,
        ty: ConcreteType<'s>,
        init_expr: ExprRef,
    ) -> VarId {
        self.local_vars.borrow_mut().push(LocalVariable {
            occurence: name.clone(),
            ty,
            init_expr_id: init_expr,
        });
        let vid = VarId::ScopeLocal(self.local_vars.borrow().len() - 1);
        self.var_id_by_name.borrow_mut().insert(name.slice, vid);
        vid
    }

    pub fn lookup_intrinsic_function(&self, name: &'s str) -> Option<&IntrinsicFunctionSymbol> {
        match self.intrinsic_symbols.get(name) {
            Some(t) => Some(t),
            None => match self.parent {
                Some(ref p) => p.lookup_intrinsic_function(name),
                None => None,
            },
        }
    }

    pub fn lookup<'x>(&'x self, name: &str) -> Option<(&Self, VarLookupResult<'x, 's>)> {
        if let Some(x) = self.intrinsic_symbols.get(name) {
            return Some((self, VarLookupResult::IntrinsicFunction(x)));
        }

        match self.var_id_by_name.borrow().get(name) {
            Some(VarId::FunctionInput(x)) => Some((
                self,
                VarLookupResult::FunctionInputVar(*x, &self.function_input_vars[*x].ty),
            )),
            Some(VarId::ScopeLocal(x)) => Some((
                self,
                VarLookupResult::ScopeLocalVar(*x, self.local_vars.borrow()[*x].ty.clone()),
            )),
            Some(VarId::IntrinsicTypeConstructor(x)) => {
                Some((self, VarLookupResult::IntrinsicTypeConstructor(*x)))
            }
            None => match self.parent {
                Some(ref p) => p.lookup(name),
                None => None,
            },
        }
    }

    pub fn lookup_user_defined_type(
        &self,
        name: &'s str,
    ) -> Option<(&Self, &(SourceRef<'s>, UserDefinedType<'s>))> {
        match self.user_defined_type_symbols.get(name) {
            Some(p) => Some((self, p)),
            None => match self.parent {
                Some(ref p) => p.lookup_user_defined_type(name),
                None => None,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderModel {
    VertexShader,
    TessellationControlShader,
    TessellationEvaluationShader,
    GeometryShader,
    FragmentShader,
    ComputeShader,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuiltinInputOutput {
    Position,
    VertexID,
    InstanceID,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SymbolAttribute {
    pub module_entry_point: bool,
    pub shader_model: Option<ShaderModel>,
    pub descriptor_set_location: Option<u32>,
    pub descriptor_set_binding: Option<u32>,
    pub input_attachment_index: Option<u32>,
    pub push_constant_offset: Option<u32>,
    pub bound_location: Option<u32>,
    pub bound_builtin_io: Option<BuiltinInputOutput>,
}
impl Default for SymbolAttribute {
    fn default() -> Self {
        Self {
            module_entry_point: false,
            shader_model: None,
            descriptor_set_location: None,
            descriptor_set_binding: None,
            input_attachment_index: None,
            push_constant_offset: None,
            bound_location: None,
            bound_builtin_io: None,
        }
    }
}
impl SymbolAttribute {
    pub const fn is_empty(&self) -> bool {
        matches!(
            self,
            SymbolAttribute {
                module_entry_point: false,
                shader_model: None,
                descriptor_set_location: None,
                descriptor_set_binding: None,
                input_attachment_index: None,
                push_constant_offset: None,
                bound_location: None,
                bound_builtin_io: None
            }
        )
    }
}
fn eval_symbol_attributes(attr: SymbolAttribute, a: Attribute) -> SymbolAttribute {
    let args = match a.arg {
        Some(AttributeArg::Single(x)) => vec![(x, None)],
        Some(AttributeArg::Multiple { arg_list, .. }) => arg_list,
        None => Vec::new(),
    };

    // TODO: User-defined Attributeとかシンボルエイリアスとかをサポートするようになったら真面目に型解決して処理する必要がある
    match a.name_token.slice {
        "DescriptorSet" => match <&[(ConstExpression, _); 1]>::try_from(&args[..]) {
            Ok([(ConstExpression::Number(n), _)]) => SymbolAttribute {
                descriptor_set_location: Some(n.slice.parse().unwrap()),
                ..attr
            },
            Err(_) => panic!("DescriptorSet attribute requires a number as argument"),
        },
        "Binding" => match <&[(ConstExpression, _); 1]>::try_from(&args[..]) {
            Ok([(ConstExpression::Number(n), _)]) => SymbolAttribute {
                descriptor_set_binding: Some(n.slice.parse().unwrap()),
                ..attr
            },
            Err(_) => panic!("Binding attribute requires a number as argument"),
        },
        "InputAttachment" => match <&[(ConstExpression, _); 1]>::try_from(&args[..]) {
            Ok([(ConstExpression::Number(n), _)]) => SymbolAttribute {
                input_attachment_index: Some(n.slice.parse().unwrap()),
                ..attr
            },
            Err(_) => panic!("InputAttachment attribute requires a number as argument"),
        },
        "PushConstant" => match <&[(ConstExpression, _); 1]>::try_from(&args[..]) {
            Ok([(ConstExpression::Number(n), _)]) => SymbolAttribute {
                push_constant_offset: Some(n.slice.parse().unwrap()),
                ..attr
            },
            Err(_) => panic!("PushConstant attribute requires a number as argument"),
        },
        "Location" => match <&[(ConstExpression, _); 1]>::try_from(&args[..]) {
            Ok([(ConstExpression::Number(n), _)]) => SymbolAttribute {
                bound_location: Some(n.slice.parse().unwrap()),
                ..attr
            },
            Err(_) => panic!("Location attribute requires a number as argument"),
        },
        "Position" => {
            if args.is_empty() {
                SymbolAttribute {
                    bound_builtin_io: Some(BuiltinInputOutput::Position),
                    ..attr
                }
            } else {
                panic!("Position attribute does not take any arguments");
            }
        }
        "VertexID" => {
            if args.is_empty() {
                SymbolAttribute {
                    bound_builtin_io: Some(BuiltinInputOutput::VertexID),
                    ..attr
                }
            } else {
                panic!("VertexID attribute does not take any arguments");
            }
        }
        "InstanceID" => {
            if args.is_empty() {
                SymbolAttribute {
                    bound_builtin_io: Some(BuiltinInputOutput::InstanceID),
                    ..attr
                }
            } else {
                panic!("InstanceID attribute does not take any arguments");
            }
        }
        "VertexShader" => {
            if args.is_empty() {
                SymbolAttribute {
                    module_entry_point: true,
                    shader_model: Some(ShaderModel::VertexShader),
                    ..attr
                }
            } else {
                panic!("VertexShader attribute does not take any arguments");
            }
        }
        "FragmentShader" => {
            if args.is_empty() {
                SymbolAttribute {
                    module_entry_point: true,
                    shader_model: Some(ShaderModel::FragmentShader),
                    ..attr
                }
            } else {
                panic!("VertexShader attribute does not take any arguments");
            }
        }
        _ => panic!("{}: Unknown attribute", a.name_token.slice),
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ConstModifiers: u8 {
        const NEGATE = 1 << 0;
        const BIT_NOT = 1 << 1;
        const LOGICAL_NOT = 1 << 2;
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DebugPrintGuard<T>(pub T);
impl<T> core::fmt::Debug for DebugPrintGuard<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(masked {}...)", core::any::type_name::<T>())
    }
}

#[repr(transparent)]
pub struct PtrEq<'a, T: 'a + ?Sized>(pub &'a T);
impl<'a, T: 'a + ?Sized> Clone for PtrEq<'a, T> {
    fn clone(&self) -> Self {
        PtrEq(self.0)
    }
}
impl<'a, T: 'a + ?Sized> Copy for PtrEq<'a, T> {}
impl<'a, T: 'a + ?Sized> core::cmp::PartialEq for PtrEq<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        core::ptr::eq(self.0, other.0)
    }
}
impl<'a, T: 'a + ?Sized> core::cmp::Eq for PtrEq<'a, T> {}
impl<'a, T: 'a + ?Sized> core::hash::Hash for PtrEq<'a, T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.0 as *const T).hash(state)
    }
}
impl<'a, T: 'a + ?Sized> core::fmt::Debug for PtrEq<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ptr<{}>({:p})", core::any::type_name::<T>(), self.0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct ExprRef(pub usize);
impl core::fmt::Debug for ExprRef {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SimplifiedExpression<'a, 's> {
    Add(ExprRef, ExprRef),
    Sub(ExprRef, ExprRef),
    Mul(ExprRef, ExprRef),
    Div(ExprRef, ExprRef),
    Rem(ExprRef, ExprRef),
    BitAnd(ExprRef, ExprRef),
    BitOr(ExprRef, ExprRef),
    BitXor(ExprRef, ExprRef),
    Eq(ExprRef, ExprRef),
    Ne(ExprRef, ExprRef),
    Gt(ExprRef, ExprRef),
    Ge(ExprRef, ExprRef),
    Lt(ExprRef, ExprRef),
    Le(ExprRef, ExprRef),
    LogAnd(ExprRef, ExprRef),
    LogOr(ExprRef, ExprRef),
    Neg(ExprRef),
    BitNot(ExprRef),
    LogNot(ExprRef),
    Funcall(ExprRef, Vec<ExprRef>),
    MemberRef(ExprRef, SourceRefSliceEq<'s>),
    LoadVar(PtrEq<'a, SymbolScope2<'a, 's>>, VarId),
    InitializeVar(PtrEq<'a, SymbolScope2<'a, 's>>, VarId),
    StoreLocal(SourceRefSliceEq<'s>, ExprRef),
    LoadByCanonicalRefPath(RefPath<'s>),
    IntrinsicFunction(&'static str, bool),
    IntrinsicTypeConstructor(IntrinsicType),
    IntrinsicFuncall(&'static str, bool, Vec<ExprRef>),
    Select(ExprRef, ExprRef, ExprRef),
    Cast(ExprRef, ConcreteType<'s>),
    Swizzle1(ExprRef, usize),
    Swizzle2(ExprRef, usize, usize),
    Swizzle3(ExprRef, usize, usize, usize),
    Swizzle4(ExprRef, usize, usize, usize, usize),
    InstantiateIntrinsicTypeClass(ExprRef, IntrinsicType),
    ConstInt(SourceRefSliceEq<'s>),
    ConstNumber(SourceRefSliceEq<'s>),
    ConstUnit,
    ConstUInt(SourceRefSliceEq<'s>, ConstModifiers),
    ConstSInt(SourceRefSliceEq<'s>, ConstModifiers),
    ConstFloat(SourceRefSliceEq<'s>, ConstModifiers),
    ConstructTuple(Vec<ExprRef>),
    ConstructIntrinsicComposite(IntrinsicType, Vec<ExprRef>),
    ScopedBlock {
        symbol_scope: PtrEq<'a, SymbolScope2<'a, 's>>,
        expressions: Vec<(SimplifiedExpression<'a, 's>, ConcreteType<'s>)>,
        returning: ExprRef,
    },
    StoreOutput(ExprRef, usize),
    DistributeOutputTuple(ExprRef, Vec<usize>),
}
impl SimplifiedExpression<'_, '_> {
    pub fn is_pure(&self) -> bool {
        match self {
            Self::Funcall(_, _)
            | Self::InitializeVar(_, _)
            | Self::StoreLocal(_, _)
            | Self::ScopedBlock { .. }
            | Self::StoreOutput(_, _)
            | Self::DistributeOutputTuple(_, _) => false,
            &Self::IntrinsicFuncall(_, is_pure, _) => is_pure,
            _ => true,
        }
    }

    pub fn relocate_ref(&mut self, relocator: impl Fn(usize) -> usize) -> bool {
        match self {
            Self::Add(ref mut l, ref mut r)
            | Self::Sub(ref mut l, ref mut r)
            | Self::Mul(ref mut l, ref mut r)
            | Self::Div(ref mut l, ref mut r)
            | Self::Rem(ref mut l, ref mut r)
            | Self::BitAnd(ref mut l, ref mut r)
            | Self::BitOr(ref mut l, ref mut r)
            | Self::BitXor(ref mut l, ref mut r)
            | Self::Eq(ref mut l, ref mut r)
            | Self::Ne(ref mut l, ref mut r)
            | Self::Gt(ref mut l, ref mut r)
            | Self::Lt(ref mut l, ref mut r)
            | Self::Ge(ref mut l, ref mut r)
            | Self::Le(ref mut l, ref mut r)
            | Self::LogAnd(ref mut l, ref mut r)
            | Self::LogOr(ref mut l, ref mut r) => {
                let (l1, r1) = (l.0, r.0);

                l.0 = relocator(l.0);
                r.0 = relocator(r.0);
                l.0 != l1 || r.0 != r1
            }
            Self::Select(ref mut c, ref mut t, ref mut e) => {
                let (c1, t1, e1) = (c.0, t.0, e.0);

                c.0 = relocator(c.0);
                t.0 = relocator(t.0);
                e.0 = relocator(e.0);
                c.0 != c1 || t.0 != t1 || e.0 != e1
            }
            Self::Neg(ref mut x)
            | Self::BitNot(ref mut x)
            | Self::LogNot(ref mut x)
            | Self::Cast(ref mut x, _)
            | Self::Swizzle1(ref mut x, _)
            | Self::Swizzle2(ref mut x, _, _)
            | Self::Swizzle3(ref mut x, _, _, _)
            | Self::Swizzle4(ref mut x, _, _, _, _)
            | Self::StoreLocal(_, ref mut x)
            | Self::InstantiateIntrinsicTypeClass(ref mut x, _)
            | Self::MemberRef(ref mut x, _)
            | Self::StoreOutput(ref mut x, _)
            | Self::DistributeOutputTuple(ref mut x, _) => {
                let x1 = x.0;

                x.0 = relocator(x.0);
                x.0 != x1
            }
            Self::Funcall(ref mut base, ref mut args) => {
                let base1 = base.0;
                base.0 = relocator(base.0);

                let mut dirty = base.0 != base1;
                for a in args {
                    let a1 = a.0;
                    a.0 = relocator(a.0);
                    dirty |= a.0 != a1;
                }
                dirty
            }
            Self::ConstructTuple(ref mut xs)
            | Self::ConstructIntrinsicComposite(_, ref mut xs)
            | Self::IntrinsicFuncall(_, _, ref mut xs) => {
                let mut dirty = false;
                for x in xs {
                    let x1 = x.0;
                    x.0 = relocator(x.0);
                    dirty |= x.0 != x1;
                }
                dirty
            }
            Self::ConstInt(_)
            | Self::ConstNumber(_)
            | Self::ConstUnit
            | Self::ConstUInt(_, _)
            | Self::ConstSInt(_, _)
            | Self::ConstFloat(_, _)
            | Self::LoadVar(_, _)
            | Self::InitializeVar(_, _)
            | Self::LoadByCanonicalRefPath(_)
            | Self::IntrinsicFunction(_, _)
            | Self::IntrinsicTypeConstructor(_)
            | Self::ScopedBlock { .. } => false,
        }
    }
}
pub struct SimplificationContext<'a, 's> {
    pub symbol_scope_arena: &'a Arena<SymbolScope2<'a, 's>>,
    pub vars: Vec<(SimplifiedExpression<'a, 's>, ConcreteType<'s>)>,
}
impl<'a, 's> SimplificationContext<'a, 's> {
    pub fn add(&mut self, expr: SimplifiedExpression<'a, 's>, ty: ConcreteType<'s>) -> ExprRef {
        self.vars.push((expr, ty));

        ExprRef(self.vars.len() - 1)
    }

    pub fn type_of(&self, expr_id: usize) -> Option<&ConcreteType<'s>> {
        self.vars.get(expr_id).map(|(_, t)| t)
    }
}
fn simplify_expression<'a, 's>(
    ast: Expression<'s>,
    ctx: &mut SimplificationContext<'a, 's>,
    symbol_scope: &'a SymbolScope2<'a, 's>,
) -> (ExprRef, ConcreteType<'s>) {
    match ast {
        Expression::Binary(left, op, right) => {
            let (left, lt) = simplify_expression(*left, ctx, symbol_scope);
            let (right, rt) = simplify_expression(*right, ctx, symbol_scope);

            let r = match op.slice {
                "+" | "-" | "*" | "/" | "%" => lt.arithmetic_compare_op_type_conversion(rt),
                // 比較演算の出力は必ずBoolになる
                "==" | "!=" | "<=" | ">=" | "<" | ">" => lt
                    .arithmetic_compare_op_type_conversion(rt)
                    .map(|(conv, _)| (conv, IntrinsicType::Bool.into())),
                "&" | "|" | "^" => lt.bitwise_op_type_conversion(rt),
                "&&" | "||" => lt.logical_op_type_conversion(rt),
                _ => None,
            };
            let (conv, res) = match r {
                Some(x) => x,
                None => {
                    eprintln!("Error: cannot apply binary op {} between terms", op.slice);
                    (BinaryOpTypeConversion::NoConversion, ConcreteType::Never)
                }
            };

            let (left, right) = match conv {
                BinaryOpTypeConversion::NoConversion => (left, right),
                BinaryOpTypeConversion::CastLeftHand => {
                    let left = ctx.add(SimplifiedExpression::Cast(left, res.clone()), res.clone());

                    (left, right)
                }
                BinaryOpTypeConversion::CastRightHand => {
                    let right =
                        ctx.add(SimplifiedExpression::Cast(right, res.clone()), res.clone());

                    (left, right)
                }
                BinaryOpTypeConversion::InstantiateAndCastLeftHand(it) => {
                    let left = ctx.add(
                        SimplifiedExpression::InstantiateIntrinsicTypeClass(left, it),
                        it.into(),
                    );
                    let left = ctx.add(SimplifiedExpression::Cast(left, res.clone()), res.clone());

                    (left, right)
                }
                BinaryOpTypeConversion::InstantiateAndCastRightHand(it) => {
                    let right = ctx.add(
                        SimplifiedExpression::InstantiateIntrinsicTypeClass(right, it),
                        it.into(),
                    );
                    let right =
                        ctx.add(SimplifiedExpression::Cast(right, res.clone()), res.clone());

                    (left, right)
                }
                BinaryOpTypeConversion::InstantiateLeftHand(it) => {
                    let left = ctx.add(
                        SimplifiedExpression::InstantiateIntrinsicTypeClass(left, it),
                        it.into(),
                    );

                    (left, right)
                }
                BinaryOpTypeConversion::InstantiateRightHand(it) => {
                    let right = ctx.add(
                        SimplifiedExpression::InstantiateIntrinsicTypeClass(right, it),
                        it.into(),
                    );

                    (left, right)
                }
                BinaryOpTypeConversion::InstantiateLeftAndCastRightHand(it) => {
                    let left = ctx.add(
                        SimplifiedExpression::InstantiateIntrinsicTypeClass(left, it),
                        it.into(),
                    );
                    let right =
                        ctx.add(SimplifiedExpression::Cast(right, res.clone()), res.clone());

                    (left, right)
                }
                BinaryOpTypeConversion::InstantiateRightAndCastLeftHand(it) => {
                    let right = ctx.add(
                        SimplifiedExpression::InstantiateIntrinsicTypeClass(right, it),
                        it.into(),
                    );
                    let left = ctx.add(SimplifiedExpression::Cast(left, res.clone()), res.clone());

                    (left, right)
                }
                BinaryOpTypeConversion::CastBoth => {
                    let left = ctx.add(SimplifiedExpression::Cast(left, res.clone()), res.clone());
                    let right =
                        ctx.add(SimplifiedExpression::Cast(right, res.clone()), res.clone());

                    (left, right)
                }
            };

            (
                ctx.add(
                    match op.slice {
                        "+" => SimplifiedExpression::Add(left, right),
                        "-" => SimplifiedExpression::Sub(left, right),
                        "*" => SimplifiedExpression::Mul(left, right),
                        "/" => SimplifiedExpression::Div(left, right),
                        "%" => SimplifiedExpression::Rem(left, right),
                        "&" => SimplifiedExpression::BitAnd(left, right),
                        "|" => SimplifiedExpression::BitOr(left, right),
                        "^" => SimplifiedExpression::BitXor(left, right),
                        "==" => SimplifiedExpression::Eq(left, right),
                        "!=" => SimplifiedExpression::Ne(left, right),
                        ">=" => SimplifiedExpression::Ge(left, right),
                        "<=" => SimplifiedExpression::Le(left, right),
                        ">" => SimplifiedExpression::Gt(left, right),
                        "<" => SimplifiedExpression::Lt(left, right),
                        "&&" => SimplifiedExpression::LogAnd(left, right),
                        "||" => SimplifiedExpression::LogOr(left, right),
                        _ => unreachable!("unknown binary op"),
                    },
                    res.clone(),
                ),
                res,
            )
        }
        Expression::Prefixed(op, expr) => {
            let (expr, et) = simplify_expression(*expr, ctx, symbol_scope);

            match op.slice {
                "+" if et.scalar_type().is_some() => (expr, et),
                "-" => match et.scalar_type() {
                    Some(IntrinsicScalarType::Bool) | Some(IntrinsicScalarType::UInt) => {
                        let target_type: ConcreteType = IntrinsicScalarType::SInt
                            .of_vector(et.vector_elements().unwrap())
                            .unwrap()
                            .into();
                        let expr = ctx.add(
                            SimplifiedExpression::Cast(expr, target_type.clone()),
                            target_type.clone(),
                        );

                        (
                            ctx.add(SimplifiedExpression::Neg(expr), target_type.clone()),
                            target_type,
                        )
                    }
                    Some(IntrinsicScalarType::UnknownIntClass) => {
                        let target_type: ConcreteType = IntrinsicType::SInt.into();
                        let expr = ctx.add(
                            SimplifiedExpression::InstantiateIntrinsicTypeClass(
                                expr,
                                IntrinsicType::SInt,
                            ),
                            target_type.clone(),
                        );

                        (
                            ctx.add(SimplifiedExpression::Neg(expr), target_type.clone()),
                            target_type,
                        )
                    }
                    Some(IntrinsicScalarType::UnknownNumberClass) => {
                        let target_type: ConcreteType = IntrinsicType::Float.into();
                        let expr = ctx.add(
                            SimplifiedExpression::InstantiateIntrinsicTypeClass(
                                expr,
                                IntrinsicType::Float,
                            ),
                            target_type.clone(),
                        );

                        (
                            ctx.add(SimplifiedExpression::Neg(expr), target_type.clone()),
                            target_type,
                        )
                    }
                    Some(_) => (ctx.add(SimplifiedExpression::Neg(expr), et.clone()), et),
                    None => panic!("Error: cannot apply prefixed - to the term"),
                },
                "!" => match et.scalar_type() {
                    Some(IntrinsicScalarType::SInt)
                    | Some(IntrinsicScalarType::UInt)
                    | Some(IntrinsicScalarType::Float) => {
                        let target_type: ConcreteType = IntrinsicScalarType::Bool
                            .of_vector(et.vector_elements().unwrap())
                            .unwrap()
                            .into();
                        let expr = ctx.add(
                            SimplifiedExpression::Cast(expr, target_type.clone()),
                            target_type.clone(),
                        );

                        (
                            ctx.add(SimplifiedExpression::LogNot(expr), target_type.clone()),
                            target_type,
                        )
                    }
                    Some(IntrinsicScalarType::UnknownIntClass) => {
                        let target_type: ConcreteType = IntrinsicType::Bool.into();
                        let expr = ctx.add(
                            SimplifiedExpression::InstantiateIntrinsicTypeClass(
                                expr,
                                IntrinsicType::UInt,
                            ),
                            target_type.clone(),
                        );
                        let expr = ctx.add(
                            SimplifiedExpression::Cast(expr, IntrinsicType::Bool.into()),
                            target_type.clone(),
                        );

                        (
                            ctx.add(SimplifiedExpression::LogNot(expr), target_type.clone()),
                            target_type,
                        )
                    }
                    Some(IntrinsicScalarType::UnknownNumberClass) => {
                        let target_type: ConcreteType = IntrinsicType::Bool.into();
                        let expr = ctx.add(
                            SimplifiedExpression::InstantiateIntrinsicTypeClass(
                                expr,
                                IntrinsicType::Float,
                            ),
                            target_type.clone(),
                        );
                        let expr = ctx.add(
                            SimplifiedExpression::Cast(expr, IntrinsicType::Bool.into()),
                            target_type.clone(),
                        );

                        (
                            ctx.add(SimplifiedExpression::LogNot(expr), target_type.clone()),
                            target_type,
                        )
                    }
                    Some(_) => (ctx.add(SimplifiedExpression::LogNot(expr), et.clone()), et),
                    None => panic!("Error: cannot apply prefixed ! to the term"),
                },
                "~" => match et.scalar_type() {
                    Some(IntrinsicScalarType::Bool) | Some(IntrinsicScalarType::SInt) => {
                        let target_type: ConcreteType = IntrinsicScalarType::UInt
                            .of_vector(et.vector_elements().unwrap())
                            .unwrap()
                            .into();
                        let expr = ctx.add(
                            SimplifiedExpression::Cast(expr, target_type.clone()),
                            target_type.clone(),
                        );

                        (
                            ctx.add(SimplifiedExpression::BitNot(expr), target_type.clone()),
                            target_type,
                        )
                    }
                    Some(IntrinsicScalarType::UnknownIntClass) => {
                        let target_type: ConcreteType = IntrinsicType::UInt.into();
                        let expr = ctx.add(
                            SimplifiedExpression::InstantiateIntrinsicTypeClass(
                                expr,
                                IntrinsicType::UInt,
                            ),
                            target_type.clone(),
                        );

                        (
                            ctx.add(SimplifiedExpression::BitNot(expr), target_type.clone()),
                            target_type,
                        )
                    }
                    Some(IntrinsicScalarType::UInt) => {
                        (ctx.add(SimplifiedExpression::BitNot(expr), et.clone()), et)
                    }
                    _ => panic!("Error: cannot apply prefixed ~ to the term"),
                },
                x => panic!("Error: cannot apply prefixed op {x} to the term"),
            }
        }
        Expression::Lifted(_, x, _) => simplify_expression(*x, ctx, symbol_scope),
        Expression::Blocked(stmts, x) => {
            let new_symbol_scope = ctx
                .symbol_scope_arena
                .alloc(SymbolScope2::new(Some(symbol_scope), false));
            let mut new_ctx = SimplificationContext {
                symbol_scope_arena: ctx.symbol_scope_arena,
                vars: Vec::new(),
            };

            for s in stmts {
                match s {
                    Statement::Let {
                        varname_token,
                        expr,
                        ..
                    } => {
                        let (res, ty) = simplify_expression(expr, &mut new_ctx, new_symbol_scope);
                        let vid = new_symbol_scope.declare_local_var(
                            SourceRef::from(&varname_token),
                            ty.clone(),
                            res,
                        );
                        new_ctx.add(
                            SimplifiedExpression::InitializeVar(PtrEq(new_symbol_scope), vid),
                            ty,
                        );
                    }
                }
            }

            let (last_id, last_ty) = simplify_expression(*x, &mut new_ctx, new_symbol_scope);
            (
                ctx.add(
                    SimplifiedExpression::ScopedBlock {
                        symbol_scope: PtrEq(new_symbol_scope),
                        expressions: new_ctx.vars,
                        returning: last_id,
                    },
                    last_ty.clone(),
                ),
                last_ty,
            )
        }
        Expression::MemberRef(base, _, name) => {
            let (base, base_ty) = simplify_expression(*base, ctx, symbol_scope);

            match base_ty {
                ConcreteType::Intrinsic(x) => match (x.scalar_type(), x.vector_elements()) {
                    (None, _) | (_, None) => panic!("cannot member ref to complex data"),
                    (_, Some(1)) => panic!("scalar value cannot be swizzled"),
                    (Some(scalar), Some(count)) => match swizzle_indices(name.slice, count) {
                        Some([Some(a), None, None, None]) => (
                            ctx.add(SimplifiedExpression::Swizzle1(base, a), scalar.into()),
                            scalar.into(),
                        ),
                        Some([Some(a), Some(b), None, None]) => (
                            ctx.add(
                                SimplifiedExpression::Swizzle2(base, a, b),
                                scalar.of_vector(2).unwrap().into(),
                            ),
                            scalar.of_vector(2).unwrap().into(),
                        ),
                        Some([Some(a), Some(b), Some(c), None]) => (
                            ctx.add(
                                SimplifiedExpression::Swizzle3(base, a, b, c),
                                scalar.of_vector(3).unwrap().into(),
                            ),
                            scalar.of_vector(3).unwrap().into(),
                        ),
                        Some([Some(a), Some(b), Some(c), Some(d)]) => (
                            ctx.add(
                                SimplifiedExpression::Swizzle4(base, a, b, c, d),
                                scalar.of_vector(4).unwrap().into(),
                            ),
                            scalar.of_vector(4).unwrap().into(),
                        ),
                        Some(_) => panic!("invalid swizzle ref"),
                        None => panic!("too many swizzle components"),
                    },
                },
                ConcreteType::UserDefined {
                    name: ty_name,
                    generic_args,
                } => {
                    let (_, (_, ty)) = symbol_scope
                        .lookup_user_defined_type(ty_name)
                        .expect("No user defined type defined");

                    match ty {
                        UserDefinedType::Struct(members) => {
                            let target_member =
                                members.iter().find(|x| x.name.0.slice == name.slice);
                            match target_member {
                                Some(x) => (
                                    ctx.add(
                                        SimplifiedExpression::MemberRef(
                                            base,
                                            SourceRefSliceEq(SourceRef::from(&name)),
                                        ),
                                        x.ty.clone(),
                                    ),
                                    x.ty.clone(),
                                ),
                                None => {
                                    panic!("Struct {ty_name} has no member named '{}'", name.slice);
                                }
                            }
                        }
                    }
                }
                ConcreteType::Struct(members) => {
                    let target_member = members.iter().find(|x| x.name.0.slice == name.slice);
                    match target_member {
                        Some(x) => (
                            ctx.add(
                                SimplifiedExpression::MemberRef(
                                    base,
                                    SourceRefSliceEq(SourceRef::from(&name)),
                                ),
                                x.ty.clone(),
                            ),
                            x.ty.clone(),
                        ),
                        None => {
                            panic!("Struct has no member named '{}'", name.slice);
                        }
                    }
                }
                _ => {
                    eprintln!("unsupported member ref op for type {base_ty:?}");

                    (
                        ctx.add(
                            SimplifiedExpression::MemberRef(
                                base,
                                SourceRefSliceEq(SourceRef::from(&name)),
                            ),
                            ConcreteType::Never,
                        ),
                        ConcreteType::Never,
                    )
                }
            }
        }
        Expression::Funcall {
            base_expr, args, ..
        } => {
            let (base_expr, base_ty) = simplify_expression(*base_expr, ctx, symbol_scope);
            let (args, arg_types): (Vec<_>, Vec<_>) = args
                .into_iter()
                .map(|(x, _)| simplify_expression(x, ctx, symbol_scope))
                .unzip();

            let res_ty = match base_ty {
                ConcreteType::IntrinsicTypeConstructor(t) => t.into(),
                ConcreteType::Function { args, output }
                    if args.iter().zip(arg_types.iter()).all(|(a, b)| a == b) =>
                {
                    output.map_or(IntrinsicType::Unit.into(), |x| *x)
                }
                ConcreteType::Function { args, .. } => {
                    eprintln!("Error: argument types mismatched({args:?} and {arg_types:?})");
                    ConcreteType::Never
                }
                _ => panic!("Error: not applyable type"),
            };

            (
                ctx.add(
                    SimplifiedExpression::Funcall(base_expr, args),
                    res_ty.clone(),
                ),
                res_ty,
            )
        }
        Expression::FuncallSingle(base_expr, arg) => {
            let (base_expr, base_ty) = simplify_expression(*base_expr, ctx, symbol_scope);
            let (arg, arg_ty) = simplify_expression(*arg, ctx, symbol_scope);

            let res_ty = match base_ty {
                ConcreteType::IntrinsicTypeConstructor(t) => t.into(),
                ConcreteType::Function { args, output } if args.len() == 1 && args[0] == arg_ty => {
                    output.map_or(IntrinsicType::Unit.into(), |x| *x)
                }
                ConcreteType::Function { args, .. } => {
                    eprintln!("Error: argument types mismatched({args:?} and [{arg_ty:?}])");
                    ConcreteType::Never
                }
                _ => panic!("Error: not applyable type"),
            };

            (
                ctx.add(
                    SimplifiedExpression::Funcall(base_expr, vec![arg]),
                    res_ty.clone(),
                ),
                res_ty,
            )
        }
        Expression::Number(t) => {
            let has_hex_prefix = t.slice.starts_with("0x") || t.slice.starts_with("0X");
            let has_float_suffix = t.slice.ends_with(['f', 'F']);
            let has_fpart = t.slice.contains('.');

            let (expr, ty) = if has_hex_prefix {
                (
                    SimplifiedExpression::ConstInt(SourceRefSliceEq(SourceRef::from(&t))),
                    ConcreteType::UnknownIntClass,
                )
            } else if has_float_suffix {
                (
                    SimplifiedExpression::ConstFloat(
                        SourceRefSliceEq(SourceRef::from(&t)),
                        ConstModifiers::empty(),
                    ),
                    IntrinsicType::Float.into(),
                )
            } else if has_fpart {
                (
                    SimplifiedExpression::ConstNumber(SourceRefSliceEq(SourceRef::from(&t))),
                    ConcreteType::UnknownNumberClass,
                )
            } else {
                (
                    SimplifiedExpression::ConstInt(SourceRefSliceEq(SourceRef::from(&t))),
                    ConcreteType::UnknownIntClass,
                )
            };

            (ctx.add(expr, ty.clone()), ty)
        }
        Expression::Var(x) => {
            let Some((scope, v)) = symbol_scope.lookup(x.slice) else {
                panic!("Error: referencing undefined symbol '{}' {x:?}", x.slice);
            };

            match v {
                VarLookupResult::IntrinsicFunction(t) => (
                    ctx.add(
                        SimplifiedExpression::IntrinsicFunction(t.name, t.is_pure),
                        t.ty.clone(),
                    ),
                    t.ty.clone(),
                ),
                VarLookupResult::IntrinsicTypeConstructor(t) => (
                    ctx.add(
                        SimplifiedExpression::IntrinsicTypeConstructor(t),
                        ConcreteType::IntrinsicTypeConstructor(t),
                    ),
                    ConcreteType::IntrinsicTypeConstructor(t),
                ),
                VarLookupResult::ScopeLocalVar(vid, ty) => (
                    ctx.add(
                        SimplifiedExpression::LoadVar(PtrEq(scope), VarId::ScopeLocal(vid)),
                        ty.clone(),
                    ),
                    ty,
                ),
                VarLookupResult::FunctionInputVar(vid, ty) => (
                    ctx.add(
                        SimplifiedExpression::LoadVar(PtrEq(scope), VarId::FunctionInput(vid)),
                        ty.clone(),
                    ),
                    ty.clone(),
                ),
            }
        }
        Expression::Tuple(_, xs, _) => {
            let (xs, xs_types): (Vec<_>, Vec<_>) = xs
                .into_iter()
                .map(|(x, _)| simplify_expression(x, ctx, symbol_scope))
                .unzip();

            let ty = ConcreteType::Tuple(xs_types);
            (
                ctx.add(SimplifiedExpression::ConstructTuple(xs), ty.clone()),
                ty,
            )
        }
        Expression::If {
            condition,
            then_expr,
            else_expr,
            ..
        } => {
            let (condition, cty) = simplify_expression(*condition, ctx, symbol_scope);
            let (then_expr, tty) = simplify_expression(*then_expr, ctx, symbol_scope);
            let (else_expr, ety) = match else_expr {
                None => (
                    ctx.add(SimplifiedExpression::ConstUnit, IntrinsicType::Unit.into()),
                    IntrinsicType::Unit.into(),
                ),
                Some(x) => simplify_expression(*x, ctx, symbol_scope),
            };

            let condition = match cty {
                ConcreteType::Intrinsic(IntrinsicType::Bool) => condition,
                _ => ctx.add(
                    SimplifiedExpression::Cast(condition, IntrinsicType::Bool.into()),
                    IntrinsicType::Bool.into(),
                ),
            };

            let res_ty = match (tty, ety) {
                (a, b) if a == b => a,
                _ => {
                    eprintln!("Error: if then block and else block has different result type");
                    ConcreteType::Never
                }
            };

            (
                ctx.add(
                    SimplifiedExpression::Select(condition, then_expr, else_expr),
                    res_ty.clone(),
                ),
                res_ty,
            )
        }
    }
}

fn promote_local_var_scope<'a, 's>(
    expr: &mut SimplifiedExpression<'a, 's>,
    old_scope: &'a SymbolScope2<'a, 's>,
    new_scope: &'a SymbolScope2<'a, 's>,
    local_var_offset: usize,
) {
    match expr {
        &mut SimplifiedExpression::LoadVar(scope, VarId::ScopeLocal(lv))
            if scope == PtrEq(old_scope) =>
        {
            *expr = SimplifiedExpression::LoadVar(
                PtrEq(new_scope),
                VarId::ScopeLocal(lv + local_var_offset),
            );
        }
        &mut SimplifiedExpression::InitializeVar(scope, VarId::ScopeLocal(lv))
            if scope == PtrEq(old_scope) =>
        {
            *expr = SimplifiedExpression::InitializeVar(
                PtrEq(new_scope),
                VarId::ScopeLocal(lv + local_var_offset),
            );
        }
        SimplifiedExpression::ScopedBlock { expressions, .. } => {
            for x in expressions.iter_mut() {
                promote_local_var_scope(&mut x.0, old_scope, new_scope, local_var_offset);
            }
        }
        _ => (),
    }
}

#[derive(Clone, Copy)]
pub enum LocalVarUsage {
    Unaccessed,
    Read,
    Write(ExprRef),
    ReadAfterWrite,
}
impl LocalVarUsage {
    pub fn mark_read(&mut self) {
        *self = match self {
            Self::Unaccessed => Self::Read,
            Self::Read => Self::Read,
            Self::Write(_) => Self::ReadAfterWrite,
            Self::ReadAfterWrite => Self::ReadAfterWrite,
        };
    }

    pub fn mark_write(&mut self, last_write: ExprRef) {
        *self = match self {
            Self::Unaccessed => Self::Write(last_write),
            Self::Read => Self::Write(last_write),
            Self::Write(_) => Self::Write(last_write),
            Self::ReadAfterWrite => Self::Write(last_write),
        };
    }
}

fn optimize_pure_expr<'a, 's>(
    expressions: &mut Vec<(SimplifiedExpression<'a, 's>, ConcreteType<'s>)>,
    scope: &'a SymbolScope2<'a, 's>,
    mut block_returning_ref: Option<&mut ExprRef>,
) -> bool {
    let mut least_one_tree_modified = false;
    let mut tree_modified = true;

    while tree_modified {
        tree_modified = false;

        // promote single scope
        if expressions.len() == 1 {
            match expressions.pop().unwrap() {
                (
                    SimplifiedExpression::ScopedBlock {
                        symbol_scope: child_scope,
                        expressions: mut scope_expr,
                        returning,
                    },
                    ty,
                ) => {
                    assert_eq!(ty, scope_expr[returning.0].1);
                    let parent_scope = child_scope.0.parent.unwrap();
                    let local_var_offset = parent_scope.merge_local_vars(child_scope.0);
                    println!("scopemerge {child_scope:?} -> {:?}", PtrEq(parent_scope));

                    for x in scope_expr.iter_mut() {
                        promote_local_var_scope(
                            &mut x.0,
                            child_scope.0,
                            parent_scope,
                            local_var_offset,
                        );
                    }

                    expressions.extend(scope_expr);
                    if let Some(ref mut b) = block_returning_ref {
                        **b = returning;
                    }

                    tree_modified = true;
                }
                (x, t) => expressions.push((x, t)),
            }
        }

        // unfold pure computation scope
        for n in 0..expressions.len() {
            match &mut expressions[n] {
                (
                    SimplifiedExpression::ScopedBlock {
                        expressions: scope_expr,
                        symbol_scope,
                        returning,
                    },
                    _,
                ) if symbol_scope.0.local_vars.borrow().is_empty()
                    && scope_expr.iter().all(|x| x.0.is_pure()) =>
                {
                    assert_eq!(returning.0, scope_expr.len() - 1);

                    for x in scope_expr.iter_mut() {
                        x.0.relocate_ref(|x| x + n);
                    }
                    let first_expr = scope_expr.pop().unwrap();
                    let (
                        SimplifiedExpression::ScopedBlock {
                            expressions: mut scope_expr,
                            ..
                        },
                        _,
                    ) = core::mem::replace(&mut expressions[n], first_expr)
                    else {
                        unreachable!();
                    };
                    let nth_shifts = scope_expr.len();
                    while let Some(x) = scope_expr.pop() {
                        expressions.insert(n, x);
                    }

                    // rewrite shifted reference
                    for m in n + nth_shifts + 1..expressions.len() {
                        expressions[m]
                            .0
                            .relocate_ref(|x| if x >= n { x + nth_shifts } else { x });
                    }

                    for l in scope.local_vars.borrow_mut().iter_mut() {
                        l.init_expr_id.0 += if l.init_expr_id.0 >= n { nth_shifts } else { 0 };
                    }

                    if let Some(ref mut ret) = block_returning_ref {
                        ret.0 += if ret.0 >= n { nth_shifts } else { 0 };
                    }

                    tree_modified = true;
                }
                _ => (),
            }
        }

        // inlining loadvar until dirtified
        let mut last_loadvar_expr_id = HashMap::new();
        let mut expr_id_alias = HashMap::new();
        let mut last_expr_ids = HashMap::new();
        for n in 0..expressions.len() {
            match &mut expressions[n].0 {
                &mut SimplifiedExpression::LoadVar(vscope, vid) => {
                    if let Some(x) =
                        last_loadvar_expr_id.get(&(vscope.0 as *const SymbolScope2, vid))
                    {
                        expr_id_alias.insert(n, *x);

                        for l in scope.local_vars.borrow_mut().iter_mut() {
                            if l.init_expr_id.0 == n {
                                l.init_expr_id.0 = *x;
                            }
                        }
                    } else {
                        last_loadvar_expr_id.insert((vscope.0 as *const SymbolScope2, vid), n);
                    }
                }
                &mut SimplifiedExpression::InitializeVar(vscope, VarId::ScopeLocal(vx)) => {
                    let init_expr_id = vscope.0.local_vars.borrow()[vx].init_expr_id;

                    expr_id_alias.insert(n, init_expr_id.0);
                    for l in scope.local_vars.borrow_mut().iter_mut() {
                        if l.init_expr_id.0 == n {
                            l.init_expr_id.0 = init_expr_id.0;
                        }
                    }
                    last_loadvar_expr_id.insert(
                        (vscope.0 as *const SymbolScope2, VarId::ScopeLocal(vx)),
                        init_expr_id.0,
                    );
                }
                x => {
                    tree_modified |=
                        x.relocate_ref(|x| expr_id_alias.get(&x).copied().unwrap_or(x));
                    if let Some(x) = last_expr_ids.get(&*x) {
                        expr_id_alias.insert(n, *x);
                        for l in scope.local_vars.borrow_mut().iter_mut() {
                            if l.init_expr_id.0 == n {
                                l.init_expr_id.0 = *x;
                            }
                        }
                    } else {
                        last_expr_ids.insert(x.clone(), n);
                    }
                }
            }
        }

        let mut referenced_expr = HashSet::new();
        let mut current_scope_var_usages = HashMap::new();
        referenced_expr.extend(block_returning_ref.as_ref().map(|x| **x));
        let expressions_head_ptr = expressions.as_ptr();
        for n in 0..expressions.len() {
            match &mut expressions[n].0 {
                &mut SimplifiedExpression::Neg(src) => match expressions[src.0].0 {
                    SimplifiedExpression::ConstSInt(ref org, mods) => {
                        expressions[n].0 = SimplifiedExpression::ConstSInt(
                            org.clone(),
                            mods | ConstModifiers::NEGATE,
                        );
                        tree_modified = true;
                    }
                    SimplifiedExpression::ConstFloat(ref org, mods) => {
                        expressions[n].0 = SimplifiedExpression::ConstFloat(
                            org.clone(),
                            mods | ConstModifiers::NEGATE,
                        );
                        tree_modified = true;
                    }
                    _ => {
                        referenced_expr.insert(src);
                    }
                },
                &mut SimplifiedExpression::BitNot(src) => match expressions[src.0].0 {
                    SimplifiedExpression::ConstUInt(ref org, mods) => {
                        expressions[n].0 = SimplifiedExpression::ConstUInt(
                            org.clone(),
                            mods | ConstModifiers::BIT_NOT,
                        );
                        tree_modified = true;
                    }
                    SimplifiedExpression::ConstSInt(ref org, mods) => {
                        expressions[n].0 = SimplifiedExpression::ConstSInt(
                            org.clone(),
                            mods | ConstModifiers::BIT_NOT,
                        );
                        tree_modified = true;
                    }
                    _ => {
                        referenced_expr.insert(src);
                    }
                },
                &mut SimplifiedExpression::LogNot(src) => match expressions[src.0].0 {
                    SimplifiedExpression::ConstUInt(ref org, mods) => {
                        expressions[n].0 = SimplifiedExpression::ConstUInt(
                            org.clone(),
                            mods | ConstModifiers::LOGICAL_NOT,
                        );
                        tree_modified = true;
                    }
                    SimplifiedExpression::ConstSInt(ref org, mods) => {
                        expressions[n].0 = SimplifiedExpression::ConstSInt(
                            org.clone(),
                            mods | ConstModifiers::LOGICAL_NOT,
                        );
                        tree_modified = true;
                    }
                    SimplifiedExpression::ConstFloat(ref org, mods) => {
                        expressions[n].0 = SimplifiedExpression::ConstFloat(
                            org.clone(),
                            mods | ConstModifiers::LOGICAL_NOT,
                        );
                        tree_modified = true;
                    }
                    _ => {
                        referenced_expr.insert(src);
                    }
                },
                &mut SimplifiedExpression::Add(left, right) => {
                    referenced_expr.extend([left, right]);
                }
                &mut SimplifiedExpression::Sub(left, right) => {
                    referenced_expr.extend([left, right]);
                }
                &mut SimplifiedExpression::Mul(left, right) => {
                    referenced_expr.extend([left, right]);
                }
                &mut SimplifiedExpression::Div(left, right) => {
                    referenced_expr.extend([left, right]);
                }
                &mut SimplifiedExpression::Rem(left, right) => {
                    referenced_expr.extend([left, right]);
                }
                &mut SimplifiedExpression::BitAnd(left, right) => {
                    referenced_expr.extend([left, right]);
                }
                &mut SimplifiedExpression::BitOr(left, right) => {
                    referenced_expr.extend([left, right]);
                }
                &mut SimplifiedExpression::BitXor(left, right) => {
                    referenced_expr.extend([left, right]);
                }
                &mut SimplifiedExpression::Eq(left, right) => {
                    referenced_expr.extend([left, right]);
                }
                &mut SimplifiedExpression::Ne(left, right) => {
                    referenced_expr.extend([left, right]);
                }
                &mut SimplifiedExpression::Gt(left, right) => {
                    referenced_expr.extend([left, right]);
                }
                &mut SimplifiedExpression::Ge(left, right) => {
                    referenced_expr.extend([left, right]);
                }
                &mut SimplifiedExpression::Lt(left, right) => {
                    referenced_expr.extend([left, right]);
                }
                &mut SimplifiedExpression::Le(left, right) => {
                    referenced_expr.extend([left, right]);
                }
                &mut SimplifiedExpression::LogAnd(left, right) => {
                    referenced_expr.extend([left, right]);
                }
                &mut SimplifiedExpression::LogOr(left, right) => {
                    referenced_expr.extend([left, right]);
                }
                &mut SimplifiedExpression::Select(c, t, e) => {
                    referenced_expr.extend([c, t, e]);
                }
                &mut SimplifiedExpression::Funcall(base, ref args) => {
                    let intrinsic_constructor =
                        match unsafe { &(&*expressions_head_ptr.add(base.0)).1 } {
                            ConcreteType::IntrinsicTypeConstructor(it) => match it {
                                IntrinsicType::Float2 => Some((IntrinsicType::Float2, 2)),
                                IntrinsicType::Float3 => Some((IntrinsicType::Float3, 3)),
                                IntrinsicType::Float4 => Some((IntrinsicType::Float4, 4)),
                                IntrinsicType::UInt2 => Some((IntrinsicType::UInt2, 2)),
                                IntrinsicType::UInt3 => Some((IntrinsicType::UInt3, 3)),
                                IntrinsicType::UInt4 => Some((IntrinsicType::UInt4, 4)),
                                IntrinsicType::SInt2 => Some((IntrinsicType::SInt2, 2)),
                                IntrinsicType::SInt3 => Some((IntrinsicType::SInt3, 3)),
                                IntrinsicType::SInt4 => Some((IntrinsicType::SInt4, 4)),
                                _ => None,
                            },
                            _ => None,
                        };

                    if let Some((it, count)) = intrinsic_constructor {
                        let mut args = args.clone();
                        referenced_expr.extend(args.iter().copied());

                        let org_arg_count = args.len();
                        let mut ins_count = 0;
                        while args.len() < count {
                            // extend by repeating
                            args.push(args[ins_count % org_arg_count]);
                            ins_count += 1;
                        }

                        expressions[n].0 =
                            SimplifiedExpression::ConstructIntrinsicComposite(it, args);
                        tree_modified = true;
                    } else {
                        let intrinsic_function =
                            match unsafe { &(&*expressions_head_ptr.add(base.0)).0 } {
                                &SimplifiedExpression::IntrinsicFunction(name, is_pure) => {
                                    Some((name, is_pure))
                                }
                                _ => None,
                            };

                        if let Some((instr, is_pure)) = intrinsic_function {
                            let args = args.clone();
                            referenced_expr.extend(args.iter().copied());

                            expressions[n].0 =
                                SimplifiedExpression::IntrinsicFuncall(instr, is_pure, args);
                            tree_modified = true;
                        } else {
                            referenced_expr.insert(base);
                            referenced_expr.extend(args.iter().copied());
                        }
                    }
                }
                &mut SimplifiedExpression::MemberRef(
                    base,
                    SourceRefSliceEq(SourceRef { slice: name, .. }),
                ) => match &expressions[base.0].0 {
                    SimplifiedExpression::LoadByCanonicalRefPath(rp) => {
                        expressions[n].0 = SimplifiedExpression::LoadByCanonicalRefPath(
                            RefPath::Member(Box::new(rp.clone()), name),
                        );
                        tree_modified = true;
                    }
                    _ => {
                        referenced_expr.insert(base);
                    }
                },
                &mut SimplifiedExpression::LoadVar(scope, VarId::FunctionInput(vx))
                    if scope.0.is_toplevel_function =>
                {
                    expressions[n].0 =
                        SimplifiedExpression::LoadByCanonicalRefPath(RefPath::FunctionInput(vx));
                    tree_modified = true;
                }
                &mut SimplifiedExpression::LoadVar(vscope, VarId::ScopeLocal(vx))
                    if vscope == PtrEq(scope) =>
                {
                    current_scope_var_usages
                        .entry(vx)
                        .or_insert(LocalVarUsage::Unaccessed)
                        .mark_read();
                }
                &mut SimplifiedExpression::LoadVar(_, _) => (),
                &mut SimplifiedExpression::InitializeVar(vscope, VarId::ScopeLocal(vx))
                    if vscope == PtrEq(scope) =>
                {
                    current_scope_var_usages
                        .entry(vx)
                        .or_insert(LocalVarUsage::Unaccessed)
                        .mark_write(ExprRef(n));
                }
                &mut SimplifiedExpression::InitializeVar(_, _) => (),
                &mut SimplifiedExpression::LoadByCanonicalRefPath(_) => (),
                &mut SimplifiedExpression::StoreLocal(_, v) => {
                    referenced_expr.insert(v);
                }
                &mut SimplifiedExpression::IntrinsicFunction(_, _) => (),
                &mut SimplifiedExpression::IntrinsicTypeConstructor(_) => (),
                &mut SimplifiedExpression::IntrinsicFuncall(_, _, ref xs) => {
                    referenced_expr.extend(xs.iter().copied());
                }
                &mut SimplifiedExpression::Cast(x, ref to) => {
                    let to_ty = to.clone();
                    let target_ty = expressions[x.0].1.clone();

                    if to_ty == target_ty {
                        // cast to same type
                        expressions[n] = expressions[x.0].clone();
                        tree_modified = true;
                    } else {
                        referenced_expr.insert(x);
                    }
                }
                &mut SimplifiedExpression::Swizzle1(src, _) => {
                    referenced_expr.insert(src);
                }
                &mut SimplifiedExpression::Swizzle2(src, _, _) => {
                    referenced_expr.insert(src);
                }
                &mut SimplifiedExpression::Swizzle3(src, _, _, _) => {
                    referenced_expr.insert(src);
                }
                &mut SimplifiedExpression::Swizzle4(src, _, _, _, _) => {
                    referenced_expr.insert(src);
                }
                &mut SimplifiedExpression::InstantiateIntrinsicTypeClass(
                    v,
                    IntrinsicType::UInt,
                ) => match &expressions[v.0].0 {
                    SimplifiedExpression::ConstInt(t) => {
                        expressions[n].0 =
                            SimplifiedExpression::ConstUInt(t.clone(), ConstModifiers::empty());
                        tree_modified = true;
                    }
                    _ => {
                        referenced_expr.insert(v);
                    }
                },
                &mut SimplifiedExpression::InstantiateIntrinsicTypeClass(
                    v,
                    IntrinsicType::SInt,
                ) => match &expressions[v.0].0 {
                    SimplifiedExpression::ConstInt(t) => {
                        expressions[n].0 =
                            SimplifiedExpression::ConstSInt(t.clone(), ConstModifiers::empty());
                        tree_modified = true;
                    }
                    _ => {
                        referenced_expr.insert(v);
                    }
                },
                &mut SimplifiedExpression::InstantiateIntrinsicTypeClass(
                    v,
                    IntrinsicType::Float,
                ) => match &expressions[v.0].0 {
                    SimplifiedExpression::ConstInt(t) => {
                        expressions[n].0 =
                            SimplifiedExpression::ConstFloat(t.clone(), ConstModifiers::empty());
                        tree_modified = true;
                    }
                    SimplifiedExpression::ConstNumber(t) => {
                        expressions[n].0 =
                            SimplifiedExpression::ConstFloat(t.clone(), ConstModifiers::empty());
                        tree_modified = true;
                    }
                    _ => {
                        referenced_expr.insert(v);
                    }
                },
                &mut SimplifiedExpression::InstantiateIntrinsicTypeClass(v, _) => {
                    referenced_expr.insert(v);
                }
                &mut SimplifiedExpression::ConstInt(_) => (),
                &mut SimplifiedExpression::ConstNumber(_) => (),
                &mut SimplifiedExpression::ConstUnit => (),
                &mut SimplifiedExpression::ConstUInt(_, _) => (),
                &mut SimplifiedExpression::ConstSInt(_, _) => (),
                &mut SimplifiedExpression::ConstFloat(_, _) => (),
                &mut SimplifiedExpression::ConstructTuple(ref xs) => {
                    referenced_expr.extend(xs.iter().copied());
                }
                &mut SimplifiedExpression::ConstructIntrinsicComposite(_, ref xs) => {
                    referenced_expr.extend(xs.iter().copied());
                }
                &mut SimplifiedExpression::StoreOutput(x, _) => {
                    referenced_expr.insert(x);
                }
                &mut SimplifiedExpression::DistributeOutputTuple(x, _) => {
                    referenced_expr.insert(x);
                }
                SimplifiedExpression::ScopedBlock {
                    ref mut expressions,
                    ref mut returning,
                    ref symbol_scope,
                } => {
                    tree_modified |=
                        optimize_pure_expr(expressions, symbol_scope.0, Some(returning));

                    for (n, x) in expressions.iter().enumerate() {
                        match x.0 {
                            SimplifiedExpression::LoadVar(vscope, VarId::ScopeLocal(vx))
                                if vscope == PtrEq(scope) =>
                            {
                                current_scope_var_usages
                                    .entry(vx)
                                    .or_insert(LocalVarUsage::Unaccessed)
                                    .mark_read();
                            }
                            SimplifiedExpression::InitializeVar(vscope, VarId::ScopeLocal(vx))
                                if vscope == PtrEq(scope) =>
                            {
                                current_scope_var_usages
                                    .entry(vx)
                                    .or_insert(LocalVarUsage::Unaccessed)
                                    .mark_write(ExprRef(n));
                            }
                            _ => (),
                        }
                    }
                }
            }
        }

        // collect stripped expression ids
        let mut stripped_ops = HashSet::new();
        for (_, t) in current_scope_var_usages.iter() {
            if let &LocalVarUsage::Write(last_write) = t {
                println!("striplastwrite: {last_write:?}");
                stripped_ops.insert(last_write.0);
            }
        }
        for n in 0..expressions.len() {
            if !referenced_expr.contains(&ExprRef(n)) && expressions[n].0.is_pure() {
                stripped_ops.insert(n);
            }
        }

        // strip expressions
        let mut stripped_ops = stripped_ops.into_iter().collect::<Vec<_>>();
        while let Some(n) = stripped_ops.pop() {
            expressions.remove(n);
            // rewrite shifted reference
            for m in n..expressions.len() {
                expressions[m]
                    .0
                    .relocate_ref(|x| if x > n { x - 1 } else { x });
            }

            if let Some(ref mut ret) = block_returning_ref {
                ret.0 -= if ret.0 > n { 1 } else { 0 };
            }

            for x in stripped_ops.iter_mut() {
                *x -= if *x > n { 1 } else { 0 };
            }

            tree_modified = true;
        }

        // strip unaccessed local vars
        let mut stripped_local_var_ids = (0..scope.local_vars.borrow().len())
            .filter(|lvid| !current_scope_var_usages.contains_key(lvid))
            .collect::<Vec<_>>();
        while let Some(n) = stripped_local_var_ids.pop() {
            scope.local_vars.borrow_mut().remove(n);

            // rewrite shifted references
            for m in 0..expressions.len() {
                match &mut expressions[m].0 {
                    &mut SimplifiedExpression::LoadVar(vscope, VarId::ScopeLocal(ref mut vx))
                        if vscope == PtrEq(scope) =>
                    {
                        *vx -= if *vx > n { 1 } else { 0 };
                    }
                    &mut SimplifiedExpression::InitializeVar(
                        vscope,
                        VarId::ScopeLocal(ref mut vx),
                    ) if vscope == PtrEq(scope) => {
                        *vx -= if *vx > n { 1 } else { 0 };
                    }
                    _ => (),
                }
            }
            for lv in scope.var_id_by_name.borrow_mut().values_mut() {
                match lv {
                    &mut VarId::ScopeLocal(ref mut vx) => {
                        *vx -= if *vx > n { 1 } else { 0 };
                    }
                    _ => (),
                }
            }
            for x in stripped_local_var_ids.iter_mut() {
                *x -= if *x > n { 1 } else { 0 };
            }

            tree_modified = true;
        }

        // println!("transformed(cont?{tree_modified}):");
        // for (n, (x, t)) in expressions.iter().enumerate() {
        //     print_simp_expr(x, t, n, 0);
        // }

        least_one_tree_modified |= tree_modified;
    }

    least_one_tree_modified
}

#[derive(Debug, Clone)]
pub struct SourceRef<'s> {
    pub slice: &'s str,
    pub line: usize,
    pub col: usize,
}
impl<'s> From<&'_ Token<'s>> for SourceRef<'s> {
    #[inline]
    fn from(value: &'_ Token<'s>) -> Self {
        Self {
            slice: value.slice,
            line: value.line,
            col: value.col,
        }
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct SourceRefSliceEq<'s>(pub SourceRef<'s>);
impl core::fmt::Debug for SourceRefSliceEq<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <SourceRef as core::fmt::Debug>::fmt(&self.0, f)
    }
}
impl core::cmp::PartialEq for SourceRefSliceEq<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.0.slice == other.0.slice
    }
}
impl core::cmp::Eq for SourceRefSliceEq<'_> {}
impl core::hash::Hash for SourceRefSliceEq<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.slice.hash(state)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntrinsicScalarType {
    Unit,
    Bool,
    UInt,
    SInt,
    Float,
    UnknownIntClass,
    UnknownNumberClass,
}
impl IntrinsicScalarType {
    #[inline(always)]
    pub const fn of_vector(self, count: u8) -> Option<IntrinsicType> {
        match (self, count) {
            (Self::Unit, 0) => Some(IntrinsicType::Unit),
            (Self::Bool, 1) => Some(IntrinsicType::Bool),
            (Self::UInt, 1) => Some(IntrinsicType::UInt),
            (Self::SInt, 1) => Some(IntrinsicType::SInt),
            (Self::Float, 1) => Some(IntrinsicType::Float),
            (Self::UInt, 2) => Some(IntrinsicType::UInt2),
            (Self::SInt, 2) => Some(IntrinsicType::SInt2),
            (Self::Float, 2) => Some(IntrinsicType::Float2),
            (Self::UInt, 3) => Some(IntrinsicType::UInt3),
            (Self::SInt, 3) => Some(IntrinsicType::SInt3),
            (Self::Float, 3) => Some(IntrinsicType::Float3),
            (Self::UInt, 4) => Some(IntrinsicType::UInt4),
            (Self::SInt, 4) => Some(IntrinsicType::SInt4),
            (Self::Float, 4) => Some(IntrinsicType::Float4),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntrinsicType {
    Unit,
    Bool,
    UInt,
    UInt2,
    UInt3,
    UInt4,
    SInt,
    SInt2,
    SInt3,
    SInt4,
    Float,
    Float2,
    Float3,
    Float4,
    Float2x2,
    Float2x3,
    Float2x4,
    Float3x2,
    Float3x3,
    Float3x4,
    Float4x2,
    Float4x3,
    Float4x4,
    Sampler1D,
    Sampler2D,
    Sampler3D,
    Texture1D,
    Texture2D,
    Texture3D,
    SubpassInput,
}
impl IntrinsicType {
    pub const fn scalar_type(&self) -> Option<IntrinsicScalarType> {
        match self {
            Self::Unit => Some(IntrinsicScalarType::Unit),
            Self::Bool => Some(IntrinsicScalarType::Bool),
            Self::UInt | Self::UInt2 | Self::UInt3 | Self::UInt4 => Some(IntrinsicScalarType::UInt),
            Self::SInt | Self::SInt2 | Self::SInt3 | Self::SInt4 => Some(IntrinsicScalarType::SInt),
            Self::Float | Self::Float2 | Self::Float3 | Self::Float4 => {
                Some(IntrinsicScalarType::Float)
            }
            _ => None,
        }
    }

    pub const fn vector_elements(&self) -> Option<u8> {
        match self {
            Self::Unit => Some(0),
            Self::Bool | Self::UInt | Self::SInt | Self::Float => Some(1),
            Self::UInt2 | Self::SInt2 | Self::Float2 => Some(2),
            Self::UInt3 | Self::SInt3 | Self::Float3 => Some(3),
            Self::UInt4 | Self::SInt4 | Self::Float4 => Some(4),
            _ => None,
        }
    }

    pub const fn can_uniform_struct_member(&self) -> bool {
        match self {
            // unit(void) cannot be a member of uniform struct
            Self::Unit => false,
            // samplers/image refs cannot be a member of uniform struct
            Self::Sampler1D
            | Self::Sampler2D
            | Self::Sampler3D
            | Self::Texture1D
            | Self::Texture2D
            | Self::Texture3D
            | Self::SubpassInput => false,
            _ => true,
        }
    }

    pub const fn std140_alignment(&self) -> Option<usize> {
        match self {
            // unit(void) cannot be a member of uniform struct
            Self::Unit => None,
            Self::Bool | Self::UInt | Self::SInt | Self::Float => Some(4),
            Self::UInt2 | Self::SInt2 | Self::Float2 => Some(8),
            Self::UInt3 | Self::SInt3 | Self::Float3 | Self::UInt4 | Self::SInt4 | Self::Float4 => {
                Some(16)
            }
            Self::Float2x2 | Self::Float2x3 | Self::Float2x4 => Some(8),
            Self::Float3x2
            | Self::Float3x3
            | Self::Float3x4
            | Self::Float4x2
            | Self::Float4x3
            | Self::Float4x4 => Some(16),
            // samplers/image refs cannot be a member of uniform struct
            Self::Sampler1D
            | Self::Sampler2D
            | Self::Sampler3D
            | Self::Texture1D
            | Self::Texture2D
            | Self::Texture3D
            | Self::SubpassInput => None,
        }
    }

    pub const fn std140_size(&self) -> Option<usize> {
        match self {
            // unit(void) cannot be a member of uniform struct
            Self::Unit => None,
            Self::Bool | Self::UInt | Self::SInt | Self::Float => Some(4),
            Self::UInt2 | Self::SInt2 | Self::Float2 => Some(8),
            Self::UInt3 | Self::SInt3 | Self::Float3 | Self::UInt4 | Self::SInt4 | Self::Float4 => {
                Some(16)
            }
            Self::Float2x2 | Self::Float2x3 | Self::Float2x4 => Some(16),
            Self::Float3x2
            | Self::Float3x3
            | Self::Float3x4
            | Self::Float4x2
            | Self::Float4x3
            | Self::Float4x4 => Some(16),
            // samplers/image refs cannot be a member of uniform struct
            Self::Sampler1D
            | Self::Sampler2D
            | Self::Sampler3D
            | Self::Texture1D
            | Self::Texture2D
            | Self::Texture3D
            | Self::SubpassInput => None,
        }
    }

    pub fn make_spv_type(&self) -> spv::Type {
        match self {
            Self::Unit => spv::Type::Void,
            Self::Bool => spv::Type::Bool,
            Self::UInt => spv::Type::Int {
                width: 32,
                signedness: false,
            },
            Self::SInt => spv::Type::Int {
                width: 32,
                signedness: true,
            },
            Self::Float => spv::Type::Float { width: 32 },
            Self::UInt2 => spv::Type::Vector {
                component_type: Box::new(Self::UInt.make_spv_type()),
                component_count: 2,
            },
            Self::UInt3 => spv::Type::Vector {
                component_type: Box::new(Self::UInt.make_spv_type()),
                component_count: 3,
            },
            Self::UInt4 => spv::Type::Vector {
                component_type: Box::new(Self::UInt.make_spv_type()),
                component_count: 4,
            },
            Self::SInt2 => spv::Type::Vector {
                component_type: Box::new(Self::SInt.make_spv_type()),
                component_count: 2,
            },
            Self::SInt3 => spv::Type::Vector {
                component_type: Box::new(Self::SInt.make_spv_type()),
                component_count: 3,
            },
            Self::SInt4 => spv::Type::Vector {
                component_type: Box::new(Self::SInt.make_spv_type()),
                component_count: 4,
            },
            Self::Float2 => spv::Type::Vector {
                component_type: Box::new(Self::Float.make_spv_type()),
                component_count: 2,
            },
            Self::Float3 => spv::Type::Vector {
                component_type: Box::new(Self::Float.make_spv_type()),
                component_count: 3,
            },
            Self::Float4 => spv::Type::Vector {
                component_type: Box::new(Self::Float.make_spv_type()),
                component_count: 4,
            },
            Self::Float2x2 => spv::Type::Matrix {
                column_type: Box::new(Self::Float2.make_spv_type()),
                column_count: 2,
            },
            Self::Float2x3 => spv::Type::Matrix {
                column_type: Box::new(Self::Float2.make_spv_type()),
                column_count: 3,
            },
            Self::Float2x4 => spv::Type::Matrix {
                column_type: Box::new(Self::Float2.make_spv_type()),
                column_count: 4,
            },
            Self::Float3x2 => spv::Type::Matrix {
                column_type: Box::new(Self::Float3.make_spv_type()),
                column_count: 2,
            },
            Self::Float3x3 => spv::Type::Matrix {
                column_type: Box::new(Self::Float3.make_spv_type()),
                column_count: 3,
            },
            Self::Float3x4 => spv::Type::Matrix {
                column_type: Box::new(Self::Float3.make_spv_type()),
                column_count: 4,
            },
            Self::Float4x2 => spv::Type::Matrix {
                column_type: Box::new(Self::Float4.make_spv_type()),
                column_count: 2,
            },
            Self::Float4x3 => spv::Type::Matrix {
                column_type: Box::new(Self::Float4.make_spv_type()),
                column_count: 3,
            },
            Self::Float4x4 => spv::Type::Matrix {
                column_type: Box::new(Self::Float4.make_spv_type()),
                column_count: 4,
            },
            Self::Sampler1D => unreachable!("deprecated"),
            Self::Sampler2D => unreachable!("deprecated"),
            Self::Sampler3D => unreachable!("deprecated"),
            Self::Texture1D => unimplemented!(),
            Self::Texture2D => unimplemented!(),
            Self::Texture3D => unimplemented!(),
            Self::SubpassInput => spv::Type::subpass_data_image_type(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConcreteType<'s> {
    Generic(Vec<usize>, Box<ConcreteType<'s>>),
    GenericVar(usize),
    Intrinsic(IntrinsicType),
    UnknownIntClass,
    UnknownNumberClass,
    UserDefined {
        name: &'s str,
        generic_args: Vec<ConcreteType<'s>>,
    },
    Struct(Vec<UserDefinedStructMember<'s>>),
    Tuple(Vec<ConcreteType<'s>>),
    Function {
        args: Vec<ConcreteType<'s>>,
        output: Option<Box<ConcreteType<'s>>>,
    },
    IntrinsicTypeConstructor(IntrinsicType),
    Never,
}
impl<'s> ConcreteType<'s> {
    pub fn build(
        symbol_scope: &SymbolScope2<'_, 's>,
        sibling_scope_opaque_symbols: &HashSet<&'s str>,
        t: Type<'s>,
    ) -> Self {
        match t.name_token.slice {
            "UInt" => Self::Intrinsic(IntrinsicType::UInt),
            "UInt2" => Self::Intrinsic(IntrinsicType::UInt2),
            "UInt3" => Self::Intrinsic(IntrinsicType::UInt3),
            "UInt4" => Self::Intrinsic(IntrinsicType::UInt4),
            "SInt" | "Int" => Self::Intrinsic(IntrinsicType::SInt),
            "SInt2" | "Int2" => Self::Intrinsic(IntrinsicType::SInt2),
            "SInt3" | "Int3" => Self::Intrinsic(IntrinsicType::SInt3),
            "SInt4" | "Int4" => Self::Intrinsic(IntrinsicType::SInt4),
            "Float" => Self::Intrinsic(IntrinsicType::Float),
            "Float2" => Self::Intrinsic(IntrinsicType::Float2),
            "Float3" => Self::Intrinsic(IntrinsicType::Float3),
            "Float4" => Self::Intrinsic(IntrinsicType::Float4),
            "Float2x2" => Self::Intrinsic(IntrinsicType::Float2x2),
            "Float2x3" => Self::Intrinsic(IntrinsicType::Float2x3),
            "Float2x4" => Self::Intrinsic(IntrinsicType::Float2x4),
            "Float3x2" => Self::Intrinsic(IntrinsicType::Float3x2),
            "Float3x3" => Self::Intrinsic(IntrinsicType::Float3x3),
            "Float3x4" => Self::Intrinsic(IntrinsicType::Float3x4),
            "Float4x2" => Self::Intrinsic(IntrinsicType::Float4x2),
            "Float4x3" => Self::Intrinsic(IntrinsicType::Float4x3),
            "Float4x4" => Self::Intrinsic(IntrinsicType::Float4x4),
            "Sampler1D" => Self::Intrinsic(IntrinsicType::Sampler1D),
            "Sampler2D" => Self::Intrinsic(IntrinsicType::Sampler2D),
            "Sampler3D" => Self::Intrinsic(IntrinsicType::Sampler3D),
            "Texture1D" => Self::Intrinsic(IntrinsicType::Texture1D),
            "Texture2D" => Self::Intrinsic(IntrinsicType::Texture2D),
            "Texture3D" => Self::Intrinsic(IntrinsicType::Texture3D),
            "SubpassInput" => Self::Intrinsic(IntrinsicType::SubpassInput),
            name => {
                if sibling_scope_opaque_symbols.contains(name) {
                    ConcreteType::UserDefined {
                        name: t.name_token.slice,
                        generic_args: t
                            .generic_args
                            .map_or_else(Vec::new, |x| x.args)
                            .into_iter()
                            .map(|x| {
                                ConcreteType::build(symbol_scope, sibling_scope_opaque_symbols, x.0)
                            })
                            .collect(),
                    }
                } else {
                    match symbol_scope.lookup_user_defined_type(name) {
                        Some(_) => ConcreteType::UserDefined {
                            name: t.name_token.slice,
                            generic_args: t
                                .generic_args
                                .map_or_else(Vec::new, |x| x.args)
                                .into_iter()
                                .map(|x| {
                                    ConcreteType::build(
                                        symbol_scope,
                                        sibling_scope_opaque_symbols,
                                        x.0,
                                    )
                                })
                                .collect(),
                        },
                        None => panic!("Error: referencing undefined type: {}", t.name_token.slice),
                    }
                }
            }
        }
    }

    pub fn instantiate(self, scope: &SymbolScope2<'_, 's>) -> Self {
        match self {
            Self::UserDefined { name, generic_args } => {
                match scope.lookup_user_defined_type(name) {
                    Some((_, (_, UserDefinedType::Struct(members)))) => Self::Struct(
                        members
                            .iter()
                            .map(|x| UserDefinedStructMember {
                                attribute: x.attribute.clone(),
                                name: x.name.clone(),
                                ty: x.ty.clone().instantiate(scope),
                            })
                            .collect(),
                    ),
                    None => Self::UserDefined { name, generic_args },
                }
            }
            _ => self,
        }
    }

    pub const fn scalar_type(&self) -> Option<IntrinsicScalarType> {
        match self {
            Self::Intrinsic(x) => x.scalar_type(),
            Self::UnknownIntClass => Some(IntrinsicScalarType::UnknownIntClass),
            Self::UnknownNumberClass => Some(IntrinsicScalarType::UnknownNumberClass),
            _ => None,
        }
    }

    pub const fn vector_elements(&self) -> Option<u8> {
        match self {
            Self::Intrinsic(x) => x.vector_elements(),
            Self::UnknownIntClass | Self::UnknownNumberClass => Some(1),
            _ => None,
        }
    }

    pub const fn can_uniform_struct_member(&self) -> bool {
        match self {
            Self::Intrinsic(it) => it.can_uniform_struct_member(),
            Self::Struct(_) | Self::Tuple(_) => true,
            _ => false,
        }
    }

    pub const fn std140_alignment(&self) -> Option<usize> {
        match self {
            Self::Intrinsic(it) => it.std140_alignment(),
            Self::Struct(_) | Self::Tuple(_) => Some(16),
            _ => None,
        }
    }

    pub fn std140_size(&self) -> Option<usize> {
        match self {
            Self::Intrinsic(it) => it.std140_size(),
            Self::Struct(xs) => xs.iter().map(|x| x.ty.std140_size()).sum(),
            Self::Tuple(xs) => xs.iter().map(|x| x.std140_size()).sum(),
            _ => None,
        }
    }

    pub fn make_spv_type(&self, scope: &SymbolScope2<'_, 's>) -> spv::Type {
        match self {
            Self::Intrinsic(it) => it.make_spv_type(),
            Self::Tuple(xs) => spv::Type::Struct {
                member_types: xs
                    .iter()
                    .scan(0, |top, x| {
                        let offs = roundup2(
                            *top,
                            x.std140_alignment().expect("cannot a member of a struct"),
                        );
                        *top = offs + x.std140_size().expect("cannot a member of a struct");

                        Some(spv::TypeStructMember {
                            ty: x.make_spv_type(scope),
                            offset: offs as _,
                            decorations: vec![],
                        })
                    })
                    .collect(),
            },
            Self::Function { args, output } => spv::Type::Function {
                return_type: Box::new(
                    output
                        .as_ref()
                        .map_or(spv::Type::Void, |o| o.make_spv_type(scope)),
                ),
                parameter_types: args.iter().map(|x| x.make_spv_type(scope)).collect(),
            },
            Self::Struct(members) => spv::Type::Struct {
                member_types: members
                    .iter()
                    .scan(0, |top, x| {
                        let offs = roundup2(
                            *top,
                            x.ty.std140_alignment()
                                .expect("cannot a member of a struct"),
                        );
                        *top = offs + x.ty.std140_size().expect("cannot a member of a struct");

                        Some(spv::TypeStructMember {
                            ty: x.ty.make_spv_type(scope),
                            offset: offs as _,
                            decorations: vec![],
                        })
                    })
                    .collect(),
            },
            Self::UserDefined { name, .. } => match scope.lookup_user_defined_type(name) {
                None => spv::Type::Opaque {
                    name: (*name).into(),
                },
                Some((_, (_, UserDefinedType::Struct(members)))) => spv::Type::Struct {
                    member_types: members
                        .iter()
                        .scan(0, |top, x| {
                            let offs = roundup2(
                                *top,
                                x.ty.std140_alignment()
                                    .expect("cannot a member of a struct"),
                            );
                            *top = offs + x.ty.std140_size().expect("cannot a member of a struct");

                            Some(spv::TypeStructMember {
                                ty: x.ty.make_spv_type(scope),
                                offset: offs as _,
                                decorations: vec![],
                            })
                        })
                        .collect(),
                },
            },
            Self::IntrinsicTypeConstructor(_) => {
                unreachable!("non-reduced intrinsic type construction")
            }
            Self::Never => unreachable!("type inference has error"),
            Self::Generic { .. } => unreachable!("uninstantiated generic type"),
            Self::GenericVar(_) => unreachable!("uninstantiated generic var"),
            Self::UnknownIntClass => unreachable!("left UnknownIntClass"),
            Self::UnknownNumberClass => unreachable!("left UnknownNumberClass"),
        }
    }
}
impl From<IntrinsicType> for ConcreteType<'_> {
    #[inline(always)]
    fn from(value: IntrinsicType) -> Self {
        Self::Intrinsic(value)
    }
}
impl From<IntrinsicScalarType> for ConcreteType<'_> {
    #[inline(always)]
    fn from(value: IntrinsicScalarType) -> Self {
        match value {
            IntrinsicScalarType::Unit => Self::Intrinsic(IntrinsicType::Unit),
            IntrinsicScalarType::Bool => Self::Intrinsic(IntrinsicType::Bool),
            IntrinsicScalarType::UInt => Self::Intrinsic(IntrinsicType::UInt),
            IntrinsicScalarType::SInt => Self::Intrinsic(IntrinsicType::SInt),
            IntrinsicScalarType::Float => Self::Intrinsic(IntrinsicType::Float),
            IntrinsicScalarType::UnknownIntClass => Self::UnknownIntClass,
            IntrinsicScalarType::UnknownNumberClass => Self::UnknownNumberClass,
        }
    }
}

pub enum BinaryOpTypeConversion {
    NoConversion,
    CastLeftHand,
    CastRightHand,
    CastBoth,
    InstantiateAndCastLeftHand(IntrinsicType),
    InstantiateAndCastRightHand(IntrinsicType),
    InstantiateRightAndCastLeftHand(IntrinsicType),
    InstantiateLeftAndCastRightHand(IntrinsicType),
    InstantiateLeftHand(IntrinsicType),
    InstantiateRightHand(IntrinsicType),
}
impl<'s> ConcreteType<'s> {
    pub fn arithmetic_compare_op_type_conversion(
        self,
        rhs: Self,
    ) -> Option<(BinaryOpTypeConversion, Self)> {
        match (self, rhs) {
            // between same type
            (a, b) if a == b => Some((BinaryOpTypeConversion::NoConversion, a)),
            // between same length vectors
            (a, b) if a.vector_elements()? == b.vector_elements()? => {
                match (a.scalar_type()?, b.scalar_type()?) {
                    // simple casting
                    (IntrinsicScalarType::Bool, IntrinsicScalarType::Float) => {
                        Some((BinaryOpTypeConversion::CastLeftHand, b))
                    }
                    (IntrinsicScalarType::UInt, IntrinsicScalarType::Float) => {
                        Some((BinaryOpTypeConversion::CastLeftHand, b))
                    }
                    (IntrinsicScalarType::SInt, IntrinsicScalarType::Float) => {
                        Some((BinaryOpTypeConversion::CastLeftHand, b))
                    }
                    (IntrinsicScalarType::Bool, IntrinsicScalarType::UInt) => {
                        Some((BinaryOpTypeConversion::CastLeftHand, b))
                    }
                    (IntrinsicScalarType::SInt, IntrinsicScalarType::UInt) => {
                        Some((BinaryOpTypeConversion::CastRightHand, a))
                    }
                    (IntrinsicScalarType::Float, IntrinsicScalarType::UInt) => {
                        Some((BinaryOpTypeConversion::CastRightHand, a))
                    }
                    (IntrinsicScalarType::Bool, IntrinsicScalarType::SInt) => {
                        Some((BinaryOpTypeConversion::CastLeftHand, b))
                    }
                    (IntrinsicScalarType::UInt, IntrinsicScalarType::SInt) => {
                        Some((BinaryOpTypeConversion::CastLeftHand, b))
                    }
                    (IntrinsicScalarType::Float, IntrinsicScalarType::SInt) => {
                        Some((BinaryOpTypeConversion::CastRightHand, a))
                    }
                    (IntrinsicScalarType::UInt, IntrinsicScalarType::Bool) => {
                        Some((BinaryOpTypeConversion::CastRightHand, a))
                    }
                    (IntrinsicScalarType::SInt, IntrinsicScalarType::Bool) => {
                        Some((BinaryOpTypeConversion::CastRightHand, a))
                    }
                    (IntrinsicScalarType::Float, IntrinsicScalarType::Bool) => {
                        Some((BinaryOpTypeConversion::CastRightHand, a))
                    }
                    // instantiate left
                    (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::Bool) => Some((
                        BinaryOpTypeConversion::InstantiateLeftAndCastRightHand(
                            IntrinsicType::UInt,
                        ),
                        IntrinsicType::UInt.into(),
                    )),
                    (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::SInt) => Some((
                        BinaryOpTypeConversion::InstantiateLeftHand(IntrinsicType::SInt),
                        b,
                    )),
                    (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::UInt) => Some((
                        BinaryOpTypeConversion::InstantiateLeftHand(IntrinsicType::UInt),
                        b,
                    )),
                    (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::Float) => Some((
                        BinaryOpTypeConversion::InstantiateAndCastLeftHand(IntrinsicType::SInt),
                        b,
                    )),
                    (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::Bool) => Some((
                        BinaryOpTypeConversion::InstantiateLeftAndCastRightHand(
                            IntrinsicType::Float,
                        ),
                        IntrinsicType::Float.into(),
                    )),
                    (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::SInt) => Some((
                        BinaryOpTypeConversion::InstantiateLeftAndCastRightHand(
                            IntrinsicType::Float,
                        ),
                        IntrinsicType::Float.into(),
                    )),
                    (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::UInt) => Some((
                        BinaryOpTypeConversion::InstantiateLeftAndCastRightHand(
                            IntrinsicType::Float,
                        ),
                        IntrinsicType::Float.into(),
                    )),
                    (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::Float) => {
                        Some((
                            BinaryOpTypeConversion::InstantiateLeftHand(IntrinsicType::Float),
                            b,
                        ))
                    }
                    // instantiate right
                    (IntrinsicScalarType::Bool, IntrinsicScalarType::UnknownIntClass) => Some((
                        BinaryOpTypeConversion::InstantiateRightAndCastLeftHand(
                            IntrinsicType::UInt,
                        ),
                        IntrinsicType::UInt.into(),
                    )),
                    (IntrinsicScalarType::SInt, IntrinsicScalarType::UnknownIntClass) => Some((
                        BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::SInt),
                        a,
                    )),
                    (IntrinsicScalarType::UInt, IntrinsicScalarType::UnknownIntClass) => Some((
                        BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::UInt),
                        a,
                    )),
                    (IntrinsicScalarType::Float, IntrinsicScalarType::UnknownIntClass) => Some((
                        BinaryOpTypeConversion::InstantiateAndCastRightHand(IntrinsicType::SInt),
                        a,
                    )),
                    (IntrinsicScalarType::Bool, IntrinsicScalarType::UnknownNumberClass) => Some((
                        BinaryOpTypeConversion::InstantiateRightAndCastLeftHand(
                            IntrinsicType::Float,
                        ),
                        IntrinsicType::Float.into(),
                    )),
                    (IntrinsicScalarType::SInt, IntrinsicScalarType::UnknownNumberClass) => Some((
                        BinaryOpTypeConversion::InstantiateRightAndCastLeftHand(
                            IntrinsicType::Float,
                        ),
                        IntrinsicType::Float.into(),
                    )),
                    (IntrinsicScalarType::UInt, IntrinsicScalarType::UnknownNumberClass) => Some((
                        BinaryOpTypeConversion::InstantiateRightAndCastLeftHand(
                            IntrinsicType::Float,
                        ),
                        IntrinsicType::Float.into(),
                    )),
                    (IntrinsicScalarType::Float, IntrinsicScalarType::UnknownNumberClass) => {
                        Some((
                            BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::Float),
                            a,
                        ))
                    }
                    // never
                    _ => None,
                }
            }
            // never
            _ => None,
        }
    }

    pub fn bitwise_op_type_conversion(self, rhs: Self) -> Option<(BinaryOpTypeConversion, Self)> {
        match (self, rhs) {
            // between same type
            (a, b) if a == b => Some((BinaryOpTypeConversion::NoConversion, a)),
            // between same length vectors
            (a, b) if a.vector_elements()? == b.vector_elements()? => {
                match (a.scalar_type()?, b.scalar_type()?) {
                    // simple casting
                    (IntrinsicScalarType::Bool, IntrinsicScalarType::UInt) => {
                        Some((BinaryOpTypeConversion::CastLeftHand, b))
                    }
                    (IntrinsicScalarType::SInt, IntrinsicScalarType::UInt) => {
                        Some((BinaryOpTypeConversion::CastLeftHand, b))
                    }
                    (IntrinsicScalarType::Bool, IntrinsicScalarType::SInt) => {
                        Some((BinaryOpTypeConversion::CastLeftHand, b))
                    }
                    (IntrinsicScalarType::UInt, IntrinsicScalarType::SInt) => {
                        Some((BinaryOpTypeConversion::CastRightHand, a))
                    }
                    (IntrinsicScalarType::UInt, IntrinsicScalarType::Bool) => {
                        Some((BinaryOpTypeConversion::CastRightHand, a))
                    }
                    (IntrinsicScalarType::SInt, IntrinsicScalarType::Bool) => {
                        Some((BinaryOpTypeConversion::CastRightHand, a))
                    }
                    // instantiate left
                    (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::Bool) => Some((
                        BinaryOpTypeConversion::InstantiateLeftAndCastRightHand(
                            IntrinsicType::UInt,
                        ),
                        IntrinsicType::UInt.into(),
                    )),
                    (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::SInt) => Some((
                        BinaryOpTypeConversion::InstantiateLeftHand(IntrinsicType::SInt),
                        IntrinsicType::SInt.into(),
                    )),
                    (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::UInt) => Some((
                        BinaryOpTypeConversion::InstantiateLeftHand(IntrinsicType::UInt),
                        b,
                    )),
                    // instantiate right
                    (IntrinsicScalarType::Bool, IntrinsicScalarType::UnknownIntClass) => Some((
                        BinaryOpTypeConversion::InstantiateRightAndCastLeftHand(
                            IntrinsicType::UInt,
                        ),
                        IntrinsicType::UInt.into(),
                    )),
                    (IntrinsicScalarType::SInt, IntrinsicScalarType::UnknownIntClass) => Some((
                        BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::SInt),
                        IntrinsicType::SInt.into(),
                    )),
                    (IntrinsicScalarType::UInt, IntrinsicScalarType::UnknownIntClass) => Some((
                        BinaryOpTypeConversion::InstantiateRightHand(IntrinsicType::UInt),
                        a,
                    )),
                    // never
                    _ => None,
                }
            }
            // never
            _ => None,
        }
    }

    pub fn logical_op_type_conversion(self, rhs: Self) -> Option<(BinaryOpTypeConversion, Self)> {
        match (self, rhs) {
            // between same type
            (a, b) if a == b => Some((BinaryOpTypeConversion::NoConversion, a)),
            // between same length vectors
            (a, b) if a.vector_elements()? == b.vector_elements()? => {
                match (a.scalar_type()?, b.scalar_type()?) {
                    // instantiate and cast
                    (IntrinsicScalarType::Bool, IntrinsicScalarType::UnknownIntClass) => Some((
                        BinaryOpTypeConversion::InstantiateAndCastRightHand(IntrinsicType::UInt),
                        a,
                    )),
                    (IntrinsicScalarType::UnknownIntClass, IntrinsicScalarType::Bool) => Some((
                        BinaryOpTypeConversion::InstantiateAndCastLeftHand(IntrinsicType::UInt),
                        a,
                    )),
                    (IntrinsicScalarType::Bool, IntrinsicScalarType::UnknownNumberClass) => Some((
                        BinaryOpTypeConversion::InstantiateAndCastRightHand(IntrinsicType::Float),
                        a,
                    )),
                    (IntrinsicScalarType::UnknownNumberClass, IntrinsicScalarType::Bool) => Some((
                        BinaryOpTypeConversion::InstantiateAndCastLeftHand(IntrinsicType::Float),
                        a,
                    )),
                    // simple casting
                    (IntrinsicScalarType::Bool, _) => {
                        Some((BinaryOpTypeConversion::CastRightHand, a))
                    }
                    (_, IntrinsicScalarType::Bool) => {
                        Some((BinaryOpTypeConversion::CastLeftHand, b))
                    }
                    // never
                    _ => None,
                }
            }
            // never
            _ => None,
        }
    }
}

const fn swizzle_indices(x: &str, src_component_count: u8) -> Option<[Option<usize>; 4]> {
    match x.as_bytes() {
        &[a] => Some([swizzle_index(a, src_component_count), None, None, None]),
        &[a, b] => Some([
            swizzle_index(a, src_component_count),
            swizzle_index(b, src_component_count),
            None,
            None,
        ]),
        &[a, b, c] => Some([
            swizzle_index(a, src_component_count),
            swizzle_index(b, src_component_count),
            swizzle_index(c, src_component_count),
            None,
        ]),
        &[a, b, c, d] => Some([
            swizzle_index(a, src_component_count),
            swizzle_index(b, src_component_count),
            swizzle_index(c, src_component_count),
            swizzle_index(d, src_component_count),
        ]),
        _ => None,
    }
}

const fn swizzle_index(x: u8, src_component_count: u8) -> Option<usize> {
    match x {
        b'r' | b'R' | b'x' | b'X' if src_component_count >= 1 => Some(0),
        b'g' | b'G' | b'y' | b'Y' if src_component_count >= 2 => Some(1),
        b'b' | b'B' | b'z' | b'Z' if src_component_count >= 3 => Some(2),
        b'a' | b'A' | b'w' | b'W' if src_component_count >= 4 => Some(3),
        _ => None,
    }
}

#[derive(Debug, Clone)]
pub struct UserDefinedStructMemberPartiallyTyped<'s> {
    pub name: SourceRef<'s>,
    pub ty: Type<'s>,
    pub attributes: Vec<Attribute<'s>>,
}

#[derive(Debug, Clone)]
pub enum UserDefinedTypePartiallyTyped<'s> {
    Struct(Vec<UserDefinedStructMemberPartiallyTyped<'s>>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UserDefinedStructMember<'s> {
    pub attribute: SymbolAttribute,
    pub name: SourceRefSliceEq<'s>,
    pub ty: ConcreteType<'s>,
}

#[derive(Debug, Clone)]
pub enum UserDefinedType<'s> {
    Struct(Vec<UserDefinedStructMember<'s>>),
}

#[derive(Debug)]
pub enum ParseErrorKind {
    ExpectedKeyword(&'static str),
    ExpectedKind(TokenKind),
    ExpectedConstExpression,
    ExpectedExpression,
    ExpectedFunctionBodyStarter,
    Outdent(IndentContext),
    ListNotPunctuated(TokenKind),
}

#[derive(Debug)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub line: usize,
    pub col: usize,
}
pub type ParseResult<T> = Result<T, ParseError>;

#[derive(Debug, Clone, Copy)]
pub enum IndentContext {
    Free,
    Exclusive(usize),
    Inclusive(usize),
}
impl IndentContext {
    #[inline]
    pub fn satisfies(self, target: usize) -> bool {
        match self {
            Self::Free => true,
            Self::Exclusive(x) => x < target,
            Self::Inclusive(x) => x <= target,
        }
    }
}

struct ParseState<'s> {
    pub token_list: Vec<Token<'s>>,
    pub token_ptr: usize,
    pub indent_context_stack: Vec<IndentContext>,
}
impl<'s> ParseState<'s> {
    #[inline]
    pub fn current_token(&self) -> Option<&Token<'s>> {
        self.token_list.get(self.token_ptr)
    }

    #[inline]
    pub fn consume_token(&mut self) {
        self.token_ptr += 1;
    }

    #[inline]
    pub fn current_indent_context(&self) -> IndentContext {
        self.indent_context_stack
            .last()
            .copied()
            .unwrap_or(IndentContext::Free)
    }

    #[inline]
    pub fn check_indent_requirements(&self) -> bool {
        self.current_token()
            .map_or(true, |t| self.current_indent_context().satisfies(t.col))
    }

    #[inline]
    pub fn push_indent_context(&mut self, ctx: IndentContext) {
        self.indent_context_stack.push(ctx)
    }

    #[inline]
    pub fn pop_indent_context(&mut self) {
        self.indent_context_stack.pop();
    }

    #[inline]
    pub fn require_in_block_next(&self) -> Result<(), ParseError> {
        if !self.check_indent_requirements() {
            return Err(self.err(ParseErrorKind::Outdent(self.current_indent_context())));
        }

        Ok(())
    }

    #[inline]
    pub fn consume_keyword(&mut self, kw: &'static str) -> Result<&Token<'s>, ParseError> {
        match self.token_list.get(self.token_ptr) {
            Some(x) if x.kind == TokenKind::Keyword && x.slice == kw => {
                self.token_ptr += 1;
                Ok(x)
            }
            t => Err(self.err_on(ParseErrorKind::ExpectedKeyword(kw), t)),
        }
    }

    #[inline]
    pub fn consume_by_kind(&mut self, kind: TokenKind) -> Result<&Token<'s>, ParseError> {
        match self.token_list.get(self.token_ptr) {
            Some(x) if x.kind == kind => {
                self.token_ptr += 1;
                Ok(x)
            }
            t => Err(self.err_on(ParseErrorKind::ExpectedKind(kind), t)),
        }
    }

    #[inline]
    pub fn consume_in_block_keyword(&mut self, kw: &'static str) -> Result<&Token<'s>, ParseError> {
        if !self.check_indent_requirements() {
            return Err(self.err(ParseErrorKind::Outdent(self.current_indent_context())));
        }

        match self.token_list.get(self.token_ptr) {
            Some(x) if x.kind == TokenKind::Keyword && x.slice == kw => {
                self.token_ptr += 1;
                Ok(x)
            }
            t => Err(self.err_on(ParseErrorKind::ExpectedKeyword(kw), t)),
        }
    }

    #[inline]
    pub fn consume_in_block_by_kind(&mut self, kind: TokenKind) -> Result<&Token<'s>, ParseError> {
        if !self.check_indent_requirements() {
            return Err(self.err(ParseErrorKind::Outdent(self.current_indent_context())));
        }

        match self.token_list.get(self.token_ptr) {
            Some(x) if x.kind == kind => {
                self.token_ptr += 1;
                Ok(x)
            }
            t => Err(self.err_on(ParseErrorKind::ExpectedKind(kind), t)),
        }
    }

    #[inline]
    pub fn err_on(&self, kind: ParseErrorKind, tref: Option<&Token<'s>>) -> ParseError {
        ParseError {
            kind,
            line: tref
                .or_else(|| self.token_list.last())
                .map_or(0, |t| t.line),
            col: tref.or_else(|| self.token_list.last()).map_or(0, |t| t.col),
        }
    }

    #[inline]
    pub fn err(&self, kind: ParseErrorKind) -> ParseError {
        self.err_on(kind, self.current_token())
    }
}

#[derive(Debug)]
pub enum ToplevelDeclaration<'s> {
    Struct(StructDeclaration<'s>),
    Function(FunctionDeclaration<'s>),
}
fn parse_toplevel_declaration<'s>(
    state: &mut ParseState<'s>,
) -> ParseResult<ToplevelDeclaration<'s>> {
    match state.current_token() {
        Some(t) if t.kind == TokenKind::Keyword && t.slice == "struct" => {
            parse_struct_declaration(state).map(ToplevelDeclaration::Struct)
        }
        _ => parse_function_declaration(state).map(ToplevelDeclaration::Function),
    }
}

#[derive(Debug, Clone)]
pub struct TypeGenericArgs<'s> {
    pub open_angle_bracket_token: Token<'s>,
    pub args: Vec<(Type<'s>, Option<Token<'s>>)>,
    pub close_angle_bracket_token: Token<'s>,
}
fn parse_type_generic_args<'s>(state: &mut ParseState<'s>) -> ParseResult<TypeGenericArgs<'s>> {
    let open_angle_bracket_token = state.consume_by_kind(TokenKind::OpenAngleBracket)?.clone();

    let mut args = Vec::new();
    let mut can_continue = true;
    while state
        .current_token()
        .is_some_and(|t| t.kind != TokenKind::CloseAngleBracket)
    {
        if !can_continue {
            return Err(state.err(ParseErrorKind::ListNotPunctuated(TokenKind::Comma)));
        }

        let t = parse_type(state)?;
        let opt_comma_token = state.consume_by_kind(TokenKind::Comma).ok().cloned();

        can_continue = opt_comma_token.is_some();
        args.push((t, opt_comma_token));
    }

    let close_angle_bracket_token = state.consume_by_kind(TokenKind::CloseAngleBracket)?.clone();

    Ok(TypeGenericArgs {
        open_angle_bracket_token,
        args,
        close_angle_bracket_token,
    })
}

#[derive(Debug, Clone)]
pub struct Type<'s> {
    pub name_token: Token<'s>,
    pub generic_args: Option<TypeGenericArgs<'s>>,
}
fn parse_type<'s>(state: &mut ParseState<'s>) -> ParseResult<Type<'s>> {
    let name_token = state.consume_by_kind(TokenKind::Identifier)?.clone();
    let generic_args = match state.current_token() {
        Some(t) if t.kind == TokenKind::OpenAngleBracket => Some(parse_type_generic_args(state)?),
        _ => None,
    };

    Ok(Type {
        name_token,
        generic_args,
    })
}

#[derive(Debug, Clone)]
pub enum ConstExpression<'s> {
    Number(Token<'s>),
}
fn parse_const_expression<'s>(state: &mut ParseState<'s>) -> ParseResult<ConstExpression<'s>> {
    match state.current_token() {
        Some(t) if t.kind == TokenKind::Number => {
            let t = t.clone();
            state.token_ptr += 1;
            Ok(ConstExpression::Number(t))
        }
        t => Err(state.err_on(ParseErrorKind::ExpectedConstExpression, t)),
    }
}
fn lookahead_const_expression(state: &ParseState) -> bool {
    state
        .current_token()
        .is_some_and(|t| t.kind == TokenKind::Number)
}

#[derive(Debug, Clone)]
pub enum AttributeArg<'s> {
    Single(ConstExpression<'s>),
    Multiple {
        open_parenthese_token: Token<'s>,
        arg_list: Vec<(ConstExpression<'s>, Option<Token<'s>>)>,
        close_parenthese_token: Token<'s>,
    },
}
fn parse_attribute_arg<'s>(state: &mut ParseState<'s>) -> ParseResult<AttributeArg<'s>> {
    match state.current_token() {
        Some(t) if t.kind == TokenKind::OpenParenthese => {
            let open_parenthese_token = state.consume_by_kind(TokenKind::OpenParenthese)?.clone();
            let mut arg_list = Vec::new();
            let mut can_continue = true;
            while state
                .current_token()
                .is_some_and(|t| t.kind != TokenKind::CloseParenthese)
            {
                if !can_continue {
                    return Err(state.err(ParseErrorKind::ListNotPunctuated(TokenKind::Comma)));
                }

                let arg = parse_const_expression(state)?;
                let opt_comma_token = state.consume_by_kind(TokenKind::Comma).ok().cloned();
                can_continue = opt_comma_token.is_some();
                arg_list.push((arg, opt_comma_token));
            }
            let close_parenthese_token = state.consume_by_kind(TokenKind::CloseParenthese)?.clone();

            Ok(AttributeArg::Multiple {
                open_parenthese_token,
                arg_list,
                close_parenthese_token,
            })
        }
        _ => parse_const_expression(state).map(AttributeArg::Single),
    }
}
fn lookahead_attribute_arg(state: &ParseState) -> bool {
    state
        .current_token()
        .is_some_and(|t| t.kind == TokenKind::OpenParenthese)
        || lookahead_const_expression(state)
}

#[derive(Debug, Clone)]
pub struct Attribute<'s> {
    pub name_token: Token<'s>,
    pub arg: Option<AttributeArg<'s>>,
}
fn parse_attribute<'s>(state: &mut ParseState<'s>) -> ParseResult<Attribute<'s>> {
    let name_token = state.consume_by_kind(TokenKind::Identifier)?.clone();
    let arg = if lookahead_attribute_arg(state) {
        Some(parse_attribute_arg(state)?)
    } else {
        None
    };

    Ok(Attribute { name_token, arg })
}

#[derive(Debug)]
pub struct AttributeList<'s> {
    pub open_bracket_token: Token<'s>,
    pub attribute_list: Vec<(Attribute<'s>, Option<Token<'s>>)>,
    pub close_bracket_token: Token<'s>,
}
fn parse_attribute_list<'s>(state: &mut ParseState<'s>) -> ParseResult<AttributeList<'s>> {
    let open_bracket_token = state.consume_by_kind(TokenKind::OpenBracket)?.clone();
    let mut attribute_list = Vec::new();
    let mut can_continue = true;
    while state
        .current_token()
        .is_some_and(|t| t.kind != TokenKind::CloseBracket)
    {
        if !can_continue {
            return Err(state.err(ParseErrorKind::ListNotPunctuated(TokenKind::Comma)));
        }

        let a = parse_attribute(state)?;
        let opt_comma_token = state.consume_by_kind(TokenKind::Comma).ok().cloned();
        can_continue = opt_comma_token.is_some();
        attribute_list.push((a, opt_comma_token));
    }
    let close_bracket_token = state.consume_by_kind(TokenKind::CloseBracket)?.clone();

    Ok(AttributeList {
        open_bracket_token,
        attribute_list,
        close_bracket_token,
    })
}

#[derive(Debug)]
pub struct StructMember<'s> {
    pub attribute_lists: Vec<AttributeList<'s>>,
    pub name_token: Token<'s>,
    pub colon_token: Token<'s>,
    pub ty: Type<'s>,
}
fn parse_struct_member<'s>(state: &mut ParseState<'s>) -> ParseResult<StructMember<'s>> {
    let mut attribute_lists = Vec::new();
    while state.check_indent_requirements()
        && state
            .current_token()
            .is_some_and(|t| t.kind == TokenKind::OpenBracket)
    {
        attribute_lists.push(parse_attribute_list(state)?);
    }
    let name_token = state
        .consume_in_block_by_kind(TokenKind::Identifier)?
        .clone();
    let colon_token = state.consume_by_kind(TokenKind::Colon)?.clone();
    let ty = parse_type(state)?;

    Ok(StructMember {
        attribute_lists,
        name_token,
        colon_token,
        ty,
    })
}

#[derive(Debug)]
pub struct StructDeclaration<'s> {
    pub decl_token: Token<'s>,
    pub name_token: Token<'s>,
    pub with_token: Option<Token<'s>>,
    pub member_list: Vec<StructMember<'s>>,
}
fn parse_struct_declaration<'s>(
    state: &mut ParseState<'s>,
) -> Result<StructDeclaration<'s>, ParseError> {
    let decl_token = state.consume_keyword("struct")?.clone();
    let name_token = state.consume_by_kind(TokenKind::Identifier)?.clone();
    let Some(with_token) = state.consume_keyword("with").ok().cloned() else {
        return Ok(StructDeclaration {
            decl_token,
            name_token,
            with_token: None,
            member_list: Vec::new(),
        });
    };

    state.push_indent_context(IndentContext::Exclusive(with_token.line_indent));
    let mut member_list = Vec::new();
    while state.check_indent_requirements() {
        member_list.push(parse_struct_member(state)?);
    }
    state.pop_indent_context();

    Ok(StructDeclaration {
        decl_token,
        name_token,
        with_token: Some(with_token),
        member_list,
    })
}

#[derive(Debug)]
pub enum FunctionDeclarationInputArguments<'s> {
    Single {
        attribute_lists: Vec<AttributeList<'s>>,
        varname_token: Token<'s>,
        colon_token: Token<'s>,
        ty: Type<'s>,
    },
    Multiple {
        open_parenthese_token: Token<'s>,
        args: Vec<(
            Vec<AttributeList<'s>>,
            Token<'s>,
            Token<'s>,
            Type<'s>,
            Option<Token<'s>>,
        )>,
        close_parenthese_token: Token<'s>,
    },
}
fn parse_function_declaration_input_arguments<'s>(
    state: &mut ParseState<'s>,
) -> ParseResult<FunctionDeclarationInputArguments<'s>> {
    match state.current_token() {
        Some(t) if t.kind == TokenKind::OpenParenthese => {
            let open_parenthese_token = t.clone();
            state.consume_token();

            let mut args = Vec::new();
            let mut can_continue = true;
            while state
                .current_token()
                .is_some_and(|t| t.kind != TokenKind::CloseParenthese)
            {
                if !can_continue {
                    return Err(state.err(ParseErrorKind::ListNotPunctuated(TokenKind::Comma)));
                }

                let mut attribute_lists = Vec::new();
                while state
                    .current_token()
                    .is_some_and(|t| t.kind == TokenKind::OpenBracket)
                {
                    attribute_lists.push(parse_attribute_list(state)?);
                }

                let name_token = state.consume_by_kind(TokenKind::Identifier)?.clone();
                let colon_token = state.consume_by_kind(TokenKind::Colon)?.clone();
                let ty = parse_type(state)?;
                let opt_comma_token = state.consume_by_kind(TokenKind::Comma).ok().cloned();

                can_continue = opt_comma_token.is_some();
                args.push((
                    attribute_lists,
                    name_token,
                    colon_token,
                    ty,
                    opt_comma_token,
                ));
            }

            let close_parenthese_token = state.consume_by_kind(TokenKind::CloseParenthese)?.clone();

            Ok(FunctionDeclarationInputArguments::Multiple {
                open_parenthese_token,
                args,
                close_parenthese_token,
            })
        }
        _ => {
            let mut attribute_lists = Vec::new();
            while state
                .current_token()
                .is_some_and(|t| t.kind == TokenKind::OpenBracket)
            {
                attribute_lists.push(parse_attribute_list(state)?);
            }

            let varname_token = state.consume_by_kind(TokenKind::Identifier)?.clone();
            let colon_token = state.consume_by_kind(TokenKind::Colon)?.clone();
            let ty = parse_type(state)?;

            Ok(FunctionDeclarationInputArguments::Single {
                attribute_lists,
                varname_token,
                colon_token,
                ty,
            })
        }
    }
}

#[derive(Debug)]
pub enum FunctionDeclarationOutput<'s> {
    Single {
        attribute_lists: Vec<AttributeList<'s>>,
        ty: Type<'s>,
    },
    Tupled {
        open_parenthese_token: Token<'s>,
        elements: Vec<(Vec<AttributeList<'s>>, Type<'s>, Option<Token<'s>>)>,
        close_parenthese_token: Token<'s>,
    },
}
fn parse_function_declaration_output<'s>(
    state: &mut ParseState<'s>,
) -> ParseResult<FunctionDeclarationOutput<'s>> {
    match state.current_token() {
        Some(t) if t.kind == TokenKind::OpenParenthese => {
            let open_parenthese_token = t.clone();
            state.consume_token();

            let mut elements = Vec::new();
            let mut can_continue = true;
            while state
                .current_token()
                .is_some_and(|t| t.kind != TokenKind::CloseParenthese)
            {
                if !can_continue {
                    return Err(state.err(ParseErrorKind::ListNotPunctuated(TokenKind::Comma)));
                }

                let mut attribute_lists = Vec::new();
                while state
                    .current_token()
                    .is_some_and(|t| t.kind == TokenKind::OpenBracket)
                {
                    attribute_lists.push(parse_attribute_list(state)?);
                }

                let ty = parse_type(state)?;
                let opt_comma_token = state.consume_by_kind(TokenKind::Comma).ok().cloned();

                can_continue = opt_comma_token.is_some();
                elements.push((attribute_lists, ty, opt_comma_token));
            }

            let close_parenthese_token = state.consume_by_kind(TokenKind::CloseParenthese)?.clone();

            Ok(FunctionDeclarationOutput::Tupled {
                open_parenthese_token,
                elements,
                close_parenthese_token,
            })
        }
        _ => {
            let mut attribute_lists = Vec::new();
            while state
                .current_token()
                .is_some_and(|t| t.kind == TokenKind::OpenBracket)
            {
                attribute_lists.push(parse_attribute_list(state)?);
            }

            let ty = parse_type(state)?;

            Ok(FunctionDeclarationOutput::Single {
                attribute_lists,
                ty,
            })
        }
    }
}

#[derive(Debug)]
pub struct FunctionDeclaration<'s> {
    pub attribute_lists: Vec<AttributeList<'s>>,
    pub fname_token: Token<'s>,
    pub input_args: FunctionDeclarationInputArguments<'s>,
    pub arrow_to_right_token: Option<Token<'s>>,
    pub output: Option<FunctionDeclarationOutput<'s>>,
    pub body_starter_token: Token<'s>,
    pub body: Expression<'s>,
}
fn parse_function_declaration<'s>(
    state: &mut ParseState<'s>,
) -> ParseResult<FunctionDeclaration<'s>> {
    let mut attribute_lists = Vec::new();
    while state
        .current_token()
        .is_some_and(|t| t.kind == TokenKind::OpenBracket)
    {
        attribute_lists.push(parse_attribute_list(state)?);
    }

    let fname_token = state.consume_by_kind(TokenKind::Identifier)?.clone();
    let input_args = parse_function_declaration_input_arguments(state)?;
    let arrow_to_right_token = state.consume_by_kind(TokenKind::ArrowToRight).ok().cloned();
    let output = if arrow_to_right_token.is_some() {
        Some(parse_function_declaration_output(state)?)
    } else {
        None
    };
    let body_starter_token = match state.current_token() {
        Some(t) if t.kind == TokenKind::Eq || (t.kind == TokenKind::Keyword && t.slice == "do") => {
            let tok = t.clone();
            state.consume_token();

            tok
        }
        t => return Err(state.err_on(ParseErrorKind::ExpectedFunctionBodyStarter, t)),
    };

    state.push_indent_context(IndentContext::Exclusive(body_starter_token.line_indent));
    let body = parse_block(state)?;
    state.pop_indent_context();

    Ok(FunctionDeclaration {
        attribute_lists,
        fname_token,
        input_args,
        arrow_to_right_token,
        output,
        body_starter_token,
        body,
    })
}

#[derive(Debug)]
pub enum Statement<'s> {
    Let {
        let_token: Token<'s>,
        varname_token: Token<'s>,
        eq_token: Token<'s>,
        expr: Expression<'s>,
    },
}

#[derive(Debug)]
pub enum Expression<'s> {
    Blocked(Vec<Statement<'s>>, Box<Expression<'s>>),
    Lifted(Token<'s>, Box<Expression<'s>>, Token<'s>),
    Binary(Box<Expression<'s>>, Token<'s>, Box<Expression<'s>>),
    Prefixed(Token<'s>, Box<Expression<'s>>),
    MemberRef(Box<Expression<'s>>, Token<'s>, Token<'s>),
    Funcall {
        base_expr: Box<Expression<'s>>,
        open_parenthese_token: Token<'s>,
        args: Vec<(Expression<'s>, Option<Token<'s>>)>,
        close_parenthese_token: Token<'s>,
    },
    FuncallSingle(Box<Expression<'s>>, Box<Expression<'s>>),
    Number(Token<'s>),
    Var(Token<'s>),
    Tuple(
        Token<'s>,
        Vec<(Expression<'s>, Option<Token<'s>>)>,
        Token<'s>,
    ),
    If {
        if_token: Token<'s>,
        condition: Box<Expression<'s>>,
        then_token: Token<'s>,
        then_expr: Box<Expression<'s>>,
        else_token: Option<Token<'s>>,
        else_expr: Option<Box<Expression<'s>>>,
    },
}
fn parse_block<'s>(state: &mut ParseState<'s>) -> ParseResult<Expression<'s>> {
    let mut statements = Vec::new();
    loop {
        state.require_in_block_next()?;

        match state.current_token() {
            Some(t) if t.kind == TokenKind::Keyword && t.slice == "let" => {
                let let_token = t.clone();
                state.consume_token();
                let varname_token = state.consume_by_kind(TokenKind::Identifier)?.clone();
                let eq_token = state.consume_by_kind(TokenKind::Eq)?.clone();
                state.push_indent_context(IndentContext::Exclusive(eq_token.line_indent));
                let expr = parse_block(state)?;
                state.pop_indent_context();

                statements.push(Statement::Let {
                    let_token,
                    varname_token,
                    eq_token,
                    expr,
                })
            }
            _ => break,
        }
    }

    let final_expr = parse_expression(state)?;
    Ok(Expression::Blocked(statements, Box::new(final_expr)))
}
fn parse_expression<'s>(state: &mut ParseState<'s>) -> ParseResult<Expression<'s>> {
    parse_expression_if(state)
}
fn parse_expression_if<'s>(state: &mut ParseState<'s>) -> ParseResult<Expression<'s>> {
    let Some(if_token) = state.consume_keyword("if").ok().cloned() else {
        return parse_expression_logical_ops(state);
    };
    state.push_indent_context(IndentContext::Inclusive(if_token.line_indent));

    let condition = parse_expression_logical_ops(state)?;
    let then_token = state.consume_in_block_keyword("then")?.clone();
    state.push_indent_context(IndentContext::Exclusive(then_token.line_indent));
    let then_expr = parse_block(state)?;
    state.pop_indent_context();
    let else_token = state.consume_in_block_keyword("else").ok().cloned();
    let else_expr = if let Some(ref e) = else_token {
        state.push_indent_context(IndentContext::Exclusive(e.line_indent));
        let x = parse_block(state)?;
        state.pop_indent_context();
        Some(x)
    } else {
        None
    };

    state.pop_indent_context();

    Ok(Expression::If {
        if_token,
        condition: Box::new(condition),
        then_token,
        then_expr: Box::new(then_expr),
        else_token,
        else_expr: else_expr.map(Box::new),
    })
}
fn parse_expression_logical_ops<'s>(state: &mut ParseState<'s>) -> ParseResult<Expression<'s>> {
    let mut expr = parse_expression_compare_ops(state)?;

    loop {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Op && t.slice == "||" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_compare_ops(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "&&" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_compare_ops(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            _ => break Ok(expr),
        }
    }
}
fn parse_expression_compare_ops<'s>(state: &mut ParseState<'s>) -> ParseResult<Expression<'s>> {
    let mut expr = parse_expression_bitwise_ops(state)?;

    loop {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Op && t.slice == "==" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_bitwise_ops(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "!=" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_bitwise_ops(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "<=" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_bitwise_ops(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == ">=" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_bitwise_ops(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::OpenAngleBracket => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_bitwise_ops(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::CloseAngleBracket => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_bitwise_ops(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            _ => break Ok(expr),
        }
    }
}
fn parse_expression_bitwise_ops<'s>(state: &mut ParseState<'s>) -> ParseResult<Expression<'s>> {
    let mut expr = parse_expression_arithmetic_ops_1(state)?;

    loop {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Op && t.slice == "|" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_arithmetic_ops_1(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "&" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_arithmetic_ops_1(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "^" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_arithmetic_ops_1(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            _ => break Ok(expr),
        }
    }
}
fn parse_expression_arithmetic_ops_1<'s>(
    state: &mut ParseState<'s>,
) -> ParseResult<Expression<'s>> {
    let mut expr = parse_expression_arithmetic_ops_2(state)?;

    loop {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Op && t.slice == "+" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_arithmetic_ops_2(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "-" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_arithmetic_ops_2(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            _ => break Ok(expr),
        }
    }
}
fn parse_expression_arithmetic_ops_2<'s>(
    state: &mut ParseState<'s>,
) -> ParseResult<Expression<'s>> {
    let mut expr = parse_expression_prefixed_ops(state)?;

    loop {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Op && t.slice == "*" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_prefixed_ops(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "/" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_prefixed_ops(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "%" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_prefixed_ops(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            _ => break Ok(expr),
        }
    }
}
fn parse_expression_prefixed_ops<'s>(state: &mut ParseState<'s>) -> ParseResult<Expression<'s>> {
    match state.current_token() {
        Some(t) if t.kind == TokenKind::Op && t.slice == "+" => {
            let op_token = t.clone();
            state.consume_token();
            let expr = parse_expression_prefixed_ops(state)?;

            Ok(Expression::Prefixed(op_token, Box::new(expr)))
        }
        Some(t) if t.kind == TokenKind::Op && t.slice == "-" => {
            let op_token = t.clone();
            state.consume_token();
            let expr = parse_expression_prefixed_ops(state)?;

            Ok(Expression::Prefixed(op_token, Box::new(expr)))
        }
        Some(t) if t.kind == TokenKind::Op && t.slice == "~" => {
            let op_token = t.clone();
            state.consume_token();
            let expr = parse_expression_prefixed_ops(state)?;

            Ok(Expression::Prefixed(op_token, Box::new(expr)))
        }
        Some(t) if t.kind == TokenKind::Op && t.slice == "!" => {
            let op_token = t.clone();
            state.consume_token();
            let expr = parse_expression_prefixed_ops(state)?;

            Ok(Expression::Prefixed(op_token, Box::new(expr)))
        }
        _ => parse_expression_suffixed_ops(state),
    }
}
fn parse_expression_suffixed_ops<'s>(state: &mut ParseState<'s>) -> ParseResult<Expression<'s>> {
    let mut expr = parse_expression_prime(state)?;

    loop {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Op && t.slice == "." => {
                let dot_token = t.clone();
                state.consume_token();
                let name_token = state.consume_by_kind(TokenKind::Identifier)?.clone();

                expr = Expression::MemberRef(Box::new(expr), dot_token, name_token);
            }
            Some(t) if t.kind == TokenKind::OpenParenthese => {
                let open_parenthese_token = t.clone();
                state.consume_token();
                let mut args = Vec::new();
                let mut can_continue = true;
                while state
                    .current_token()
                    .is_some_and(|t| t.kind != TokenKind::CloseParenthese)
                {
                    if !can_continue {
                        return Err(state.err(ParseErrorKind::ListNotPunctuated(TokenKind::Comma)));
                    }

                    let arg = parse_expression(state)?;
                    let opt_comma_token = state.consume_by_kind(TokenKind::Comma).ok().cloned();
                    can_continue = opt_comma_token.is_some();
                    args.push((arg, opt_comma_token));
                }
                let close_parenthese_token =
                    state.consume_by_kind(TokenKind::CloseParenthese)?.clone();

                expr = Expression::Funcall {
                    base_expr: Box::new(expr),
                    open_parenthese_token,
                    args,
                    close_parenthese_token,
                }
            }
            _ => {
                break Ok(
                    if let Ok(arg) = parse_expression_funcall_single_arg(state) {
                        Expression::FuncallSingle(Box::new(expr), Box::new(arg))
                    } else {
                        expr
                    },
                );
            }
        }
    }
}
fn parse_expression_funcall_single_arg<'s>(
    state: &mut ParseState<'s>,
) -> ParseResult<Expression<'s>> {
    let mut expr = parse_expression_prime(state)?;

    loop {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Op && t.slice == "." => {
                let dot_token = t.clone();
                state.consume_token();
                let name_token = state.consume_by_kind(TokenKind::Identifier)?.clone();

                expr = Expression::MemberRef(Box::new(expr), dot_token, name_token);
            }
            Some(t) if t.kind == TokenKind::OpenParenthese => {
                let open_parenthese_token = t.clone();
                state.consume_token();
                let mut args = Vec::new();
                let mut can_continue = true;
                while state
                    .current_token()
                    .is_some_and(|t| t.kind != TokenKind::CloseParenthese)
                {
                    if !can_continue {
                        return Err(state.err(ParseErrorKind::ListNotPunctuated(TokenKind::Comma)));
                    }

                    let arg = parse_expression(state)?;
                    let opt_comma_token = state.consume_by_kind(TokenKind::Comma).ok().cloned();
                    can_continue = opt_comma_token.is_some();
                    args.push((arg, opt_comma_token));
                }
                let close_parenthese_token =
                    state.consume_by_kind(TokenKind::CloseParenthese)?.clone();

                expr = Expression::Funcall {
                    base_expr: Box::new(expr),
                    open_parenthese_token,
                    args,
                    close_parenthese_token,
                }
            }
            _ => break Ok(expr),
        }
    }
}
fn parse_expression_prime<'s>(state: &mut ParseState<'s>) -> ParseResult<Expression<'s>> {
    match state.current_token() {
        Some(t) if t.kind == TokenKind::Number => {
            let tok = t.clone();
            state.consume_token();

            Ok(Expression::Number(tok))
        }
        Some(t) if t.kind == TokenKind::Identifier => {
            let tok = t.clone();
            state.consume_token();

            Ok(Expression::Var(tok))
        }
        Some(t) if t.kind == TokenKind::OpenParenthese => {
            let open_parenthese_token = t.clone();
            state.consume_token();

            let mut expressions = Vec::new();
            let mut can_continue = true;
            while state
                .current_token()
                .is_some_and(|t| t.kind != TokenKind::CloseParenthese)
            {
                if !can_continue {
                    return Err(state.err(ParseErrorKind::ListNotPunctuated(TokenKind::Comma)));
                }

                let e = parse_expression(state)?;
                let opt_comma_token = state.consume_by_kind(TokenKind::Comma).ok().cloned();
                can_continue = opt_comma_token.is_some();
                expressions.push((e, opt_comma_token));
            }

            let close_parenthese_token = state.consume_by_kind(TokenKind::CloseParenthese)?.clone();

            if expressions.len() == 1 && expressions.last().is_some_and(|(_, t)| t.is_none()) {
                // single and not terminated by ",": lifted expression
                Ok(Expression::Lifted(
                    open_parenthese_token,
                    Box::new(expressions.pop().unwrap().0),
                    close_parenthese_token,
                ))
            } else {
                Ok(Expression::Tuple(
                    open_parenthese_token,
                    expressions,
                    close_parenthese_token,
                ))
            }
        }
        t => Err(state.err_on(ParseErrorKind::ExpectedExpression, t)),
    }
}

#[derive(Debug)]
pub enum TokenizerErrorKind {
    IncompleteHexLiteral,
}

#[derive(Debug)]
pub struct TokenizerError {
    pub kind: TokenizerErrorKind,
    pub line: usize,
    pub col: usize,
}

struct Tokenizer<'s> {
    pub source: &'s str,
    pub line: usize,
    pub col: usize,
    pub current_line_indent: usize,
}
impl<'s> Tokenizer<'s> {
    pub fn populate_line_indent(&mut self) {
        let (line_indent_chars, line_indent_bytes) = self
            .source
            .chars()
            .take_while(|&c| c.is_whitespace() && c != '\n')
            .fold((0, 0), |(a, b), c| (a + 1, b + c.len_utf8()));
        self.current_line_indent = line_indent_chars;
        self.col = line_indent_chars;
        self.source = &self.source[line_indent_bytes..];
    }

    pub fn next_token(&mut self) -> Result<Option<Token<'s>>, TokenizerError> {
        let (head_space_chars, head_space_bytes) = self
            .source
            .chars()
            .take_while(|&c| c.is_whitespace() && c != '\n')
            .fold((0, 0), |(a, b), c| (a + 1, b + c.len_utf8()));
        self.col += head_space_chars;
        self.source = &self.source[head_space_bytes..];

        while self.source.starts_with('\n') {
            let lf_count = self
                .source
                .chars()
                .take_while(|&c| c == '\n' || c == '\r')
                .fold(0, |a, c| a + (if c == '\n' { 1 } else { 0 }));
            self.line += lf_count;
            self.source = self.source.trim_start_matches(|c| c == '\n' || c == '\r');
            self.populate_line_indent();
        }

        if self.source.is_empty() {
            return Ok(None);
        }

        if self.source.starts_with('#') {
            // line comment
            self.source = self.source.trim_start_matches(|c| c != '\n');
            return self.next_token();
        }

        let double_byte_tok = if self.source.as_bytes().len() >= 2 {
            match &self.source.as_bytes()[..2] {
                b"->" => Some(TokenKind::ArrowToRight),
                b"==" | b"!=" | b"<=" | b">=" | b"&&" | b"||" => Some(TokenKind::Op),
                _ => None,
            }
        } else {
            None
        };

        if let Some(k) = double_byte_tok {
            let tk = Token {
                slice: &self.source[..2],
                kind: k,
                line: self.line,
                col: self.col,
                line_indent: self.current_line_indent,
            };
            self.source = &self.source[2..];
            self.col += 2;
            return Ok(Some(tk));
        }

        let single_tok = match self.source.as_bytes()[0] {
            b'[' => Some(TokenKind::OpenBracket),
            b']' => Some(TokenKind::CloseBracket),
            b'(' => Some(TokenKind::OpenParenthese),
            b')' => Some(TokenKind::CloseParenthese),
            b'<' => Some(TokenKind::OpenAngleBracket),
            b'>' => Some(TokenKind::CloseAngleBracket),
            b',' => Some(TokenKind::Comma),
            b':' => Some(TokenKind::Colon),
            b'=' => Some(TokenKind::Eq),
            b'+' | b'-' | b'*' | b'/' | b'%' | b'&' | b'|' | b'^' | b'~' | b'!' | b'.' => {
                Some(TokenKind::Op)
            }
            _ => None,
        };
        if let Some(k) = single_tok {
            let tk = Token {
                slice: &self.source[..1],
                kind: k,
                line: self.line,
                col: self.col,
                line_indent: self.current_line_indent,
            };
            self.source = &self.source[1..];
            self.col += 1;
            return Ok(Some(tk));
        }

        if self.source.starts_with("0x") || self.source.starts_with("0X") {
            // hexlit
            let hexpart_count = self.source[2..]
                .chars()
                .take_while(|c| {
                    ('0'..='9').contains(c) || ('a'..='f').contains(c) || ('A'..='F').contains(c)
                })
                .count();
            if hexpart_count == 0 {
                return Err(TokenizerError {
                    kind: TokenizerErrorKind::IncompleteHexLiteral,
                    line: self.line,
                    col: self.col,
                });
            }

            let tk = Token {
                slice: &self.source[..2 + hexpart_count],
                kind: TokenKind::Number,
                line: self.line,
                col: self.col,
                line_indent: self.current_line_indent,
            };
            self.source = &self.source[2 + hexpart_count..];
            self.col += 2 + hexpart_count;
            return Ok(Some(tk));
        }

        if self.source.starts_with(|c: char| ('0'..='9').contains(&c)) {
            let ipart_count = self
                .source
                .chars()
                .take_while(|c| ('0'..='9').contains(c))
                .count();
            let extended_count = if self.source[ipart_count..].starts_with('.') {
                1 + self.source[ipart_count + 1..]
                    .chars()
                    .take_while(|c| ('0'..='9').contains(c))
                    .count()
            } else {
                0
            };

            let has_float_suffix =
                self.source[ipart_count + extended_count..].starts_with(['f', 'F']);
            let total_length = ipart_count + extended_count + if has_float_suffix { 1 } else { 0 };

            let tk = Token {
                slice: &self.source[..total_length],
                kind: TokenKind::Number,
                line: self.line,
                col: self.col,
                line_indent: self.current_line_indent,
            };
            self.source = &self.source[total_length..];
            self.col += total_length;
            return Ok(Some(tk));
        }

        let (ident_char_count, ident_byte_count) = self
            .source
            .chars()
            .take_while(|&c| !"=!\"#%&'()[]?><.,;:@=~-^|\\ \t\r\n".contains(c))
            .fold((0, 0), |(a, b), c| (a + 1, b + c.len_utf8()));
        assert!(
            ident_byte_count > 0,
            "empty identifier token(src: {}...)",
            &self.source[..8]
        );
        let tk = Token {
            slice: &self.source[..ident_byte_count],
            kind: match &self.source[..ident_byte_count] {
                "struct" | "with" | "if" | "else" | "then" | "do" | "let" => TokenKind::Keyword,
                _ => TokenKind::Identifier,
            },
            line: self.line,
            col: self.col,
            line_indent: self.current_line_indent,
        };
        self.source = &self.source[ident_byte_count..];
        self.col += ident_char_count;
        Ok(Some(tk))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Identifier,
    Keyword,
    Op,
    Number,
    OpenBracket,
    CloseBracket,
    OpenParenthese,
    CloseParenthese,
    OpenAngleBracket,
    CloseAngleBracket,
    Comma,
    Colon,
    ArrowToRight,
    Eq,
}

#[derive(Debug, Clone)]
pub struct Token<'s> {
    pub slice: &'s str,
    pub kind: TokenKind,
    pub line: usize,
    pub col: usize,
    pub line_indent: usize,
}
