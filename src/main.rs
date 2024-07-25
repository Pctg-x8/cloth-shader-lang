use std::{collections::HashSet, io::Write};

use clap::Parser;
use codegen::{
    emit_entry_point_spv_ops, emit_function_body_spv_ops, entrypoint::ShaderEntryPointDescription,
    SpvFunctionBodyEmissionContext, SpvModuleEmissionContext, SpvSectionLocalId,
};
use concrete_type::{ConcreteType, IntrinsicType, UserDefinedStructMember};
use ir::{
    expr::{simplify_expression, SimplificationContext, SimplifiedExpression},
    opt::optimize_pure_expr,
    FunctionBody,
};
use parser::{
    FunctionDeclarationInputArguments, FunctionDeclarationOutput, ParseState, Tokenizer,
    ToplevelDeclaration,
};
use scope::SymbolScope;
use source_ref::{SourceRef, SourceRefSliceEq};
use symbol::{
    meta::{eval_symbol_attributes, SymbolAttribute},
    UserDefinedFunctionSymbol,
};
use typed_arena::Arena;

mod parser;
mod spirv;
use spirv as spv;
mod codegen;
mod concrete_type;
mod ir;
mod ref_path;
mod scope;
mod source_ref;
mod symbol;
mod utils;

#[derive(Parser)]
pub struct Args {
    pub file_name: std::path::PathBuf,
}

fn main() {
    let args = Args::parse();

    let src = std::fs::read_to_string(&args.file_name).expect("Failed to load source");
    let mut tokenizer = Tokenizer::new(&src);
    let mut tokens = Vec::new();
    while let Some(t) = tokenizer.next_token().unwrap() {
        tokens.push(t);
    }

    let mut parse_state = ParseState::new(tokens);
    let mut tlds = Vec::new();
    while parse_state.current_token().is_some() {
        let tld = ToplevelDeclaration::parse(&mut parse_state).unwrap();
        // println!("tld: {tld:#?}");
        tlds.push(tld);
    }

    let symbol_scope_arena = Arena::new();
    let global_symbol_scope = symbol_scope_arena.alloc(SymbolScope::new_intrinsics());

    let mut toplevel_opaque_types = HashSet::new();
    let mut user_defined_function_nodes = Vec::new();
    for tld in tlds.iter() {
        match tld {
            ToplevelDeclaration::Struct(s) => {
                toplevel_opaque_types.insert(s.name_token.slice);
            }
            ToplevelDeclaration::Function(f) => user_defined_function_nodes.push(f),
        }
    }

    let top_scope = symbol_scope_arena.alloc(SymbolScope::new(Some(global_symbol_scope), false));
    for tld in tlds.iter() {
        match tld {
            ToplevelDeclaration::Struct(s) => {
                top_scope.declare_struct(
                    SourceRef::from(&s.name_token),
                    s.member_list
                        .iter()
                        .map(|x| UserDefinedStructMember {
                            attribute: x
                                .iter_attributes()
                                .fold(SymbolAttribute::default(), |attrs, a| {
                                    eval_symbol_attributes(attrs, a.clone())
                                }),
                            name: SourceRefSliceEq(SourceRef::from(&x.name_token)),
                            ty: ConcreteType::build(
                                global_symbol_scope,
                                &toplevel_opaque_types,
                                x.ty.clone(),
                            ),
                        })
                        .collect(),
                );
            }
            ToplevelDeclaration::Function(_) => {}
        }
    }
    for f in user_defined_function_nodes {
        top_scope.declare_function(UserDefinedFunctionSymbol {
            occurence: SourceRef::from(&f.fname_token),
            attribute: f
                .iter_attributes()
                .fold(SymbolAttribute::default(), |attrs, a| {
                    eval_symbol_attributes(attrs, a.clone())
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
                        .flat_map(|xs| xs.attribute_list.iter().map(|(a, _)| a))
                        .fold(SymbolAttribute::default(), |attrs, a| {
                            eval_symbol_attributes(attrs, a.clone())
                        }),
                    SourceRef::from(varname_token),
                    ConcreteType::build(global_symbol_scope, &toplevel_opaque_types, ty.clone())
                        .instantiate(&top_scope),
                )],
                FunctionDeclarationInputArguments::Multiple { args, .. } => args
                    .iter()
                    .map(|(attribute_lists, varname_token, _, ty, _)| {
                        (
                            attribute_lists
                                .iter()
                                .flat_map(|xs| xs.attribute_list.iter().map(|(a, _)| a))
                                .fold(SymbolAttribute::default(), |attrs, a| {
                                    eval_symbol_attributes(attrs, a.clone())
                                }),
                            SourceRef::from(varname_token),
                            ConcreteType::build(
                                global_symbol_scope,
                                &toplevel_opaque_types,
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
                        .flat_map(|xs| xs.attribute_list.iter().map(|(a, _)| a))
                        .fold(SymbolAttribute::default(), |attrs, a| {
                            eval_symbol_attributes(attrs, a.clone())
                        }),
                    ConcreteType::build(global_symbol_scope, &toplevel_opaque_types, ty.clone())
                        .instantiate(&top_scope),
                )],
                Some(FunctionDeclarationOutput::Tupled { elements, .. }) => elements
                    .iter()
                    .map(|(attribute_lists, ty, _)| {
                        (
                            attribute_lists
                                .iter()
                                .flat_map(|xs| xs.attribute_list.iter().map(|(a, _)| a))
                                .fold(SymbolAttribute::default(), |attrs, a| {
                                    eval_symbol_attributes(attrs, a.clone())
                                }),
                            ConcreteType::build(
                                global_symbol_scope,
                                &toplevel_opaque_types,
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
                let fn_symbol = top_scope
                    .user_defined_function_symbol(f.fname_token.slice)
                    .unwrap();
                let function_symbol_scope =
                    symbol_scope_arena.alloc(SymbolScope::new(Some(top_scope), true));
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
                if fn_symbol.attribute.module_entry_point {
                    match fn_symbol.output.len() {
                        0 => panic!("module entry point must output at least one value"),
                        1 => {
                            if last_var_type != fn_symbol.output[0].1 {
                                panic!("Error: output type mismatching");
                            }

                            last_var_id = simplify_context.add(
                                SimplifiedExpression::StoreOutput(last_var_id, 0),
                                IntrinsicType::Unit.into(),
                            );
                            last_var_type = IntrinsicType::Unit.into();
                        }
                        _ => {
                            if last_var_type
                                != ConcreteType::Tuple(
                                    fn_symbol.output.iter().map(|x| x.1.clone()).collect(),
                                )
                            {
                                panic!("Error: output type mismatching");
                            }

                            last_var_id = simplify_context.add(
                                SimplifiedExpression::DistributeOutputTuple(
                                    last_var_id,
                                    (0..fn_symbol.output.len()).collect(),
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

    let entry_points = top_scope
        .iter_user_defined_function_symbols()
        .filter_map(|f| {
            if !f.attribute.module_entry_point {
                return None;
            }

            Some(ShaderEntryPointDescription::extract(f, &top_scope))
        })
        .collect::<Vec<_>>();

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
    for e in entry_points {
        let body = top_scope
            .user_defined_function_body(e.name)
            .expect("cannot emit entry point without body");
        let entry_point_maps = emit_entry_point_spv_ops(&e, &mut spv_context);
        let mut body_context =
            SpvFunctionBodyEmissionContext::new(&mut spv_context, entry_point_maps);
        let main_label_id = body_context.new_id();
        body_context.ops.push(spv::Instruction::Label {
            result: main_label_id,
        });
        emit_function_body_spv_ops(&body.expressions, body.returning, &mut body_context);
        body_context.ops.push(spv::Instruction::Return);
        let SpvFunctionBodyEmissionContext {
            latest_id: body_latest_id,
            ops: body_ops,
            entry_point_maps,
            ..
        } = body_context;

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
        spv_context.latest_function_id += body_latest_id;
        spv_context
            .function_ops
            .extend(body_ops.into_iter().map(|x| {
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
                execution_model: e.execution_model,
                entry_point: fnid,
                name: e.name.into(),
                interface: entry_point_maps
                    .iter_interface_global_vars()
                    .copied()
                    .collect(),
            });
        spv_context
            .execution_mode_ops
            .extend(e.execution_mode_modifiers.iter().map(|m| match m {
                spv::ExecutionModeModifier::OriginUpperLeft => spv::Instruction::ExecutionMode {
                    entry_point: fnid,
                    mode: spv::ExecutionMode::OriginUpperLeft,
                    args: Vec::new(),
                },
            }));
    }

    let (module_ops, max_id) = spv_context.serialize_ops();

    let outfile = std::fs::File::options()
        .create(true)
        .write(true)
        .truncate(true)
        .open("out.spv")
        .expect("Failed to open outfile");
    let mut writer = std::io::BufWriter::new(outfile);
    module_header(max_id + 1)
        .serialize(&mut writer)
        .expect("Failed to serialize module header");
    for x in module_ops.iter() {
        x.serialize_binary(&mut writer)
            .expect("Failed to serialize op");
    }
    writer.flush().expect("Failed to flush bufwriter");
}

#[inline]
const fn module_header(bound: spv::Id) -> spv::BinaryModuleHeader {
    spv::BinaryModuleHeader {
        magic_number: spv::BinaryModuleHeader::MAGIC_NUMBER,
        major_version: 1,
        minor_version: 4,
        generator_magic_number: 0,
        bound,
    }
}

// fn print_simp_expr(x: &SimplifiedExpression, ty: &ConcreteType, vid: usize, nested: usize) {
//     match x {
//         SimplifiedExpression::ScopedBlock {
//             expressions,
//             returning,
//             symbol_scope,
//         } => {
//             println!("  {}%{vid}: {ty:?} = Scope {{", "  ".repeat(nested));
//             println!("  {}Function Inputs:", "  ".repeat(nested + 1));
//             for (n, a) in symbol_scope.0.function_input_vars.iter().enumerate() {
//                 println!(
//                     "  {}  {n} = {}: {:?}",
//                     "  ".repeat(nested + 1),
//                     a.occurence.slice,
//                     a.ty
//                 );
//             }
//             println!("  {}Local Vars:", "  ".repeat(nested + 1));
//             for (n, a) in symbol_scope.0.local_vars.borrow().iter().enumerate() {
//                 println!(
//                     "  {}  {n} = {}: {:?}",
//                     "  ".repeat(nested + 1),
//                     a.occurence.slice,
//                     a.ty
//                 );
//             }
//             for (n, (x, t)) in expressions.iter().enumerate() {
//                 print_simp_expr(x, t, n, nested + 1);
//             }
//             println!("  {}returning {returning:?}", "  ".repeat(nested + 1));
//             println!("  {}}}", "  ".repeat(nested));
//         }
//         _ => println!("  {}%{vid}: {ty:?} = {x:?}", "  ".repeat(nested)),
//     }
// }
