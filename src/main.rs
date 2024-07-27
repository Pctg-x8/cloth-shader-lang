use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    io::Write,
};

use clap::Parser;
use codegen::{
    emit_entry_point_spv_ops, emit_function_body_spv_ops, entrypoint::ShaderEntryPointDescription,
    SpvFunctionBodyEmissionContext, SpvModuleEmissionContext, SpvSectionLocalId,
};
use concrete_type::{ConcreteType, IntrinsicType, UserDefinedStructMember};
use ir::{
    expr::{simplify_expression, SimplificationContext, SimplifiedExpression},
    opt::{inline_function1, optimize_pure_expr},
    ExprRef, FunctionBody,
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
use utils::PtrEq;
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

struct UserFunctionCallGraph<'a, 's, 'g> {
    scope: PtrEq<'a, SymbolScope<'a, 's>>,
    name: &'s str,
    calling: RefCell<
        HashMap<(PtrEq<'a, SymbolScope<'a, 's>>, &'s str), &'g UserFunctionCallGraph<'a, 's, 'g>>,
    >,
}
impl<'a, 's, 'g> UserFunctionCallGraph<'a, 's, 'g> {
    pub fn new(scope: &'a SymbolScope<'a, 's>, name: &'s str) -> Self {
        Self {
            scope: PtrEq(scope),
            name,
            calling: RefCell::new(HashMap::new()),
        }
    }
}

fn collect_function_call_graph<'a, 's, 'g>(
    body: &[(SimplifiedExpression<'a, 's>, ConcreteType<'s>)],
    graph_root: &'g UserFunctionCallGraph<'a, 's, 'g>,
    graph_arena: &'g Arena<UserFunctionCallGraph<'a, 's, 'g>>,
) -> HashMap<ExprRef, (PtrEq<'a, SymbolScope<'a, 's>>, &'s str)> {
    let mut last_reffunction = HashMap::<ExprRef, (PtrEq<'a, SymbolScope<'a, 's>>, &'s str)>::new();

    for (n, (expr, _)) in body.iter().enumerate() {
        match expr {
            &SimplifiedExpression::RefFunction(scope, f) => {
                last_reffunction.insert(ExprRef(n), (scope, f));
            }
            &SimplifiedExpression::Funcall(base, _) => {
                if let Some(&(scope, name)) = last_reffunction.get(&base) {
                    let child_graph = graph_arena.alloc(UserFunctionCallGraph::new(scope.0, name));
                    graph_root
                        .calling
                        .borrow_mut()
                        .insert((scope, name), child_graph);
                    if let Some(body) = scope.0.user_defined_function_body(name) {
                        collect_function_call_graph(
                            &body.borrow().expressions,
                            child_graph,
                            graph_arena,
                        );
                    }
                }
            }
            &SimplifiedExpression::ScopedBlock {
                ref expressions,
                returning,
                ..
            } => {
                let mut inner_reffunctions =
                    collect_function_call_graph(expressions, graph_root, graph_arena);
                if let Some((scope, name)) = inner_reffunctions.remove(&returning) {
                    // escaping reffunction value
                    last_reffunction.insert(ExprRef(n), (scope, name));
                }
            }
            _ => (),
        }
    }

    last_reffunction
}

fn print_call_graph<'a, 's, 'g>(root: &'g UserFunctionCallGraph<'a, 's, 'g>) {
    fn rec<'a, 's, 'g>(
        current: &'g UserFunctionCallGraph<'a, 's, 'g>,
        text_stack: &mut Vec<String>,
        occurence_stack: &mut Vec<(PtrEq<'a, SymbolScope<'a, 's>>, &'s str)>,
    ) {
        text_stack.push(format!("{}@{:?}", current.name, current.scope));
        if occurence_stack.contains(&(current.scope, current.name)) {
            println!("{} (looped)", text_stack.join(" - "));
        } else if current.calling.borrow().is_empty() {
            println!("{}", text_stack.join(" - "));
        } else {
            occurence_stack.push((current.scope, current.name));
            for v in current.calling.borrow().values() {
                rec(v, text_stack, occurence_stack)
            }
            occurence_stack.pop();
        }
        text_stack.pop();
    }

    rec(root, &mut Vec::new(), &mut Vec::new());
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
                                panic!(
                                    "Error: output type mismatching({last_var_type:?} /= {:?})",
                                    fn_symbol.output[0].1
                                );
                            }

                            let output = fn_symbol.flatten_output(function_symbol_scope);
                            match output.len() {
                                0 => panic!("module entry point must output at least one value"),
                                1 => {
                                    last_var_id = simplify_context.add(
                                        SimplifiedExpression::StoreOutput(last_var_id, 0),
                                        IntrinsicType::Unit.into(),
                                    );
                                }
                                _ => {
                                    last_var_id = simplify_context.add(
                                        SimplifiedExpression::FlattenAndDistributeOutputComposite(
                                            last_var_id,
                                            (0..output.len()).collect(),
                                        ),
                                        IntrinsicType::Unit.into(),
                                    );
                                }
                            }
                            last_var_type = IntrinsicType::Unit.into();
                        }
                        _ => {
                            if last_var_type
                                != ConcreteType::Tuple(
                                    fn_symbol.output.iter().map(|x| x.1.clone()).collect(),
                                )
                            {
                                panic!(
                                    "Error: output type mismatching({last_var_type:?} /= {:?})",
                                    ConcreteType::Tuple(
                                        fn_symbol.output.iter().map(|x| x.1.clone()).collect(),
                                    )
                                );
                            }

                            let output = fn_symbol.flatten_output(function_symbol_scope);
                            match output.len() {
                                0 => panic!("module entry point must output at least one value"),
                                1 => {
                                    last_var_id = simplify_context.add(
                                        SimplifiedExpression::StoreOutput(last_var_id, 0),
                                        IntrinsicType::Unit.into(),
                                    );
                                }
                                _ => {
                                    last_var_id = simplify_context.add(
                                        SimplifiedExpression::FlattenAndDistributeOutputComposite(
                                            last_var_id,
                                            (0..output.len()).collect(),
                                        ),
                                        IntrinsicType::Unit.into(),
                                    );
                                }
                            }
                            last_var_type = IntrinsicType::Unit.into();
                        }
                    }
                }

                println!("ir body({}):", f.fname_token.slice);
                for (n, (x, t)) in simplify_context.vars.iter().enumerate() {
                    ir::expr::print_simp_expr(&mut std::io::stdout(), x, t, n, 0);
                }

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

    let function_call_graph_arena = Arena::new();
    let call_graph_per_entry = top_scope
        .iter_user_defined_function_symbols()
        .filter_map(|f| {
            if !f.attribute.module_entry_point {
                return None;
            }

            let graph_root =
                function_call_graph_arena.alloc(UserFunctionCallGraph::new(top_scope, f.name()));
            let Some(body) = top_scope.user_defined_function_body(f.name()) else {
                panic!("entry point function must have its body");
            };
            collect_function_call_graph(
                &body.borrow().expressions,
                &*graph_root,
                &function_call_graph_arena,
            );
            Some(&*graph_root)
        })
        .collect::<Vec<_>>();
    println!("call graphs:");
    for c in call_graph_per_entry.iter() {
        print_call_graph(c);
    }

    let mut spv_context = SpvModuleEmissionContext::new();
    spv_context
        .capabilities
        .insert(spv::asm::Capability::Shader);
    spv_context
        .capabilities
        .insert(spv::asm::Capability::InputAttachment);
    for e in entry_points {
        let body = top_scope
            .user_defined_function_body(e.name)
            .expect("cannot emit entry point without body");
        optimize(&mut body.borrow_mut(), &symbol_scope_arena);
        let entry_point_maps = emit_entry_point_spv_ops(&e, &mut spv_context);
        let mut body_context =
            SpvFunctionBodyEmissionContext::new(&mut spv_context, entry_point_maps);
        let main_label_id = body_context.new_id();
        body_context.ops.push(spv::Instruction::Label {
            result: main_label_id,
        });
        emit_function_body_spv_ops(&body.borrow().expressions, &mut body_context);
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
            function_control: spv::asm::FunctionControl::empty(),
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
                    mode: spv::asm::ExecutionMode::OriginUpperLeft,
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

fn optimize<'a, 's>(body: &mut FunctionBody<'a, 's>, scope_arena: &'a Arena<SymbolScope<'a, 's>>) {
    inline_function1(&mut body.expressions);

    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "inline function1:").unwrap();
    for (n, (x, t)) in body.expressions.iter().enumerate() {
        ir::expr::print_simp_expr(&mut stdout, x, t, n, 0);
    }
    stdout.flush().unwrap();
    drop(stdout);

    optimize_pure_expr(
        &mut body.expressions,
        body.symbol_scope,
        scope_arena,
        Some(&mut body.returning),
    );

    let mut stdout = std::io::stdout().lock();
    writeln!(stdout, "optimized:").unwrap();
    for (n, (x, t)) in body.expressions.iter().enumerate() {
        ir::expr::print_simp_expr(&mut stdout, x, t, n, 0);
    }
    writeln!(
        stdout,
        "returning {:?}: {:?}",
        body.returning, body.returning_type
    )
    .unwrap();
    stdout.flush().unwrap();
}
