use std::{collections::HashSet, io::Write};

use clap::Parser;
use codegen::{
    emit_entry_point_spv_ops, entrypoint::ShaderEntryPointDescription,
    SpvFunctionBodyEmissionContext, SpvModuleEmissionContext, SpvSectionLocalId,
};
use concrete_type::{ConcreteType, UserDefinedStructMember};
use ir::{
    block::{Block, BlockFlowInstruction, BlockGenerationContext},
    expr::simplify_expression,
    opt::{inline_function1, optimize_pure_expr},
    FunctionBody,
};
use parser::{
    CompilationUnit, FunctionDeclarationInputArguments, FunctionDeclarationOutput, ParseState,
    Tokenizer, ToplevelDeclaration,
};
use ref_path::RefPath;
use scope::SymbolScope;
use source_ref::{SourceRef, SourceRefSliceEq};
use symbol::{
    meta::{eval_symbol_attributes, SymbolAttribute},
    UserDefinedFunctionSymbol,
};
use typed_arena::Arena;

mod const_expr;
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

    let compilation_unit = CompilationUnit::parse(ParseState::new(tokens)).unwrap();

    let symbol_scope_arena = Arena::new();
    let global_symbol_scope = symbol_scope_arena.alloc(SymbolScope::new_intrinsics());

    let mut toplevel_opaque_types = HashSet::new();
    let mut user_defined_function_nodes = Vec::new();
    for tld in compilation_unit.declarations.iter() {
        match tld {
            ToplevelDeclaration::Struct(s) => {
                toplevel_opaque_types.insert(s.name_token.slice);
            }
            ToplevelDeclaration::Function(f) => user_defined_function_nodes.push(f),
        }
    }

    let top_scope = symbol_scope_arena.alloc(SymbolScope::new(Some(global_symbol_scope), false));
    for tld in compilation_unit.declarations.iter() {
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
                            mutable: x.mut_token.is_some(),
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
                    mut_token,
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
                    mut_token.is_some(),
                    SourceRef::from(varname_token),
                    ConcreteType::build(global_symbol_scope, &toplevel_opaque_types, ty.clone())
                        .instantiate(&top_scope),
                )],
                FunctionDeclarationInputArguments::Multiple { args, .. } => args
                    .iter()
                    .map(|(attribute_lists, mut_token, varname_token, _, ty, _)| {
                        (
                            attribute_lists
                                .iter()
                                .flat_map(|xs| xs.attribute_list.iter().map(|(a, _)| a))
                                .fold(SymbolAttribute::default(), |attrs, a| {
                                    eval_symbol_attributes(attrs, a.clone())
                                }),
                            mut_token.is_some(),
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

    for d in compilation_unit.declarations {
        match d {
            ToplevelDeclaration::Function(f) => {
                let fn_symbol = top_scope
                    .user_defined_function_symbol(f.fname_token.slice)
                    .unwrap();
                let function_symbol_scope =
                    symbol_scope_arena.alloc(SymbolScope::new(Some(top_scope), true));
                match f.input_args {
                    FunctionDeclarationInputArguments::Single {
                        varname_token,
                        ty,
                        mut_token,
                        ..
                    } => {
                        function_symbol_scope.declare_function_input(
                            SourceRef::from(&varname_token),
                            ConcreteType::build(function_symbol_scope, &HashSet::new(), ty)
                                .instantiate(&function_symbol_scope),
                            mut_token.is_some(),
                        );
                    }
                    FunctionDeclarationInputArguments::Multiple { args, .. } => {
                        for (_, mut_token, n, _, ty, _) in args {
                            function_symbol_scope.declare_function_input(
                                SourceRef::from(&n),
                                ConcreteType::build(function_symbol_scope, &HashSet::new(), ty)
                                    .instantiate(&function_symbol_scope),
                                mut_token.is_some(),
                            );
                        }
                    }
                }
                let mut block_generation_context = BlockGenerationContext::new(&symbol_scope_arena);
                let last = simplify_expression(
                    f.body,
                    &mut block_generation_context,
                    function_symbol_scope,
                );
                let final_return_block = block_generation_context.add(Block {
                    instructions: vec![],
                    flow: BlockFlowInstruction::Return(last.result),
                });
                assert!(
                    block_generation_context.try_chain(last.end_block, final_return_block),
                    "function body multiple out"
                );
                println!(
                    "Generated({}, last_start_block={} last_eval_block={final_return_block}):",
                    f.fname_token.slice, last.start_block
                );
                let mut o = std::io::stdout().lock();
                block_generation_context.dump_blocks(&mut o).unwrap();
                o.flush().unwrap();
                drop(o);
                if fn_symbol.attribute.module_entry_point {
                    unimplemented!("module entry point epilogue");
                    // match fn_symbol.output.len() {
                    //     0 => (),
                    //     // 0 => panic!("module entry point must output at least one value"),
                    //     1 => {
                    //         if last_var.ty != fn_symbol.output[0].1 {
                    //             panic!(
                    //                 "Error: output type mismatching({:?} /= {:?})",
                    //                 last_var.ty, fn_symbol.output[0].1
                    //             );
                    //         }

                    //         let output = fn_symbol.flatten_output(function_symbol_scope);
                    //         match output.len() {
                    //             0 => panic!("module entry point must output at least one value"),
                    //             1 => {
                    //                 last_var = simplify_context
                    //                     .add(
                    //                         SimplifiedExpression::StoreOutput(last_var.id, 0),
                    //                         IntrinsicType::Unit.into(),
                    //                     )
                    //                     .typed(IntrinsicType::Unit.into());
                    //             }
                    //             _ => {
                    //                 last_var = simplify_context.add(
                    //                     SimplifiedExpression::FlattenAndDistributeOutputComposite(
                    //                         last_var.id,
                    //                         (0..output.len()).collect(),
                    //                     ),
                    //                     IntrinsicType::Unit.into(),
                    //                 ).typed(IntrinsicType::Unit.into());
                    //             }
                    //         }
                    //     }
                    //     _ => {
                    //         if last_var.ty
                    //             != ConcreteType::Tuple(
                    //                 fn_symbol.output.iter().map(|x| x.1.clone()).collect(),
                    //             )
                    //         {
                    //             panic!(
                    //                 "Error: output type mismatching({:?} /= {:?})",
                    //                 last_var.ty,
                    //                 ConcreteType::Tuple(
                    //                     fn_symbol.output.iter().map(|x| x.1.clone()).collect(),
                    //                 )
                    //             );
                    //         }

                    //         let output = fn_symbol.flatten_output(function_symbol_scope);
                    //         match output.len() {
                    //             0 => panic!("module entry point must output at least one value"),
                    //             1 => {
                    //                 last_var = simplify_context
                    //                     .add(
                    //                         SimplifiedExpression::StoreOutput(last_var.id, 0),
                    //                         IntrinsicType::Unit.into(),
                    //                     )
                    //                     .typed(IntrinsicType::Unit.into());
                    //             }
                    //             _ => {
                    //                 last_var = simplify_context.add(
                    //                     SimplifiedExpression::FlattenAndDistributeOutputComposite(
                    //                         last_var.id,
                    //                         (0..output.len()).collect(),
                    //                     ),
                    //                     IntrinsicType::Unit.into(),
                    //                 ).typed(IntrinsicType::Unit.into());
                    //             }
                    //         }
                    //     }
                    // }
                }

                top_scope.attach_function_body(
                    f.fname_token.slice,
                    FunctionBody {
                        symbol_scope: function_symbol_scope,
                        registers: block_generation_context.registers,
                        blocks: block_generation_context.blocks,
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
    spv_context
        .capabilities
        .insert(spv::asm::Capability::Shader);
    spv_context
        .capabilities
        .insert(spv::asm::Capability::InputAttachment);
    for e in entry_points {
        let refpaths = e
            .global_variables
            .inputs
            .iter()
            .map(|x| &x.original_refpath)
            .chain(
                e.global_variables
                    .uniforms
                    .iter()
                    .map(|x| &x.original_refpath),
            )
            .chain(
                e.global_variables
                    .push_constants
                    .iter()
                    .map(|x| &x.original_refpath),
            )
            .chain(
                e.global_variables
                    .workgroup_shared_vars
                    .iter()
                    .map(|x| &x.original_refpath),
            )
            .cloned()
            .collect::<HashSet<_>>();
        let body = top_scope
            .user_defined_function_body(e.name)
            .expect("cannot emit entry point without body");
        optimize(&mut body.borrow_mut(), &symbol_scope_arena, &refpaths);
        println!("refpaths: {refpaths:?}");
        let entry_point_maps = emit_entry_point_spv_ops(&e, &mut spv_context);
        let mut body_context =
            SpvFunctionBodyEmissionContext::new(&mut spv_context, entry_point_maps);
        let main_label_id = body_context.new_id();
        body_context.ops.push(spv::Instruction::Label {
            result: main_label_id,
        });
        unimplemented!("emit_function_body_spv_ops");
        // emit_function_body_spv_ops(&body.borrow().expressions, &mut body_context);
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

fn optimize<'a, 's>(
    body: &mut FunctionBody<'a, 's>,
    scope_arena: &'a Arena<SymbolScope<'a, 's>>,
    refpaths: &HashSet<RefPath>,
) {
    unimplemented!("optimize");
    // while inline_function1(&mut body.expressions, scope_arena) {}

    // let mut stdout = std::io::stdout().lock();
    // writeln!(stdout, "inline function1:").unwrap();
    // for (n, (x, t)) in body.expressions.iter().enumerate() {
    //     ir::expr::print_simp_expr(&mut stdout, x, t, n, 0);
    // }
    // stdout.flush().unwrap();
    // drop(stdout);

    // optimize_pure_expr(
    //     &mut body.expressions,
    //     body.symbol_scope,
    //     scope_arena,
    //     Some(&mut body.returning.id),
    //     refpaths,
    // );

    // let mut stdout = std::io::stdout().lock();
    // writeln!(stdout, "optimized:").unwrap();
    // for (n, (x, t)) in body.expressions.iter().enumerate() {
    //     ir::expr::print_simp_expr(&mut stdout, x, t, n, 0);
    // }
    // writeln!(stdout, "returning {:?}", body.returning).unwrap();
    // stdout.flush().unwrap();
}
