use std::{
    collections::{HashMap, HashSet},
    io::Write,
};

use clap::Parser;
use codegen::{
    emit_block, emit_entry_point_spv_ops, emit_entry_point_spv_ops2, emit_shader_interface_vars,
    entrypoint::ShaderEntryPointDescription, SpvFunctionBodyEmissionContext,
    SpvModuleEmissionContext, SpvSectionLocalId,
};
use concrete_type::{ConcreteType, IntrinsicType, UserDefinedStructMember};
use ir::{
    block::{
        dump_blocks, dump_registers, parse_incoming_flows, Block, BlockFlowInstruction,
        BlockGenerationContext, BlockInstruction, BlockInstructionEmissionContext, BlockRef,
        RegisterRef,
    },
    expr::simplify_expression,
    opt::{
        apply_local_var_states, block_aliasing, build_register_state_map,
        build_scope_local_var_state, collect_scope_local_memory_usages, convert_static_path_ref,
        deconstruct_effectless_phi, extract_constants, fold_const_ops, inline_function2,
        merge_simple_goto_blocks, optimize_pure_expr, promote_instantiate_const,
        resolve_intrinsic_funcalls, resolve_register_aliases, resolve_shader_io_ref_binds,
        strip_unreachable_blocks, strip_unreferenced_registers, strip_write_only_local_memory,
        track_scope_local_var_aliases, transform_swizzle_component_store, unify_constants,
        unref_swizzle_ref_loads,
    },
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
    meta::{eval_symbol_attributes, ShaderModel, SymbolAttribute},
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
mod ir2;
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
                None => vec![(SymbolAttribute::default(), IntrinsicType::Unit.into())],
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
                let mut block_instruction_emission_context = BlockInstructionEmissionContext::new();
                let last = simplify_expression(
                    f.body,
                    &mut block_generation_context,
                    &mut block_instruction_emission_context,
                    function_symbol_scope,
                );
                let final_return_block = block_generation_context
                    .add(Block::flow_only(BlockFlowInstruction::Return(last.result)));
                assert!(
                    block_generation_context.try_chain(last.end_block, final_return_block),
                    "function body multiple out"
                );
                println!(
                    "Generated({}, last_start_block={} last_eval_block={}):",
                    f.fname_token.slice, last.start_block.0, final_return_block.0
                );
                let mut o = std::io::stdout().lock();
                block_generation_context
                    .dump_blocks(
                        &mut o,
                        &block_instruction_emission_context.registers,
                        &block_instruction_emission_context.instructions,
                    )
                    .unwrap();
                o.flush().unwrap();
                drop(o);

                let mut const_instructions =
                    extract_constants(&mut block_instruction_emission_context.instructions);

                println!("ConstInstPromotion:");
                println!("Constants:");
                for (k, v) in const_instructions.iter() {
                    println!("  r{}: {v:?}", k.0);
                }
                let mut o = std::io::stdout().lock();
                block_generation_context
                    .dump_blocks(
                        &mut o,
                        &block_instruction_emission_context.registers,
                        &block_instruction_emission_context.instructions,
                    )
                    .unwrap();
                o.flush().unwrap();
                drop(o);

                let mut last_ir_document = build_ir_document(
                    &block_instruction_emission_context.registers,
                    &const_instructions,
                    &block_generation_context.blocks,
                    &block_instruction_emission_context.instructions,
                );
                fn perform_log(
                    fname: &parser::Token,
                    step_name: &str,
                    modified: bool,
                    last_irdoc: &mut String,
                    blocks: &[Block],
                    instructions: &HashMap<RegisterRef, BlockInstruction>,
                    registers: &[ConcreteType],
                    const_map: &HashMap<RegisterRef, BlockInstruction>,
                ) {
                    print_step_header(fname.slice, step_name, modified);

                    if modified {
                        let next_irdoc =
                            build_ir_document(registers, const_map, blocks, instructions);

                        let diff = similar::TextDiff::from_lines(last_irdoc, &next_irdoc);
                        let mut o = std::io::stdout().lock();
                        for c in diff.iter_all_changes() {
                            let sign = match c.tag() {
                                similar::ChangeTag::Equal => " ",
                                similar::ChangeTag::Insert => "+",
                                similar::ChangeTag::Delete => "-",
                            };

                            write!(o, "{sign}{c}").unwrap();
                        }
                        o.flush().unwrap();
                        drop(o);

                        *last_irdoc = next_irdoc;
                    }
                }

                let mut needs_reopt = true;
                while needs_reopt {
                    needs_reopt = false;

                    loop {
                        let modified = promote_instantiate_const(
                            &mut block_instruction_emission_context.instructions,
                            &mut const_instructions,
                        );
                        needs_reopt = needs_reopt || modified;

                        perform_log(
                            &f.fname_token,
                            "PromoteInstantiateConst",
                            modified,
                            &mut last_ir_document,
                            &block_generation_context.blocks,
                            &block_instruction_emission_context.instructions,
                            &block_instruction_emission_context.registers,
                            &const_instructions,
                        );

                        if !modified {
                            break;
                        }
                    }

                    loop {
                        let modified = fold_const_ops(
                            &mut block_instruction_emission_context.instructions,
                            &mut const_instructions,
                        );
                        needs_reopt = needs_reopt || modified;

                        perform_log(
                            &f.fname_token,
                            "FoldConst",
                            modified,
                            &mut last_ir_document,
                            &block_generation_context.blocks,
                            &block_instruction_emission_context.instructions,
                            &block_instruction_emission_context.registers,
                            &const_instructions,
                        );

                        if !modified {
                            break;
                        }
                    }
                }

                {
                    let modified = unify_constants(
                        &mut block_generation_context.blocks,
                        &mut block_instruction_emission_context.instructions,
                        &const_instructions,
                    );

                    perform_log(
                        &f.fname_token,
                        "UnifyConstants",
                        modified,
                        &mut last_ir_document,
                        &block_generation_context.blocks,
                        &block_instruction_emission_context.instructions,
                        &block_instruction_emission_context.registers,
                        &const_instructions,
                    );
                }

                loop {
                    let modified = unref_swizzle_ref_loads(
                        &mut block_generation_context.blocks,
                        &mut block_instruction_emission_context.instructions,
                        &const_instructions,
                        &mut block_instruction_emission_context.registers,
                    );

                    perform_log(
                        &f.fname_token,
                        "UnrefSwizleRefLoads",
                        modified,
                        &mut last_ir_document,
                        &block_generation_context.blocks,
                        &block_instruction_emission_context.instructions,
                        &block_instruction_emission_context.registers,
                        &const_instructions,
                    );

                    if !modified {
                        break;
                    }
                }

                loop {
                    let modified = transform_swizzle_component_store(
                        &mut block_generation_context.blocks,
                        &mut block_instruction_emission_context.instructions,
                        &mut block_instruction_emission_context.registers,
                    );

                    perform_log(
                        &f.fname_token,
                        "TransformSwizzleComponentStore",
                        modified,
                        &mut last_ir_document,
                        &block_generation_context.blocks,
                        &block_instruction_emission_context.instructions,
                        &block_instruction_emission_context.registers,
                        &const_instructions,
                    );

                    if !modified {
                        break;
                    }
                }

                let local_scope_var_aliases = track_scope_local_var_aliases(
                    &block_generation_context.blocks,
                    &block_instruction_emission_context.instructions,
                    &const_instructions,
                );
                println!("Scope Var Aliases:");
                let mut sorted = local_scope_var_aliases.iter().collect::<Vec<_>>();
                sorted.sort_by_key(|p| p.0 .0);
                for (b, a) in sorted {
                    println!("  After b{}:", b.0);
                    for ((scope, id), r) in a.iter() {
                        println!("    {id} at {scope:?} = r{}", r.0);
                    }
                }

                let scope_local_var_states = build_scope_local_var_state(
                    &mut block_generation_context.blocks,
                    &mut block_instruction_emission_context.instructions,
                    &local_scope_var_aliases,
                    &mut block_instruction_emission_context.registers,
                );
                println!("Scope Local Var States:");
                let mut sorted = scope_local_var_states.iter().collect::<Vec<_>>();
                sorted.sort_by_key(|p| p.0 .0);
                for (b, a) in sorted {
                    println!("  Head of b{}:", b.0);
                    for ((scope, id), r) in a.iter() {
                        println!("    {id} at {scope:?} = {r:?}");
                    }
                }

                loop {
                    let modified = apply_local_var_states(
                        &mut block_generation_context.blocks,
                        &mut block_instruction_emission_context.instructions,
                        &const_instructions,
                        &scope_local_var_states,
                    );

                    perform_log(
                        &f.fname_token,
                        "ApplyLocalVarStates",
                        modified,
                        &mut last_ir_document,
                        &block_generation_context.blocks,
                        &block_instruction_emission_context.instructions,
                        &block_instruction_emission_context.registers,
                        &const_instructions,
                    );

                    if !modified {
                        break;
                    }
                }

                let local_memory_usages = collect_scope_local_memory_usages(
                    &block_generation_context.blocks,
                    &block_instruction_emission_context.instructions,
                    &const_instructions,
                );
                println!("LocalMemoryUsages:");
                for ((scope, id), usage) in local_memory_usages.iter() {
                    println!("  {id} @ {scope:?}: {usage:?}");
                }

                let modified = strip_write_only_local_memory(
                    &mut block_generation_context.blocks,
                    &const_instructions,
                    &local_memory_usages,
                );
                perform_log(
                    &f.fname_token,
                    "StripWriteOnlyLocalMemory",
                    modified,
                    &mut last_ir_document,
                    &block_generation_context.blocks,
                    &block_instruction_emission_context.instructions,
                    &block_instruction_emission_context.registers,
                    &const_instructions,
                );

                {
                    let modified = resolve_intrinsic_funcalls(
                        &mut block_generation_context.blocks,
                        &mut block_instruction_emission_context,
                        &const_instructions,
                    );

                    perform_log(
                        &f.fname_token,
                        "ResolveIntrinsicFuncalls",
                        modified,
                        &mut last_ir_document,
                        &block_generation_context.blocks,
                        &block_instruction_emission_context.instructions,
                        &block_instruction_emission_context.registers,
                        &const_instructions,
                    );
                }

                let mut needs_reopt = true;
                while needs_reopt {
                    needs_reopt = false;

                    loop {
                        let modified = merge_simple_goto_blocks(
                            &mut block_generation_context.blocks,
                            &mut block_instruction_emission_context.instructions,
                        );
                        needs_reopt |= modified;

                        perform_log(
                            &f.fname_token,
                            "MergeSimpleGotoBlocks",
                            modified,
                            &mut last_ir_document,
                            &block_generation_context.blocks,
                            &block_instruction_emission_context.instructions,
                            &block_instruction_emission_context.registers,
                            &const_instructions,
                        );

                        if !modified {
                            break;
                        }
                    }

                    loop {
                        let modified = block_aliasing(
                            &mut block_generation_context.blocks,
                            &mut block_instruction_emission_context.instructions,
                        );
                        needs_reopt |= modified;

                        perform_log(
                            &f.fname_token,
                            "BlockAliasing",
                            modified,
                            &mut last_ir_document,
                            &block_generation_context.blocks,
                            &block_instruction_emission_context.instructions,
                            &block_instruction_emission_context.registers,
                            &const_instructions,
                        );

                        if !modified {
                            break;
                        }
                    }

                    {
                        let modified = deconstruct_effectless_phi(
                            &mut block_instruction_emission_context.instructions,
                        );
                        needs_reopt = needs_reopt || modified;

                        perform_log(
                            &f.fname_token,
                            "DeconstructEffectlessPhi",
                            modified,
                            &mut last_ir_document,
                            &block_generation_context.blocks,
                            &block_instruction_emission_context.instructions,
                            &block_instruction_emission_context.registers,
                            &const_instructions,
                        );
                    }

                    loop {
                        let register_state_map = build_register_state_map(
                            &block_generation_context.blocks,
                            &mut block_instruction_emission_context.instructions,
                        );
                        let modified = resolve_register_aliases(
                            &mut block_generation_context.blocks,
                            &mut block_instruction_emission_context.instructions,
                            &register_state_map,
                        );
                        needs_reopt = needs_reopt || modified;

                        perform_log(
                            &f.fname_token,
                            "ResolveRegisterAliases",
                            modified,
                            &mut last_ir_document,
                            &block_generation_context.blocks,
                            &block_instruction_emission_context.instructions,
                            &block_instruction_emission_context.registers,
                            &const_instructions,
                        );

                        if !modified {
                            break;
                        }
                    }
                }

                loop {
                    let modified = strip_unreachable_blocks(
                        &mut block_generation_context.blocks,
                        &mut block_instruction_emission_context.instructions,
                    );

                    perform_log(
                        &f.fname_token,
                        "StripUnreachableBlocks",
                        modified,
                        &mut last_ir_document,
                        &block_generation_context.blocks,
                        &block_instruction_emission_context.instructions,
                        &block_instruction_emission_context.registers,
                        &const_instructions,
                    );

                    if !modified {
                        break;
                    }
                }

                loop {
                    let modified = strip_unreferenced_registers(
                        &mut block_generation_context.blocks,
                        &mut block_instruction_emission_context.instructions,
                        &mut block_instruction_emission_context.registers,
                        &mut const_instructions,
                    );

                    perform_log(
                        &f.fname_token,
                        "StripUnreferencedRegisters",
                        modified,
                        &mut last_ir_document,
                        &block_generation_context.blocks,
                        &block_instruction_emission_context.instructions,
                        &block_instruction_emission_context.registers,
                        &const_instructions,
                    );

                    if !modified {
                        break;
                    }
                }

                println!("Optimized({:?}):", f.fname_token);
                println!("Registers:");
                let mut o = std::io::stdout().lock();
                for (n, r) in block_instruction_emission_context
                    .registers
                    .iter()
                    .enumerate()
                {
                    match const_instructions.get(&RegisterRef(n)) {
                        Some(c) => writeln!(o, "  r{n}: {r:?} = {c:?}").unwrap(),
                        None => writeln!(o, "  r{n}: {r:?}").unwrap(),
                    }
                }
                dump_blocks(
                    &mut o,
                    &block_generation_context.blocks,
                    &block_instruction_emission_context.instructions,
                )
                .unwrap();
                o.flush().unwrap();
                drop(o);

                if false {
                    let incoming_flows = parse_incoming_flows(&block_generation_context.blocks);
                    println!("Incoming Flows:");
                    for (src, dst) in incoming_flows.iter() {
                        println!(
                            "  b{} <- {}",
                            src.0,
                            dst.iter()
                                .map(|b| format!("b{}", b.0))
                                .collect::<Vec<_>>()
                                .join(", ")
                        );
                    }
                }

                if fn_symbol.attribute.module_entry_point {
                    match fn_symbol.attribute.shader_model.unwrap() {
                        ShaderModel::ComputeShader => {
                            assert_eq!(
                                fn_symbol.output,
                                vec![(SymbolAttribute::default(), IntrinsicType::Unit.into())]
                            );
                        }
                        _ => {
                            unimplemented!("non-compute-shader module entry point epilogue");
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
                    }
                }

                top_scope.attach_function_body(
                    f.fname_token.slice,
                    FunctionBody {
                        symbol_scope: function_symbol_scope,
                        registers: block_instruction_emission_context.registers,
                        constants: const_instructions,
                        instructions: block_instruction_emission_context.instructions,
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

    struct OptimizedShaderEntryPoint<'s> {
        pub info: ShaderEntryPointDescription<'s>,
        pub ir2: ir2::Function<'s>,
    }

    let optimized_entry_points = entry_points
        .into_iter()
        .map(|ep| {
            let sym = top_scope.user_defined_function_symbol(ep.name).unwrap();
            let body = top_scope
                .user_defined_function_body(ep.name)
                .expect("cannot emit entry point without body");
            optimize(&ep, sym, &mut body.borrow_mut(), &symbol_scope_arena);
            let ir2 = ir2::reconstruct(
                &body.borrow().blocks,
                &body.borrow().instructions,
                &body.borrow().registers,
                &body.borrow().constants,
            );

            println!("IR2({}):", ep.name);
            let mut o = std::io::stdout().lock();
            ir2::Inst::dump_list(&ir2.instructions, &mut o).unwrap();
            o.flush().unwrap();
            drop(o);

            OptimizedShaderEntryPoint { info: ep, ir2 }
        })
        .collect::<Vec<_>>();

    let mut spv_context = SpvModuleEmissionContext::new();
    spv_context
        .capabilities
        .insert(spv::asm::Capability::Shader);
    spv_context
        .capabilities
        .insert(spv::asm::Capability::InputAttachment);
    for e in optimized_entry_points {
        unimplemented!("codegen");

        // let shader_if = emit_shader_interface_vars(sym, &body.borrow(), &mut spv_context);
        // let mut body_context = SpvFunctionBodyEmissionContext::new(&mut spv_context);
        // let main_label_id = body_context.new_id();
        // body_context.ops.push(spv::Instruction::Label {
        //     result: main_label_id,
        // });
        // emit_block(
        //     &shader_if,
        //     &body.borrow().constants,
        //     &body.borrow().blocks,
        //     BlockRef(0),
        //     &mut body_context,
        // );
        // // body_context.ops.push(spv::Instruction::Return);
        // let SpvFunctionBodyEmissionContext {
        //     latest_id: body_latest_id,
        //     ops: body_ops,
        //     ..
        // } = body_context;

        // let fn_result_ty = spv_context.request_type_id(spv::Type::Void);
        // let fnty = spv_context.request_type_id(spv::Type::Function {
        //     return_type: Box::new(spv::Type::Void),
        //     parameter_types: Vec::new(),
        // });
        // let fnid = spv_context.new_function_id();
        // spv_context.function_ops.push(spv::Instruction::Function {
        //     result_type: fn_result_ty,
        //     result: fnid,
        //     function_control: spv::asm::FunctionControl::empty(),
        //     function_type: fnty,
        // });
        // let fnid_offset = spv_context.latest_function_id;
        // spv_context.latest_function_id += body_latest_id;
        // spv_context
        //     .function_ops
        //     .extend(body_ops.into_iter().map(|x| {
        //         x.relocate(|id| match id {
        //             SpvSectionLocalId::CurrentFunction(x) => {
        //                 SpvSectionLocalId::Function(x + fnid_offset)
        //             }
        //             x => x,
        //         })
        //     }));
        // spv_context.function_ops.push(spv::Instruction::FunctionEnd);
        // spv_context
        //     .entry_point_ops
        //     .push(spv::Instruction::EntryPoint {
        //         execution_model: e.execution_model,
        //         entry_point: fnid,
        //         name: e.name.into(),
        //         interface: shader_if.iter_interface_global_vars().copied().collect(),
        //     });
        // spv_context
        //     .execution_mode_ops
        //     .extend(e.execution_mode_modifiers.iter().map(|m| match m {
        //         spv::ExecutionModeModifier::OriginUpperLeft => spv::Instruction::ExecutionMode {
        //             entry_point: fnid,
        //             mode: spv::asm::ExecutionMode::OriginUpperLeft,
        //             args: Vec::new(),
        //         },
        //     }));
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

#[inline(always)]
fn print_step_header(fname: &str, step_name: &str, modified: bool) {
    println!(
        "{step_name}({fname}, {}):",
        if modified { "performed" } else { "skip" }
    );
}
fn build_ir_document(
    registers: &[ConcreteType],
    const_map: &HashMap<RegisterRef, BlockInstruction>,
    blocks: &[Block],
    instructions: &HashMap<RegisterRef, BlockInstruction>,
) -> String {
    use std::io::Write;

    let mut sink = Vec::new();
    for (rn, rt) in registers.iter().enumerate() {
        match const_map.get(&RegisterRef(rn)) {
            Some(c) => writeln!(sink, "  r{rn}: {rt:?} = {c:?}").unwrap(),
            None => writeln!(sink, "  r{rn}: {rt:?}").unwrap(),
        }
    }

    dump_blocks(&mut sink, blocks, instructions).unwrap();

    unsafe { String::from_utf8_unchecked(sink) }
}

fn optimize<'a, 's>(
    ep: &ShaderEntryPointDescription,
    f: &UserDefinedFunctionSymbol<'s>,
    body: &mut FunctionBody<'a, 's>,
    scope_arena: &'a Arena<SymbolScope<'a, 's>>,
) {
    let refpath_binds = ep
        .global_variables
        .inputs
        .iter()
        .map(|v| (&v.original_refpath, &v.decorations))
        .chain(
            ep.global_variables
                .uniforms
                .iter()
                .map(|v| (&v.original_refpath, &v.decorations)),
        )
        .collect::<HashMap<_, _>>();

    let mut last_ir_document = build_ir_document(
        &body.registers,
        &body.constants,
        &body.blocks,
        &body.instructions,
    );
    fn perform_log(
        f: &UserDefinedFunctionSymbol,
        step_name: &str,
        modified: bool,
        last_irdoc: &mut String,
        body: &FunctionBody,
    ) {
        print_step_header(&f.name(), step_name, modified);

        if modified {
            let new_irdoc = build_ir_document(
                &body.registers,
                &body.constants,
                &body.blocks,
                &body.instructions,
            );

            let mut o = std::io::stdout().lock();
            let diff = similar::TextDiff::from_lines(last_irdoc, &new_irdoc);
            for c in diff.iter_all_changes() {
                let sign = match c.tag() {
                    similar::ChangeTag::Equal => " ",
                    similar::ChangeTag::Insert => "+",
                    similar::ChangeTag::Delete => "-",
                };

                write!(o, "{sign} {c}").unwrap();
            }
            o.flush().unwrap();
            drop(o);

            *last_irdoc = new_irdoc;
        }
    }

    loop {
        let modified = inline_function2(
            &mut body.blocks,
            &mut body.instructions,
            &mut body.constants,
            scope_arena,
            body.symbol_scope,
            &mut body.registers,
        );

        perform_log(f, "InlineFunction", modified, &mut last_ir_document, body);

        if !modified {
            break;
        }
    }

    println!("refpath binds: {refpath_binds:#?}");

    loop {
        let modified = unref_swizzle_ref_loads(
            &mut body.blocks,
            &mut body.instructions,
            &body.constants,
            &mut body.registers,
        );

        perform_log(
            f,
            "UnrefSwizzleRefLoads",
            modified,
            &mut last_ir_document,
            body,
        );

        if !modified {
            break;
        }
    }

    loop {
        let modified = transform_swizzle_component_store(
            &mut body.blocks,
            &mut body.instructions,
            &mut body.registers,
        );

        perform_log(
            f,
            "TransformSwizzleComponentStore",
            modified,
            &mut last_ir_document,
            body,
        );

        if !modified {
            break;
        }
    }

    let local_scope_var_aliases =
        track_scope_local_var_aliases(&body.blocks, &body.instructions, &body.constants);
    println!("Scope Var Aliases:");
    let mut sorted = local_scope_var_aliases.iter().collect::<Vec<_>>();
    sorted.sort_by_key(|p| p.0 .0);
    for (b, a) in sorted {
        println!("  After b{}:", b.0);
        for ((scope, id), r) in a.iter() {
            println!("    {id} at {scope:?} = r{}", r.0);
        }
    }

    let scope_local_var_states = build_scope_local_var_state(
        &mut body.blocks,
        &mut body.instructions,
        &local_scope_var_aliases,
        &mut body.registers,
    );
    println!("Scope Local Var States:");
    let mut sorted = scope_local_var_states.iter().collect::<Vec<_>>();
    sorted.sort_by_key(|p| p.0 .0);
    for (b, a) in sorted {
        println!("  Head of b{}:", b.0);
        for ((scope, id), r) in a.iter() {
            println!("    {id} at {scope:?} = {r:?}");
        }
    }

    loop {
        let modified = apply_local_var_states(
            &mut body.blocks,
            &mut body.instructions,
            &body.constants,
            &scope_local_var_states,
        );

        perform_log(
            f,
            "ApplyLocalVarStates",
            modified,
            &mut last_ir_document,
            body,
        );

        if !modified {
            break;
        }
    }

    let local_memory_usages =
        collect_scope_local_memory_usages(&body.blocks, &body.instructions, &body.constants);
    println!("LocalMemoryUsages:");
    for ((scope, id), usage) in local_memory_usages.iter() {
        println!("  {id} @ {scope:?}: {usage:?}");
    }

    let modified =
        strip_write_only_local_memory(&mut body.blocks, &body.constants, &local_memory_usages);

    perform_log(
        f,
        "StripWriteOnlyLocalMemory",
        modified,
        &mut last_ir_document,
        body,
    );

    loop {
        let modified = resolve_shader_io_ref_binds(
            &f.inputs,
            body.symbol_scope,
            &refpath_binds,
            &mut body.blocks,
            &mut body.instructions,
            &mut body.constants,
            &body.registers,
        );

        perform_log(
            f,
            "ResolveShaderIORefBinds",
            modified,
            &mut last_ir_document,
            body,
        );

        if !modified {
            break;
        }
    }

    let mut needs_reopt = true;
    while needs_reopt {
        needs_reopt = false;

        loop {
            let modified = promote_instantiate_const(&mut body.instructions, &mut body.constants);
            needs_reopt = needs_reopt || modified;

            perform_log(
                f,
                "PromoteInstantiateConst",
                modified,
                &mut last_ir_document,
                body,
            );

            if !modified {
                break;
            }
        }

        loop {
            let modified = fold_const_ops(&mut body.instructions, &mut body.constants);
            needs_reopt = needs_reopt || modified;

            perform_log(f, "FoldConstants", modified, &mut last_ir_document, body);

            if !modified {
                break;
            }
        }

        loop {
            let modified = unify_constants(
                &mut body.blocks,
                &mut body.instructions,
                &mut body.constants,
            );
            needs_reopt = needs_reopt || modified;

            perform_log(f, "UnifyConstants", modified, &mut last_ir_document, body);

            if !modified {
                break;
            }
        }

        loop {
            let modified = merge_simple_goto_blocks(&mut body.blocks, &mut body.instructions);
            needs_reopt |= modified;

            perform_log(
                f,
                "MergeSimpleGotoBlocks",
                modified,
                &mut last_ir_document,
                body,
            );

            if !modified {
                break;
            }
        }

        loop {
            let modified = block_aliasing(&mut body.blocks, &mut body.instructions);
            needs_reopt = needs_reopt || modified;

            perform_log(f, "BlockAliasing", modified, &mut last_ir_document, body);

            if !modified {
                break;
            }
        }

        loop {
            let modified = deconstruct_effectless_phi(&mut body.instructions);
            needs_reopt = needs_reopt || modified;

            perform_log(
                f,
                "DeconstructEffectlessPhi",
                modified,
                &mut last_ir_document,
                body,
            );

            if !modified {
                break;
            }
        }

        loop {
            let register_state_map = build_register_state_map(&body.blocks, &mut body.instructions);
            let modified = resolve_register_aliases(
                &mut body.blocks,
                &mut body.instructions,
                &register_state_map,
            );
            needs_reopt = needs_reopt || modified;

            perform_log(
                f,
                "ResolveRegisterAliases",
                modified,
                &mut last_ir_document,
                body,
            );

            if !modified {
                break;
            }
        }
    }

    loop {
        let modified = strip_unreachable_blocks(&mut body.blocks, &mut body.instructions);
        needs_reopt = needs_reopt || modified;

        perform_log(
            f,
            "StripUnreachableBlocks",
            modified,
            &mut last_ir_document,
            body,
        );

        if !modified {
            break;
        }
    }

    loop {
        let modified = strip_unreferenced_registers(
            &mut body.blocks,
            &mut body.instructions,
            &mut body.registers,
            &mut body.constants,
        );

        perform_log(
            f,
            "StripUnreferencedRegisters",
            modified,
            &mut last_ir_document,
            body,
        );

        if !modified {
            break;
        }
    }
}
