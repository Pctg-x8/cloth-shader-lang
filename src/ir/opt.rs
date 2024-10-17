mod r#const;

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::{
    concrete_type::ConcreteType,
    ir::block::BlockFlowInstruction,
    ref_path::RefPath,
    scope::{SymbolScope, VarId},
    symbol::{
        meta::{BuiltinInputOutput, SymbolAttribute},
        UserDefinedFunctionSymbol,
    },
    utils::PtrEq,
};

pub use self::r#const::*;

use super::block::{
    BaseRegisters, Block, BlockConstInstruction, BlockInstruction, BlockPureInstruction, BlockRef,
    BlockifiedProgram, Constants, ImpureInstructionMap, PureInstructions, RegisterAliasMap,
    RegisterRef,
};

/// 同じPure命令を若い番号のregisterにまとめる
pub fn unify_pure_instructions(prg: &BlockifiedProgram) -> RegisterAliasMap {
    let mut inst_to_register_map = HashMap::new();
    let mut register_alias_map = HashMap::new();
    for (r, v) in prg.pure_instructions.iter().enumerate() {
        match inst_to_register_map.entry(v) {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(r);
            }
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let last_occurence = *e.get();
                if last_occurence > r {
                    // 若い番号を優先する
                    let mapped = e.insert(r);
                    register_alias_map.insert(RegisterRef::Pure(mapped), RegisterRef::Pure(r));
                } else {
                    register_alias_map
                        .insert(RegisterRef::Pure(r), RegisterRef::Pure(last_occurence));
                }
            }
        }
    }

    register_alias_map
}

/// 同じブロックにあるLoad命令を若い番号のレジスタにまとめる
pub fn unify_same_block_load_instructions(prg: &BlockifiedProgram) -> RegisterAliasMap {
    let mut register_alias_map = HashMap::new();

    for b in prg.blocks.iter() {
        let mut load_to_register_map = HashMap::new();

        for r in b.eval_impure_registers.iter().copied() {
            match prg.impure_instructions[&r] {
                BlockInstruction::LoadRef(lr) => {
                    match load_to_register_map.entry(lr) {
                        std::collections::hash_map::Entry::Vacant(e) => {
                            e.insert(r);
                        }
                        std::collections::hash_map::Entry::Occupied(mut e) => {
                            let last_occurence = *e.get();
                            if last_occurence > r {
                                // 若い番号を優先する
                                let mapped = e.insert(r);
                                register_alias_map
                                    .insert(RegisterRef::Impure(mapped), RegisterRef::Impure(r));
                            } else {
                                register_alias_map.insert(
                                    RegisterRef::Impure(r),
                                    RegisterRef::Impure(last_occurence),
                                );
                            }
                        }
                    }
                }
                _ => (),
            }
        }
    }

    register_alias_map
}

/// 使われていないPureレジスタを削除
pub fn strip_unreferenced_pure_instructions(prg: &mut BlockifiedProgram) -> bool {
    let mut unreferenced = prg
        .collect_unreferenced_pure_registers()
        .into_iter()
        .collect::<Vec<_>>();
    unreferenced.sort_by(|a, b| b.cmp(a));
    println!("[StripPureInst] Targets: {unreferenced:?}");
    let mut register_alias_map = HashMap::new();
    for n in unreferenced {
        register_alias_map.insert(
            RegisterRef::Pure(prg.pure_instructions.len() - 1),
            RegisterRef::Pure(n),
        );
        println!(
            "[StripPureInst] swap remove {} -> {}",
            prg.pure_instructions.len() - 1,
            n
        );
        prg.pure_instructions.swap_remove(n);
    }

    prg.apply_register_alias(&register_alias_map);
    !register_alias_map.is_empty()
}

/// 使われていないImpure命令とレジスタを削除
pub fn strip_unreferenced_impure_instructions(prg: &mut BlockifiedProgram) -> bool {
    let unreferenced = prg.collect_unreferenced_impure_registers();

    for b in prg.blocks.iter_mut() {
        b.eval_impure_registers
            .retain(|n| !unreferenced.contains(n));
    }

    let mut unreferenced = unreferenced.into_iter().collect::<Vec<_>>();
    unreferenced.sort_by(|a, b| b.cmp(a));
    println!("[StripImpureInst] Targets: {unreferenced:?}");
    let mut register_alias_map = HashMap::new();
    let mut modified = false;
    for n in unreferenced {
        let last_register = prg.impure_registers.len() - 1;
        if last_register != n {
            // 最後のレジスタではない（入れ替えが発生する）
            register_alias_map.insert(RegisterRef::Impure(last_register), RegisterRef::Impure(n));
            println!("[StripImpureInst] swap remove {last_register} -> {n}");
            if let Some(last_inst) = prg.impure_instructions.remove(&last_register) {
                // Note: ない場合がある（BlockFlowInstructionのdestレジスタなど）
                prg.impure_instructions.insert(n, last_inst);
            }
            prg.impure_registers.swap_remove(n);

            for b in prg.blocks.iter_mut() {
                if b.eval_impure_registers.remove(&last_register) {
                    b.eval_impure_registers.insert(n);
                }
                b.flow.relocate_dest_register(|r| {
                    if *r == RegisterRef::Impure(last_register) {
                        *r = RegisterRef::Impure(n);
                    }
                });
            }
        } else {
            println!("[StripImpureInst] last remove {n}");
            prg.impure_instructions.remove(&n);
            prg.impure_registers.pop();
        }

        modified = true;
    }

    prg.apply_register_alias(&register_alias_map);
    modified
}

pub fn collect_block_incomings(blocks: &[Block]) -> HashMap<BlockRef, HashSet<BlockRef>> {
    fn process(
        blocks: &[Block],
        processing: BlockRef,
        incoming: BlockRef,
        loop_stack: &mut Vec<(BlockRef, BlockRef)>,
        merge_stack: &mut Vec<BlockRef>,
        collect: &mut HashMap<BlockRef, HashSet<BlockRef>>,
    ) {
        match collect.entry(processing) {
            std::collections::hash_map::Entry::Vacant(e) => {
                // 初めての処理
                e.insert(HashSet::new()).insert(incoming);
            }
            std::collections::hash_map::Entry::Occupied(mut e) => {
                // 以前にも処理したことがある
                e.get_mut().insert(incoming);
                return;
            }
        }

        match blocks[processing.0].flow {
            BlockFlowInstruction::Goto(next) => {
                process(blocks, next, processing, loop_stack, merge_stack, collect)
            }
            BlockFlowInstruction::StoreRef {
                after: Some(next), ..
            } => process(blocks, next, processing, loop_stack, merge_stack, collect),
            BlockFlowInstruction::StoreRef { after: None, .. } => (),
            BlockFlowInstruction::Funcall {
                after_return: Some(next),
                ..
            } => process(blocks, next, processing, loop_stack, merge_stack, collect),
            BlockFlowInstruction::Funcall {
                after_return: None, ..
            } => (),
            BlockFlowInstruction::IntrinsicImpureFuncall {
                after_return: Some(next),
                ..
            } => process(blocks, next, processing, loop_stack, merge_stack, collect),
            BlockFlowInstruction::IntrinsicImpureFuncall {
                after_return: None, ..
            } => (),
            BlockFlowInstruction::Return(_) => (),
            BlockFlowInstruction::Conditional {
                r#true,
                r#false,
                merge,
                ..
            } => {
                merge_stack.push(merge);
                process(
                    blocks,
                    r#true,
                    processing,
                    loop_stack,
                    &mut merge_stack.clone(),
                    collect,
                );
                process(
                    blocks,
                    r#false,
                    processing,
                    loop_stack,
                    &mut merge_stack.clone(),
                    collect,
                );
                merge_stack.pop();
            }
            BlockFlowInstruction::ConditionalEnd => {
                let m = merge_stack.pop().expect("not in conditional branch");
                process(blocks, m, processing, loop_stack, merge_stack, collect);
            }
            BlockFlowInstruction::ConditionalLoop {
                r#break,
                r#continue,
                body,
                ..
            } => {
                loop_stack.push((r#continue, r#break));
                process(blocks, body, processing, loop_stack, merge_stack, collect);
                loop_stack.pop();

                process(
                    blocks,
                    r#break,
                    processing,
                    loop_stack,
                    merge_stack,
                    collect,
                );
            }
            BlockFlowInstruction::Continue => {
                let &(next, _) = loop_stack.last().expect("continue outside a loop");
                let mut new_stack = loop_stack[..loop_stack.len() - 1].to_vec();
                process(
                    blocks,
                    next,
                    processing,
                    &mut new_stack,
                    merge_stack,
                    collect,
                );
            }
            BlockFlowInstruction::Break => {
                let &(_, next) = loop_stack.last().expect("break outside a loop");
                let mut new_stack = loop_stack[..loop_stack.len() - 1].to_vec();
                process(
                    blocks,
                    next,
                    processing,
                    &mut new_stack,
                    merge_stack,
                    collect,
                );
            }
            BlockFlowInstruction::Undetermined => (),
        }
    }

    let mut collect = HashMap::new();
    let mut loop_stack = Vec::new();
    let mut merge_stack = Vec::new();

    match blocks[0].flow {
        BlockFlowInstruction::Goto(next) => {
            process(
                blocks,
                next,
                BlockRef(0),
                &mut loop_stack,
                &mut merge_stack,
                &mut collect,
            );
        }
        BlockFlowInstruction::StoreRef {
            after: Some(next), ..
        } => process(
            blocks,
            next,
            BlockRef(0),
            &mut loop_stack,
            &mut merge_stack,
            &mut collect,
        ),
        BlockFlowInstruction::StoreRef { after: None, .. } => (),
        BlockFlowInstruction::Funcall {
            after_return: Some(next),
            ..
        } => process(
            blocks,
            next,
            BlockRef(0),
            &mut loop_stack,
            &mut merge_stack,
            &mut collect,
        ),
        BlockFlowInstruction::Funcall {
            after_return: None, ..
        } => (),
        BlockFlowInstruction::IntrinsicImpureFuncall {
            after_return: Some(next),
            ..
        } => process(
            blocks,
            next,
            BlockRef(0),
            &mut loop_stack,
            &mut merge_stack,
            &mut collect,
        ),
        BlockFlowInstruction::IntrinsicImpureFuncall {
            after_return: None, ..
        } => (),
        BlockFlowInstruction::Return(_) => (),
        BlockFlowInstruction::Conditional {
            r#true,
            r#false,
            merge,
            ..
        } => {
            merge_stack.push(merge);
            process(
                blocks,
                r#true,
                BlockRef(0),
                &mut loop_stack,
                &mut merge_stack.clone(),
                &mut collect,
            );
            process(
                blocks,
                r#false,
                BlockRef(0),
                &mut loop_stack,
                &mut merge_stack.clone(),
                &mut collect,
            );
            merge_stack.pop();
        }
        BlockFlowInstruction::ConditionalEnd => {
            let m = merge_stack.pop().expect("not in conditional branch");
            process(
                blocks,
                m,
                BlockRef(0),
                &mut loop_stack,
                &mut merge_stack,
                &mut collect,
            );
        }
        BlockFlowInstruction::ConditionalLoop {
            r#break,
            r#continue,
            body,
            ..
        } => {
            loop_stack.push((r#continue, r#break));
            process(
                blocks,
                body,
                BlockRef(0),
                &mut loop_stack,
                &mut merge_stack,
                &mut collect,
            );
            loop_stack.pop();

            process(
                blocks,
                r#break,
                BlockRef(0),
                &mut loop_stack,
                &mut merge_stack,
                &mut collect,
            );
        }
        BlockFlowInstruction::Continue => {
            let &(next, _) = loop_stack.last().expect("continue outside a loop");
            let mut new_stack = loop_stack[..loop_stack.len() - 1].to_vec();
            process(
                blocks,
                next,
                BlockRef(0),
                &mut new_stack,
                &mut merge_stack,
                &mut collect,
            );
        }
        BlockFlowInstruction::Break => {
            let &(_, next) = loop_stack.last().expect("break outside a loop");
            let mut new_stack = loop_stack[..loop_stack.len() - 1].to_vec();
            process(
                blocks,
                next,
                BlockRef(0),
                &mut new_stack,
                &mut merge_stack,
                &mut collect,
            );
        }
        BlockFlowInstruction::Undetermined => (),
    }

    collect
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LocalMemoryIdentifier<'a, 's> {
    Var(PtrEq<'a, SymbolScope<'a, 's>>, usize),
    FunArg(PtrEq<'a, SymbolScope<'a, 's>>, usize),
}

pub fn collect_block_local_memory_stores<'a, 's>(
    prg: &BlockifiedProgram<'a, 's>,
) -> HashMap<BlockRef, (LocalMemoryIdentifier<'a, 's>, RegisterRef)> {
    let mut collect = HashMap::new();

    for (n, b) in prg.blocks.iter().enumerate() {
        match b.flow {
            BlockFlowInstruction::StoreRef {
                ptr: RegisterRef::Const(r),
                value,
                ..
            } => match prg.constants[r].inst {
                BlockConstInstruction::ScopeLocalVarRef(scope, vid) => {
                    collect.insert(BlockRef(n), (LocalMemoryIdentifier::Var(scope, vid), value));
                }
                BlockConstInstruction::FunctionInputVarRef(scope, id) => {
                    collect.insert(
                        BlockRef(n),
                        (LocalMemoryIdentifier::FunArg(scope, id), value),
                    );
                }
                _ => (),
            },
            _ => (),
        }
    }

    collect
}

/// ローカルメモリ（ローカル変数/引数）への代入をトラックして、前のブロックからのphi命令を生成して固有のレジスタを割り当てる
///
/// 無駄なphiは後工程で消すので、ここでは一旦何も考えずに個々のブロックですべてのローカルメモリに対してphi命令を生成して唯一のレジスタを割り当てます
/// （ローカルメモリに紐づくレジスタが途中で変わると後のブロック全てで変える必要があり大変面倒なため）
pub fn propagate_local_memory_stores<'a, 's>(
    prg: &mut BlockifiedProgram<'a, 's>,
    block_local_memory_stores: &HashMap<BlockRef, (LocalMemoryIdentifier<'a, 's>, RegisterRef)>,
) -> HashMap<BlockRef, HashMap<LocalMemoryIdentifier<'a, 's>, RegisterRef>> {
    struct BlockMetadata<'a, 's> {
        pub local_mem_aliased_registers: HashMap<LocalMemoryIdentifier<'a, 's>, RegisterRef>,
    }

    fn process<'a, 's>(
        prg: &mut BlockifiedProgram<'a, 's>,
        block_local_memory_stores: &HashMap<BlockRef, (LocalMemoryIdentifier<'a, 's>, RegisterRef)>,
        processing: BlockRef,
        incoming: BlockRef,
        meta: &mut HashMap<BlockRef, BlockMetadata<'a, 's>>,
        loop_stack: &mut Vec<(BlockRef, BlockRef)>,
        merge_stack: &mut Vec<BlockRef>,
    ) {
        // 流入ブロックでの最終的なローカルメモリマップ（流入ブロックにおけるincoming state + 流入ブロックの変数代入操作）
        let incoming_stores = meta[&incoming]
            .local_mem_aliased_registers
            .iter()
            .chain(
                block_local_memory_stores
                    .get(&incoming)
                    .map(|(a, b)| (a, b)),
            )
            .map(|(k, v)| (k.clone(), *v))
            .collect::<Vec<_>>();

        match meta.entry(processing) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                // 再訪
                let bm = e.get_mut();

                for (k, v) in incoming_stores {
                    let ty = v.ty(prg).clone();

                    match bm.local_mem_aliased_registers.entry(k.clone()) {
                        std::collections::hash_map::Entry::Vacant(e) => {
                            // 新しいLocalMemStore
                            prg.impure_registers.push(ty);
                            let vreg = prg.impure_registers.len() - 1;
                            prg.impure_instructions.insert(
                                vreg,
                                BlockInstruction::Phi([(incoming, v)].into_iter().collect()),
                            );
                            prg.blocks[processing.0].eval_impure_registers.insert(vreg);
                            e.insert(RegisterRef::Impure(vreg));
                        }
                        std::collections::hash_map::Entry::Occupied(e) => {
                            // すでにあるLocalMemStore
                            let RegisterRef::Impure(vreg) = e.get() else {
                                unreachable!("not imported entry");
                            };
                            assert_eq!(prg.impure_registers[*vreg], ty);
                            let BlockInstruction::Phi(ref mut incomings) =
                                prg.impure_instructions.get_mut(&vreg).unwrap()
                            else {
                                unreachable!("not a phi entry");
                            };

                            incomings.insert(incoming, v);
                        }
                    }
                }

                return;
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                // 初
                let bm = e.insert(BlockMetadata {
                    local_mem_aliased_registers: HashMap::new(),
                });

                for (k, v) in incoming_stores {
                    let ty = v.ty(prg).clone();

                    match bm.local_mem_aliased_registers.entry(k.clone()) {
                        std::collections::hash_map::Entry::Vacant(e) => {
                            // 新しいLocalMemStore
                            prg.impure_registers.push(ty);
                            let vreg = prg.impure_registers.len() - 1;
                            prg.impure_instructions.insert(
                                vreg,
                                BlockInstruction::Phi([(incoming, v)].into_iter().collect()),
                            );
                            prg.blocks[processing.0].eval_impure_registers.insert(vreg);
                            e.insert(RegisterRef::Impure(vreg));
                        }
                        std::collections::hash_map::Entry::Occupied(e) => {
                            // すでにあるLocalMemStore
                            let RegisterRef::Impure(vreg) = e.get() else {
                                unreachable!("not imported entry");
                            };
                            assert_eq!(prg.impure_registers[*vreg], ty);
                            let BlockInstruction::Phi(ref mut incomings) =
                                prg.impure_instructions.get_mut(&vreg).unwrap()
                            else {
                                unreachable!("not a phi entry");
                            };

                            incomings.insert(incoming, v);
                        }
                    }
                }

                match prg.blocks[processing.0].flow {
                    BlockFlowInstruction::Goto(next) => process(
                        prg,
                        block_local_memory_stores,
                        next,
                        processing,
                        meta,
                        loop_stack,
                        merge_stack,
                    ),
                    BlockFlowInstruction::StoreRef {
                        after: Some(next), ..
                    } => process(
                        prg,
                        block_local_memory_stores,
                        next,
                        processing,
                        meta,
                        loop_stack,
                        merge_stack,
                    ),
                    BlockFlowInstruction::StoreRef { after: None, .. } => (),
                    BlockFlowInstruction::Funcall {
                        after_return: Some(after_return),
                        ..
                    } => process(
                        prg,
                        block_local_memory_stores,
                        after_return,
                        processing,
                        meta,
                        loop_stack,
                        merge_stack,
                    ),
                    BlockFlowInstruction::Funcall {
                        after_return: None, ..
                    } => (),
                    BlockFlowInstruction::IntrinsicImpureFuncall {
                        after_return: Some(after_return),
                        ..
                    } => process(
                        prg,
                        block_local_memory_stores,
                        after_return,
                        processing,
                        meta,
                        loop_stack,
                        merge_stack,
                    ),
                    BlockFlowInstruction::IntrinsicImpureFuncall {
                        after_return: None, ..
                    } => (),
                    BlockFlowInstruction::Conditional {
                        r#true,
                        r#false,
                        merge,
                        ..
                    } => {
                        merge_stack.push(merge);
                        process(
                            prg,
                            block_local_memory_stores,
                            r#true,
                            processing,
                            meta,
                            loop_stack,
                            &mut merge_stack.clone(),
                        );
                        process(
                            prg,
                            block_local_memory_stores,
                            r#false,
                            processing,
                            meta,
                            loop_stack,
                            &mut merge_stack.clone(),
                        );
                        merge_stack.pop();
                    }
                    BlockFlowInstruction::ConditionalEnd => {
                        let m = merge_stack.pop().expect("not in conditional branch");
                        process(
                            prg,
                            block_local_memory_stores,
                            m,
                            processing,
                            meta,
                            loop_stack,
                            merge_stack,
                        );
                    }
                    BlockFlowInstruction::ConditionalLoop {
                        r#break,
                        r#continue,
                        body,
                        ..
                    } => {
                        loop_stack.push((r#continue, r#break));
                        process(
                            prg,
                            block_local_memory_stores,
                            body,
                            processing,
                            meta,
                            loop_stack,
                            merge_stack,
                        );
                        loop_stack.pop();
                        process(
                            prg,
                            block_local_memory_stores,
                            r#break,
                            processing,
                            meta,
                            loop_stack,
                            merge_stack,
                        );
                    }
                    BlockFlowInstruction::Continue => {
                        let &(c, _) = loop_stack.last().expect("continue not in a loop");
                        let mut new_loop_stack = loop_stack[..loop_stack.len() - 1].to_vec();
                        process(
                            prg,
                            block_local_memory_stores,
                            c,
                            processing,
                            meta,
                            &mut new_loop_stack,
                            merge_stack,
                        );
                    }
                    BlockFlowInstruction::Break => {
                        let &(_, b) = loop_stack.last().expect("break not in a loop");
                        let mut new_loop_stack = loop_stack[..loop_stack.len() - 1].to_vec();
                        process(
                            prg,
                            block_local_memory_stores,
                            b,
                            processing,
                            meta,
                            &mut new_loop_stack,
                            merge_stack,
                        );
                    }
                    BlockFlowInstruction::Return(_) | BlockFlowInstruction::Undetermined => (),
                }
            }
        }
    }

    let mut meta = HashMap::new();
    let mut loop_stack = Vec::new();
    let mut merge_stack = Vec::new();

    // b0 has empty state
    meta.insert(
        BlockRef(0),
        BlockMetadata {
            local_mem_aliased_registers: HashMap::new(),
        },
    );

    match prg.blocks[0].flow {
        BlockFlowInstruction::Goto(next) => process(
            prg,
            block_local_memory_stores,
            next,
            BlockRef(0),
            &mut meta,
            &mut loop_stack,
            &mut merge_stack,
        ),
        BlockFlowInstruction::StoreRef {
            after: Some(next), ..
        } => process(
            prg,
            block_local_memory_stores,
            next,
            BlockRef(0),
            &mut meta,
            &mut loop_stack,
            &mut merge_stack,
        ),
        BlockFlowInstruction::StoreRef { after: None, .. } => (),
        BlockFlowInstruction::Funcall {
            after_return: Some(after_return),
            ..
        } => process(
            prg,
            block_local_memory_stores,
            after_return,
            BlockRef(0),
            &mut meta,
            &mut loop_stack,
            &mut merge_stack,
        ),
        BlockFlowInstruction::Funcall {
            after_return: None, ..
        } => (),
        BlockFlowInstruction::IntrinsicImpureFuncall {
            after_return: Some(after_return),
            ..
        } => process(
            prg,
            block_local_memory_stores,
            after_return,
            BlockRef(0),
            &mut meta,
            &mut loop_stack,
            &mut merge_stack,
        ),
        BlockFlowInstruction::IntrinsicImpureFuncall {
            after_return: None, ..
        } => (),
        BlockFlowInstruction::Conditional {
            r#true,
            r#false,
            merge,
            ..
        } => {
            merge_stack.push(merge);
            process(
                prg,
                block_local_memory_stores,
                r#true,
                BlockRef(0),
                &mut meta,
                &mut loop_stack,
                &mut merge_stack.clone(),
            );
            process(
                prg,
                block_local_memory_stores,
                r#false,
                BlockRef(0),
                &mut meta,
                &mut loop_stack,
                &mut merge_stack.clone(),
            );
            merge_stack.pop();
        }
        BlockFlowInstruction::ConditionalEnd => {
            let m = merge_stack.pop().expect("not in conditional branch");
            process(
                prg,
                block_local_memory_stores,
                m,
                BlockRef(0),
                &mut meta,
                &mut loop_stack,
                &mut merge_stack,
            );
        }
        BlockFlowInstruction::ConditionalLoop {
            r#break,
            r#continue,
            body,
            ..
        } => {
            loop_stack.push((r#continue, r#break));
            process(
                prg,
                block_local_memory_stores,
                body,
                BlockRef(0),
                &mut meta,
                &mut loop_stack,
                &mut merge_stack,
            );
            loop_stack.pop();
            process(
                prg,
                block_local_memory_stores,
                r#break,
                BlockRef(0),
                &mut meta,
                &mut loop_stack,
                &mut merge_stack,
            );
        }
        BlockFlowInstruction::Continue => {
            let &(c, _) = loop_stack.last().expect("continue not in a loop");
            let mut new_loop_stack = loop_stack[..loop_stack.len() - 1].to_vec();
            process(
                prg,
                block_local_memory_stores,
                c,
                BlockRef(0),
                &mut meta,
                &mut new_loop_stack,
                &mut merge_stack,
            );
        }
        BlockFlowInstruction::Break => {
            let &(_, b) = loop_stack.last().expect("break not in a loop");
            let mut new_loop_stack = loop_stack[..loop_stack.len() - 1].to_vec();
            process(
                prg,
                block_local_memory_stores,
                b,
                BlockRef(0),
                &mut meta,
                &mut new_loop_stack,
                &mut merge_stack,
            );
        }
        BlockFlowInstruction::Return(_) | BlockFlowInstruction::Undetermined => (),
    }

    meta.into_iter()
        .map(|(k, v)| (k, v.local_mem_aliased_registers))
        .collect()
}

/// ローカルメモリのLoadを対応するレジスタへのエイリアスに変える
pub fn replace_local_memory_load<'a, 's>(
    prg: &mut BlockifiedProgram<'a, 's>,
    per_block_local_memory_register_map: &HashMap<
        BlockRef,
        HashMap<LocalMemoryIdentifier<'a, 's>, RegisterRef>,
    >,
) {
    for (bx, b) in prg.blocks.iter().enumerate() {
        let local_memory_register_map = &per_block_local_memory_register_map[&BlockRef(bx)];

        for x in b.eval_impure_registers.iter() {
            match prg.impure_instructions[x] {
                BlockInstruction::LoadRef(RegisterRef::Const(n)) => match prg.constants[n].inst {
                    BlockConstInstruction::ScopeLocalVarRef(scope, vid) => {
                        if let Some(&r) =
                            local_memory_register_map.get(&LocalMemoryIdentifier::Var(scope, vid))
                        {
                            match r {
                                RegisterRef::Impure(n) if b.eval_impure_registers.contains(&n) => {
                                    // Impure（phi）かつ、同ブロック内で評価されるものであれば直接それに変える
                                    prg.impure_instructions
                                        .insert(*x, prg.impure_instructions[&n].clone());
                                }
                                _ => {
                                    todo!("not impure register or evaluated at external block");
                                    // impure_instructions
                                    //     .insert(*x, BlockPureInstruction::RegisterAlias(r));
                                }
                            }
                        }
                    }
                    BlockConstInstruction::FunctionInputVarRef(scope, id) => {
                        if let Some(&r) =
                            local_memory_register_map.get(&LocalMemoryIdentifier::FunArg(scope, id))
                        {
                            match r {
                                RegisterRef::Impure(n) if b.eval_impure_registers.contains(&n) => {
                                    // Impure（phi）かつ、同ブロック内で評価されるものであれば直接それに変える
                                    prg.impure_instructions
                                        .insert(*x, prg.impure_instructions[&n].clone());
                                }
                                _ => {
                                    todo!("not impure register or evaluated at external block");
                                    // impure_instructions
                                    //     .insert(*x, BlockPureInstruction::RegisterAlias(r));
                                }
                            }
                        }
                    }
                    _ => (),
                },
                _ => (),
            }
        }
    }
}

/// 一つもLoad命令がないローカルメモリのStore命令を削除する
///
/// 本当はRead after WriteじゃないStore命令（つまりStore -> Storeが連続しているもの）も消したいが、フローを含めて正しくRead after Writeを検出するのが難しそうなので一旦諦める
pub fn strip_never_load_local_memory_stores(prg: &mut BlockifiedProgram) -> bool {
    let mut loaded_local_memories = HashSet::new();
    for b in prg.blocks.iter() {
        for x in b.eval_impure_registers.iter() {
            match prg.impure_instructions[x] {
                BlockInstruction::LoadRef(RegisterRef::Const(n)) => match prg.constants[n].inst {
                    BlockConstInstruction::ScopeLocalVarRef(scope, vid) => {
                        loaded_local_memories.insert(LocalMemoryIdentifier::Var(scope, vid));
                    }
                    BlockConstInstruction::FunctionInputVarRef(scope, id) => {
                        loaded_local_memories.insert(LocalMemoryIdentifier::FunArg(scope, id));
                    }
                    _ => (),
                },
                _ => (),
            }
        }
    }

    let mut modified = false;
    for b in prg.blocks.iter_mut() {
        match b.flow {
            BlockFlowInstruction::StoreRef {
                ptr: RegisterRef::Const(n),
                after,
                ..
            } => {
                let loaded = match prg.constants[n].inst {
                    BlockConstInstruction::ScopeLocalVarRef(scope, vid) => {
                        loaded_local_memories.contains(&LocalMemoryIdentifier::Var(scope, vid))
                    }
                    BlockConstInstruction::FunctionInputVarRef(scope, id) => {
                        loaded_local_memories.contains(&LocalMemoryIdentifier::FunArg(scope, id))
                    }
                    // 上記以外は一旦保守的にLoadあるものとして判定
                    _ => true,
                };

                if !loaded {
                    // Load命令がないのでStoreを剥がす
                    b.flow = match after {
                        Some(n) => BlockFlowInstruction::Goto(n),
                        None => BlockFlowInstruction::Undetermined,
                    };
                    modified = true;
                }
            }
            _ => (),
        }
    }

    modified
}

/// 解決先レジスタがすべて同じなphi命令をエイリアスに置き換える
pub fn deconstruct_effectless_phi(prg: &mut BlockifiedProgram) -> bool {
    let mut register_alias_map = HashMap::new();

    let new_impure_instruction_sink = HashMap::with_capacity(prg.impure_instructions.len());
    for (r, x) in core::mem::replace(&mut prg.impure_instructions, new_impure_instruction_sink) {
        match x {
            BlockInstruction::Phi(xs) => {
                let unique_registers = xs
                    .values()
                    .fold(HashSet::new(), |mut h, &r| {
                        h.insert(r);
                        h
                    })
                    .into_iter()
                    .collect::<Vec<_>>();

                if let &[r1] = &unique_registers[..] {
                    // 唯一にunifyできた
                    prg.pure_instructions.push(
                        BlockPureInstruction::RegisterAlias(r1)
                            .typed(prg.impure_registers[r].clone()),
                    );
                    register_alias_map.insert(
                        RegisterRef::Impure(r),
                        RegisterRef::Pure(prg.pure_instructions.len() - 1),
                    );
                    for b in prg.blocks.iter_mut() {
                        b.eval_impure_registers.remove(&r);
                    }
                } else {
                    prg.impure_instructions.insert(r, BlockInstruction::Phi(xs));
                }
            }
            _ => {
                prg.impure_instructions.insert(r, x);
            }
        }
    }

    prg.apply_parallel_register_alias(&register_alias_map);
    !register_alias_map.is_empty()
}

/// 単純なGotoのブロックをマージしてまとめる
pub fn merge_simple_goto_blocks(prg: &mut BlockifiedProgram) -> bool {
    let mut modified = false;

    for n in 0..prg.blocks.len() {
        if let BlockFlowInstruction::Goto(next) = prg.blocks[n].flow {
            println!("[MergeSimpleGoto] b{n}->b{next}", next = next.0);
            let (current, merged) = unsafe {
                (
                    &mut *prg.blocks.as_mut_ptr().add(n),
                    &*prg.blocks.as_ptr().add(next.0),
                )
            };

            if !merged.has_block_dependent_instructions(&prg.impure_instructions)
                && !merged.is_loop_term_block()
                && !merged.is_branch_term_block()
            {
                current
                    .eval_impure_registers
                    .extend(merged.eval_impure_registers.iter().copied());
                current.flow = merged.flow.clone();

                match merged.flow {
                    BlockFlowInstruction::Goto(new_after)
                    | BlockFlowInstruction::Funcall {
                        after_return: Some(new_after),
                        ..
                    }
                    | BlockFlowInstruction::StoreRef {
                        after: Some(new_after),
                        ..
                    }
                    | BlockFlowInstruction::IntrinsicImpureFuncall {
                        after_return: Some(new_after),
                        ..
                    } => {
                        // 新しいとび先にphiがあれば、元のとび先のエントリと同じものを今のブロックからのものとして追加
                        println!("[MergeSimpleGoto] rechain: phi redirect b{n}->b{next}->b{new_after} => b{n}->b{new_after}", next = next.0, new_after = new_after.0);
                        prg.blocks[new_after.0].dup_phi_incoming(
                            &mut prg.impure_instructions,
                            next,
                            BlockRef(n),
                        );
                    }
                    BlockFlowInstruction::Conditional {
                        r#true: new_true_after,
                        r#false: new_false_after,
                        ..
                    } => {
                        // 新しいとび先にphiがあれば、元のとび先のエントリと同じものを今のブロックからのものとして追加
                        println!("[MergeSimpleGoto] rechain: phi redirect b{n}->b{next}->b{new_after} => b{n}->b{new_after}", next = next.0, new_after = new_true_after.0);
                        prg.blocks[new_true_after.0].dup_phi_incoming(
                            &mut prg.impure_instructions,
                            next,
                            BlockRef(n),
                        );
                        println!("[MergeSimpleGoto] rechain: phi redirect b{n}->b{next}->b{new_after} => b{n}->b{new_after}", next = next.0, new_after = new_false_after.0);
                        prg.blocks[new_false_after.0].dup_phi_incoming(
                            &mut prg.impure_instructions,
                            next,
                            BlockRef(n),
                        );
                    }
                    BlockFlowInstruction::ConditionalLoop {
                        r#break: new_break_after,
                        body: new_body_after,
                        ..
                    } => {
                        // 新しいとび先にphiがあれば、元のとび先のエントリと同じものを今のブロックからのものとして追加
                        println!("[MergeSimpleGoto] rechain: phi redirect b{n}->b{next}->b{new_after} => b{n}->b{new_after}", next = next.0, new_after = new_break_after.0);
                        prg.blocks[new_break_after.0].dup_phi_incoming(
                            &mut prg.impure_instructions,
                            next,
                            BlockRef(n),
                        );
                        println!("[MergeSimpleGoto] rechain: phi redirect b{n}->b{next}->b{new_after} => b{n}->b{new_after}", next = next.0, new_after = new_body_after.0);
                        prg.blocks[new_body_after.0].dup_phi_incoming(
                            &mut prg.impure_instructions,
                            next,
                            BlockRef(n),
                        );
                    }
                    BlockFlowInstruction::Break | BlockFlowInstruction::Continue => {
                        unreachable!("break/continue cannot determine new_after")
                    }
                    BlockFlowInstruction::ConditionalEnd => {
                        unreachable!("ConditionalEnd marker cannot determine new_after")
                    }
                    BlockFlowInstruction::Return(_)
                    | BlockFlowInstruction::Undetermined
                    | BlockFlowInstruction::StoreRef { after: None, .. }
                    | BlockFlowInstruction::Funcall {
                        after_return: None, ..
                    }
                    | BlockFlowInstruction::IntrinsicImpureFuncall {
                        after_return: None, ..
                    } => (),
                }

                modified = true;
            }
        }
    }

    modified
}

pub fn rechain_blocks(prg: &mut BlockifiedProgram) {
    let new_blocks = Vec::with_capacity(prg.blocks.len());
    let mut old_blocks = core::mem::replace(&mut prg.blocks, new_blocks);
    let mut block_relocate_map = HashMap::new();

    fn collect(
        old_blocks: &mut Vec<Block>,
        current_block: BlockRef,
        sink_blocks: &mut Vec<Block>,
        block_relocate_map: &mut HashMap<BlockRef, BlockRef>,
        merge_stack: &mut Vec<BlockRef>,
    ) {
        if block_relocate_map.contains_key(&current_block) {
            // すでに訪問済み
            return;
        }

        sink_blocks.push(core::mem::replace(
            &mut old_blocks[current_block.0],
            Block::empty(),
        ));
        block_relocate_map.insert(current_block, BlockRef(sink_blocks.len() - 1));
        match unsafe { &sink_blocks.last().unwrap_unchecked().flow } {
            BlockFlowInstruction::Undetermined
            | BlockFlowInstruction::Return(_)
            | BlockFlowInstruction::Break
            | BlockFlowInstruction::Continue
            | BlockFlowInstruction::StoreRef { after: None, .. }
            | BlockFlowInstruction::Funcall {
                after_return: None, ..
            }
            | BlockFlowInstruction::IntrinsicImpureFuncall {
                after_return: None, ..
            } => (),
            &BlockFlowInstruction::Goto(next)
            | &BlockFlowInstruction::StoreRef {
                after: Some(next), ..
            }
            | &BlockFlowInstruction::Funcall {
                after_return: Some(next),
                ..
            }
            | &BlockFlowInstruction::IntrinsicImpureFuncall {
                after_return: Some(next),
                ..
            } => {
                collect(
                    old_blocks,
                    next,
                    sink_blocks,
                    block_relocate_map,
                    merge_stack,
                );
            }
            &BlockFlowInstruction::Conditional {
                r#true,
                r#false,
                merge,
                ..
            } => {
                merge_stack.push(merge);
                collect(
                    old_blocks,
                    r#true,
                    sink_blocks,
                    block_relocate_map,
                    &mut merge_stack.clone(),
                );
                collect(
                    old_blocks,
                    r#false,
                    sink_blocks,
                    block_relocate_map,
                    &mut merge_stack.clone(),
                );
                merge_stack.pop();
            }
            BlockFlowInstruction::ConditionalEnd => {
                let m = merge_stack.pop().expect("not in conditional branch");
                collect(old_blocks, m, sink_blocks, block_relocate_map, merge_stack);
            }
            &BlockFlowInstruction::ConditionalLoop { r#break, body, .. } => {
                // continueはみなくていい（他のブロックからの流入ですでに訪問済みのはず）
                collect(
                    old_blocks,
                    body,
                    sink_blocks,
                    block_relocate_map,
                    merge_stack,
                );
                collect(
                    old_blocks,
                    r#break,
                    sink_blocks,
                    block_relocate_map,
                    merge_stack,
                );
            }
        }
    }
    collect(
        &mut old_blocks,
        BlockRef(0),
        &mut prg.blocks,
        &mut block_relocate_map,
        &mut Vec::new(),
    );

    for b in prg.blocks.iter_mut() {
        for x in b.eval_impure_registers.iter() {
            match prg.impure_instructions.get_mut(x).unwrap() {
                &mut BlockInstruction::Phi(ref mut incomings) => {
                    *incomings = incomings
                        .into_iter()
                        .filter_map(|(b, &mut r)| match block_relocate_map.get(b) {
                            Some(&nb) => Some((nb, r)),
                            None => None,
                        })
                        .collect();
                }
                &mut BlockInstruction::LoadRef(_) => (),
            }
        }

        b.flow.relocate_block_ref(|r| {
            if let Some(&nr) = block_relocate_map.get(r) {
                *r = nr;
            }
        });
    }
}

/// 組み込み関数呼び出しを探して命令を変換する
pub fn resolve_intrinsic_funcalls(prg: &mut BlockifiedProgram) -> bool {
    let mut register_alias_map = HashMap::new();
    let mut modified = false;

    for b in prg.blocks.iter_mut() {
        match b.flow {
            BlockFlowInstruction::Funcall {
                result: RegisterRef::Impure(result),
                callee: RegisterRef::Const(callee),
                ref args,
                after_return,
            } => match prg.constants.get(callee).map(|x| &x.inst) {
                Some(&BlockConstInstruction::IntrinsicTypeConstructorRef(ty)) => {
                    // 組み込み型の値コンストラクタ呼び出し

                    prg.pure_instructions.push(
                        BlockPureInstruction::ConstructIntrinsicComposite(ty, args.clone())
                            .typed(prg.impure_registers[result].clone()),
                    );
                    b.flow = BlockFlowInstruction::Goto(after_return.unwrap());
                    register_alias_map.insert(
                        RegisterRef::Impure(result),
                        RegisterRef::Pure(prg.pure_instructions.len() - 1),
                    );
                    modified = true;
                }
                Some(BlockConstInstruction::IntrinsicFunctionRef(overloads)) => {
                    let selected_overload = overloads
                        .iter()
                        .find(|o| {
                            o.args
                                .iter()
                                .zip(args.iter())
                                .all(|(def, call)| match call {
                                    &RegisterRef::Const(call) => def == &prg.constants[call].ty,
                                    &RegisterRef::Pure(call) => {
                                        def == &prg.pure_instructions[call].ty
                                    }
                                    &RegisterRef::Impure(call) => {
                                        def == &prg.impure_registers[call]
                                    }
                                })
                        })
                        .expect("Error: no matching overload found");

                    if selected_overload.is_pure {
                        // 純粋関数はinstruction化

                        prg.pure_instructions.push(
                            BlockPureInstruction::PureIntrinsicCall(
                                selected_overload.name,
                                args.clone(),
                            )
                            .typed(prg.impure_registers[result].clone()),
                        );
                        register_alias_map.insert(
                            RegisterRef::Impure(result),
                            RegisterRef::Pure(prg.pure_instructions.len() - 1),
                        );
                        b.flow = BlockFlowInstruction::Goto(after_return.unwrap());
                        modified = true;
                    } else {
                        // 非純粋ならFlow

                        b.flow = BlockFlowInstruction::IntrinsicImpureFuncall {
                            identifier: selected_overload.name,
                            args: args.clone(),
                            result: RegisterRef::Impure(result),
                            after_return,
                        };
                        modified = true;
                    }
                }
                _ => (),
            },
            _ => (),
        }
    }

    prg.apply_register_alias(&register_alias_map);
    modified
}

/// Load (SwizzleRef)を元refのLoad + Swizzleに分解する
pub fn unref_swizzle_ref_loads<'a, 's>(prg: &mut BlockifiedProgram<'a, 's>) -> bool {
    let mut register_alias_map = HashMap::new();

    'flp: for bx in 0..prg.blocks.len() {
        let (r, src, indices, r_ty) = 'find: {
            for r in prg.blocks[bx].eval_impure_registers.iter() {
                match prg.impure_instructions.get(r) {
                    Some(&BlockInstruction::LoadRef(RegisterRef::Pure(rload))) => {
                        match prg.pure_instructions[rload].inst {
                            BlockPureInstruction::SwizzleRef(src, ref indices) => {
                                break 'find (
                                    *r,
                                    src,
                                    indices.clone(),
                                    prg.impure_registers[*r].clone(),
                                );
                            }
                            _ => (),
                        }
                    }
                    _ => (),
                }
            }

            continue 'flp;
        };

        let src_value_ty = match src {
            RegisterRef::Const(src) => prg.constants[src]
                .ty
                .as_dereferenced()
                .expect("cannot dereference source of SwizzleRef")
                .clone(),
            RegisterRef::Pure(src) => prg.pure_instructions[src]
                .ty
                .as_dereferenced()
                .expect("cannot dereference source of SwizzleRef")
                .clone(),
            RegisterRef::Impure(_) => unreachable!("SwizzleRef applied for impure source?"),
        };
        let src_value_reg =
            prg.add_impure_instruction(BlockInstruction::LoadRef(src).typed(src_value_ty));
        let swizzled_reg = prg.add_pure_instruction(
            BlockPureInstruction::Swizzle(src_value_reg, indices.clone()).typed(r_ty),
        );

        prg.blocks[bx]
            .eval_impure_registers
            .insert(src_value_reg.as_id());
        prg.blocks[bx].eval_impure_registers.remove(&r);
        register_alias_map.insert(RegisterRef::Impure(r), swizzled_reg);
    }

    prg.apply_register_alias(&register_alias_map);
    !register_alias_map.is_empty()
}

/// SwizzleRefへのStoreをSwizzleRefを使わない形に変形する
pub fn transform_swizzle_component_store(prg: &mut BlockifiedProgram) -> bool {
    let mut modified = false;

    for bx in 0..prg.blocks.len() {
        match prg.blocks[bx].flow {
            BlockFlowInstruction::StoreRef {
                ptr: RegisterRef::Pure(ptr),
                value,
                after,
            } => {
                let (source, index) =
                    match prg.pure_instructions[ptr].inst {
                        BlockPureInstruction::SwizzleRef(
                            RegisterRef::Const(source),
                            ref indices,
                        ) if indices.len() == 1 => (RegisterRef::Const(source), indices[0]),
                        BlockPureInstruction::SwizzleRef(
                            RegisterRef::Pure(source),
                            ref indices,
                        ) if indices.len() == 1 => (RegisterRef::Pure(source), indices[0]),
                        _ => {
                            continue;
                        }
                    };

                let source_value_ty = source
                    .ty(prg)
                    .as_dereferenced()
                    .expect("cannot dereference swizzleRef source?")
                    .clone();
                let src_value_reg = prg.add_evaluated_impure_instruction(
                    BlockInstruction::LoadRef(source).typed(source_value_ty.clone()),
                    BlockRef(bx),
                );
                let inserted_value_reg = prg.add_pure_instruction(
                    BlockPureInstruction::CompositeInsert {
                        value,
                        source: src_value_reg,
                        index,
                    }
                    .typed(source_value_ty),
                );

                prg.blocks[bx].flow = BlockFlowInstruction::StoreRef {
                    ptr: source,
                    value: inserted_value_reg,
                    after,
                };
                modified = true;
            }
            _ => (),
        }
    }

    modified
}

pub fn inline_function1<'a, 's>(
    prg: &mut BlockifiedProgram<'a, 's>,
    root_scope: &'a SymbolScope<'a, 's>,
) -> bool {
    let mut modified = false;

    let mut bx = 0;
    let block_lim = prg.blocks.len();
    while bx < block_lim {
        match prg.blocks[bx].flow {
            BlockFlowInstruction::Funcall {
                result,
                callee: RegisterRef::Const(callee),
                ref args,
                after_return,
            } => {
                let (user_function_body, user_function_symbol) = match prg.constants[callee].inst {
                    BlockConstInstruction::UserDefinedFunctionRef(defscope, ref name) => {
                        match defscope.0.user_defined_function_body(name.0.slice) {
                            Some(udf) => (
                                udf,
                                defscope
                                    .0
                                    .user_defined_function_symbol(name.0.slice)
                                    .expect("body defined but symbol info not found"),
                            ),
                            None => {
                                // bodyがないfunction(今はないけどプロトタイプ宣言のみとか)はinline化しない
                                bx += 1;
                                continue;
                            }
                        }
                    }
                    _ => {
                        // calleeの指している先がユーザ定義関数参照ではない（そんなことある？）
                        bx += 1;
                        continue;
                    }
                };

                println!(
                    "[InlineFunction1] Inlining {:?}",
                    user_function_symbol.occurence
                );

                let ufb = user_function_body.borrow();
                let args = args.clone();

                // 引数をすべて新たに作ったローカル変数に代入する（こうするとRefが取れる）
                let (arg_local_vids, arg_store_blocks): (Vec<_>, Vec<_>) =
                    args.iter()
                        .zip(user_function_symbol.inputs.iter())
                        .map(|(&r, a)| {
                            let VarId::ScopeLocal(lvid) =
                                root_scope.declare_anon_local_var(a.3.clone(), a.1)
                            else {
                                unreachable!();
                            };
                            let vptr = prg.add_constant(
                                BlockConstInstruction::ScopeLocalVarRef(PtrEq(root_scope), lvid)
                                    .typed(if a.1 {
                                        a.3.clone().mutable_ref()
                                    } else {
                                        a.3.clone().imm_ref()
                                    }),
                            );

                            (
                                lvid,
                                Block::flow_only(BlockFlowInstruction::StoreRef {
                                    ptr: vptr,
                                    value: r,
                                    after: None,
                                }),
                            )
                        })
                        .unzip();
                let arg_store_perform_block_range = prg.append_block_sequence(arg_store_blocks);

                // インライン元のレジスタを全部インライン先に取り込む
                let base_registers = BaseRegisters {
                    r#const: prg.constants.len(),
                    pure: prg.pure_instructions.len(),
                    impure: prg.impure_registers.len(),
                };
                let base_execution_block = prg.blocks.len();
                prg.constants
                    .extend(ufb.program.constants.iter().cloned().map(|x| {
                        match x.inst {
                            BlockConstInstruction::FunctionInputVarRef(scope, id)
                                if scope == PtrEq(ufb.symbol_scope) =>
                            {
                                // 引数
                                BlockConstInstruction::ScopeLocalVarRef(
                                    PtrEq(root_scope),
                                    arg_local_vids[id],
                                )
                                .typed(x.ty)
                            }
                            _ => x,
                        }
                    }));
                let new_pure_instructions = ufb
                    .program
                    .pure_instructions
                    .iter()
                    .cloned()
                    .map(|mut x| {
                        x.inst.relocate_register(|r| {
                            *r = r.based_on(&base_registers);
                        });
                        x
                    })
                    .collect::<Vec<_>>();
                prg.pure_instructions.extend(new_pure_instructions);
                prg.impure_registers
                    .extend(ufb.program.impure_registers.iter().cloned());
                let new_impure_instructions = ufb
                    .program
                    .impure_instructions
                    .iter()
                    .map(|(&r, x)| {
                        let mut x = x.clone();
                        x.relocate_register(|r| {
                            *r = r.based_on(&base_registers);
                        });
                        x.relocate_block_ref(|b| b.0 += base_execution_block);
                        (r + base_registers.impure, x)
                    })
                    .collect::<Vec<_>>();
                prg.impure_instructions.extend(new_impure_instructions);

                // インライン元の実行ブロックを全部インライン先の末尾につなげる
                let mut return_phi_map = BTreeMap::new();
                prg.blocks.extend(
                    user_function_body
                        .borrow()
                        .program
                        .blocks
                        .iter()
                        .cloned()
                        .enumerate()
                        .map(|(bx, mut b)| {
                            b.eval_impure_registers = b
                                .eval_impure_registers
                                .into_iter()
                                .map(|x| x + base_registers.impure)
                                .collect();
                            b.flow
                                .relocate_register(|r| *r = r.based_on(&base_registers));
                            b.flow
                                .relocate_dest_register(|r| *r = r.based_on(&base_registers));
                            b.flow.relocate_block_ref(|b| b.0 += base_execution_block);

                            b.flow = match b.flow {
                                BlockFlowInstruction::Return(v) => {
                                    // インラインされた関数のReturnはresultへのAlias+Gotoにする
                                    // 複数Returnがあり得るので、resultはPhiになる
                                    return_phi_map.insert(BlockRef(bx + base_execution_block), v);

                                    match after_return {
                                        Some(a) => BlockFlowInstruction::Goto(a),
                                        None => BlockFlowInstruction::Undetermined,
                                    }
                                }
                                x => x,
                            };

                            b
                        }),
                );

                // return先ブロックでresultをphiにして評価させる
                if let Some(a) = after_return {
                    let RegisterRef::Impure(r) = result else {
                        unreachable!("impure funcall result must be impure register");
                    };
                    prg.impure_instructions.insert(r, BlockInstruction::Phi(return_phi_map));
                    prg.blocks[a.0].eval_impure_registers.insert(r);
                } else {
                    eprintln!("warn: impure funcall destination block is unknown, return value may not be evaluated");
                }

                // 引数セットアップブロックと実行ブロックを繋げる
                assert!(prg.blocks[arg_store_perform_block_range.end().0]
                    .try_set_next(BlockRef(base_execution_block)));

                // 関数呼び出しをGotoに変換
                prg.blocks[bx].flow =
                    BlockFlowInstruction::Goto(*arg_store_perform_block_range.start());
                modified = true;
                bx += 1;
                continue;
            }
            _ => {
                // 呼び出しフローでなければなにもしない
                bx += 1;
                continue;
            }
        }
    }

    modified
}

fn resolve_refpath(prg: &BlockifiedProgram, r: RegisterRef) -> Option<RefPath> {
    match r {
        RegisterRef::Const(r) => match prg.constants[r].inst {
            BlockConstInstruction::FunctionInputVarRef(_scope, id) => {
                Some(RefPath::FunctionInput(id))
            }
            _ => None,
        },
        RegisterRef::Pure(r) => match prg.pure_instructions[r].inst {
            BlockPureInstruction::MemberRef(src, ref name) => {
                let member_index = match src.ty(prg).as_dereferenced()? {
                    &ConcreteType::Struct(ref members) => {
                        members.iter().position(|m| &m.name == name)?
                    }
                    _ => return None,
                };

                Some(RefPath::Member(
                    Box::new(resolve_refpath(prg, src)?),
                    member_index,
                ))
            }
            _ => None,
        },
        RegisterRef::Impure(_) => None,
    }
}

pub struct DescriptorBound {
    pub set: u32,
    pub binding: u32,
}
pub struct PushConstantBound {
    pub offset: u32,
}
pub struct BuiltinBound(pub BuiltinInputOutput);

/// シェーダ入力変数への参照を専用の参照命令に置き換える
pub fn replace_shader_input_refs<'a, 's>(
    prg: &mut BlockifiedProgram<'a, 's>,
    root_scope: PtrEq<'a, SymbolScope<'a, 's>>,
    function_symbol: &UserDefinedFunctionSymbol<'s>,
) {
    let mut register_alias_map = HashMap::new();

    for x in prg.constants.iter_mut() {
        match x.inst {
            BlockConstInstruction::FunctionInputVarRef(scope, id) if scope == root_scope => {
                let descriptor_bound = match function_symbol.inputs[id].0 {
                    SymbolAttribute {
                        descriptor_set_location: Some(set),
                        descriptor_set_binding: Some(binding),
                        ..
                    } => Some(DescriptorBound { set, binding }),
                    _ => None,
                };
                let push_constant_bound = match function_symbol.inputs[id].0 {
                    SymbolAttribute {
                        push_constant_offset: Some(offset),
                        ..
                    } => Some(PushConstantBound { offset }),
                    _ => None,
                };
                let builtin_bound = match function_symbol.inputs[id].0 {
                    SymbolAttribute {
                        bound_builtin_io: Some(io),
                        ..
                    } => Some(BuiltinBound(io)),
                    _ => None,
                };
                let workgroup_shared = function_symbol.inputs[id].0.workgroup_shared;

                match (
                    descriptor_bound,
                    push_constant_bound,
                    builtin_bound,
                    workgroup_shared,
                ) {
                    (Some(b), None, None, false) => {
                        x.inst = BlockConstInstruction::DescriptorRef {
                            set: b.set,
                            binding: b.binding,
                        };
                    }
                    (None, Some(b), None, false) => {
                        x.inst = BlockConstInstruction::PushConstantRef(b.offset);
                    }
                    (None, None, Some(b), false) => {
                        x.inst = BlockConstInstruction::BuiltinIORef(b.0);
                    }
                    (None, None, None, true) => {
                        x.inst = BlockConstInstruction::WorkgroupSharedMemoryRef(
                            RefPath::FunctionInput(id),
                        );
                    }
                    (None, None, None, false) => (),
                    _ => panic!("conflicting bound attributes"),
                }
            }
            _ => (),
        }
    }

    for r in 0..prg.pure_instructions.len() {
        match prg.pure_instructions[r].inst {
            BlockPureInstruction::MemberRef(src, ref name) => {
                let member = match src
                    .ty(prg)
                    .as_dereferenced()
                    .expect("cannot dereference a source of a MemberRef")
                {
                    &ConcreteType::Struct(ref members) => members
                        .iter()
                        .find(|m| &m.name == name)
                        .expect("no member in source struct"),
                    _ => continue,
                };

                let descriptor_bound = match member.attribute {
                    SymbolAttribute {
                        descriptor_set_location: Some(set),
                        descriptor_set_binding: Some(binding),
                        ..
                    } => Some(DescriptorBound { set, binding }),
                    _ => None,
                };
                let push_constant_bound = match member.attribute {
                    SymbolAttribute {
                        push_constant_offset: Some(offset),
                        ..
                    } => Some(PushConstantBound { offset }),
                    _ => None,
                };
                let builtin_bound = match member.attribute {
                    SymbolAttribute {
                        bound_builtin_io: Some(io),
                        ..
                    } => Some(BuiltinBound(io)),
                    _ => None,
                };
                let workgroup_shared = member.attribute.workgroup_shared;

                match (
                    descriptor_bound,
                    push_constant_bound,
                    builtin_bound,
                    workgroup_shared,
                ) {
                    (Some(b), None, None, false) => {
                        let cr = prg.add_constant(
                            BlockConstInstruction::DescriptorRef {
                                set: b.set,
                                binding: b.binding,
                            }
                            .typed(prg.pure_instructions[r].ty.clone()),
                        );
                        register_alias_map.insert(RegisterRef::Pure(r), cr);
                    }
                    (None, Some(b), None, false) => {
                        let cr = prg.add_constant(
                            BlockConstInstruction::PushConstantRef(b.offset)
                                .typed(prg.pure_instructions[r].ty.clone()),
                        );
                        register_alias_map.insert(RegisterRef::Pure(r), cr);
                    }
                    (None, None, Some(b), false) => {
                        let cr = prg.add_constant(
                            BlockConstInstruction::BuiltinIORef(b.0)
                                .typed(prg.pure_instructions[r].ty.clone()),
                        );
                        register_alias_map.insert(RegisterRef::Pure(r), cr);
                    }
                    (None, None, None, true) => {
                        let cr = prg.add_constant(
                            BlockConstInstruction::WorkgroupSharedMemoryRef(
                                resolve_refpath(prg, RegisterRef::Pure(r))
                                    .expect("register does not have any RefPath"),
                            )
                            .typed(prg.pure_instructions[r].ty.clone()),
                        );
                        register_alias_map.insert(RegisterRef::Pure(r), cr);
                    }
                    (None, None, None, false) => (),
                    _ => panic!("conflicting bound attributes"),
                }
            }
            _ => (),
        }
    }

    prg.apply_register_alias(&register_alias_map);
}

/*
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    io::Write,
};

use typed_arena::Arena;

use crate::{
    concrete_type::{ConcreteType, IntrinsicType},
    ir::block::{BaseRegisters, RegisterRef},
    parser::FunctionDeclarationInputArguments,
    ref_path::RefPath,
    scope::{self, SymbolScope, VarId},
    source_ref::{SourceRef, SourceRefSliceEq},
    spirv::Decorate,
    symbol::{
        meta::{BuiltinInputOutput, SymbolAttribute},
        FunctionInputVariable,
    },
    utils::PtrEq,
};

use super::{
    block::{
        Block, BlockFlowInstruction, BlockGenerationContext, BlockInstruction,
        BlockInstructionEmissionContext, BlockRef, Constants, ImpureInstructionMap,
        IntrinsicBinaryOperation, PureInstructionMap, RegisterAliasMap,
    },
    expr::{ConstModifiers, ScopeCaptureSource, SimplifiedExpression},
    ConstFloatLiteral, ConstNumberLiteral, ConstSIntLiteral, ConstUIntLiteral, ExprRef,
    LosslessConst,
};

pub fn split_instructions<'a, 's>(
    impure_instructions: &mut ImpureInstructionMap<'a, 's>,
) -> (Constants<'s>, PureInstructionMap<'a, 's>, RegisterAliasMap) {
    let mut constants = Vec::new();
    let mut pure_instructions = HashMap::new();
    let mut register_alias_map = HashMap::new();
    let mut pure_last_index = 0;

    for (r, x) in core::mem::replace(
        impure_instructions,
        HashMap::with_capacity(impure_instructions.len()),
    ) {
        let x = match x.try_into_const_inst() {
            Ok(x) => {
                constants.push(x);
                register_alias_map.insert(
                    RegisterRef::Impure(r),
                    RegisterRef::Const(constants.len() - 1),
                );
                continue;
            }
            Err(x) => x,
        };

        if x.is_pure() {
            pure_instructions.insert(pure_last_index, x);
            register_alias_map.insert(RegisterRef::Impure(r), RegisterRef::Pure(pure_last_index));
            pure_last_index += 1;
        } else {
            impure_instructions.insert(r, x);
        }
    }

    (constants, pure_instructions, register_alias_map)
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

fn replace_inlined_function_input_refs<'a, 's>(
    expressions: &mut [(SimplifiedExpression<'a, 's>, ConcreteType<'s>)],
    function_scope: &'a SymbolScope<'a, 's>,
    substitutions: &[usize],
) {
    for n in 0..expressions.len() {
        match &mut expressions[n].0 {
            &mut SimplifiedExpression::LoadRef(r) => {
                let function_input_index = match &expressions[r.0].0 {
                    &SimplifiedExpression::VarRef(vscope, VarId::FunctionInput(n))
                        if vscope == PtrEq(function_scope) =>
                    {
                        Some(n)
                    }
                    _ => None,
                };

                if let Some(f) = function_input_index {
                    expressions[n].0 = SimplifiedExpression::AliasScopeCapture(substitutions[f]);
                }
            }
            &mut SimplifiedExpression::ScopedBlock {
                ref mut capturing,
                ref mut expressions,
                ..
            } => {
                let finput_capture_offset = capturing.len();
                capturing.extend(
                    substitutions
                        .iter()
                        .map(|&x| ScopeCaptureSource::Capture(x)),
                );
                let substitutions = substitutions
                    .iter()
                    .enumerate()
                    .map(|(x, _)| x + finput_capture_offset)
                    .collect::<Vec<_>>();
                replace_inlined_function_input_refs(expressions, function_scope, &substitutions);
            }
            &mut SimplifiedExpression::LoopBlock {
                ref mut capturing,
                ref mut expressions,
                ..
            } => {
                let finput_capture_offset = capturing.len();
                capturing.extend(
                    substitutions
                        .iter()
                        .map(|&x| ScopeCaptureSource::Capture(x)),
                );
                let substitutions = substitutions
                    .iter()
                    .enumerate()
                    .map(|(x, _)| x + finput_capture_offset)
                    .collect::<Vec<_>>();
                replace_inlined_function_input_refs(expressions, function_scope, &substitutions);
            }
            _ => (),
        }
    }
}

pub fn strip_unreachable_blocks<'a, 's>(
    blocks: &mut Vec<Block>,
    mod_instructions: &mut HashMap<RegisterRef, BlockInstruction<'a, 's>>,
) -> bool {
    let mut incomings = HashSet::new();
    // block 0 always have incoming
    incomings.insert(BlockRef(0));
    for b in blocks.iter() {
        match b.flow {
            BlockFlowInstruction::Goto(next) => {
                incomings.insert(next);
            }
            BlockFlowInstruction::Conditional {
                r#true,
                r#false,
                merge,
                ..
            } => {
                incomings.extend([r#true, r#false, merge]);
            }
            BlockFlowInstruction::Funcall { after_return, .. } => {
                incomings.extend(after_return);
            }
            BlockFlowInstruction::IntrinsicImpureFuncall { after_return, .. } => {
                incomings.extend(after_return);
            }
            BlockFlowInstruction::StoreRef { after, .. } => {
                incomings.extend(after);
            }
            BlockFlowInstruction::ConditionalLoop {
                r#break,
                r#continue,
                body,
                ..
            } => {
                incomings.extend([r#break, r#continue, body]);
            }
            BlockFlowInstruction::Undetermined
            | BlockFlowInstruction::Return(_)
            | BlockFlowInstruction::Break
            | BlockFlowInstruction::Continue => (),
        }
    }

    let mut refblock_ids = incomings.iter().map(|x| x.0).collect::<Vec<_>>();
    refblock_ids.sort();
    println!("BlockReferences: {refblock_ids:?}");

    let dropped = (0..blocks.len())
        .rev()
        .map(BlockRef)
        .filter(|x| !incomings.contains(&x))
        .collect::<Vec<_>>();
    let modified = !dropped.is_empty();
    for n in dropped {
        blocks.remove(n.0);

        // shift refs after n
        for b in blocks.iter_mut() {
            for r in b.eval_impure_registers.iter() {
                match mod_instructions.get_mut(r) {
                    Some(BlockInstruction::Phi(ref mut incoming_selectors)) => {
                        for (k, v) in core::mem::replace(incoming_selectors, BTreeMap::new()) {
                            if k == n {
                                // drop
                                continue;
                            }

                            incoming_selectors
                                .insert(if k.0 > n.0 { BlockRef(k.0 - 1) } else { k }, v);
                        }
                    }
                    _ => (),
                }
            }

            b.flow.relocate_next_block(|r| {
                r.0 -= if r.0 > n.0 { 1 } else { 0 };
            });
        }
    }

    modified
}

pub fn block_aliasing<'a, 's>(
    blocks: &mut [Block],
    mod_instructions: &mut HashMap<RegisterRef, BlockInstruction<'a, 's>>,
) -> bool {
    let block_aliases_to = blocks
        .iter()
        .enumerate()
        .filter_map(|(n, b)| match b {
            &Block {
                eval_impure_registers: ref instructions,
                flow: BlockFlowInstruction::Goto(to),
            } if instructions.is_empty() => Some((BlockRef(n), to)),
            _ => None,
        })
        .collect::<HashMap<_, _>>();

    for (from, to) in block_aliases_to.iter() {
        println!("[block alias] b{} = b{}", from.0, to.0);
    }

    let mut modified = false;
    for n in 0..blocks.len() {
        match blocks[n].flow {
            BlockFlowInstruction::Goto(ref mut next)
            | BlockFlowInstruction::StoreRef {
                after: Some(ref mut next),
                ..
            }
            | BlockFlowInstruction::Funcall {
                after_return: Some(ref mut next),
                ..
            }
            | BlockFlowInstruction::IntrinsicImpureFuncall {
                after_return: Some(ref mut next),
                ..
            } => {
                if let Some(&skip) = block_aliases_to.get(next) {
                    let from = core::mem::replace(next, skip);
                    blocks[skip.0].dup_phi_incoming(mod_instructions, from, BlockRef(n));

                    modified = true;
                }
            }
            BlockFlowInstruction::Conditional {
                ref mut r#true,
                ref mut r#false,
                ..
            } => {
                let old_true = block_aliases_to
                    .get(r#true)
                    .map(|&skip| (core::mem::replace(r#true, skip), skip));
                let old_false = block_aliases_to
                    .get(r#false)
                    .map(|&skip| (core::mem::replace(r#false, skip), skip));

                if let Some((from, skip)) = old_true {
                    blocks[skip.0].dup_phi_incoming(mod_instructions, from, BlockRef(n));
                    modified = true;
                }
                if let Some((from, skip)) = old_false {
                    blocks[skip.0].dup_phi_incoming(mod_instructions, from, BlockRef(n));
                    modified = true;
                }
            }
            BlockFlowInstruction::ConditionalLoop {
                ref mut r#break,
                ref mut body,
                ..
            } => {
                let old_break = block_aliases_to
                    .get(r#break)
                    .map(|&skip| (core::mem::replace(r#break, skip), skip));
                let old_body = block_aliases_to
                    .get(body)
                    .map(|&skip| (core::mem::replace(body, skip), skip));

                if let Some((from, skip)) = old_break {
                    blocks[skip.0].dup_phi_incoming(mod_instructions, from, BlockRef(n));
                    modified = true;
                }
                if let Some((from, skip)) = old_body {
                    blocks[skip.0].dup_phi_incoming(mod_instructions, from, BlockRef(n));
                    modified = true;
                }
            }
            _ => (),
        }
    }

    modified
}

pub fn inline_function2<'a, 's>(
    blocks: &mut Vec<Block>,
    impure_instructions: &mut ImpureInstructionMap<'a, 's>,
    pure_instructions: &mut PureInstructionMap<'a, 's>,
    constants: &mut Constants<'s>,
    scope_arena: &'a Arena<SymbolScope<'a, 's>>,
    function_root_scope: &'a SymbolScope<'a, 's>,
    pure_registers: &mut Vec<ConcreteType<'s>>,
    impure_registers: &mut Vec<ConcreteType<'s>>,
) -> bool {
    let mut modified = false;

    let mut n = 0;
    while n < blocks.len() {
        if let BlockFlowInstruction::Funcall {
            callee: RegisterRef::Pure(callee),
            ref args,
            result,
            after_return,
        } = blocks[n].flow
        {
            if let BlockInstruction::UserDefinedFunctionRef(scope, name) =
                pure_instructions[&callee]
            {
                let target_function_symbol =
                    scope.0.user_defined_function_symbol(name.0.slice).unwrap();
                let target_function_body =
                    scope.0.user_defined_function_body(name.0.slice).unwrap();

                println!("[Inlining Function] {name:?} at {scope:?} ({args:?})");
                println!("  symbol = {target_function_symbol:#?}");
                println!("  body: {target_function_body:#?}");

                // 定数は単純に合体
                let inlined_const_base_register = constants.len();
                constants.extend(target_function_body.borrow().constants.iter().cloned());

                // 引数は変数化する
                let tmp_scope = scope_arena.alloc(function_root_scope.new_child());
                let mut arg_store_set = Vec::with_capacity(args.len());
                let mut function_input_remap = HashMap::new();
                for (n, (ee, def)) in args
                    .iter()
                    .zip(target_function_symbol.inputs.iter())
                    .enumerate()
                {
                    let VarId::ScopeLocal(varid) =
                        tmp_scope.declare_anon_local_var(def.3.clone(), def.1)
                    else {
                        unreachable!();
                    };
                    pure_registers.push(def.3.clone().mutable_ref());
                    let ptr = RegisterRef::Pure(pure_registers.len() - 1);
                    arg_store_set.push((*ee, ptr, varid));
                    function_input_remap.insert(n, varid);
                }

                // Pure/Impureはレジスタだけ合体させる
                let inlined_pure_base_register = pure_registers.len();
                pure_registers.extend(target_function_body.borrow().pure_registers.iter().cloned());
                let inlined_impure_base_register = impure_registers.len();
                impure_registers.extend(
                    target_function_body
                        .borrow()
                        .impure_registers
                        .iter()
                        .cloned(),
                );

                let base_registers = BaseRegisters {
                    r#const: inlined_const_base_register,
                    pure: inlined_pure_base_register,
                    impure: inlined_impure_base_register,
                };

                // relocateしながらPure命令列を合体
                pure_instructions.extend(
                    target_function_body
                        .borrow()
                        .pure_instructions
                        .iter()
                        .map(|(res, x)| {
                            let mut x2 = x.clone();
                            x2.relocate_register(|r| {
                                *r = r.based_on(&base_registers);
                            });

                            (res + base_registers.pure, x2)
                        }),
                );

                dbg!(n);
                let mut after_blocks = blocks.split_off(n + 1);

                let mut setup_blocks = arg_store_set
                    .into_iter()
                    .enumerate()
                    .map(|(n, (a, ptr, _))| {
                        Block::flow_only(BlockFlowInstruction::StoreRef {
                            ptr,
                            value: a,
                            after: Some(BlockRef(n + 1)),
                        })
                    })
                    .collect::<Vec<_>>();

                let expand_block_base = n + setup_blocks.len() + 1;
                dbg!(expand_block_base);
                let exit_block =
                    BlockRef(expand_block_base + target_function_body.borrow().blocks.len());
                let after_block_before_base = n + 1;
                let after_block_base = exit_block.0 + 1;
                dbg!(after_block_base);
                let mut exit_block_incomings = BTreeMap::new();
                let inserted_blocks = target_function_body
                    .borrow()
                    .blocks
                    .iter()
                    .enumerate()
                    .map(|(n, b)| {
                        let mut nb = b.clone();
                        let instruction_count = nb.eval_impure_registers.len();
                        for k in core::mem::replace(
                            &mut nb.eval_impure_registers,
                            HashSet::with_capacity(instruction_count),
                        ) {
                            let Some(mut x) =
                                target_function_body.borrow().instructions.get(&k).cloned()
                            else {
                                continue;
                            };

                            x.relocate_register(|r| r.0 += register_offset);
                            x.relocate_block_ref(|b| b.0 += expand_block_base);

                            mod_instructions.insert(RegisterRef(k.0 + register_offset), x);
                            nb.eval_impure_registers
                                .insert(RegisterRef(k.0 + register_offset));
                        }

                        match nb.flow {
                            BlockFlowInstruction::Goto(ref mut next) => {
                                next.0 += expand_block_base;
                            }
                            BlockFlowInstruction::StoreRef {
                                ref mut ptr,
                                ref mut value,
                                ref mut after,
                            } => {
                                ptr.0 += register_offset;
                                value.0 += register_offset;
                                if let Some(ref mut after) = after {
                                    after.0 += expand_block_base;
                                }
                            }
                            BlockFlowInstruction::Funcall {
                                ref mut callee,
                                ref mut args,
                                ref mut result,
                                ref mut after_return,
                            } => {
                                callee.0 += register_offset;
                                for x in args.iter_mut() {
                                    x.0 += register_offset;
                                }
                                result.0 += register_offset;
                                if let Some(ref mut after_return) = after_return {
                                    after_return.0 += expand_block_base;
                                }
                            }
                            BlockFlowInstruction::IntrinsicImpureFuncall {
                                ref mut args,
                                ref mut result,
                                ref mut after_return,
                                ..
                            } => {
                                for x in args.iter_mut() {
                                    x.0 += register_offset;
                                }
                                result.0 += register_offset;
                                if let Some(ref mut after_return) = after_return {
                                    after_return.0 += expand_block_base;
                                }
                            }
                            BlockFlowInstruction::Conditional {
                                ref mut source,
                                ref mut r#true,
                                ref mut r#false,
                                ref mut merge,
                            } => {
                                source.0 += register_offset;
                                r#true.0 += expand_block_base;
                                r#false.0 += expand_block_base;
                                merge.0 += expand_block_base;
                            }
                            BlockFlowInstruction::ConditionalLoop {
                                ref mut condition,
                                ref mut r#break,
                                ref mut r#continue,
                                ref mut body,
                            } => {
                                condition.0 += register_offset;
                                r#break.0 += expand_block_base;
                                r#continue.0 += expand_block_base;
                                body.0 += expand_block_base;
                            }
                            BlockFlowInstruction::Return(r) => {
                                exit_block_incomings.insert(
                                    BlockRef(n + expand_block_base),
                                    RegisterRef(r.0 + register_offset),
                                );

                                nb.flow = BlockFlowInstruction::Goto(exit_block);
                            }
                            BlockFlowInstruction::Break
                            | BlockFlowInstruction::Continue
                            | BlockFlowInstruction::Undetermined => (),
                        }

                        nb
                    })
                    .collect::<Vec<_>>();
                mod_instructions.insert(result, BlockInstruction::Phi(exit_block_incomings));
                let funcall_merge_block = Block {
                    eval_impure_registers: [result].into_iter().collect(),
                    flow: match after_return {
                        Some(b) => BlockFlowInstruction::Goto(BlockRef(
                            b.0 - after_block_before_base + after_block_base,
                        )),
                        None => BlockFlowInstruction::Undetermined,
                    },
                };

                for b in blocks.iter_mut() {
                    for r in b.eval_impure_registers.iter() {
                        if let Some(x) = mod_instructions.get_mut(r) {
                            x.relocate_block_ref(|b| {
                                b.0 += if b.0 > n {
                                    after_block_base - after_block_before_base
                                } else {
                                    0
                                }
                            });
                        }
                    }

                    b.flow.relocate_next_block(|b| {
                        b.0 += if b.0 > n {
                            after_block_base - after_block_before_base
                        } else {
                            0
                        }
                    });
                }

                for b in after_blocks.iter_mut() {
                    for r in b.eval_impure_registers.iter() {
                        if let Some(x) = mod_instructions.get_mut(r) {
                            x.relocate_block_ref(|b| {
                                b.0 += if b.0 > n {
                                    after_block_base - after_block_before_base
                                } else {
                                    0
                                }
                            });
                        }
                    }

                    b.flow.relocate_next_block(|b| {
                        b.0 += if b.0 > n {
                            after_block_base - after_block_before_base
                        } else {
                            0
                        }
                    });
                }

                let setup_blocks_base = n + 1;
                dbg!(setup_blocks_base);

                blocks.last_mut().unwrap().flow =
                    BlockFlowInstruction::Goto(BlockRef(setup_blocks_base));
                for b in setup_blocks.iter_mut() {
                    b.flow.relocate_next_block(|b| b.0 += setup_blocks_base);
                }
                match setup_blocks.last_mut().unwrap().flow {
                    BlockFlowInstruction::StoreRef { ref mut after, .. } => {
                        *after = Some(BlockRef(expand_block_base));
                    }
                    _ => unreachable!(),
                }

                blocks.extend(
                    setup_blocks
                        .into_iter()
                        .chain(inserted_blocks.into_iter())
                        .chain(core::iter::once(funcall_merge_block))
                        .chain(after_blocks.into_iter()),
                );

                modified = true;
            }
        }

        n += 1;
    }

    modified
}

pub struct DescriptorBound {
    pub set: u32,
    pub binding: u32,
}
pub struct PushConstantBound {
    pub offset: u32,
}
pub struct BuiltinBound(pub BuiltinInputOutput);

pub fn resolve_shader_io_ref_binds<'a, 's>(
    function_input: &[(SymbolAttribute, bool, SourceRef<'s>, ConcreteType<'s>)],
    function_root_scope: &'a SymbolScope<'a, 's>,
    refpath_binds: &HashMap<&RefPath, &Vec<Decorate>>,
    blocks: &mut [Block],
    mod_instructions: &mut HashMap<RegisterRef, BlockInstruction<'a, 's>>,
    pure_instructions: &mut PureInstructionMap<'a, 's>,
    registers: &[ConcreteType<'s>],
    pure_registers: &[ConcreteType<'s>],
) -> bool {
    let mut modified = false;

    for v in pure_instructions.values_mut() {
        match v {
            BlockInstruction::FunctionInputVarRef(scope, x)
                if *scope == PtrEq(function_root_scope) =>
            {
                let input_attr = &function_input[*x].0;

                let descriptor_bound = match input_attr {
                    &SymbolAttribute {
                        descriptor_set_location: Some(set),
                        descriptor_set_binding: Some(binding),
                        ..
                    } => Some(DescriptorBound { set, binding }),
                    _ => None,
                };
                let push_constant_bound = match input_attr {
                    &SymbolAttribute {
                        push_constant_offset: Some(offset),
                        ..
                    } => Some(PushConstantBound { offset }),
                    _ => None,
                };
                let builtin_bound = match input_attr {
                    &SymbolAttribute {
                        bound_builtin_io: Some(builtin),
                        ..
                    } => Some(BuiltinBound(builtin)),
                    _ => None,
                };
                let workgroup_shared = input_attr.workgroup_shared;

                match (
                    descriptor_bound,
                    push_constant_bound,
                    builtin_bound,
                    workgroup_shared,
                ) {
                    (Some(d), None, None, false) => {
                        *v = BlockInstruction::DescriptorRef {
                            set: d.set,
                            binding: d.binding,
                        };
                        modified = true;
                    }
                    (None, Some(p), None, false) => {
                        *v = BlockInstruction::PushConstantRef(p.offset);
                        modified = true;
                    }
                    (None, None, Some(b), false) => {
                        *v = BlockInstruction::BuiltinIORef(b.0);
                        modified = true;
                    }
                    (None, None, None, true) => {
                        *v = BlockInstruction::WorkgroupSharedMemoryRef(RefPath::FunctionInput(*x));
                        modified = true;
                    }
                    (None, None, None, false) => (),
                    _ => panic!("Error: conflicting shader io attributes"),
                }
            }
            _ => (),
        }
    }

    for (r, x) in core::mem::replace(
        pure_instructions,
        HashMap::with_capacity(pure_instructions.len()),
    ) {
        match x {
            BlockInstruction::MemberRef(RegisterRef::Pure(src), ref member) => {
                let Some(ConcreteType::Struct(type_members)) =
                    pure_registers[src].as_dereferenced()
                else {
                    panic!(
                        "Error: cannot ref member of this type: {:?}",
                        pure_registers[src]
                    );
                };
                let member_index = type_members.iter().position(|m| &m.name == member).unwrap();
                let member_info = &type_members[member_index];

                let descriptor_bound = match member_info.attribute {
                    SymbolAttribute {
                        descriptor_set_location: Some(set),
                        descriptor_set_binding: Some(binding),
                        ..
                    } => Some(DescriptorBound { set, binding }),
                    _ => None,
                };
                let push_constant_bound = match member_info.attribute {
                    SymbolAttribute {
                        push_constant_offset: Some(offset),
                        ..
                    } => Some(PushConstantBound { offset }),
                    _ => None,
                };
                let builtin_bound = match member_info.attribute {
                    SymbolAttribute {
                        bound_builtin_io: Some(builtin),
                        ..
                    } => Some(BuiltinBound(builtin)),
                    _ => None,
                };
                let workgroup_shared = member_info.attribute.workgroup_shared;

                match (
                    descriptor_bound,
                    push_constant_bound,
                    builtin_bound,
                    workgroup_shared,
                ) {
                    (Some(d), None, None, false) => {
                        pure_instructions.insert(
                            r,
                            BlockInstruction::DescriptorRef {
                                set: d.set,
                                binding: d.binding,
                            },
                        );
                        modified = true;
                    }
                    (None, Some(p), None, false) => {
                        pure_instructions.insert(r, BlockInstruction::PushConstantRef(p.offset));
                        modified = true;
                    }
                    (None, None, Some(b), false) => {
                        pure_instructions.insert(r, BlockInstruction::BuiltinIORef(b.0));
                        modified = true;
                    }
                    (None, None, None, true) => match pure_instructions[&src] {
                        BlockInstruction::FunctionInputVarRef(scope, id)
                            if scope == PtrEq(function_root_scope) =>
                        {
                            pure_instructions.insert(
                                r,
                                BlockInstruction::WorkgroupSharedMemoryRef(RefPath::Member(
                                    Box::new(RefPath::FunctionInput(id)),
                                    member_index,
                                )),
                            );
                            modified = true;
                        }
                        _ => unimplemented!("deep refpath resolving"),
                    },
                    (None, None, None, false) => {
                        // no replacement
                        pure_instructions.insert(r, x);
                    }
                    _ => panic!("Error: conflicting shader io attributes"),
                }
            }
            _ => {
                // no replacement
                pure_instructions.insert(r, x);
            }
        }
    }

    modified
}

pub fn build_register_state_map<'a, 's>(
    blocks: &[Block],
    mod_instructions: &HashMap<RegisterRef, BlockInstruction<'a, 's>>,
) -> HashMap<BlockRef, HashMap<RegisterRef, BlockInstruction<'a, 's>>> {
    let mut state_map = HashMap::new();
    let mut loop_stack = Vec::new();
    let mut processed = HashSet::new();

    // b0は何もない状態
    state_map.insert(BlockRef(0), HashMap::new());

    fn process<'a, 's>(
        blocks: &[Block],
        mod_instructions: &HashMap<RegisterRef, BlockInstruction<'a, 's>>,
        n: BlockRef,
        incoming: BlockRef,
        state_map: &mut HashMap<BlockRef, HashMap<RegisterRef, BlockInstruction<'a, 's>>>,
        loop_stack: &mut Vec<(BlockRef, BlockRef)>,
        processed: &mut HashSet<BlockRef>,
    ) {
        let incoming_state = state_map.remove(&incoming).unwrap();

        let block_state_map = state_map.entry(n).or_insert_with(HashMap::new);
        let mut modified = false;
        for (r, x) in incoming_state.iter() {
            match block_state_map.entry(*r) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(x.clone());
                    modified = true;
                }
                std::collections::hash_map::Entry::Occupied(e) => {
                    assert!(
                        e.get() == x,
                        "conflicting register content: {:?} | {:?}",
                        e.get(),
                        x
                    );
                }
            }
        }
        for r in blocks[incoming.0].eval_impure_registers.iter() {
            let Some(x) = mod_instructions.get(r) else {
                continue;
            };

            match block_state_map.entry(*r) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(x.clone());
                    modified = true;
                }
                std::collections::hash_map::Entry::Occupied(e) => {
                    assert!(
                        e.get() == x,
                        "conflicting register content? {:?} | {:?}",
                        e.get(),
                        x
                    );
                }
            }
        }

        state_map.insert(incoming, incoming_state);

        if !modified && processed.contains(&n) {
            return;
        }

        processed.insert(n);
        match blocks[n.0].flow {
            BlockFlowInstruction::Goto(next) => process(
                blocks,
                mod_instructions,
                next,
                n,
                state_map,
                loop_stack,
                processed,
            ),
            BlockFlowInstruction::StoreRef {
                after: Some(after), ..
            } => process(
                blocks,
                mod_instructions,
                after,
                n,
                state_map,
                loop_stack,
                processed,
            ),
            BlockFlowInstruction::StoreRef { .. } => (),
            BlockFlowInstruction::Funcall {
                after_return: Some(after_return),
                ..
            } => process(
                blocks,
                mod_instructions,
                after_return,
                n,
                state_map,
                loop_stack,
                processed,
            ),
            BlockFlowInstruction::Funcall { .. } => (),
            BlockFlowInstruction::IntrinsicImpureFuncall {
                after_return: Some(after_return),
                ..
            } => process(
                blocks,
                mod_instructions,
                after_return,
                n,
                state_map,
                loop_stack,
                processed,
            ),
            BlockFlowInstruction::IntrinsicImpureFuncall { .. } => (),
            BlockFlowInstruction::Conditional {
                r#true, r#false, ..
            } => {
                process(
                    blocks,
                    mod_instructions,
                    r#true,
                    n,
                    state_map,
                    loop_stack,
                    processed,
                );
                process(
                    blocks,
                    mod_instructions,
                    r#false,
                    n,
                    state_map,
                    loop_stack,
                    processed,
                );
            }
            BlockFlowInstruction::ConditionalLoop {
                r#break,
                r#continue,
                body,
                ..
            } => {
                loop_stack.push((r#break, r#continue));
                process(
                    blocks,
                    mod_instructions,
                    body,
                    n,
                    state_map,
                    loop_stack,
                    processed,
                );
                loop_stack.pop();
                process(
                    blocks,
                    mod_instructions,
                    r#break,
                    n,
                    state_map,
                    loop_stack,
                    processed,
                );
            }
            BlockFlowInstruction::Break => {
                let &(brk, _) = loop_stack.last().unwrap();
                process(
                    blocks,
                    mod_instructions,
                    brk,
                    n,
                    state_map,
                    loop_stack,
                    processed,
                );
            }
            BlockFlowInstruction::Continue => {
                let &(_, cont) = loop_stack.last().unwrap();
                process(
                    blocks,
                    mod_instructions,
                    cont,
                    n,
                    state_map,
                    loop_stack,
                    processed,
                );
            }
            BlockFlowInstruction::Return(_) | BlockFlowInstruction::Undetermined => (),
        }
    }

    match blocks[0].flow {
        BlockFlowInstruction::Goto(next) => process(
            blocks,
            mod_instructions,
            next,
            BlockRef(0),
            &mut state_map,
            &mut loop_stack,
            &mut processed,
        ),
        BlockFlowInstruction::StoreRef {
            after: Some(after), ..
        } => process(
            blocks,
            mod_instructions,
            after,
            BlockRef(0),
            &mut state_map,
            &mut loop_stack,
            &mut processed,
        ),
        BlockFlowInstruction::StoreRef { .. } => (),
        BlockFlowInstruction::Funcall {
            after_return: Some(after_return),
            ..
        } => process(
            blocks,
            mod_instructions,
            after_return,
            BlockRef(0),
            &mut state_map,
            &mut loop_stack,
            &mut processed,
        ),
        BlockFlowInstruction::Funcall { .. } => (),
        BlockFlowInstruction::IntrinsicImpureFuncall {
            after_return: Some(after_return),
            ..
        } => process(
            blocks,
            mod_instructions,
            after_return,
            BlockRef(0),
            &mut state_map,
            &mut loop_stack,
            &mut processed,
        ),
        BlockFlowInstruction::IntrinsicImpureFuncall { .. } => (),
        BlockFlowInstruction::Conditional {
            r#true, r#false, ..
        } => {
            process(
                blocks,
                mod_instructions,
                r#true,
                BlockRef(0),
                &mut state_map,
                &mut loop_stack,
                &mut processed,
            );
            process(
                blocks,
                mod_instructions,
                r#false,
                BlockRef(0),
                &mut state_map,
                &mut loop_stack,
                &mut processed,
            );
        }
        BlockFlowInstruction::ConditionalLoop {
            r#break,
            r#continue,
            body,
            ..
        } => {
            loop_stack.push((r#break, r#continue));
            process(
                blocks,
                mod_instructions,
                body,
                BlockRef(0),
                &mut state_map,
                &mut loop_stack,
                &mut processed,
            );
            loop_stack.pop();
            process(
                blocks,
                mod_instructions,
                r#break,
                BlockRef(0),
                &mut state_map,
                &mut loop_stack,
                &mut processed,
            );
        }
        BlockFlowInstruction::Break => {
            let &(brk, _) = loop_stack.last().unwrap();
            process(
                blocks,
                mod_instructions,
                brk,
                BlockRef(0),
                &mut state_map,
                &mut loop_stack,
                &mut processed,
            );
        }
        BlockFlowInstruction::Continue => {
            let &(_, cont) = loop_stack.last().unwrap();
            process(
                blocks,
                mod_instructions,
                cont,
                BlockRef(0),
                &mut state_map,
                &mut loop_stack,
                &mut processed,
            );
        }
        BlockFlowInstruction::Return(_) | BlockFlowInstruction::Undetermined => (),
    }

    state_map
}

pub fn resolve_register_aliases<'a, 's>(
    blocks: &mut [Block],
    mod_instructions: &mut HashMap<RegisterRef, BlockInstruction<'a, 's>>,
    register_state_map: &HashMap<BlockRef, HashMap<RegisterRef, BlockInstruction<'a, 's>>>,
) -> bool {
    let mut modified = false;

    for (n, b) in blocks.iter_mut().enumerate() {
        let Some(state_map) = register_state_map.get(&BlockRef(n)) else {
            continue;
        };

        let in_block_map = b
            .eval_impure_registers
            .iter()
            .filter_map(|r| mod_instructions.get(r).map(|x| (*r, x.clone())))
            .collect::<HashMap<_, _>>();

        for r in b.eval_impure_registers.iter() {
            let m = match mod_instructions.get_mut(r) {
                Some(x) => x.relocate_register(|x| loop {
                    match state_map.get(x).or_else(|| in_block_map.get(x)) {
                        Some(&BlockInstruction::RegisterAlias(to)) => {
                            *x = to;
                        }
                        _ => break,
                    }
                }),
                None => false,
            };
            modified = modified || m;
        }

        let m = b.flow.relocate_register(|x| loop {
            match state_map.get(x).or_else(|| in_block_map.get(x)) {
                Some(&BlockInstruction::RegisterAlias(to)) => {
                    *x = to;
                }
                _ => break,
            }
        });
        modified = modified || m;
    }

    modified
}

pub fn strip_unreferenced_registers(
    blocks: &mut [Block],
    mod_instructions: &mut HashMap<RegisterRef, BlockInstruction>,
    registers: &mut Vec<ConcreteType>,
    const_map: &mut HashMap<RegisterRef, BlockInstruction>,
) -> bool {
    let mut referenced_registers = HashSet::new();
    for b in blocks.iter() {
        for r in b.eval_impure_registers.iter() {
            let Some(x) = mod_instructions.get(r) else {
                continue;
            };

            match x {
                BlockInstruction::ConstUnit
                | BlockInstruction::ConstInt(_)
                | BlockInstruction::ConstNumber(_)
                | BlockInstruction::ConstSInt(_)
                | BlockInstruction::ConstUInt(_)
                | BlockInstruction::ConstFloat(_)
                | BlockInstruction::ImmBool(_)
                | BlockInstruction::ImmSInt(_)
                | BlockInstruction::ImmUInt(_)
                | BlockInstruction::ImmInt(_)
                | BlockInstruction::ScopeLocalVarRef(_, _)
                | BlockInstruction::FunctionInputVarRef(_, _)
                | BlockInstruction::IntrinsicTypeConstructorRef(_)
                | BlockInstruction::IntrinsicFunctionRef(_)
                | BlockInstruction::UserDefinedFunctionRef(_, _)
                | BlockInstruction::BuiltinIORef(_)
                | BlockInstruction::DescriptorRef { .. }
                | BlockInstruction::PushConstantRef(_)
                | BlockInstruction::WorkgroupSharedMemoryRef(_)
                | BlockInstruction::StaticPathRef(_) => (),
                &BlockInstruction::MemberRef(x, _)
                | &BlockInstruction::TupleRef(x, _)
                | &BlockInstruction::IntrinsicUnaryOp(x, _)
                | &BlockInstruction::Cast(x, _)
                | &BlockInstruction::PromoteIntToNumber(x)
                | &BlockInstruction::InstantiateIntrinsicTypeClass(x, _)
                | &BlockInstruction::LoadRef(x)
                | &BlockInstruction::Swizzle(x, _)
                | &BlockInstruction::SwizzleRef(x, _)
                | &BlockInstruction::RegisterAlias(x) => {
                    referenced_registers.insert(x);
                }
                BlockInstruction::IntrinsicBinaryOp(x, _, y)
                | BlockInstruction::ArrayRef {
                    source: x,
                    index: y,
                } => {
                    referenced_registers.extend([x, y]);
                }
                BlockInstruction::ConstructTuple(ref xs)
                | BlockInstruction::ConstructStruct(ref xs)
                | BlockInstruction::ConstructIntrinsicComposite(_, ref xs)
                | BlockInstruction::PureIntrinsicCall(_, ref xs) => {
                    referenced_registers.extend(xs.iter().copied());
                }
                BlockInstruction::Phi(ref xs) => {
                    referenced_registers.extend(xs.values().copied());
                }
                &BlockInstruction::PureFuncall(callee, ref xs) => {
                    referenced_registers.insert(callee);
                    referenced_registers.extend(xs.iter().copied());
                }
                BlockInstruction::CompositeInsert { value, source, .. } => {
                    referenced_registers.extend([value, source]);
                }
            }
        }

        match b.flow {
            BlockFlowInstruction::StoreRef { ptr, value, .. } => {
                referenced_registers.extend([ptr, value]);
            }
            BlockFlowInstruction::Funcall {
                callee, ref args, ..
            } => {
                referenced_registers.insert(callee);
                referenced_registers.extend(args.iter().copied());
            }
            BlockFlowInstruction::IntrinsicImpureFuncall { ref args, .. } => {
                referenced_registers.extend(args.iter().copied());
            }
            BlockFlowInstruction::Conditional { source, .. } => {
                referenced_registers.insert(source);
            }
            BlockFlowInstruction::ConditionalLoop { condition, .. } => {
                referenced_registers.insert(condition);
            }
            BlockFlowInstruction::Return(r) => {
                referenced_registers.insert(r);
            }
            BlockFlowInstruction::Break
            | BlockFlowInstruction::Continue
            | BlockFlowInstruction::Undetermined
            | BlockFlowInstruction::Goto(_) => (),
        }
    }

    let mut stripped_registers = (0..registers.len())
        .map(RegisterRef)
        .filter(|x| !referenced_registers.contains(x))
        .collect::<Vec<_>>();
    let mut modified = false;
    while let Some(stripped) = stripped_registers.pop() {
        let swapped_register = RegisterRef(registers.len() - 1);
        registers.swap_remove(stripped.0);
        modified = true;

        const_map.remove(&stripped);
        if swapped_register != stripped {
            if let Some(s) = const_map.remove(&swapped_register) {
                const_map.insert(stripped, s);
            }
        }
        for b in blocks.iter_mut() {
            b.eval_impure_registers.remove(&stripped);

            if swapped_register != stripped {
                if let Some(x) = mod_instructions.remove(&swapped_register) {
                    mod_instructions.insert(stripped, x);
                }
                if b.eval_impure_registers.remove(&swapped_register) {
                    b.eval_impure_registers.insert(stripped);
                }

                for r in b.eval_impure_registers.iter() {
                    if let Some(x) = mod_instructions.get_mut(r) {
                        x.relocate_register(|r| {
                            if r == &swapped_register {
                                *r = stripped;
                            }
                        });
                    }
                }

                b.flow.relocate_register(|r| {
                    if r == &swapped_register {
                        *r = stripped;
                    }
                });
                b.flow.relocate_result_register(|r| {
                    if r == &swapped_register {
                        *r = stripped;
                    }
                });
            }
        }
    }

    modified
}

#[derive(Debug, Clone)]
pub enum IncomingRegister {
    Bulk(BTreeMap<BlockRef, RegisterRef>),
    Phied(RegisterRef),
}

pub fn build_scope_local_var_state<'a, 's>(
    blocks: &mut [Block],
    mod_instructions: &mut HashMap<RegisterRef, BlockInstruction<'a, 's>>,
    scope_local_var_stores_per_block: &HashMap<
        BlockRef,
        HashMap<(PtrEq<'a, SymbolScope<'a, 's>>, usize), RegisterRef>,
    >,
    registers: &mut Vec<ConcreteType<'s>>,
) -> HashMap<BlockRef, HashMap<(PtrEq<'a, SymbolScope<'a, 's>>, usize), IncomingRegister>> {
    let mut state_map = HashMap::new();
    let mut loop_stack = Vec::new();
    let mut processed = HashSet::new();
    // b0は何もない状態
    state_map.insert(BlockRef(0), HashMap::new());

    fn process<'a, 's>(
        blocks: &mut [Block],
        mod_instructions: &mut HashMap<RegisterRef, BlockInstruction<'a, 's>>,
        n: BlockRef,
        incoming: BlockRef,
        scope_local_var_stores_per_block: &HashMap<
            BlockRef,
            HashMap<(PtrEq<'a, SymbolScope<'a, 's>>, usize), RegisterRef>,
        >,
        registers: &mut Vec<ConcreteType<'s>>,
        state_map: &mut HashMap<
            BlockRef,
            HashMap<(PtrEq<'a, SymbolScope<'a, 's>>, usize), IncomingRegister>,
        >,
        processed: &mut HashSet<BlockRef>,
        loop_stack: &mut Vec<(BlockRef, BlockRef)>,
    ) {
        let mut incoming_state = state_map.remove(&incoming).unwrap();
        let incoming_local_stores = &scope_local_var_stores_per_block[&incoming];

        let block_state_map = state_map.entry(n).or_insert_with(HashMap::new);
        let mut modified = false;
        for (&(scope, vid), r_incoming) in incoming_state.iter_mut() {
            match r_incoming {
                IncomingRegister::Phied(rp) => {
                    let existing = block_state_map
                        .entry((scope, vid))
                        .or_insert_with(|| IncomingRegister::Bulk(BTreeMap::new()));

                    match existing {
                        IncomingRegister::Phied(p) => {
                            let Some(BlockInstruction::Phi(ref mut incomings)) =
                                mod_instructions.get_mut(&p)
                            else {
                                unreachable!("no phi instruction reference");
                            };

                            let old_rp = incomings.insert(incoming, *rp);
                            modified = modified || old_rp != Some(*rp);
                        }
                        IncomingRegister::Bulk(rs) => {
                            let old = rs.insert(incoming, *rp);
                            modified = modified || old != Some(*rp);
                        }
                    }
                }
                IncomingRegister::Bulk(r_incomings) => {
                    let existing = block_state_map
                        .entry((scope, vid))
                        .or_insert_with(|| IncomingRegister::Bulk(BTreeMap::new()));

                    match existing {
                        IncomingRegister::Phied(p) => {
                            let r = if r_incomings.len() != 1 {
                                let register_type =
                                    registers[r_incomings.first_key_value().unwrap().1 .0].clone();
                                assert!(r_incomings
                                    .values()
                                    .all(|r| registers[r.0] == register_type));
                                registers.push(register_type);
                                let rp = RegisterRef(registers.len() - 1);
                                let phi_incomings = r_incomings.clone();

                                mod_instructions.insert(rp, BlockInstruction::Phi(phi_incomings));
                                blocks[incoming.0].eval_impure_registers.insert(rp);
                                *r_incoming = IncomingRegister::Phied(rp);

                                rp
                            } else {
                                *r_incomings.first_key_value().unwrap().1
                            };

                            let Some(BlockInstruction::Phi(ref mut incomings)) =
                                mod_instructions.get_mut(&p)
                            else {
                                unreachable!("no phi instruction reference");
                            };

                            let old_rp = incomings.insert(incoming, r);
                            modified = modified || old_rp != Some(r);
                        }
                        IncomingRegister::Bulk(rs) => {
                            let r = if r_incomings.len() != 1 {
                                let register_type =
                                    registers[r_incomings.first_key_value().unwrap().1 .0].clone();
                                assert!(r_incomings
                                    .values()
                                    .all(|r| registers[r.0] == register_type));
                                registers.push(register_type);
                                let rp = RegisterRef(registers.len() - 1);
                                let phi_incomings = r_incomings.clone();

                                mod_instructions.insert(rp, BlockInstruction::Phi(phi_incomings));
                                blocks[incoming.0].eval_impure_registers.insert(rp);
                                r_incomings.clear();
                                *r_incoming = IncomingRegister::Phied(rp);

                                rp
                            } else {
                                *r_incomings.first_key_value().unwrap().1
                            };

                            let old_r = rs.insert(incoming, r);
                            modified = modified || old_r != Some(r);
                        }
                    }
                }
            }
        }
        for (&(scope, vid), &r) in incoming_local_stores {
            let existing = block_state_map
                .entry((scope, vid))
                .or_insert_with(|| IncomingRegister::Bulk(BTreeMap::new()));

            match existing {
                IncomingRegister::Phied(p) => {
                    unimplemented!("phi distribute");
                }
                IncomingRegister::Bulk(rs) => {
                    let old_r = rs.insert(incoming, r);
                    modified = modified || old_r != Some(r);
                }
            }
        }

        // もどす
        state_map.insert(incoming, incoming_state);

        if processed.contains(&n) && !modified {
            // 2回目以降で変化がない場合はこれ以上やっても意味がない
            return;
        }

        processed.insert(n);

        match blocks[n.0].flow {
            BlockFlowInstruction::Goto(next) => process(
                blocks,
                mod_instructions,
                next,
                n,
                scope_local_var_stores_per_block,
                registers,
                state_map,
                processed,
                loop_stack,
            ),
            BlockFlowInstruction::StoreRef {
                after: Some(after), ..
            } => process(
                blocks,
                mod_instructions,
                after,
                n,
                scope_local_var_stores_per_block,
                registers,
                state_map,
                processed,
                loop_stack,
            ),
            BlockFlowInstruction::StoreRef { .. } => (),
            BlockFlowInstruction::Funcall {
                after_return: Some(after_return),
                ..
            } => process(
                blocks,
                mod_instructions,
                after_return,
                n,
                scope_local_var_stores_per_block,
                registers,
                state_map,
                processed,
                loop_stack,
            ),
            BlockFlowInstruction::Funcall { .. } => (),
            BlockFlowInstruction::IntrinsicImpureFuncall {
                after_return: Some(after_return),
                ..
            } => process(
                blocks,
                mod_instructions,
                after_return,
                n,
                scope_local_var_stores_per_block,
                registers,
                state_map,
                processed,
                loop_stack,
            ),
            BlockFlowInstruction::IntrinsicImpureFuncall { .. } => (),
            BlockFlowInstruction::Conditional {
                r#true, r#false, ..
            } => {
                process(
                    blocks,
                    mod_instructions,
                    r#true,
                    n,
                    scope_local_var_stores_per_block,
                    registers,
                    state_map,
                    processed,
                    loop_stack,
                );
                process(
                    blocks,
                    mod_instructions,
                    r#false,
                    n,
                    scope_local_var_stores_per_block,
                    registers,
                    state_map,
                    processed,
                    loop_stack,
                );
            }
            BlockFlowInstruction::ConditionalLoop {
                r#break,
                r#continue,
                body,
                ..
            } => {
                loop_stack.push((r#break, r#continue));
                process(
                    blocks,
                    mod_instructions,
                    body,
                    n,
                    scope_local_var_stores_per_block,
                    registers,
                    state_map,
                    processed,
                    loop_stack,
                );
                loop_stack.pop();
                process(
                    blocks,
                    mod_instructions,
                    r#break,
                    n,
                    scope_local_var_stores_per_block,
                    registers,
                    state_map,
                    processed,
                    loop_stack,
                );
            }
            BlockFlowInstruction::Break => {
                let &(brk, _) = loop_stack.last().unwrap();
                process(
                    blocks,
                    mod_instructions,
                    brk,
                    n,
                    scope_local_var_stores_per_block,
                    registers,
                    state_map,
                    processed,
                    loop_stack,
                );
            }
            BlockFlowInstruction::Continue => {
                let &(_, cont) = loop_stack.last().unwrap();
                process(
                    blocks,
                    mod_instructions,
                    cont,
                    n,
                    scope_local_var_stores_per_block,
                    registers,
                    state_map,
                    processed,
                    loop_stack,
                );
            }
            BlockFlowInstruction::Return(_) | BlockFlowInstruction::Undetermined => (),
        }
    }

    match blocks[0].flow {
        BlockFlowInstruction::Goto(next) => process(
            blocks,
            mod_instructions,
            next,
            BlockRef(0),
            scope_local_var_stores_per_block,
            registers,
            &mut state_map,
            &mut processed,
            &mut loop_stack,
        ),
        BlockFlowInstruction::StoreRef {
            after: Some(after), ..
        } => process(
            blocks,
            mod_instructions,
            after,
            BlockRef(0),
            scope_local_var_stores_per_block,
            registers,
            &mut state_map,
            &mut processed,
            &mut loop_stack,
        ),
        BlockFlowInstruction::StoreRef { .. } => (),
        BlockFlowInstruction::Funcall {
            after_return: Some(after_return),
            ..
        } => process(
            blocks,
            mod_instructions,
            after_return,
            BlockRef(0),
            scope_local_var_stores_per_block,
            registers,
            &mut state_map,
            &mut processed,
            &mut loop_stack,
        ),
        BlockFlowInstruction::Funcall { .. } => (),
        BlockFlowInstruction::IntrinsicImpureFuncall {
            after_return: Some(after_return),
            ..
        } => process(
            blocks,
            mod_instructions,
            after_return,
            BlockRef(0),
            scope_local_var_stores_per_block,
            registers,
            &mut state_map,
            &mut processed,
            &mut loop_stack,
        ),
        BlockFlowInstruction::IntrinsicImpureFuncall { .. } => (),
        BlockFlowInstruction::Conditional {
            r#true, r#false, ..
        } => {
            process(
                blocks,
                mod_instructions,
                r#true,
                BlockRef(0),
                scope_local_var_stores_per_block,
                registers,
                &mut state_map,
                &mut processed,
                &mut loop_stack,
            );
            process(
                blocks,
                mod_instructions,
                r#false,
                BlockRef(0),
                scope_local_var_stores_per_block,
                registers,
                &mut state_map,
                &mut processed,
                &mut loop_stack,
            );
        }
        BlockFlowInstruction::ConditionalLoop {
            r#break,
            r#continue,
            body,
            ..
        } => {
            loop_stack.push((r#break, r#continue));
            process(
                blocks,
                mod_instructions,
                body,
                BlockRef(0),
                scope_local_var_stores_per_block,
                registers,
                &mut state_map,
                &mut processed,
                &mut loop_stack,
            );
            loop_stack.pop();
            process(
                blocks,
                mod_instructions,
                r#break,
                BlockRef(0),
                scope_local_var_stores_per_block,
                registers,
                &mut state_map,
                &mut processed,
                &mut loop_stack,
            );
        }
        BlockFlowInstruction::Break => panic!("Error: break on top of function scope"),
        BlockFlowInstruction::Continue => panic!("Error: continue on top of function scope"),
        BlockFlowInstruction::Return(_) | BlockFlowInstruction::Undetermined => (),
    }

    state_map
}

pub fn apply_local_var_states<'a, 's>(
    blocks: &mut [Block],
    mod_instructions: &mut HashMap<RegisterRef, BlockInstruction<'a, 's>>,
    const_map: &HashMap<RegisterRef, BlockInstruction<'a, 's>>,
    state_map: &HashMap<
        BlockRef,
        HashMap<(PtrEq<'a, SymbolScope<'a, 's>>, usize), IncomingRegister>,
    >,
) -> bool {
    let mut modified = false;

    for n in 0..blocks.len() {
        for r in blocks[n].eval_impure_registers.iter() {
            match mod_instructions.get_mut(r) {
                Some(x @ &mut BlockInstruction::LoadRef(ptr)) => {
                    if let Some(&BlockInstruction::ScopeLocalVarRef(scope, vid)) =
                        const_map.get(&ptr)
                    {
                        match state_map[&BlockRef(n)].get(&(scope, vid)) {
                            Some(IncomingRegister::Bulk(xs)) => {
                                *x = BlockInstruction::Phi(xs.clone());
                                modified = true;
                            }
                            Some(IncomingRegister::Phied(p)) => {
                                *x = BlockInstruction::RegisterAlias(*p);
                                modified = true;
                            }
                            _ => (),
                        }
                    }
                }
                _ => (),
            }
        }
    }

    modified
}

pub fn track_scope_local_var_aliases<'a, 's>(
    blocks: &[Block],
    mod_instructions: &HashMap<RegisterRef, BlockInstruction<'a, 's>>,
    const_map: &HashMap<RegisterRef, BlockInstruction<'a, 's>>,
) -> HashMap<BlockRef, HashMap<(PtrEq<'a, SymbolScope<'a, 's>>, usize), RegisterRef>> {
    let mut processed = HashSet::new();
    let mut aliases_per_block = HashMap::new();

    fn process<'a, 's>(
        blocks: &[Block],
        mod_instructions: &HashMap<RegisterRef, BlockInstruction<'a, 's>>,
        n: BlockRef,
        const_map: &HashMap<RegisterRef, BlockInstruction<'a, 's>>,
        processed: &mut HashSet<BlockRef>,
        aliases_per_block: &mut HashMap<
            BlockRef,
            HashMap<(PtrEq<'a, SymbolScope<'a, 's>>, usize), RegisterRef>,
        >,
    ) {
        if processed.contains(&n) {
            return;
        }

        processed.insert(n);
        let mut outgoing_aliases = HashMap::new();
        match blocks[n.0].flow {
            BlockFlowInstruction::Goto(next) => {
                process(
                    blocks,
                    mod_instructions,
                    next,
                    const_map,
                    processed,
                    aliases_per_block,
                );
            }
            BlockFlowInstruction::StoreRef { ptr, value, after } => {
                if let Some(BlockInstruction::ScopeLocalVarRef(scope, vid)) = const_map.get(&ptr) {
                    outgoing_aliases.insert((*scope, *vid), value);
                }

                if let Some(after) = after {
                    process(
                        blocks,
                        mod_instructions,
                        after,
                        const_map,
                        processed,
                        aliases_per_block,
                    );
                }
            }
            BlockFlowInstruction::Funcall { after_return, .. } => {
                if let Some(after) = after_return {
                    process(
                        blocks,
                        mod_instructions,
                        after,
                        const_map,
                        processed,
                        aliases_per_block,
                    );
                }
            }
            BlockFlowInstruction::IntrinsicImpureFuncall { after_return, .. } => {
                if let Some(after) = after_return {
                    process(
                        blocks,
                        mod_instructions,
                        after,
                        const_map,
                        processed,
                        aliases_per_block,
                    );
                }
            }
            BlockFlowInstruction::Conditional {
                r#true, r#false, ..
            } => {
                process(
                    blocks,
                    mod_instructions,
                    r#true,
                    const_map,
                    processed,
                    aliases_per_block,
                );
                process(
                    blocks,
                    mod_instructions,
                    r#false,
                    const_map,
                    processed,
                    aliases_per_block,
                );
            }
            BlockFlowInstruction::ConditionalLoop { r#break, body, .. } => {
                process(
                    blocks,
                    mod_instructions,
                    body,
                    const_map,
                    processed,
                    aliases_per_block,
                );
                process(
                    blocks,
                    mod_instructions,
                    r#break,
                    const_map,
                    processed,
                    aliases_per_block,
                );
            }
            BlockFlowInstruction::Break | BlockFlowInstruction::Continue => (),
            BlockFlowInstruction::Return(_) | BlockFlowInstruction::Undetermined => (),
        }

        aliases_per_block.insert(n, outgoing_aliases);
    }

    process(
        blocks,
        mod_instructions,
        BlockRef(0),
        const_map,
        &mut processed,
        &mut aliases_per_block,
    );
    aliases_per_block
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalMemoryUsage {
    Read,
    Write,
    ReadWrite,
}
pub fn collect_scope_local_memory_usages<'a, 's>(
    blocks: &[Block],
    mod_instructions: &HashMap<RegisterRef, BlockInstruction<'a, 's>>,
    const_map: &HashMap<RegisterRef, BlockInstruction<'a, 's>>,
) -> HashMap<(PtrEq<'a, SymbolScope<'a, 's>>, usize), LocalMemoryUsage> {
    let mut usage_map = HashMap::new();

    for b in blocks {
        for r in b.eval_impure_registers.iter() {
            match mod_instructions.get(r) {
                Some(BlockInstruction::LoadRef(ptr)) => {
                    if let Some(&BlockInstruction::ScopeLocalVarRef(scope, id)) =
                        const_map.get(&ptr)
                    {
                        let e = usage_map
                            .entry((scope, id))
                            .or_insert(LocalMemoryUsage::Read);
                        match e {
                            LocalMemoryUsage::Write => *e = LocalMemoryUsage::ReadWrite,
                            LocalMemoryUsage::Read | LocalMemoryUsage::ReadWrite => (),
                        }
                    } else if let Some(&BlockInstruction::SwizzleRef(src, _)) =
                        mod_instructions.get(ptr)
                    {
                        if let Some(&BlockInstruction::ScopeLocalVarRef(scope, id)) =
                            const_map.get(&src)
                        {
                            let e = usage_map
                                .entry((scope, id))
                                .or_insert(LocalMemoryUsage::Read);
                            match e {
                                LocalMemoryUsage::Write => *e = LocalMemoryUsage::ReadWrite,
                                LocalMemoryUsage::Read | LocalMemoryUsage::ReadWrite => (),
                            }
                        }
                    }
                }
                _ => (),
            }
        }

        match b.flow {
            BlockFlowInstruction::StoreRef { ptr, .. } => {
                if let Some(&BlockInstruction::ScopeLocalVarRef(scope, id)) = const_map.get(&ptr) {
                    let e = usage_map
                        .entry((scope, id))
                        .or_insert(LocalMemoryUsage::Write);
                    match e {
                        LocalMemoryUsage::Read => *e = LocalMemoryUsage::ReadWrite,
                        LocalMemoryUsage::Write | LocalMemoryUsage::ReadWrite => (),
                    }
                }
            }
            _ => (),
        }
    }

    usage_map
}

pub fn strip_write_only_local_memory<'a, 's>(
    blocks: &mut [Block],
    const_map: &HashMap<RegisterRef, BlockInstruction<'a, 's>>,
    usage_map: &HashMap<(PtrEq<'a, SymbolScope<'a, 's>>, usize), LocalMemoryUsage>,
) -> bool {
    let mut modified = false;

    for b in blocks.iter_mut() {
        match b.flow {
            BlockFlowInstruction::StoreRef { ptr, after, .. } => {
                if let Some(&BlockInstruction::ScopeLocalVarRef(scope, id)) = const_map.get(&ptr) {
                    if usage_map.get(&(scope, id)) == Some(&LocalMemoryUsage::Write) {
                        // never read local memory store
                        b.flow = BlockFlowInstruction::Goto(after.unwrap());
                        modified = true;
                    }
                }
            }
            _ => (),
        }
    }

    modified
}

fn flatten_composite_outputs_rec<'a, 's>(
    generated_sink: &mut Vec<(SimplifiedExpression<'a, 's>, ConcreteType<'s>)>,
    scope: &'a SymbolScope<'a, 's>,
    source_type: &ConcreteType<'s>,
    source_id: ExprRef,
    left_slot_numbers: &mut Vec<usize>,
) {
    match source_type {
        &ConcreteType::Intrinsic(_) => {
            // can output directly
            let slot_number = left_slot_numbers.remove(0);
            generated_sink.push((
                SimplifiedExpression::StoreOutput(source_id, slot_number),
                IntrinsicType::Unit.into(),
            ));
        }
        &ConcreteType::Tuple(ref xs) => {
            // flatten more
            for (n, x) in xs.iter().enumerate() {
                generated_sink.push((SimplifiedExpression::TupleRef(source_id, n), x.clone()));
                let new_source_id = ExprRef(generated_sink.len() - 1);
                flatten_composite_outputs_rec(
                    generated_sink,
                    scope,
                    x,
                    new_source_id,
                    left_slot_numbers,
                );
            }
        }
        &ConcreteType::Struct(ref members) => {
            // flatten more
            for x in members {
                generated_sink.push((
                    SimplifiedExpression::MemberRef(source_id, x.name.clone()),
                    x.ty.clone(),
                ));
                let new_source_id = ExprRef(generated_sink.len() - 1);
                flatten_composite_outputs_rec(
                    generated_sink,
                    scope,
                    &x.ty,
                    new_source_id,
                    left_slot_numbers,
                );
            }
        }
        &ConcreteType::UserDefined { name, .. } => match scope.lookup_user_defined_type(name) {
            Some((_, (_, crate::concrete_type::UserDefinedType::Struct(members)))) => {
                // flatten more
                for x in members {
                    generated_sink.push((
                        SimplifiedExpression::MemberRef(source_id, x.name.clone()),
                        x.ty.clone(),
                    ));
                    let new_source_id = ExprRef(generated_sink.len() - 1);
                    flatten_composite_outputs_rec(
                        generated_sink,
                        scope,
                        &x.ty,
                        new_source_id,
                        left_slot_numbers,
                    );
                }
            }
            None => panic!("Error: cannot output this type: {source_type:?}"),
        },
        _ => panic!("Error: cannot output this type: {source_type:?}"),
    }
}

pub fn flatten_composite_outputs<'a, 's>(
    expressions: &mut [(SimplifiedExpression<'a, 's>, ConcreteType<'s>)],
    scope: &'a SymbolScope<'a, 's>,
    scope_arena: &'a Arena<SymbolScope<'a, 's>>,
) -> bool {
    let mut tree_modified = false;

    for n in 0..expressions.len() {
        match &expressions[n].0 {
            &SimplifiedExpression::FlattenAndDistributeOutputComposite(src, ref slot_numbers) => {
                let mut slot_numbers = slot_numbers.clone();
                let mut generated = Vec::new();
                generated.push((
                    SimplifiedExpression::AliasScopeCapture(0),
                    expressions[src.0].1.clone(),
                ));
                let new_scope = scope_arena.alloc(SymbolScope::new(Some(scope), false));
                flatten_composite_outputs_rec(
                    &mut generated,
                    new_scope,
                    &expressions[src.0].1,
                    ExprRef(0),
                    &mut slot_numbers,
                );
                expressions[n] = (
                    SimplifiedExpression::ScopedBlock {
                        capturing: vec![ScopeCaptureSource::Expr(src)],
                        symbol_scope: PtrEq(new_scope),
                        returning: ExprRef(generated.len() - 1),
                        expressions: generated,
                    },
                    IntrinsicType::Unit.into(),
                );
                tree_modified = true;
            }
            _ => (),
        }
    }

    tree_modified
}

fn promote_single_scope<'a, 's>(
    expressions: &mut Vec<(SimplifiedExpression<'a, 's>, ConcreteType<'s>)>,
    block_returning_ref: &mut Option<&mut ExprRef>,
) -> bool {
    if expressions.len() != 1 {
        // SimplifiedExpressionが複数ある
        return false;
    }

    match expressions.pop().unwrap() {
        (
            SimplifiedExpression::ScopedBlock {
                symbol_scope: child_scope,
                expressions: mut scope_expr,
                returning,
                capturing,
            },
            ty,
        ) if capturing.is_empty() => {
            // キャプチャのないスコープはそのまま展開可能（キャプチャあっても展開可能かも）
            assert_eq!(ty, scope_expr[returning.0].1);
            let parent_scope = child_scope.0.parent.unwrap();
            let local_var_offset = parent_scope.merge_local_vars(child_scope.0);
            println!("scopemerge {child_scope:?} -> {:?}", PtrEq(parent_scope));

            for x in scope_expr.iter_mut() {
                promote_local_var_scope(&mut x.0, child_scope.0, parent_scope, local_var_offset);
            }

            expressions.extend(scope_expr);
            if let Some(ref mut b) = block_returning_ref {
                **b = returning;
            }

            println!("[single scope promotion]");

            true
        }
        (x, t) => {
            expressions.push((x, t));
            false
        }
    }
}

fn construct_refpath<'a, 's>(
    body: &[(SimplifiedExpression<'a, 's>, ConcreteType<'s>)],
    expr_id: ExprRef,
    function_scope: &'a SymbolScope<'a, 's>,
) -> Option<RefPath> {
    match &body[expr_id.0].0 {
        &SimplifiedExpression::VarRef(vscope, VarId::FunctionInput(n))
            if vscope == PtrEq(function_scope) =>
        {
            Some(RefPath::FunctionInput(n))
        }
        &SimplifiedExpression::MemberRef(base, ref name) => {
            let composite_index = match &body[base.0].1 {
                ConcreteType::Ref(inner) => match &**inner {
                    ConcreteType::Struct(members) => {
                        members.iter().position(|m| &m.name == name).unwrap()
                    }
                    _ => unreachable!(),
                },
                ConcreteType::MutableRef(inner) => match &**inner {
                    ConcreteType::Struct(members) => {
                        members.iter().position(|m| &m.name == name).unwrap()
                    }
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            };

            Some(RefPath::Member(
                Box::new(construct_refpath(body, base, function_scope)?),
                composite_index,
            ))
        }
        _ => None,
    }
}

pub fn optimize_pure_expr<'a, 's>(
    expressions: &mut Vec<(SimplifiedExpression<'a, 's>, ConcreteType<'s>)>,
    scope: &'a SymbolScope<'a, 's>,
    scope_arena: &'a Arena<SymbolScope<'a, 's>>,
    mut block_returning_ref: Option<&mut ExprRef>,
    refpaths: &HashSet<RefPath>,
) -> bool {
    let mut least_one_tree_modified = false;
    let mut tree_modified = true;

    // println!("opt input:");
    // for (n, (x, t)) in expressions.iter().enumerate() {
    //     super::expr::print_simp_expr(&mut std::io::stdout(), x, t, n, 0);
    // }

    while tree_modified {
        tree_modified = false;

        tree_modified |= promote_single_scope(expressions, &mut block_returning_ref);
        tree_modified |= flatten_composite_outputs(expressions, scope, scope_arena);

        // reference canonicalize
        for n in 0..expressions.len() {
            match &mut expressions[n].0 {
                &mut SimplifiedExpression::VarRef(_, _)
                | &mut SimplifiedExpression::MemberRef(_, _) => {
                    let refpath = construct_refpath(
                        expressions,
                        ExprRef(n),
                        scope.nearest_function_scope().unwrap(),
                    );
                    if let Some(refpath) = refpath {
                        if refpaths.contains(&refpath) {
                            expressions[n].0 = SimplifiedExpression::CanonicalPathRef(refpath);
                            tree_modified = true;
                        }
                    }
                }
                _ => (),
            }
        }

        // construct chained ref
        let mut expressions_head_ptr = expressions.as_ptr();
        let mut n = 0;
        while n < expressions.len() {
            match &mut expressions[n].0 {
                &mut SimplifiedExpression::ArrayRef(base, x) => {
                    match &expressions[base.0].0 {
                        &SimplifiedExpression::ChainedRef(base, ref xs) => {
                            expressions[n].0 = SimplifiedExpression::ChainedRef(
                                base,
                                xs.iter().copied().chain(core::iter::once(x)).collect(),
                            );
                        }
                        _ => {
                            expressions[n].0 = SimplifiedExpression::ChainedRef(base, vec![x]);
                        }
                    }

                    tree_modified = true;
                    n += 1;
                }
                &mut SimplifiedExpression::MemberRef(base, ref name) => {
                    // base ref op should be appeared before
                    assert!(base.0 < n);

                    let composite_index = match unsafe { &(*expressions_head_ptr.add(base.0)).1 } {
                        ConcreteType::Ref(inner) => match &**inner {
                            ConcreteType::Struct(members) => {
                                members.iter().position(|m| &m.name == name).unwrap()
                            }
                            _ => unreachable!("{n} {inner:?} {:?}", unsafe {
                                &(*expressions_head_ptr.add(base.0)).0
                            }),
                        },
                        ConcreteType::MutableRef(inner) => match &**inner {
                            ConcreteType::Struct(members) => {
                                members.iter().position(|m| &m.name == name).unwrap()
                            }
                            _ => unreachable!("{n} {inner:?} {:?}", unsafe {
                                &(*expressions_head_ptr.add(base.0)).0
                            }),
                        },
                        t => unreachable!("{n} {t:?}"),
                    };

                    expressions.insert(
                        n,
                        (
                            SimplifiedExpression::ConstUIntImm(composite_index as _),
                            IntrinsicType::UInt.into(),
                        ),
                    );
                    expressions_head_ptr = expressions.as_ptr();
                    let composite_index_expr = ExprRef(n);
                    // rewrite shifted reference
                    for m in n..expressions.len() {
                        expressions[m]
                            .0
                            .relocate_ref(|x| if x >= n { x + 1 } else { x });
                    }

                    if let Some(ref mut ret) = block_returning_ref {
                        ret.0 += if ret.0 >= n { 1 } else { 0 };
                    }

                    match &expressions[base.0].0 {
                        &SimplifiedExpression::ChainedRef(base, ref xs) => {
                            expressions[n + 1].0 = SimplifiedExpression::ChainedRef(
                                base,
                                xs.iter()
                                    .copied()
                                    .chain(core::iter::once(composite_index_expr))
                                    .collect(),
                            );
                        }
                        _ => {
                            expressions[n + 1].0 =
                                SimplifiedExpression::ChainedRef(base, vec![composite_index_expr]);
                        }
                    }

                    tree_modified = true;
                    n += 2;
                }
                _ => {
                    n += 1;
                }
            }
        }

        // combine same ref
        let mut last_ref = HashMap::new();
        for n in 0..expressions.len() {
            match &mut expressions[n].0 {
                &mut SimplifiedExpression::VarRef(_, _)
                | &mut SimplifiedExpression::ArrayRef(_, _)
                | &mut SimplifiedExpression::MemberRef(_, _)
                | &mut SimplifiedExpression::TupleRef(_, _)
                | &mut SimplifiedExpression::ChainedRef(_, _)
                | &mut SimplifiedExpression::CanonicalPathRef(_)
                | &mut SimplifiedExpression::AliasScopeCapture(_) => {
                    match last_ref.entry(expressions[n].0.clone()) {
                        std::collections::hash_map::Entry::Occupied(e) => {
                            expressions[n].0 = SimplifiedExpression::Alias(*e.get());
                            tree_modified = true;
                        }
                        std::collections::hash_map::Entry::Vacant(e) => {
                            e.insert(ExprRef(n));
                        }
                    }
                }
                _ => (),
            }
        }

        // caching loadref until dirtified
        let mut loaded_refs = HashMap::new();
        let mut last_localvar_stored_refs = HashMap::new();
        let mut store_expr_refs_loaded_after = HashSet::new();
        for n in 0..expressions.len() {
            match &mut expressions[n].0 {
                &mut SimplifiedExpression::LoadRef(r) => {
                    match loaded_refs.entry(r) {
                        std::collections::hash_map::Entry::Occupied(e) => {
                            expressions[n].0 = SimplifiedExpression::Alias(*e.get());
                            tree_modified = true;
                        }
                        std::collections::hash_map::Entry::Vacant(e) => {
                            e.insert(ExprRef(n));
                        }
                    }

                    if let SimplifiedExpression::VarRef(_, VarId::ScopeLocal(_)) =
                        expressions[r.0].0
                    {
                        if let Some(&last_store) = last_localvar_stored_refs.get(&r) {
                            store_expr_refs_loaded_after.insert(ExprRef(last_store));
                        }
                    }
                }
                &mut SimplifiedExpression::StoreRef(r, x) => {
                    loaded_refs.insert(r, x);

                    if let SimplifiedExpression::VarRef(_, VarId::ScopeLocal(_)) =
                        expressions[r.0].0
                    {
                        last_localvar_stored_refs.insert(r, n);
                    }
                }
                _ => {
                    if !ExprRef(n).is_pure(expressions) {
                        // impureな式をまたぐ場合は改めてLoadしなおす
                        loaded_refs.clear();
                    }
                }
            }
        }

        // strip localvar store refs which is never load after
        for n in 0..expressions.len() {
            match &mut expressions[n].0 {
                &mut SimplifiedExpression::StoreRef(r, _)
                    if !store_expr_refs_loaded_after.contains(&ExprRef(n)) =>
                {
                    if let SimplifiedExpression::VarRef(_, VarId::ScopeLocal(_)) =
                        expressions[r.0].0
                    {
                        expressions[n].0 = SimplifiedExpression::Nop;
                    }
                }
                _ => (),
            }
        }

        // resolve alias
        let mut aliased_expr = HashMap::new();
        for n in 0..expressions.len() {
            match &mut expressions[n].0 {
                &mut SimplifiedExpression::Alias(a) => {
                    aliased_expr.insert(n, a);
                }
                x => {
                    tree_modified |= x.relocate_ref(|x| aliased_expr.get(&x).map_or(x, |a| a.0));
                }
            }
        }

        // inlining loadvar until dirtified / resolving expression aliases
        // let mut localvar_equivalent_expr_id = HashMap::new();
        // let mut expr_id_alias = HashMap::new();
        // let mut last_expr_ids = HashMap::new();
        // for n in 0..expressions.len() {
        //     match &mut expressions[n].0 {
        //         &mut SimplifiedExpression::LoadVar(vscope, vid) => {
        //             if let Some(x) =
        //                 localvar_equivalent_expr_id.get(&(vscope.0 as *const SymbolScope, vid))
        //             {
        //                 expr_id_alias.insert(n, *x);
        //                 scope.relocate_local_var_init_expr(
        //                     |r| if r.0 == n { ExprRef(*x) } else { r },
        //                 );
        //             }

        //             localvar_equivalent_expr_id.insert((vscope.0 as *const SymbolScope, vid), n);
        //         }
        //         &mut SimplifiedExpression::InitializeVar(vscope, VarId::ScopeLocal(vx)) => {
        //             let init_expr_id = vscope.0.init_expr_id(vx).unwrap();

        //             expr_id_alias.insert(n, init_expr_id.0);
        //             scope.relocate_local_var_init_expr(|r| if r.0 == n { init_expr_id } else { r });
        //             localvar_equivalent_expr_id
        //                 .insert((vscope.as_ptr(), VarId::ScopeLocal(vx)), init_expr_id.0);
        //         }
        //         &mut SimplifiedExpression::StoreVar(vscope, vx, store) => {
        //             let last_load_id = localvar_equivalent_expr_id
        //                 .insert((vscope.as_ptr(), VarId::ScopeLocal(vx)), store.0);
        //             if let Some(lld) = last_load_id {
        //                 expr_id_alias.retain(|_, v| *v != lld);
        //             }
        //         }
        //         &mut SimplifiedExpression::Alias(x) => {
        //             expr_id_alias.insert(n, x.0);
        //             scope.relocate_local_var_init_expr(|r| if r.0 == n { x } else { r });
        //         }
        //         x => {
        //             tree_modified |=
        //                 x.relocate_ref(|x| expr_id_alias.get(&x).copied().unwrap_or(x));
        //             if let Some(x) = last_expr_ids.get(&*x) {
        //                 expr_id_alias.insert(n, *x);
        //                 scope.relocate_local_var_init_expr(
        //                     |r| if r.0 == n { ExprRef(*x) } else { r },
        //                 );
        //             } else {
        //                 last_expr_ids.insert(x.clone(), n);
        //             }
        //         }
        //     }
        // }

        let mut referenced_expr = HashSet::new();
        let mut current_scope_var_usages = HashMap::new();
        referenced_expr.extend(block_returning_ref.as_ref().map(|x| **x));
        let expressions_head_ptr = expressions.as_ptr();
        for n in 0..expressions.len() {
            match &mut expressions[n].0 {
                &mut SimplifiedExpression::Nop => (),
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
                &mut SimplifiedExpression::ShiftLeft(left, right) => {
                    referenced_expr.extend([left, right]);
                }
                &mut SimplifiedExpression::ShiftRight(left, right) => {
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
                &mut SimplifiedExpression::Pow(left, right) => {
                    referenced_expr.extend([left, right]);
                }
                &mut SimplifiedExpression::Select(c, t, e) => {
                    referenced_expr.extend([c, t, e]);
                }
                &mut SimplifiedExpression::Funcall(base, ref args) => {
                    let intrinsic_constructor =
                        match unsafe { &(&*expressions_head_ptr.add(base.0)).1 } {
                            ConcreteType::IntrinsicTypeConstructor(it) => match it {
                                IntrinsicType::Float2 => Some(IntrinsicType::Float2),
                                IntrinsicType::Float3 => Some(IntrinsicType::Float3),
                                IntrinsicType::Float4 => Some(IntrinsicType::Float4),
                                IntrinsicType::UInt2 => Some(IntrinsicType::UInt2),
                                IntrinsicType::UInt3 => Some(IntrinsicType::UInt3),
                                IntrinsicType::UInt4 => Some(IntrinsicType::UInt4),
                                IntrinsicType::SInt2 => Some(IntrinsicType::SInt2),
                                IntrinsicType::SInt3 => Some(IntrinsicType::SInt3),
                                IntrinsicType::SInt4 => Some(IntrinsicType::SInt4),
                                IntrinsicType::Float => Some(IntrinsicType::Float),
                                IntrinsicType::SInt => Some(IntrinsicType::SInt),
                                IntrinsicType::UInt => Some(IntrinsicType::UInt),
                                _ => None,
                            },
                            _ => None,
                        };

                    if let Some(it) = intrinsic_constructor {
                        let args = args.clone();
                        referenced_expr.extend(args.iter().copied());

                        match &args[..] {
                            &[a] if unsafe { &(*expressions_head_ptr.add(a.0)).1 }
                                .vector_elements()
                                == it.vector_elements() =>
                            {
                                // same components construct => casting
                                expressions[n].0 = SimplifiedExpression::Cast(a, it.into())
                            }
                            _ => {
                                expressions[n].0 =
                                    SimplifiedExpression::ConstructIntrinsicComposite(it, args);
                            }
                        }

                        tree_modified = true;
                    } else {
                        let intrinsic_function = match unsafe {
                            &(&*expressions_head_ptr.add(base.0)).0
                        } {
                            &SimplifiedExpression::IntrinsicFunctions(ref overloads) => {
                                let matching_overloads = overloads.iter().find(|f| {
                                    f.args.iter().zip(args.iter()).all(|x| {
                                        x.0 == unsafe { &(&*expressions_head_ptr.add(x.1 .0)).1 }
                                    })
                                });

                                match matching_overloads {
                                    Some(f) => Some((f.name, f.is_pure, f.output.clone())),
                                    None => panic!("Error: no matching overloads found"),
                                }
                            }
                            _ => None,
                        };

                        if let Some((instr, is_pure, output)) = intrinsic_function {
                            let args = args.clone();
                            referenced_expr.extend(args.iter().copied());

                            expressions[n].0 =
                                SimplifiedExpression::IntrinsicFuncall(instr, is_pure, args);
                            expressions[n].1 = output;
                            tree_modified = true;
                        } else {
                            referenced_expr.insert(base);
                            referenced_expr.extend(args.iter().copied());
                        }
                    }
                }
                &mut SimplifiedExpression::LoadRef(ptr) => {
                    referenced_expr.insert(ptr);
                }
                // &mut SimplifiedExpression::LoadVar(scope, VarId::FunctionInput(vx))
                //     if scope.0.is_toplevel_function =>
                // {
                //     expressions[n].0 =
                //         SimplifiedExpression::LoadByCanonicalRefPath(RefPath::FunctionInput(vx));
                //     tree_modified = true;
                // }
                // &mut SimplifiedExpression::LoadVar(vscope, VarId::ScopeLocal(vx))
                //     if vscope == PtrEq(scope) =>
                // {
                //     current_scope_var_usages
                //         .entry(vx)
                //         .or_insert(LocalVarUsage::Unaccessed)
                //         .mark_read();
                // }
                // &mut SimplifiedExpression::LoadVar(_, _) => (),
                &mut SimplifiedExpression::LoadByCanonicalRefPath(_) => (),
                &mut SimplifiedExpression::StoreRef(ptr, v) => {
                    referenced_expr.extend([ptr, v]);
                }
                &mut SimplifiedExpression::VarRef(vscope, VarId::ScopeLocal(vx))
                    if vscope == PtrEq(scope) =>
                {
                    current_scope_var_usages
                        .entry(vx)
                        .or_insert(LocalVarUsage::Unaccessed)
                        .mark_read();
                }
                &mut SimplifiedExpression::VarRef(_, _) => (),
                &mut SimplifiedExpression::TupleRef(base, index) => match &expressions[base.0].0 {
                    &SimplifiedExpression::ConstructTuple(ref xs) => {
                        expressions[n].0 = SimplifiedExpression::Alias(xs[index]);
                        tree_modified = true;
                    }
                    _ => {
                        referenced_expr.insert(base);
                    }
                },
                &mut SimplifiedExpression::MemberRef(
                    base,
                    SourceRefSliceEq(SourceRef { slice: name, .. }),
                ) => match &expressions[base.0].0 {
                    SimplifiedExpression::LoadByCanonicalRefPath(rp) => {
                        let member_index = match &expressions[base.0].1 {
                            &ConcreteType::Struct(ref member) => {
                                match member.iter().position(|x| x.name.0.slice == name){
                                    Some(x) => x,
                                    None => panic!("Error: struct type does not contains member named '{name}'"),
                                }
                            }
                            &ConcreteType::UserDefined { .. } => panic!("not instantiated"),
                            _ => unreachable!("Error: cannot apply MemberRef for non-struct types"),
                        };

                        expressions[n].0 = SimplifiedExpression::LoadByCanonicalRefPath(
                            RefPath::Member(Box::new(rp.clone()), member_index),
                        );
                        tree_modified = true;
                    }
                    SimplifiedExpression::ConstructStructValue(ref xs) => {
                        let member_index = match &expressions[base.0].1 {
                            &ConcreteType::Struct(ref member) => {
                                match member.iter().position(|x| x.name.0.slice == name){
                                    Some(x) => x,
                                    None => panic!("Error: struct type does not contains member named '{name}'"),
                                }
                            }
                            &ConcreteType::UserDefined { .. } => panic!("not instantiated"),
                            _ => unreachable!("Error: cannot apply MemberRef for non-struct types"),
                        };

                        expressions[n].0 = SimplifiedExpression::Alias(xs[member_index]);
                        tree_modified = true;
                    }
                    _ => {
                        referenced_expr.insert(base);
                    }
                },
                &mut SimplifiedExpression::ArrayRef(base, x) => {
                    referenced_expr.extend([base, x]);
                }
                &mut SimplifiedExpression::SwizzleRef1(src, _) => {
                    referenced_expr.insert(src);
                }
                &mut SimplifiedExpression::SwizzleRef2(src, _, _) => {
                    referenced_expr.insert(src);
                }
                &mut SimplifiedExpression::SwizzleRef3(src, _, _, _) => {
                    referenced_expr.insert(src);
                }
                &mut SimplifiedExpression::SwizzleRef4(src, _, _, _, _) => {
                    referenced_expr.insert(src);
                }
                &mut SimplifiedExpression::ChainedRef(base, ref xs) => {
                    referenced_expr.insert(base);
                    referenced_expr.extend(xs.iter().copied());
                }
                &mut SimplifiedExpression::CanonicalPathRef(_) => (),
                // &mut SimplifiedExpression::StoreVar(vscope, vx, v) if vscope == PtrEq(scope) => {
                //     referenced_expr.insert(v);
                //     current_scope_var_usages
                //         .entry(vx)
                //         .or_insert(LocalVarUsage::Unaccessed)
                //         .mark_write(ExprRef(n));
                // }
                // &mut SimplifiedExpression::StoreVar(_, _, v) => {
                //     referenced_expr.insert(v);
                // }
                &mut SimplifiedExpression::RefFunction(_, _) => (),
                &mut SimplifiedExpression::IntrinsicFunctions(_) => (),
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
                &mut SimplifiedExpression::VectorShuffle4(v1, v2, _, _, _, _) => {
                    referenced_expr.extend([v1, v2]);
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
                &mut SimplifiedExpression::ConstUIntImm(_) => (),
                &mut SimplifiedExpression::ConstUInt(_, _) => (),
                &mut SimplifiedExpression::ConstSInt(_, _) => (),
                &mut SimplifiedExpression::ConstFloat(_, _) => (),
                &mut SimplifiedExpression::ConstIntToNumber(x) => {
                    if let &SimplifiedExpression::ConstInt(ref t) = &expressions[x.0].0 {
                        expressions[n].0 = SimplifiedExpression::ConstNumber(t.clone());
                    } else {
                        referenced_expr.insert(x);
                    }
                }
                &mut SimplifiedExpression::ConstructTuple(ref xs) => {
                    referenced_expr.extend(xs.iter().copied());
                }
                &mut SimplifiedExpression::ConstructIntrinsicComposite(
                    IntrinsicType::Float4,
                    ref xs,
                ) => match &xs[..] {
                    &[a, b] => match (&expressions[a.0], &expressions[b.0]) {
                        // TODO: 他パターンは後で実装
                        (
                            &(_, ConcreteType::Intrinsic(IntrinsicType::Float3)),
                            &(SimplifiedExpression::Swizzle1(src2, d), _),
                        ) => {
                            // splice float3
                            expressions[n].0 =
                                SimplifiedExpression::VectorShuffle4(a, src2, 0, 1, 2, d + 3);
                            referenced_expr.extend([a, src2]);
                            tree_modified = true;
                        }
                        (
                            &(_, ConcreteType::Intrinsic(IntrinsicType::Float3)),
                            &(_, ConcreteType::Intrinsic(IntrinsicType::Float)),
                        ) => {
                            // decomposite float3 and construct
                            expressions[n].0 = SimplifiedExpression::ScopedBlock {
                                capturing: vec![
                                    ScopeCaptureSource::Expr(a),
                                    ScopeCaptureSource::Expr(b),
                                ],
                                symbol_scope: PtrEq(
                                    scope_arena.alloc(SymbolScope::new(Some(scope), false)),
                                ),
                                expressions: vec![
                                    (
                                        SimplifiedExpression::AliasScopeCapture(0),
                                        IntrinsicType::Float3.into(),
                                    ),
                                    (
                                        SimplifiedExpression::Swizzle1(ExprRef(0), 0),
                                        IntrinsicType::Float.into(),
                                    ),
                                    (
                                        SimplifiedExpression::Swizzle1(ExprRef(0), 1),
                                        IntrinsicType::Float.into(),
                                    ),
                                    (
                                        SimplifiedExpression::Swizzle1(ExprRef(0), 2),
                                        IntrinsicType::Float.into(),
                                    ),
                                    (
                                        SimplifiedExpression::AliasScopeCapture(1),
                                        IntrinsicType::Float.into(),
                                    ),
                                    (
                                        SimplifiedExpression::ConstructIntrinsicComposite(
                                            IntrinsicType::Float4,
                                            vec![ExprRef(1), ExprRef(2), ExprRef(3), ExprRef(4)],
                                        ),
                                        IntrinsicType::Float4.into(),
                                    ),
                                ],
                                returning: ExprRef(5),
                            };
                            referenced_expr.extend([a, b]);
                            tree_modified = true;
                        }
                        _ => {
                            referenced_expr.extend([a, b]);
                        }
                    },
                    _ => {
                        referenced_expr.extend(xs.iter().copied());
                    }
                },
                &mut SimplifiedExpression::ConstructIntrinsicComposite(_, ref xs) => {
                    referenced_expr.extend(xs.iter().copied());
                }
                &mut SimplifiedExpression::StoreOutput(x, _) => {
                    referenced_expr.insert(x);
                }
                &mut SimplifiedExpression::FlattenAndDistributeOutputComposite(x, _) => {
                    referenced_expr.insert(x);
                }
                &mut SimplifiedExpression::ConstructStructValue(ref xs) => {
                    referenced_expr.extend(xs.iter().copied());
                }
                SimplifiedExpression::ScopedBlock {
                    ref mut expressions,
                    ref mut returning,
                    ref symbol_scope,
                    ref capturing,
                } => {
                    tree_modified |= optimize_pure_expr(
                        expressions,
                        symbol_scope.0,
                        scope_arena,
                        Some(returning),
                        refpaths,
                    );

                    referenced_expr.extend(capturing.iter().filter_map(|x| match x {
                        ScopeCaptureSource::Expr(x) => Some(x),
                        _ => None,
                    }));

                    for (n, x) in expressions.iter().enumerate() {
                        match x.0 {
                            // SimplifiedExpression::LoadVar(vscope, VarId::ScopeLocal(vx))
                            //     if vscope == PtrEq(scope) =>
                            // {
                            //     current_scope_var_usages
                            //         .entry(vx)
                            //         .or_insert(LocalVarUsage::Unaccessed)
                            //         .mark_read();
                            // }
                            _ => (),
                        }
                    }
                }
                SimplifiedExpression::LoopBlock {
                    ref mut expressions,
                    ref symbol_scope,
                    ref capturing,
                } => {
                    tree_modified |= optimize_pure_expr(
                        expressions,
                        symbol_scope.0,
                        scope_arena,
                        None,
                        refpaths,
                    );

                    referenced_expr.extend(capturing.iter().filter_map(|x| match x {
                        ScopeCaptureSource::Expr(x) => Some(x),
                        _ => None,
                    }));

                    for (n, x) in expressions.iter().enumerate() {
                        match x.0 {
                            // SimplifiedExpression::LoadVar(vscope, VarId::ScopeLocal(vx))
                            //     if vscope == PtrEq(scope) =>
                            // {
                            //     current_scope_var_usages
                            //         .entry(vx)
                            //         .or_insert(LocalVarUsage::Unaccessed)
                            //         .mark_read();
                            // }
                            _ => (),
                        }
                    }
                }
                &mut SimplifiedExpression::BreakLoop(x) => {
                    referenced_expr.insert(x);
                }
                &mut SimplifiedExpression::AliasScopeCapture(_) => (),
                &mut SimplifiedExpression::Alias(x) => {
                    referenced_expr.insert(x);
                }
            }
        }

        // collect stripped expression ids
        let mut stripped_ops = HashSet::new();
        for (_, t) in current_scope_var_usages.iter() {
            if let &LocalVarUsage::Write(last_write) = t {
                stripped_ops.insert(last_write.0);
            }
        }
        for n in 0..expressions.len() {
            if !referenced_expr.contains(&ExprRef(n)) && ExprRef(n).is_pure(expressions) {
                stripped_ops.insert(n);
            }
        }
        let mut referenced_expr = referenced_expr.into_iter().collect::<Vec<_>>();

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

            for x in referenced_expr.iter_mut() {
                x.0 -= if x.0 > n { 1 } else { 0 };
            }

            for x in stripped_ops.iter_mut() {
                *x -= if *x > n { 1 } else { 0 };
            }

            tree_modified = true;
        }

        // strip unaccessed local vars
        let mut stripped_local_var_ids = scope
            .all_local_var_ids()
            .filter(|lvid| !current_scope_var_usages.contains_key(lvid))
            .collect::<Vec<_>>();
        while let Some(n) = stripped_local_var_ids.pop() {
            scope.remove_local_var_by_id(n);

            // rewrite shifted references
            for m in 0..expressions.len() {
                match &mut expressions[m].0 {
                    &mut SimplifiedExpression::VarRef(vscope, VarId::ScopeLocal(ref mut vx))
                        if vscope == PtrEq(scope) =>
                    {
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

        // unfold unreferenced computation scope
        for n in 0..expressions.len() {
            match &mut expressions[n] {
                (
                    SimplifiedExpression::ScopedBlock {
                        expressions: scope_expr,
                        symbol_scope,
                        returning,
                        capturing,
                    },
                    _,
                ) if !symbol_scope.0.has_local_vars()
                    && (!referenced_expr.contains(&ExprRef(n))
                        || block_returning_ref.as_ref().is_some_and(|x| x.0 == n)) =>
                {
                    let returning_rel = returning.0;

                    // relocate scope local ids and unlift scope capture refs
                    for x in scope_expr.iter_mut() {
                        x.0.relocate_ref(|x| x + n);
                        match &mut x.0 {
                            &mut SimplifiedExpression::AliasScopeCapture(n) => {
                                x.0 = match capturing[n] {
                                    ScopeCaptureSource::Expr(x) => SimplifiedExpression::Alias(x),
                                    ScopeCaptureSource::Capture(x) => {
                                        SimplifiedExpression::AliasScopeCapture(x)
                                    }
                                };
                            }
                            _ => (),
                        }
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
                        expressions[m].0.relocate_ref(|x| {
                            if x == n {
                                n + returning_rel
                            } else if x > n {
                                x + nth_shifts
                            } else {
                                x
                            }
                        });
                    }

                    if let Some(ref mut ret) = block_returning_ref {
                        if ret.0 == n {
                            ret.0 = n + returning_rel;
                        } else if ret.0 > n {
                            ret.0 += nth_shifts;
                        }
                    }

                    tree_modified = true;
                }
                _ => (),
            }
        }

        // println!("transformed(cont?{tree_modified}):");
        // for (n, (x, t)) in expressions.iter().enumerate() {
        //     super::expr::print_simp_expr(&mut std::io::stdout(), x, t, n, 0);
        // }

        least_one_tree_modified |= tree_modified;
    }

    least_one_tree_modified
}

fn promote_local_var_scope<'a, 's>(
    expr: &mut SimplifiedExpression<'a, 's>,
    old_scope: &'a SymbolScope<'a, 's>,
    new_scope: &'a SymbolScope<'a, 's>,
    local_var_offset: usize,
) {
    match expr {
        &mut SimplifiedExpression::VarRef(vscope, VarId::ScopeLocal(lv))
            if vscope == PtrEq(old_scope) =>
        {
            *expr = SimplifiedExpression::VarRef(PtrEq(new_scope), VarId::ScopeLocal(lv));
        }
        SimplifiedExpression::ScopedBlock { expressions, .. } => {
            for x in expressions.iter_mut() {
                promote_local_var_scope(&mut x.0, old_scope, new_scope, local_var_offset);
            }
        }
        _ => (),
    }
}

*/
