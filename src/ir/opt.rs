use std::{
    collections::{BTreeMap, HashMap, HashSet},
    io::Write,
};

use typed_arena::Arena;

use crate::{
    concrete_type::{ConcreteType, IntrinsicType},
    ir::block::RegisterRef,
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
        Block, BlockFlowInstruction, BlockGenerationContext, BlockInstruction, BlockRef,
        IntrinsicBinaryOperation,
    },
    expr::{ConstModifiers, ScopeCaptureSource, SimplifiedExpression},
    ConstFloatLiteral, ConstNumberLiteral, ConstSIntLiteral, ConstUIntLiteral, ExprRef,
    InstantiatedConst,
};

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

pub fn merge_simple_goto_blocks(blocks: &mut [Block]) -> bool {
    let mut modified = false;

    for n in 0..blocks.len() {
        if let BlockFlowInstruction::Goto(next) = blocks[n].flow {
            println!("[MergeSimpleGoto] b{n}->b{next}", next = next.0);
            let (current, merged) = unsafe {
                (
                    &mut *blocks.as_mut_ptr().add(n),
                    &*blocks.as_ptr().add(next.0),
                )
            };

            if !merged.has_block_dependent_instructions() && !merged.is_loop_term_block() {
                current
                    .instructions
                    .extend(merged.instructions.iter().map(|(&r, x)| (r, x.clone())));
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
                        for x in blocks[new_after.0].instructions.values_mut() {
                            x.dup_phi_incoming(next, BlockRef(n));
                        }
                    }
                    BlockFlowInstruction::Conditional {
                        r#true: new_true_after,
                        r#false: new_false_after,
                        ..
                    } => {
                        // 新しいとび先にphiがあれば、元のとび先のエントリと同じものを今のブロックからのものとして追加
                        println!("[MergeSimpleGoto] rechain: phi redirect b{n}->b{next}->b{new_after} => b{n}->b{new_after}", next = next.0, new_after = new_true_after.0);
                        for x in blocks[new_true_after.0].instructions.values_mut() {
                            x.dup_phi_incoming(next, BlockRef(n));
                        }
                        println!("[MergeSimpleGoto] rechain: phi redirect b{n}->b{next}->b{new_after} => b{n}->b{new_after}", next = next.0, new_after = new_false_after.0);
                        for x in blocks[new_false_after.0].instructions.values_mut() {
                            x.dup_phi_incoming(next, BlockRef(n));
                        }
                    }
                    BlockFlowInstruction::ConditionalLoop {
                        r#break: new_break_after,
                        body: new_body_after,
                        ..
                    } => {
                        // 新しいとび先にphiがあれば、元のとび先のエントリと同じものを今のブロックからのものとして追加
                        println!("[MergeSimpleGoto] rechain: phi redirect b{n}->b{next}->b{new_after} => b{n}->b{new_after}", next = next.0, new_after = new_break_after.0);
                        for x in blocks[new_break_after.0].instructions.values_mut() {
                            x.dup_phi_incoming(next, BlockRef(n));
                        }
                        println!("[MergeSimpleGoto] rechain: phi redirect b{n}->b{next}->b{new_after} => b{n}->b{new_after}", next = next.0, new_after = new_body_after.0);
                        for x in blocks[new_body_after.0].instructions.values_mut() {
                            x.dup_phi_incoming(next, BlockRef(n));
                        }
                    }
                    BlockFlowInstruction::Break | BlockFlowInstruction::Continue => {
                        unreachable!("break/continue cannot determine new_after")
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

pub fn strip_unreachable_blocks(blocks: &mut Vec<Block>) -> bool {
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
            for x in b.instructions.values_mut() {
                match x {
                    BlockInstruction::Phi(ref mut incoming_selectors) => {
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

pub fn promote_instantiate_const<'a, 's>(
    instructions: &mut HashMap<RegisterRef, BlockInstruction<'a, 's>>,
    const_map: &mut HashMap<RegisterRef, BlockInstruction<'a, 's>>,
) -> bool {
    let mut modified = false;

    for (r, x) in core::mem::replace(instructions, HashMap::with_capacity(instructions.len())) {
        match x {
            BlockInstruction::InstantiateIntrinsicTypeClass(src, ty) => match const_map.get(&src) {
                Some(BlockInstruction::ConstInt(l)) => match ty {
                    IntrinsicType::UInt => {
                        const_map.insert(
                            r,
                            BlockInstruction::ConstUInt(ConstUIntLiteral(l.0.clone(), l.1)),
                        );
                        modified = true;
                    }
                    IntrinsicType::SInt => {
                        const_map.insert(
                            r,
                            BlockInstruction::ConstSInt(ConstSIntLiteral(l.0.clone(), l.1)),
                        );
                        modified = true;
                    }
                    IntrinsicType::Float => {
                        const_map.insert(
                            r,
                            BlockInstruction::ConstFloat(ConstFloatLiteral(l.0.clone(), l.1)),
                        );
                        modified = true;
                    }
                    _ => unreachable!(),
                },
                Some(BlockInstruction::ConstNumber(l)) => match ty {
                    IntrinsicType::Float => {
                        const_map.insert(
                            r,
                            BlockInstruction::ConstFloat(ConstFloatLiteral(l.0.clone(), l.1)),
                        );
                        modified = true;
                    }
                    IntrinsicType::UInt | IntrinsicType::SInt => {
                        panic!("number -> int can cause precision dropping")
                    }
                    _ => unreachable!(),
                },
                _ => {
                    instructions.insert(r, x);
                }
            },
            BlockInstruction::PromoteIntToNumber(value) => match const_map.get(&value) {
                Some(BlockInstruction::ConstInt(l)) => {
                    const_map.insert(
                        r,
                        BlockInstruction::ConstNumber(ConstNumberLiteral(l.0.clone(), l.1)),
                    );
                    modified = true;
                }
                _ => {
                    instructions.insert(r, x);
                }
            },
            _ => {
                instructions.insert(r, x);
            }
        }
    }

    modified
}

pub fn extract_constants<'a, 's>(
    instructions: &mut HashMap<RegisterRef, BlockInstruction<'a, 's>>,
) -> HashMap<RegisterRef, BlockInstruction<'a, 's>> {
    let mut const_instructions = HashMap::new();

    for (r, x) in core::mem::replace(instructions, HashMap::with_capacity(instructions.len())) {
        match x {
            BlockInstruction::ConstInt { .. }
            | BlockInstruction::ConstNumber { .. }
            | BlockInstruction::ConstUInt { .. }
            | BlockInstruction::ConstSInt { .. }
            | BlockInstruction::ConstFloat { .. }
            | BlockInstruction::ConstUnit
            | BlockInstruction::ScopeLocalVarRef { .. }
            | BlockInstruction::FunctionInputVarRef { .. }
            | BlockInstruction::UserDefinedFunctionRef { .. }
            | BlockInstruction::IntrinsicFunctionRef { .. }
            | BlockInstruction::IntrinsicTypeConstructorRef { .. } => {
                const_instructions.insert(r, x);
            }
            _ => {
                instructions.insert(r, x);
            }
        }
    }

    const_instructions
}

pub fn fold_const_ops<'a, 's>(
    instructions: &mut HashMap<RegisterRef, BlockInstruction<'a, 's>>,
    const_map: &mut HashMap<RegisterRef, BlockInstruction<'a, 's>>,
) -> bool {
    let mut modified = false;

    for (res, x) in core::mem::replace(instructions, HashMap::with_capacity(instructions.len())) {
        match x {
            BlockInstruction::IntrinsicBinaryOp(left, op, right) => {
                let Some(left) = const_map
                    .get(&left)
                    .and_then(BlockInstruction::try_instantiate_const)
                else {
                    instructions.insert(res, x);
                    continue;
                };
                let Some(right) = const_map
                    .get(&right)
                    .and_then(BlockInstruction::try_instantiate_const)
                else {
                    instructions.insert(res, x);
                    continue;
                };

                match op {
                    IntrinsicBinaryOperation::Add => match (left, right) {
                        (InstantiatedConst::Int(l), InstantiatedConst::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmInt(l + r));
                            modified = true;
                        }
                        (InstantiatedConst::SInt(l), InstantiatedConst::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmSInt(l + r));
                            modified = true;
                        }
                        (InstantiatedConst::UInt(l), InstantiatedConst::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmUInt(l + r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::Sub => match (left, right) {
                        (InstantiatedConst::Int(l), InstantiatedConst::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmInt(l - r));
                            modified = true;
                        }
                        (InstantiatedConst::SInt(l), InstantiatedConst::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmSInt(l - r));
                            modified = true;
                        }
                        (InstantiatedConst::UInt(l), InstantiatedConst::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmUInt(l - r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::Mul => match (left, right) {
                        (InstantiatedConst::Int(l), InstantiatedConst::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmInt(l * r));
                            modified = true;
                        }
                        (InstantiatedConst::SInt(l), InstantiatedConst::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmSInt(l * r));
                            modified = true;
                        }
                        (InstantiatedConst::UInt(l), InstantiatedConst::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmUInt(l * r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::Div => match (left, right) {
                        (InstantiatedConst::Int(l), InstantiatedConst::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmInt(l / r));
                            modified = true;
                        }
                        (InstantiatedConst::SInt(l), InstantiatedConst::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmSInt(l / r));
                            modified = true;
                        }
                        (InstantiatedConst::UInt(l), InstantiatedConst::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmUInt(l / r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::Rem => match (left, right) {
                        (InstantiatedConst::Int(l), InstantiatedConst::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmInt(l % r));
                            modified = true;
                        }
                        (InstantiatedConst::SInt(l), InstantiatedConst::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmSInt(l % r));
                            modified = true;
                        }
                        (InstantiatedConst::UInt(l), InstantiatedConst::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmUInt(l % r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::Pow => match (left, right) {
                        (InstantiatedConst::Int(l), InstantiatedConst::Int(r)) => {
                            const_map.insert(
                                res,
                                BlockInstruction::ImmInt(l.pow(r.try_into().unwrap())),
                            );
                            modified = true;
                        }
                        (InstantiatedConst::SInt(l), InstantiatedConst::SInt(r)) => {
                            const_map.insert(
                                res,
                                BlockInstruction::ImmSInt(l.pow(r.try_into().unwrap())),
                            );
                            modified = true;
                        }
                        (InstantiatedConst::UInt(l), InstantiatedConst::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmUInt(l.pow(r)));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::BitAnd => match (left, right) {
                        (InstantiatedConst::Int(l), InstantiatedConst::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmInt(l & r));
                            modified = true;
                        }
                        (InstantiatedConst::SInt(l), InstantiatedConst::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmSInt(l & r));
                            modified = true;
                        }
                        (InstantiatedConst::UInt(l), InstantiatedConst::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmUInt(l & r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::BitOr => match (left, right) {
                        (InstantiatedConst::Int(l), InstantiatedConst::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmInt(l | r));
                            modified = true;
                        }
                        (InstantiatedConst::SInt(l), InstantiatedConst::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmSInt(l | r));
                            modified = true;
                        }
                        (InstantiatedConst::UInt(l), InstantiatedConst::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmUInt(l | r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::BitXor => match (left, right) {
                        (InstantiatedConst::Int(l), InstantiatedConst::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmInt(l ^ r));
                            modified = true;
                        }
                        (InstantiatedConst::SInt(l), InstantiatedConst::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmSInt(l ^ r));
                            modified = true;
                        }
                        (InstantiatedConst::UInt(l), InstantiatedConst::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmUInt(l ^ r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::LeftShift => match (left, right) {
                        (InstantiatedConst::Int(l), InstantiatedConst::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmInt(l << r));
                            modified = true;
                        }
                        (InstantiatedConst::SInt(l), InstantiatedConst::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmSInt(l << r));
                            modified = true;
                        }
                        (InstantiatedConst::UInt(l), InstantiatedConst::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmUInt(l << r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::RightShift => match (left, right) {
                        (InstantiatedConst::Int(l), InstantiatedConst::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmInt(l >> r));
                            modified = true;
                        }
                        (InstantiatedConst::SInt(l), InstantiatedConst::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmSInt(l >> r));
                            modified = true;
                        }
                        (InstantiatedConst::UInt(l), InstantiatedConst::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmUInt(l >> r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::Eq => match (left, right) {
                        (InstantiatedConst::Int(l), InstantiatedConst::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l == r));
                            modified = true;
                        }
                        (InstantiatedConst::SInt(l), InstantiatedConst::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l == r));
                            modified = true;
                        }
                        (InstantiatedConst::UInt(l), InstantiatedConst::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l == r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::Ne => match (left, right) {
                        (InstantiatedConst::Int(l), InstantiatedConst::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l != r));
                            modified = true;
                        }
                        (InstantiatedConst::SInt(l), InstantiatedConst::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l != r));
                            modified = true;
                        }
                        (InstantiatedConst::UInt(l), InstantiatedConst::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l != r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::Lt => match (left, right) {
                        (InstantiatedConst::Int(l), InstantiatedConst::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l < r));
                            modified = true;
                        }
                        (InstantiatedConst::SInt(l), InstantiatedConst::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l < r));
                            modified = true;
                        }
                        (InstantiatedConst::UInt(l), InstantiatedConst::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l < r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::Gt => match (left, right) {
                        (InstantiatedConst::Int(l), InstantiatedConst::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l > r));
                            modified = true;
                        }
                        (InstantiatedConst::SInt(l), InstantiatedConst::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l > r));
                            modified = true;
                        }
                        (InstantiatedConst::UInt(l), InstantiatedConst::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l > r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::Le => match (left, right) {
                        (InstantiatedConst::Int(l), InstantiatedConst::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l <= r));
                            modified = true;
                        }
                        (InstantiatedConst::SInt(l), InstantiatedConst::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l <= r));
                            modified = true;
                        }
                        (InstantiatedConst::UInt(l), InstantiatedConst::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l <= r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::Ge => match (left, right) {
                        (InstantiatedConst::Int(l), InstantiatedConst::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l >= r));
                            modified = true;
                        }
                        (InstantiatedConst::SInt(l), InstantiatedConst::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l >= r));
                            modified = true;
                        }
                        (InstantiatedConst::UInt(l), InstantiatedConst::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l >= r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::LogAnd => match (left, right) {
                        (InstantiatedConst::Bool(l), InstantiatedConst::Bool(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l && r));
                            modified = true;
                        }
                        (InstantiatedConst::Int(l), InstantiatedConst::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l != 0 && r != 0));
                            modified = true;
                        }
                        (InstantiatedConst::SInt(l), InstantiatedConst::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l != 0 && r != 0));
                            modified = true;
                        }
                        (InstantiatedConst::UInt(l), InstantiatedConst::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l != 0 && r != 0));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::LogOr => match (left, right) {
                        (InstantiatedConst::Bool(l), InstantiatedConst::Bool(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l || r));
                            modified = true;
                        }
                        (InstantiatedConst::Int(l), InstantiatedConst::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l != 0 || r != 0));
                            modified = true;
                        }
                        (InstantiatedConst::SInt(l), InstantiatedConst::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l != 0 || r != 0));
                            modified = true;
                        }
                        (InstantiatedConst::UInt(l), InstantiatedConst::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l != 0 || r != 0));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                }
            }
            _ => {
                instructions.insert(res, x);
            }
        }
    }

    modified
}

pub fn block_aliasing(blocks: &mut [Block]) -> bool {
    let block_aliases_to = blocks
        .iter()
        .enumerate()
        .filter_map(|(n, b)| match b {
            &Block {
                ref instructions,
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
            BlockFlowInstruction::Goto(ref mut next) => {
                if let Some(&skip) = block_aliases_to.get(next) {
                    // skip先のphiにnextからのものがあったらコピー
                    for x in blocks[skip.0].instructions.values_mut() {
                        x.dup_phi_incoming(*next, BlockRef(n));
                    }

                    *next = skip;
                    modified = true;
                }
            }
            BlockFlowInstruction::StoreRef {
                after: Some(ref mut next),
                ..
            } => {
                if let Some(&skip) = block_aliases_to.get(next) {
                    // skip先のphiにnextからのものがあったらコピー
                    for x in blocks[skip.0].instructions.values_mut() {
                        x.dup_phi_incoming(*next, BlockRef(n));
                    }

                    *next = skip;
                    modified = true;
                }
            }
            BlockFlowInstruction::Funcall {
                after_return: Some(ref mut next),
                ..
            } => {
                if let Some(&skip) = block_aliases_to.get(next) {
                    // skip先のphiにnextからのものがあったらコピー
                    for x in blocks[skip.0].instructions.values_mut() {
                        x.dup_phi_incoming(*next, BlockRef(n));
                    }

                    *next = skip;
                    modified = true;
                }
            }
            BlockFlowInstruction::IntrinsicImpureFuncall {
                after_return: Some(ref mut next),
                ..
            } => {
                if let Some(&skip) = block_aliases_to.get(next) {
                    // skip先のphiにnextからのものがあったらコピー
                    for x in blocks[skip.0].instructions.values_mut() {
                        x.dup_phi_incoming(*next, BlockRef(n));
                    }

                    *next = skip;
                    modified = true;
                }
            }
            BlockFlowInstruction::Conditional {
                ref mut r#true,
                ref mut r#false,
                ..
            } => {
                if let Some(&skip) = block_aliases_to.get(r#true) {
                    // skip先のphiに分岐先からのものがあったらコピー
                    for x in blocks[skip.0].instructions.values_mut() {
                        x.dup_phi_incoming(*r#true, BlockRef(n));
                    }

                    *r#true = skip;
                    modified = true;
                }

                if let Some(&skip) = block_aliases_to.get(r#false) {
                    // skip先のphiに分岐先からのものがあったらコピー
                    for x in blocks[skip.0].instructions.values_mut() {
                        x.dup_phi_incoming(*r#false, BlockRef(n));
                    }

                    *r#false = skip;
                    modified = true;
                }
            }
            BlockFlowInstruction::ConditionalLoop {
                ref mut r#break,
                ref mut body,
                ..
            } => {
                if let Some(&skip) = block_aliases_to.get(r#break) {
                    // skip先のphiに分岐先からのものがあったらコピー
                    for x in blocks[skip.0].instructions.values_mut() {
                        x.dup_phi_incoming(*r#break, BlockRef(n));
                    }

                    *r#break = skip;
                    modified = true;
                }

                if let Some(&skip) = block_aliases_to.get(body) {
                    // skip先のphiに分岐先からのものがあったらコピー
                    for x in blocks[skip.0].instructions.values_mut() {
                        x.dup_phi_incoming(*body, BlockRef(n));
                    }

                    *body = skip;
                    modified = true;
                }
            }
            _ => (),
        }
    }

    modified
}

pub fn resolve_intrinsic_funcalls<'a, 's>(
    block_ctx: &mut BlockGenerationContext<'a, 's>,
    const_map: &HashMap<RegisterRef, BlockInstruction<'a, 's>>,
) -> bool {
    let mut modified = false;

    for n in 0..block_ctx.blocks.len() {
        match block_ctx.blocks[n].flow {
            BlockFlowInstruction::Funcall {
                result,
                callee,
                ref args,
                after_return,
            } => match const_map.get(&callee) {
                Some(&BlockInstruction::IntrinsicTypeConstructorRef(ty)) => {
                    let inst = BlockInstruction::ConstructIntrinsicComposite(ty, args.clone());

                    block_ctx.blocks[n].instructions.insert(result, inst);
                    block_ctx.blocks[n].flow = BlockFlowInstruction::Goto(after_return.unwrap());
                    modified = true;
                }
                Some(BlockInstruction::IntrinsicFunctionRef(overloads)) => {
                    let selected_overload = overloads
                        .iter()
                        .find(|o| {
                            o.args
                                .iter()
                                .zip(args.iter())
                                .all(|(def, call)| def == call.ty(block_ctx))
                        })
                        .expect("Error: no matching overload found");

                    if selected_overload.is_pure {
                        let inst = BlockInstruction::PureIntrinsicCall(
                            selected_overload.name,
                            args.clone(),
                        );

                        block_ctx.blocks[n].instructions.insert(result, inst);
                        block_ctx.blocks[n].flow =
                            BlockFlowInstruction::Goto(after_return.unwrap());
                        modified = true;
                    } else {
                        block_ctx.blocks[n].flow = BlockFlowInstruction::IntrinsicImpureFuncall {
                            identifier: selected_overload.name,
                            args: args.clone(),
                            result,
                            after_return,
                        };
                    }
                }
                _ => (),
            },
            _ => (),
        }
    }

    modified
}

pub fn merge_constants<'a, 's>(
    blocks: &mut [Block<'a, 's>],
    const_map: &mut HashMap<RegisterRef, BlockInstruction<'a, 's>>,
) -> bool {
    let mut const_to_register_map = HashMap::new();
    let mut register_rewrite_map = HashMap::new();
    for (r, v) in const_map.iter() {
        match const_to_register_map.entry(v) {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(r);
            }
            std::collections::hash_map::Entry::Occupied(mut e) => {
                if e.get().0 > r.0 {
                    // 若い番号を優先する
                    let mapped = e.insert(r);
                    register_rewrite_map.insert(mapped, r);
                } else {
                    register_rewrite_map.insert(r, e.get());
                }
            }
        }
    }

    if !register_rewrite_map.is_empty() {
        println!("register rewrite map:");
        for (from, to) in register_rewrite_map.iter() {
            println!("  r{} -> r{}", from.0, to.0);
        }
    }

    let mut modified = blocks.iter_mut().fold(false, |m, b| {
        let m1 = b.relocate_register(|r| {
            while let Some(&&nr) = register_rewrite_map.get(r) {
                *r = nr;
            }
        });

        m || m1
    });

    let alive_const_registers = const_to_register_map
        .into_values()
        .copied()
        .collect::<HashSet<_>>();
    let old_const_map_len = const_map.len();
    const_map.retain(|k, _| alive_const_registers.contains(k));
    modified = modified || const_map.len() != old_const_map_len;

    modified
}

pub fn inline_function2<'a, 's>(
    blocks: &mut Vec<Block<'a, 's>>,
    const_map: &mut HashMap<RegisterRef, BlockInstruction<'a, 's>>,
    scope_arena: &'a Arena<SymbolScope<'a, 's>>,
    function_root_scope: &'a SymbolScope<'a, 's>,
    registers: &mut Vec<ConcreteType<'s>>,
) -> bool {
    let mut modified = false;

    let mut n = 0;
    while n < blocks.len() {
        if let BlockFlowInstruction::Funcall {
            callee,
            ref args,
            result,
            after_return,
        } = blocks[n].flow
        {
            if let Some(BlockInstruction::UserDefinedFunctionRef(scope, name)) =
                const_map.get(&callee)
            {
                let target_function_symbol =
                    scope.0.user_defined_function_symbol(name.0.slice).unwrap();
                let target_function_body =
                    scope.0.user_defined_function_body(name.0.slice).unwrap();

                println!("[Inlining Function] {name:?} at {scope:?} ({args:?})");
                println!("  symbol = {target_function_symbol:#?}");
                println!("  body: {target_function_body:#?}");

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
                    registers.push(def.3.clone().mutable_ref());
                    let ptr = RegisterRef(registers.len() - 1);
                    arg_store_set.push((*ee, ptr, varid));
                    function_input_remap.insert(n, varid);
                }

                let register_offset = registers.len();
                registers.extend(target_function_body.borrow().registers.iter().cloned());
                const_map.extend(
                    target_function_body
                        .borrow()
                        .constants
                        .iter()
                        .map(|(k, v)| {
                            if let BlockInstruction::FunctionInputVarRef(scope, id) = v {
                                if *scope == PtrEq(target_function_body.borrow().symbol_scope) {
                                    let Some(&varid) = function_input_remap.get(&id) else {
                                        unreachable!("no function arg to var remap");
                                    };

                                    return (
                                        RegisterRef(k.0 + register_offset),
                                        BlockInstruction::ScopeLocalVarRef(PtrEq(tmp_scope), varid),
                                    );
                                }
                            }

                            (RegisterRef(k.0 + register_offset), v.clone())
                        }),
                );
                const_map.extend(arg_store_set.iter().map(|(_, ptr, varid)| {
                    (
                        *ptr,
                        BlockInstruction::ScopeLocalVarRef(PtrEq(tmp_scope), *varid),
                    )
                }));

                dbg!(n);
                let mut after_blocks = blocks.split_off(n + 1);

                let mut setup_blocks = arg_store_set
                    .into_iter()
                    .enumerate()
                    .map(|(n, (a, ptr, _))| Block {
                        instructions: HashMap::new(),
                        flow: BlockFlowInstruction::StoreRef {
                            ptr,
                            value: a,
                            after: Some(BlockRef(n + 1)),
                        },
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
                        let instruction_count = nb.instructions.len();
                        for (k, mut v) in core::mem::replace(
                            &mut nb.instructions,
                            HashMap::with_capacity(instruction_count),
                        ) {
                            v.relocate_register(|r| {
                                r.0 += register_offset;
                            });
                            v.relocate_block_ref(|b| b.0 += expand_block_base);

                            nb.instructions
                                .insert(RegisterRef(k.0 + register_offset), v);
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
                let funcall_merge_block = Block {
                    instructions: [(result, BlockInstruction::Phi(exit_block_incomings))]
                        .into_iter()
                        .collect(),
                    flow: match after_return {
                        Some(b) => BlockFlowInstruction::Goto(BlockRef(
                            b.0 - after_block_before_base + after_block_base,
                        )),
                        None => BlockFlowInstruction::Undetermined,
                    },
                };

                for b in blocks.iter_mut() {
                    for x in b.instructions.values_mut() {
                        x.relocate_block_ref(|b| {
                            b.0 += if b.0 > n {
                                after_block_base - after_block_before_base
                            } else {
                                0
                            }
                        });
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
                    for x in b.instructions.values_mut() {
                        x.relocate_block_ref(|b| {
                            b.0 += if b.0 > n {
                                after_block_base - after_block_before_base
                            } else {
                                0
                            }
                        });
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
    blocks: &mut [Block<'a, 's>],
    const_map: &mut HashMap<RegisterRef, BlockInstruction<'a, 's>>,
    registers: &[ConcreteType<'s>],
) -> bool {
    let mut modified = false;

    for v in const_map.values_mut() {
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

    for b in blocks.iter_mut() {
        let instruction_count = b.instructions.len();
        for (r, x) in core::mem::replace(
            &mut b.instructions,
            HashMap::with_capacity(instruction_count),
        ) {
            match x {
                BlockInstruction::MemberRef(src, member) => {
                    let Some(ConcreteType::Struct(type_members)) =
                        registers[src.0].as_dereferenced()
                    else {
                        panic!(
                            "Error: cannot ref member of this type: {:?}",
                            registers[src.0]
                        );
                    };
                    let member_index = type_members.iter().position(|m| m.name == member).unwrap();
                    let member_record = &type_members[member_index];

                    let descriptor_bound = match member_record.attribute {
                        SymbolAttribute {
                            descriptor_set_location: Some(set),
                            descriptor_set_binding: Some(binding),
                            ..
                        } => Some(DescriptorBound { set, binding }),
                        _ => None,
                    };
                    let push_constant_bound = match member_record.attribute {
                        SymbolAttribute {
                            push_constant_offset: Some(offset),
                            ..
                        } => Some(PushConstantBound { offset }),
                        _ => None,
                    };
                    let builtin_bound = match member_record.attribute {
                        SymbolAttribute {
                            bound_builtin_io: Some(builtin),
                            ..
                        } => Some(BuiltinBound(builtin)),
                        _ => None,
                    };
                    let workgroup_shared = member_record.attribute.workgroup_shared;

                    match (
                        descriptor_bound,
                        push_constant_bound,
                        builtin_bound,
                        workgroup_shared,
                    ) {
                        (Some(d), None, None, false) => {
                            const_map.insert(
                                r,
                                BlockInstruction::DescriptorRef {
                                    set: d.set,
                                    binding: d.binding,
                                },
                            );
                            modified = true;
                        }
                        (None, Some(p), None, false) => {
                            const_map.insert(r, BlockInstruction::PushConstantRef(p.offset));
                            modified = true;
                        }
                        (None, None, Some(b), false) => {
                            const_map.insert(r, BlockInstruction::BuiltinIORef(b.0));
                            modified = true;
                        }
                        (None, None, None, true) => match const_map.get(&src) {
                            Some(&BlockInstruction::FunctionInputVarRef(scope, id))
                                if scope == PtrEq(function_root_scope) =>
                            {
                                const_map.insert(
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
                            b.instructions
                                .insert(r, BlockInstruction::MemberRef(src, member));
                        }
                        _ => panic!("Error: conflicting shader io attributes"),
                    }
                }
                _ => {
                    b.instructions.insert(r, x);
                }
            }
        }
    }

    modified
}

pub fn convert_static_path_ref<'a, 's>(
    function_scope: &'a SymbolScope<'a, 's>,
    instructions: &mut HashMap<RegisterRef, BlockInstruction<'a, 's>>,
    const_map: &mut HashMap<RegisterRef, BlockInstruction<'a, 's>>,
    registers: &[ConcreteType<'s>],
) -> bool {
    let mut modified = false;

    for x in const_map.values_mut() {
        match x {
            &mut BlockInstruction::FunctionInputVarRef(scope, vid)
                if scope == PtrEq(function_scope) =>
            {
                *x = BlockInstruction::StaticPathRef(RefPath::FunctionInput(vid));
                modified = true;
            }
            _ => (),
        }
    }

    for (r, x) in core::mem::replace(instructions, HashMap::with_capacity(instructions.len())) {
        match x {
            BlockInstruction::MemberRef(src, member) => {
                let ConcreteType::Struct(type_members) =
                    registers[src.0].as_dereferenced().unwrap()
                else {
                    unreachable!("Error: cannot reference member for this type");
                };
                let member_index = type_members.iter().position(|m| m.name == member).unwrap();

                match const_map.get(&src) {
                    Some(BlockInstruction::StaticPathRef(path)) => {
                        let path = RefPath::Member(Box::new(path.clone()), member_index);
                        const_map.insert(r, BlockInstruction::StaticPathRef(path));
                        modified = true;
                    }
                    _ => {
                        instructions.insert(r, BlockInstruction::MemberRef(src, member));
                    }
                }
            }
            _ => {
                instructions.insert(r, x);
            }
        }
    }

    modified
}

pub fn build_register_state_map<'a, 's>(
    blocks: &[Block<'a, 's>],
) -> HashMap<BlockRef, HashMap<RegisterRef, BlockInstruction<'a, 's>>> {
    let mut state_map = HashMap::new();
    let mut loop_stack = Vec::new();
    let mut processed = HashSet::new();

    // b0は何もない状態
    state_map.insert(BlockRef(0), HashMap::new());

    fn process<'a, 's>(
        blocks: &[Block<'a, 's>],
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
        for (r, x) in blocks[incoming.0].instructions.iter() {
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

        state_map.insert(incoming, incoming_state);

        if !modified && processed.contains(&n) {
            return;
        }

        processed.insert(n);
        match blocks[n.0].flow {
            BlockFlowInstruction::Goto(next) => {
                process(blocks, next, n, state_map, loop_stack, processed)
            }
            BlockFlowInstruction::StoreRef {
                after: Some(after), ..
            } => process(blocks, after, n, state_map, loop_stack, processed),
            BlockFlowInstruction::StoreRef { .. } => (),
            BlockFlowInstruction::Funcall {
                after_return: Some(after_return),
                ..
            } => process(blocks, after_return, n, state_map, loop_stack, processed),
            BlockFlowInstruction::Funcall { .. } => (),
            BlockFlowInstruction::IntrinsicImpureFuncall {
                after_return: Some(after_return),
                ..
            } => process(blocks, after_return, n, state_map, loop_stack, processed),
            BlockFlowInstruction::IntrinsicImpureFuncall { .. } => (),
            BlockFlowInstruction::Conditional {
                r#true, r#false, ..
            } => {
                process(blocks, r#true, n, state_map, loop_stack, processed);
                process(blocks, r#false, n, state_map, loop_stack, processed);
            }
            BlockFlowInstruction::ConditionalLoop {
                r#break,
                r#continue,
                body,
                ..
            } => {
                loop_stack.push((r#break, r#continue));
                process(blocks, body, n, state_map, loop_stack, processed);
                loop_stack.pop();
                process(blocks, r#break, n, state_map, loop_stack, processed);
            }
            BlockFlowInstruction::Break => {
                let &(brk, _) = loop_stack.last().unwrap();
                process(blocks, brk, n, state_map, loop_stack, processed);
            }
            BlockFlowInstruction::Continue => {
                let &(_, cont) = loop_stack.last().unwrap();
                process(blocks, cont, n, state_map, loop_stack, processed);
            }
            BlockFlowInstruction::Return(_) | BlockFlowInstruction::Undetermined => (),
        }
    }

    match blocks[0].flow {
        BlockFlowInstruction::Goto(next) => process(
            blocks,
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
                r#true,
                BlockRef(0),
                &mut state_map,
                &mut loop_stack,
                &mut processed,
            );
            process(
                blocks,
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
                body,
                BlockRef(0),
                &mut state_map,
                &mut loop_stack,
                &mut processed,
            );
            loop_stack.pop();
            process(
                blocks,
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
    blocks: &mut [Block<'a, 's>],
    register_state_map: &HashMap<BlockRef, HashMap<RegisterRef, BlockInstruction<'a, 's>>>,
) -> bool {
    let mut modified = false;

    for (n, b) in blocks.iter_mut().enumerate() {
        let Some(state_map) = register_state_map.get(&BlockRef(n)) else {
            continue;
        };

        let in_block_map = b
            .instructions
            .iter()
            .map(|(r, x)| (*r, x.clone()))
            .collect::<HashMap<_, _>>();

        for x in b.instructions.values_mut() {
            let m = x.relocate_register(|x| loop {
                match state_map.get(x).or_else(|| in_block_map.get(x)) {
                    Some(&BlockInstruction::RegisterAlias(to)) => {
                        *x = to;
                    }
                    _ => break,
                }
            });
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
    registers: &mut Vec<ConcreteType>,
    const_map: &mut HashMap<RegisterRef, BlockInstruction>,
) -> bool {
    let mut referenced_registers = HashSet::new();
    for b in blocks.iter() {
        for x in b.instructions.values() {
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
                &BlockInstruction::IntrinsicBinaryOp(x, _, y)
                | &BlockInstruction::ArrayRef {
                    source: x,
                    index: y,
                } => {
                    referenced_registers.extend([x, y]);
                }
                &BlockInstruction::ConstructTuple(ref xs)
                | &BlockInstruction::ConstructStruct(ref xs)
                | &BlockInstruction::ConstructIntrinsicComposite(_, ref xs)
                | &BlockInstruction::PureIntrinsicCall(_, ref xs) => {
                    referenced_registers.extend(xs.iter().copied());
                }
                &BlockInstruction::Phi(ref xs) => {
                    referenced_registers.extend(xs.values().copied());
                }
                &BlockInstruction::PureFuncall(callee, ref xs) => {
                    referenced_registers.insert(callee);
                    referenced_registers.extend(xs.iter().copied());
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
            b.instructions.remove(&stripped);

            if swapped_register != stripped {
                if let Some(s) = b.instructions.remove(&swapped_register) {
                    b.instructions.insert(stripped, s);
                }

                for x in b.instructions.values_mut() {
                    x.relocate_register(|r| {
                        if r == &swapped_register {
                            *r = stripped;
                        }
                    });
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
    blocks: &mut [Block<'a, 's>],
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
        blocks: &mut [Block<'a, 's>],
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
                                blocks[n.0].instructions.get_mut(&p)
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

                                blocks[incoming.0]
                                    .instructions
                                    .insert(rp, BlockInstruction::Phi(phi_incomings));
                                *r_incoming = IncomingRegister::Phied(rp);

                                rp
                            } else {
                                *r_incomings.first_key_value().unwrap().1
                            };

                            let Some(BlockInstruction::Phi(ref mut incomings)) =
                                blocks[n.0].instructions.get_mut(&p)
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

                                blocks[incoming.0]
                                    .instructions
                                    .insert(rp, BlockInstruction::Phi(phi_incomings));
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
    blocks: &mut [Block<'a, 's>],
    const_map: &HashMap<RegisterRef, BlockInstruction<'a, 's>>,
    state_map: &HashMap<
        BlockRef,
        HashMap<(PtrEq<'a, SymbolScope<'a, 's>>, usize), IncomingRegister>,
    >,
) -> bool {
    let mut modified = false;

    for n in 0..blocks.len() {
        for x in blocks[n].instructions.values_mut() {
            match x {
                BlockInstruction::LoadRef(ptr) => {
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

pub fn deconstruct_effectless_phi(
    instructions: &mut HashMap<RegisterRef, BlockInstruction>,
) -> bool {
    let mut modified = false;
    for (r, x) in instructions.iter_mut() {
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
                match &unique_registers[..] {
                    &[r] => {
                        *x = BlockInstruction::RegisterAlias(r);
                        modified = true;
                    }
                    &[r1, r2] => {
                        if r1 == *r && r2 != *r {
                            *x = BlockInstruction::RegisterAlias(r2);
                            modified = true;
                        } else if r1 != *r && r2 == *r {
                            *x = BlockInstruction::RegisterAlias(r1);
                            modified = true;
                        }
                    }
                    _ => (),
                }
            }
            _ => (),
        }
    }

    modified
}

pub fn track_scope_local_var_aliases<'a, 's>(
    blocks: &[Block<'a, 's>],
    const_map: &HashMap<RegisterRef, BlockInstruction<'a, 's>>,
) -> HashMap<BlockRef, HashMap<(PtrEq<'a, SymbolScope<'a, 's>>, usize), RegisterRef>> {
    let mut processed = HashSet::new();
    let mut aliases_per_block = HashMap::new();

    fn process<'a, 's>(
        blocks: &[Block<'a, 's>],
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
                process(blocks, next, const_map, processed, aliases_per_block);
            }
            BlockFlowInstruction::StoreRef { ptr, value, after } => {
                if let Some(BlockInstruction::ScopeLocalVarRef(scope, vid)) = const_map.get(&ptr) {
                    outgoing_aliases.insert((*scope, *vid), value);
                }

                if let Some(after) = after {
                    process(blocks, after, const_map, processed, aliases_per_block);
                }
            }
            BlockFlowInstruction::Funcall { after_return, .. } => {
                if let Some(after) = after_return {
                    process(blocks, after, const_map, processed, aliases_per_block);
                }
            }
            BlockFlowInstruction::IntrinsicImpureFuncall { after_return, .. } => {
                if let Some(after) = after_return {
                    process(blocks, after, const_map, processed, aliases_per_block);
                }
            }
            BlockFlowInstruction::Conditional {
                r#true, r#false, ..
            } => {
                process(blocks, r#true, const_map, processed, aliases_per_block);
                process(blocks, r#false, const_map, processed, aliases_per_block);
            }
            BlockFlowInstruction::ConditionalLoop { r#break, body, .. } => {
                process(blocks, body, const_map, processed, aliases_per_block);
                process(blocks, r#break, const_map, processed, aliases_per_block);
            }
            BlockFlowInstruction::Break | BlockFlowInstruction::Continue => (),
            BlockFlowInstruction::Return(_) | BlockFlowInstruction::Undetermined => (),
        }

        aliases_per_block.insert(n, outgoing_aliases);
    }

    process(
        blocks,
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
    blocks: &[Block<'a, 's>],
    const_map: &HashMap<RegisterRef, BlockInstruction<'a, 's>>,
) -> HashMap<(PtrEq<'a, SymbolScope<'a, 's>>, usize), LocalMemoryUsage> {
    let mut usage_map = HashMap::new();

    for b in blocks {
        for x in b.instructions.values() {
            match x {
                BlockInstruction::LoadRef(ptr) => {
                    if let Some(&BlockInstruction::ScopeLocalVarRef(scope, id)) = const_map.get(ptr)
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
    blocks: &mut [Block<'a, 's>],
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
