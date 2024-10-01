use std::collections::{HashMap, HashSet};

use crate::{
    concrete_type::{ConcreteType, IntrinsicType},
    ir::{
        block::{
            Block, BlockConstInstruction, BlockPureInstruction, Constants, ImpureInstructionMap,
            IntrinsicBinaryOperation, PureInstructions, RegisterAliasMap, RegisterRef,
        },
        ConstFloatLiteral, ConstNumberLiteral, ConstSIntLiteral, ConstUIntLiteral, LosslessConst,
    },
};

/// 定数のInstantiateIntrinsicTypeClassを演算して定数化する
pub fn promote_instantiate_const<'a, 's>(
    pure_instructions: &mut PureInstructions<'s>,
    constants: &mut Constants<'a, 's>,
) -> RegisterAliasMap {
    let mut register_alias_map = HashMap::new();
    let mut pure_instruction_register_shifts = 0;

    for (n, x) in core::mem::replace(
        pure_instructions,
        Vec::with_capacity(pure_instructions.len()),
    )
    .into_iter()
    .enumerate()
    {
        let promoted = match x.inst {
            BlockPureInstruction::InstantiateIntrinsicTypeClass(RegisterRef::Const(src), ty) => {
                match constants[src].inst {
                    BlockConstInstruction::LitInt(ref l) => match ty {
                        IntrinsicType::UInt => Some(
                            BlockConstInstruction::LitUInt(ConstUIntLiteral(l.0.clone(), l.1))
                                .typed(ty.into()),
                        ),
                        IntrinsicType::SInt => Some(
                            BlockConstInstruction::LitSInt(ConstSIntLiteral(l.0.clone(), l.1))
                                .typed(ty.into()),
                        ),
                        IntrinsicType::Float => Some(
                            BlockConstInstruction::LitFloat(ConstFloatLiteral(l.0.clone(), l.1))
                                .typed(ty.into()),
                        ),
                        _ => None,
                    },
                    BlockConstInstruction::LitNum(ref l) => match ty {
                        IntrinsicType::Float => Some(
                            BlockConstInstruction::LitFloat(ConstFloatLiteral(l.0.clone(), l.1))
                                .typed(ty.into()),
                        ),
                        IntrinsicType::UInt | IntrinsicType::SInt => {
                            eprintln!("not promoted: number -> int can cause unintended precision dropping");
                            None
                        }
                        _ => None,
                    },
                    BlockConstInstruction::ImmInt(v) => match ty {
                        IntrinsicType::UInt => Some(
                            BlockConstInstruction::ImmUInt(v.try_into().expect("cannot promote"))
                                .typed(ty.into()),
                        ),
                        IntrinsicType::SInt => Some(
                            BlockConstInstruction::ImmSInt(v.try_into().expect("cannot promote"))
                                .typed(ty.into()),
                        ),
                        _ => None,
                    },
                    _ => None,
                }
            }
            BlockPureInstruction::PromoteIntToNumber(RegisterRef::Const(value)) => {
                match constants[value].inst {
                    BlockConstInstruction::LitInt(ref l) => Some(
                        BlockConstInstruction::LitNum(ConstNumberLiteral(l.0.clone(), l.1))
                            .typed(ConcreteType::UnknownNumberClass),
                    ),
                    _ => None,
                }
            }
            _ => None,
        };

        if let Some(c) = promoted {
            // 定数化した
            constants.push(c);
            register_alias_map.insert(
                RegisterRef::Pure(n),
                RegisterRef::Const(constants.len() - 1),
            );
            pure_instruction_register_shifts += 1;
        } else {
            // Pure命令のまま
            pure_instructions.push(x);
            if pure_instruction_register_shifts > 0 {
                // レジスタシフトが発生している状況
                register_alias_map.insert(
                    RegisterRef::Pure(n),
                    RegisterRef::Pure(n - pure_instruction_register_shifts),
                );
            }
        }
    }

    register_alias_map
}

pub fn fold_const_ops<'a, 's>(
    pure_instructions: &mut PureInstructions,
    constants: &mut Constants,
) -> RegisterAliasMap {
    let mut register_alias_map = HashMap::new();
    let mut pure_register_shifts = 0;

    for (n, x) in core::mem::replace(
        pure_instructions,
        Vec::with_capacity(pure_instructions.len()),
    )
    .into_iter()
    .enumerate()
    {
        match x.inst {
            BlockPureInstruction::IntrinsicBinaryOp(
                RegisterRef::Const(left),
                op,
                RegisterRef::Const(right),
            ) => {
                let Some(left) = constants[left].inst.try_instantiate_lossless_const() else {
                    // 左辺が演算可能な定数じゃない
                    pure_instructions.push(x);
                    if pure_register_shifts > 0 {
                        // レジスタシフトが発生している状況
                        register_alias_map.insert(
                            RegisterRef::Pure(n),
                            RegisterRef::Pure(n - pure_register_shifts),
                        );
                    }
                    continue;
                };
                let Some(right) = constants[right].inst.try_instantiate_lossless_const() else {
                    // 右辺が演算可能な定数じゃない
                    pure_instructions.push(x);
                    if pure_register_shifts > 0 {
                        // レジスタシフトが発生している状況
                        register_alias_map.insert(
                            RegisterRef::Pure(n),
                            RegisterRef::Pure(n - pure_register_shifts),
                        );
                    }
                    continue;
                };

                let reduced = match op {
                    IntrinsicBinaryOperation::Add => match (left, right) {
                        (LosslessConst::Int(l), LosslessConst::Int(r)) => Some(
                            BlockConstInstruction::ImmInt(l + r)
                                .typed(ConcreteType::UnknownIntClass),
                        ),
                        (LosslessConst::SInt(l), LosslessConst::SInt(r)) => Some(
                            BlockConstInstruction::ImmSInt(l + r).typed(IntrinsicType::SInt.into()),
                        ),
                        (LosslessConst::UInt(l), LosslessConst::UInt(r)) => Some(
                            BlockConstInstruction::ImmUInt(l + r).typed(IntrinsicType::UInt.into()),
                        ),
                        _ => None,
                    },
                    IntrinsicBinaryOperation::Sub => match (left, right) {
                        (LosslessConst::Int(l), LosslessConst::Int(r)) => Some(
                            BlockConstInstruction::ImmInt(l - r)
                                .typed(ConcreteType::UnknownIntClass),
                        ),
                        (LosslessConst::SInt(l), LosslessConst::SInt(r)) => Some(
                            BlockConstInstruction::ImmSInt(l - r).typed(IntrinsicType::SInt.into()),
                        ),
                        (LosslessConst::UInt(l), LosslessConst::UInt(r)) => Some(
                            BlockConstInstruction::ImmUInt(l - r).typed(IntrinsicType::UInt.into()),
                        ),
                        _ => None,
                    },
                    IntrinsicBinaryOperation::Mul => match (left, right) {
                        (LosslessConst::Int(l), LosslessConst::Int(r)) => Some(
                            BlockConstInstruction::ImmInt(l * r)
                                .typed(ConcreteType::UnknownIntClass),
                        ),
                        (LosslessConst::SInt(l), LosslessConst::SInt(r)) => Some(
                            BlockConstInstruction::ImmSInt(l * r).typed(IntrinsicType::SInt.into()),
                        ),
                        (LosslessConst::UInt(l), LosslessConst::UInt(r)) => Some(
                            BlockConstInstruction::ImmUInt(l * r).typed(IntrinsicType::UInt.into()),
                        ),
                        _ => None,
                    },
                    IntrinsicBinaryOperation::Div => match (left, right) {
                        (LosslessConst::Int(l), LosslessConst::Int(r)) => Some(
                            BlockConstInstruction::ImmInt(l / r)
                                .typed(ConcreteType::UnknownIntClass),
                        ),
                        (LosslessConst::SInt(l), LosslessConst::SInt(r)) => Some(
                            BlockConstInstruction::ImmSInt(l / r).typed(IntrinsicType::SInt.into()),
                        ),
                        (LosslessConst::UInt(l), LosslessConst::UInt(r)) => Some(
                            BlockConstInstruction::ImmUInt(l / r).typed(IntrinsicType::UInt.into()),
                        ),
                        _ => None,
                    },
                    IntrinsicBinaryOperation::Rem => match (left, right) {
                        (LosslessConst::Int(l), LosslessConst::Int(r)) => Some(
                            BlockConstInstruction::ImmInt(l % r)
                                .typed(ConcreteType::UnknownIntClass),
                        ),
                        (LosslessConst::SInt(l), LosslessConst::SInt(r)) => Some(
                            BlockConstInstruction::ImmSInt(l % r).typed(IntrinsicType::SInt.into()),
                        ),
                        (LosslessConst::UInt(l), LosslessConst::UInt(r)) => Some(
                            BlockConstInstruction::ImmUInt(l % r).typed(IntrinsicType::UInt.into()),
                        ),
                        _ => None,
                    },
                    IntrinsicBinaryOperation::Pow => match (left, right) {
                        (LosslessConst::Int(l), LosslessConst::Int(r)) => Some(
                            BlockConstInstruction::ImmInt(l.pow(r.try_into().unwrap()))
                                .typed(ConcreteType::UnknownIntClass),
                        ),
                        (LosslessConst::SInt(l), LosslessConst::SInt(r)) => Some(
                            BlockConstInstruction::ImmSInt(l.pow(r.try_into().unwrap()))
                                .typed(IntrinsicType::SInt.into()),
                        ),
                        (LosslessConst::UInt(l), LosslessConst::UInt(r)) => Some(
                            BlockConstInstruction::ImmUInt(l.pow(r))
                                .typed(IntrinsicType::UInt.into()),
                        ),
                        _ => None,
                    },
                    IntrinsicBinaryOperation::BitAnd => match (left, right) {
                        (LosslessConst::Int(l), LosslessConst::Int(r)) => Some(
                            BlockConstInstruction::ImmInt(l & r)
                                .typed(ConcreteType::UnknownIntClass),
                        ),
                        (LosslessConst::SInt(l), LosslessConst::SInt(r)) => Some(
                            BlockConstInstruction::ImmSInt(l & r).typed(IntrinsicType::SInt.into()),
                        ),
                        (LosslessConst::UInt(l), LosslessConst::UInt(r)) => Some(
                            BlockConstInstruction::ImmUInt(l & r).typed(IntrinsicType::UInt.into()),
                        ),
                        _ => None,
                    },
                    IntrinsicBinaryOperation::BitOr => match (left, right) {
                        (LosslessConst::Int(l), LosslessConst::Int(r)) => Some(
                            BlockConstInstruction::ImmInt(l | r)
                                .typed(ConcreteType::UnknownIntClass),
                        ),
                        (LosslessConst::SInt(l), LosslessConst::SInt(r)) => Some(
                            BlockConstInstruction::ImmSInt(l | r).typed(IntrinsicType::SInt.into()),
                        ),
                        (LosslessConst::UInt(l), LosslessConst::UInt(r)) => Some(
                            BlockConstInstruction::ImmUInt(l | r).typed(IntrinsicType::UInt.into()),
                        ),
                        _ => None,
                    },
                    IntrinsicBinaryOperation::BitXor => match (left, right) {
                        (LosslessConst::Int(l), LosslessConst::Int(r)) => Some(
                            BlockConstInstruction::ImmInt(l ^ r)
                                .typed(ConcreteType::UnknownIntClass),
                        ),
                        (LosslessConst::SInt(l), LosslessConst::SInt(r)) => Some(
                            BlockConstInstruction::ImmSInt(l ^ r).typed(IntrinsicType::SInt.into()),
                        ),
                        (LosslessConst::UInt(l), LosslessConst::UInt(r)) => Some(
                            BlockConstInstruction::ImmUInt(l ^ r).typed(IntrinsicType::UInt.into()),
                        ),
                        _ => None,
                    },
                    IntrinsicBinaryOperation::LeftShift => match (left, right) {
                        (LosslessConst::Int(l), LosslessConst::Int(r)) => Some(
                            BlockConstInstruction::ImmInt(l << r)
                                .typed(ConcreteType::UnknownIntClass),
                        ),
                        (LosslessConst::SInt(l), LosslessConst::SInt(r)) => Some(
                            BlockConstInstruction::ImmSInt(l << r)
                                .typed(IntrinsicType::SInt.into()),
                        ),
                        (LosslessConst::UInt(l), LosslessConst::UInt(r)) => Some(
                            BlockConstInstruction::ImmUInt(l << r)
                                .typed(IntrinsicType::UInt.into()),
                        ),
                        _ => None,
                    },
                    IntrinsicBinaryOperation::RightShift => match (left, right) {
                        (LosslessConst::Int(l), LosslessConst::Int(r)) => Some(
                            BlockConstInstruction::ImmInt(l >> r)
                                .typed(ConcreteType::UnknownIntClass),
                        ),
                        (LosslessConst::SInt(l), LosslessConst::SInt(r)) => Some(
                            BlockConstInstruction::ImmSInt(l >> r)
                                .typed(IntrinsicType::SInt.into()),
                        ),
                        (LosslessConst::UInt(l), LosslessConst::UInt(r)) => Some(
                            BlockConstInstruction::ImmUInt(l >> r)
                                .typed(IntrinsicType::UInt.into()),
                        ),
                        _ => None,
                    },
                    IntrinsicBinaryOperation::Eq => match (left, right) {
                        (LosslessConst::Int(l), LosslessConst::Int(r)) => Some(
                            BlockConstInstruction::ImmBool(l == r)
                                .typed(IntrinsicType::Bool.into()),
                        ),
                        (LosslessConst::SInt(l), LosslessConst::SInt(r)) => Some(
                            BlockConstInstruction::ImmBool(l == r)
                                .typed(IntrinsicType::Bool.into()),
                        ),
                        (LosslessConst::UInt(l), LosslessConst::UInt(r)) => Some(
                            BlockConstInstruction::ImmBool(l == r)
                                .typed(IntrinsicType::Bool.into()),
                        ),
                        _ => None,
                    },
                    IntrinsicBinaryOperation::Ne => match (left, right) {
                        (LosslessConst::Int(l), LosslessConst::Int(r)) => Some(
                            BlockConstInstruction::ImmBool(l != r)
                                .typed(IntrinsicType::Bool.into()),
                        ),
                        (LosslessConst::SInt(l), LosslessConst::SInt(r)) => Some(
                            BlockConstInstruction::ImmBool(l != r)
                                .typed(IntrinsicType::Bool.into()),
                        ),
                        (LosslessConst::UInt(l), LosslessConst::UInt(r)) => Some(
                            BlockConstInstruction::ImmBool(l != r)
                                .typed(IntrinsicType::Bool.into()),
                        ),
                        _ => None,
                    },
                    IntrinsicBinaryOperation::Lt => match (left, right) {
                        (LosslessConst::Int(l), LosslessConst::Int(r)) => Some(
                            BlockConstInstruction::ImmBool(l < r).typed(IntrinsicType::Bool.into()),
                        ),
                        (LosslessConst::SInt(l), LosslessConst::SInt(r)) => Some(
                            BlockConstInstruction::ImmBool(l < r).typed(IntrinsicType::Bool.into()),
                        ),
                        (LosslessConst::UInt(l), LosslessConst::UInt(r)) => Some(
                            BlockConstInstruction::ImmBool(l < r).typed(IntrinsicType::Bool.into()),
                        ),
                        _ => None,
                    },
                    IntrinsicBinaryOperation::Gt => match (left, right) {
                        (LosslessConst::Int(l), LosslessConst::Int(r)) => Some(
                            BlockConstInstruction::ImmBool(l > r).typed(IntrinsicType::Bool.into()),
                        ),
                        (LosslessConst::SInt(l), LosslessConst::SInt(r)) => Some(
                            BlockConstInstruction::ImmBool(l > r).typed(IntrinsicType::Bool.into()),
                        ),
                        (LosslessConst::UInt(l), LosslessConst::UInt(r)) => Some(
                            BlockConstInstruction::ImmBool(l > r).typed(IntrinsicType::Bool.into()),
                        ),
                        _ => None,
                    },
                    IntrinsicBinaryOperation::Le => match (left, right) {
                        (LosslessConst::Int(l), LosslessConst::Int(r)) => Some(
                            BlockConstInstruction::ImmBool(l <= r)
                                .typed(IntrinsicType::Bool.into()),
                        ),
                        (LosslessConst::SInt(l), LosslessConst::SInt(r)) => Some(
                            BlockConstInstruction::ImmBool(l <= r)
                                .typed(IntrinsicType::Bool.into()),
                        ),
                        (LosslessConst::UInt(l), LosslessConst::UInt(r)) => Some(
                            BlockConstInstruction::ImmBool(l <= r)
                                .typed(IntrinsicType::Bool.into()),
                        ),
                        _ => None,
                    },
                    IntrinsicBinaryOperation::Ge => match (left, right) {
                        (LosslessConst::Int(l), LosslessConst::Int(r)) => Some(
                            BlockConstInstruction::ImmBool(l >= r)
                                .typed(IntrinsicType::Bool.into()),
                        ),
                        (LosslessConst::SInt(l), LosslessConst::SInt(r)) => Some(
                            BlockConstInstruction::ImmBool(l >= r)
                                .typed(IntrinsicType::Bool.into()),
                        ),
                        (LosslessConst::UInt(l), LosslessConst::UInt(r)) => Some(
                            BlockConstInstruction::ImmBool(l >= r)
                                .typed(IntrinsicType::Bool.into()),
                        ),
                        _ => None,
                    },
                    IntrinsicBinaryOperation::LogAnd => match (left, right) {
                        (LosslessConst::Bool(l), LosslessConst::Bool(r)) => Some(
                            BlockConstInstruction::ImmBool(l && r)
                                .typed(IntrinsicType::Bool.into()),
                        ),
                        (LosslessConst::Int(l), LosslessConst::Int(r)) => Some(
                            BlockConstInstruction::ImmBool(l != 0 && r != 0)
                                .typed(IntrinsicType::Bool.into()),
                        ),
                        (LosslessConst::SInt(l), LosslessConst::SInt(r)) => Some(
                            BlockConstInstruction::ImmBool(l != 0 && r != 0)
                                .typed(IntrinsicType::Bool.into()),
                        ),
                        (LosslessConst::UInt(l), LosslessConst::UInt(r)) => Some(
                            BlockConstInstruction::ImmBool(l != 0 && r != 0)
                                .typed(IntrinsicType::Bool.into()),
                        ),
                        _ => None,
                    },
                    IntrinsicBinaryOperation::LogOr => match (left, right) {
                        (LosslessConst::Bool(l), LosslessConst::Bool(r)) => Some(
                            BlockConstInstruction::ImmBool(l || r)
                                .typed(IntrinsicType::Bool.into()),
                        ),
                        (LosslessConst::Int(l), LosslessConst::Int(r)) => Some(
                            BlockConstInstruction::ImmBool(l != 0 || r != 0)
                                .typed(IntrinsicType::Bool.into()),
                        ),
                        (LosslessConst::SInt(l), LosslessConst::SInt(r)) => Some(
                            BlockConstInstruction::ImmBool(l != 0 || r != 0)
                                .typed(IntrinsicType::Bool.into()),
                        ),
                        (LosslessConst::UInt(l), LosslessConst::UInt(r)) => Some(
                            BlockConstInstruction::ImmBool(l != 0 || r != 0)
                                .typed(IntrinsicType::Bool.into()),
                        ),
                        _ => None,
                    },
                };

                if let Some(c) = reduced {
                    constants.push(c);
                    register_alias_map.insert(
                        RegisterRef::Pure(n),
                        RegisterRef::Const(constants.len() - 1),
                    );
                    pure_register_shifts += 1;
                } else {
                    pure_instructions.push(x);
                    if pure_register_shifts > 0 {
                        // レジスタシフトが発生している状況
                        register_alias_map.insert(
                            RegisterRef::Pure(n),
                            RegisterRef::Pure(n - pure_register_shifts),
                        );
                    }
                }
            }
            _ => {
                pure_instructions.push(x);
                if pure_register_shifts > 0 {
                    // レジスタシフトが発生している状況
                    register_alias_map.insert(
                        RegisterRef::Pure(n),
                        RegisterRef::Pure(n - pure_register_shifts),
                    );
                }
            }
        }
    }

    register_alias_map
}

/// 同じconstant命令を若い番号のregisterにまとめる
pub fn unify_constants<'a, 's>(constants: &Constants<'a, 's>) -> RegisterAliasMap {
    let mut const_to_register_map = HashMap::new();
    let mut register_alias_map = HashMap::new();
    for (r, v) in constants.iter().enumerate() {
        match const_to_register_map.entry(v) {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(r);
            }
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let last_occurence = *e.get();
                if last_occurence > r {
                    // 若い番号を優先する
                    let mapped = e.insert(r);
                    register_alias_map.insert(RegisterRef::Const(mapped), RegisterRef::Const(r));
                } else {
                    register_alias_map
                        .insert(RegisterRef::Const(r), RegisterRef::Const(last_occurence));
                }
            }
        }
    }

    register_alias_map
}

/// 使われていないConstレジスタを削除
pub fn strip_unreferenced_const(
    constants: &mut Constants,
    pure_instructions: &PureInstructions,
    impure_instructions: &ImpureInstructionMap,
    blocks: &[Block],
) -> RegisterAliasMap {
    let mut unreferenced_constants = (0..constants.len()).collect::<HashSet<_>>();

    for x in pure_instructions {
        x.inst.enumerate_ref_registers(|r| {
            if let RegisterRef::Const(n) = r {
                unreferenced_constants.remove(&n);
            }
        });
    }
    for x in impure_instructions.values() {
        x.enumerate_ref_registers(|r| {
            if let RegisterRef::Const(n) = r {
                unreferenced_constants.remove(&n);
            }
        });
    }
    for b in blocks {
        b.flow.enumerate_ref_registers(|r| {
            if let RegisterRef::Const(n) = r {
                unreferenced_constants.remove(&n);
            }
        })
    }

    let mut unreferenced_constants = unreferenced_constants.into_iter().collect::<Vec<_>>();
    unreferenced_constants.sort_by(|a, b| b.cmp(a));
    println!("[StripConstant] Targets: {unreferenced_constants:?}");
    let mut register_alias_map = HashMap::new();
    for n in unreferenced_constants {
        register_alias_map.insert(
            RegisterRef::Const(constants.len() - 1),
            RegisterRef::Const(n),
        );
        println!(
            "[StripConstant] swap remove {} -> {}",
            constants.len() - 1,
            n
        );
        constants.swap_remove(n);
    }

    register_alias_map
}
