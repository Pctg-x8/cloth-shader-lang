use std::collections::HashMap;

use crate::{
    concrete_type::IntrinsicType,
    ir::{
        block::{Block, BlockInstruction, IntrinsicBinaryOperation, RegisterRef},
        Const, ConstFloatLiteral, ConstNumberLiteral, ConstSIntLiteral, ConstUIntLiteral,
    },
};

/// 定数命令を抽出
pub fn extract_constants<'a, 's>(
    instructions: &mut HashMap<RegisterRef, BlockInstruction<'a, 's>>,
) -> HashMap<RegisterRef, BlockInstruction<'a, 's>> {
    let mut const_instructions = HashMap::new();

    for (r, x) in core::mem::replace(instructions, HashMap::with_capacity(instructions.len())) {
        if x.is_const() {
            const_instructions.insert(r, x);
        } else {
            instructions.insert(r, x);
        }
    }

    const_instructions
}

/// 定数のInstantiateIntrinsicTypeClassを演算して定数化する
pub fn promote_instantiate_const<'a, 's>(
    instructions: &mut HashMap<RegisterRef, BlockInstruction<'a, 's>>,
    const_map: &mut HashMap<RegisterRef, BlockInstruction<'a, 's>>,
) -> bool {
    let mut modified = false;

    for (r, x) in core::mem::replace(instructions, HashMap::with_capacity(instructions.len())) {
        let transformed = match x {
            BlockInstruction::InstantiateIntrinsicTypeClass(src, ty) => {
                match const_map.get(&src) {
                    Some(BlockInstruction::ConstInt(l)) => match ty {
                        IntrinsicType::UInt => {
                            const_map.insert(
                                r,
                                BlockInstruction::ConstUInt(ConstUIntLiteral(l.0.clone(), l.1)),
                            );
                            true
                        }
                        IntrinsicType::SInt => {
                            const_map.insert(
                                r,
                                BlockInstruction::ConstSInt(ConstSIntLiteral(l.0.clone(), l.1)),
                            );
                            true
                        }
                        IntrinsicType::Float => {
                            const_map.insert(
                                r,
                                BlockInstruction::ConstFloat(ConstFloatLiteral(l.0.clone(), l.1)),
                            );
                            true
                        }
                        _ => false,
                    },
                    Some(BlockInstruction::ConstNumber(l)) => match ty {
                        IntrinsicType::Float => {
                            const_map.insert(
                                r,
                                BlockInstruction::ConstFloat(ConstFloatLiteral(l.0.clone(), l.1)),
                            );
                            true
                        }
                        IntrinsicType::UInt | IntrinsicType::SInt => {
                            eprintln!("not promoted: number -> int can cause unintended precision dropping");
                            false
                        }
                        _ => false,
                    },
                    Some(&BlockInstruction::ImmInt(v)) => match ty {
                        IntrinsicType::UInt => {
                            const_map.insert(
                                r,
                                BlockInstruction::ImmUInt(v.try_into().expect("cannot promote")),
                            );
                            true
                        }
                        IntrinsicType::SInt => {
                            const_map.insert(
                                r,
                                BlockInstruction::ImmSInt(v.try_into().expect("cannot promote")),
                            );
                            true
                        }
                        _ => false,
                    },
                    _ => false,
                }
            }
            BlockInstruction::PromoteIntToNumber(value) => match const_map.get(&value) {
                Some(BlockInstruction::ConstInt(l)) => {
                    const_map.insert(
                        r,
                        BlockInstruction::ConstNumber(ConstNumberLiteral(l.0.clone(), l.1)),
                    );
                    true
                }
                _ => false,
            },
            _ => false,
        };

        if !transformed {
            // 変換なし
            instructions.insert(r, x);
        }

        modified = modified || transformed;
    }

    modified
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
                    // 左辺が定数じゃない
                    instructions.insert(res, x);
                    continue;
                };
                let Some(right) = const_map
                    .get(&right)
                    .and_then(BlockInstruction::try_instantiate_const)
                else {
                    // 右辺が定数じゃない
                    instructions.insert(res, x);
                    continue;
                };

                match op {
                    IntrinsicBinaryOperation::Add => match (left, right) {
                        (Const::Int(l), Const::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmInt(l + r));
                            modified = true;
                        }
                        (Const::SInt(l), Const::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmSInt(l + r));
                            modified = true;
                        }
                        (Const::UInt(l), Const::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmUInt(l + r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::Sub => match (left, right) {
                        (Const::Int(l), Const::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmInt(l - r));
                            modified = true;
                        }
                        (Const::SInt(l), Const::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmSInt(l - r));
                            modified = true;
                        }
                        (Const::UInt(l), Const::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmUInt(l - r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::Mul => match (left, right) {
                        (Const::Int(l), Const::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmInt(l * r));
                            modified = true;
                        }
                        (Const::SInt(l), Const::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmSInt(l * r));
                            modified = true;
                        }
                        (Const::UInt(l), Const::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmUInt(l * r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::Div => match (left, right) {
                        (Const::Int(l), Const::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmInt(l / r));
                            modified = true;
                        }
                        (Const::SInt(l), Const::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmSInt(l / r));
                            modified = true;
                        }
                        (Const::UInt(l), Const::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmUInt(l / r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::Rem => match (left, right) {
                        (Const::Int(l), Const::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmInt(l % r));
                            modified = true;
                        }
                        (Const::SInt(l), Const::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmSInt(l % r));
                            modified = true;
                        }
                        (Const::UInt(l), Const::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmUInt(l % r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::Pow => match (left, right) {
                        (Const::Int(l), Const::Int(r)) => {
                            const_map.insert(
                                res,
                                BlockInstruction::ImmInt(l.pow(r.try_into().unwrap())),
                            );
                            modified = true;
                        }
                        (Const::SInt(l), Const::SInt(r)) => {
                            const_map.insert(
                                res,
                                BlockInstruction::ImmSInt(l.pow(r.try_into().unwrap())),
                            );
                            modified = true;
                        }
                        (Const::UInt(l), Const::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmUInt(l.pow(r)));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::BitAnd => match (left, right) {
                        (Const::Int(l), Const::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmInt(l & r));
                            modified = true;
                        }
                        (Const::SInt(l), Const::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmSInt(l & r));
                            modified = true;
                        }
                        (Const::UInt(l), Const::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmUInt(l & r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::BitOr => match (left, right) {
                        (Const::Int(l), Const::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmInt(l | r));
                            modified = true;
                        }
                        (Const::SInt(l), Const::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmSInt(l | r));
                            modified = true;
                        }
                        (Const::UInt(l), Const::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmUInt(l | r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::BitXor => match (left, right) {
                        (Const::Int(l), Const::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmInt(l ^ r));
                            modified = true;
                        }
                        (Const::SInt(l), Const::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmSInt(l ^ r));
                            modified = true;
                        }
                        (Const::UInt(l), Const::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmUInt(l ^ r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::LeftShift => match (left, right) {
                        (Const::Int(l), Const::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmInt(l << r));
                            modified = true;
                        }
                        (Const::SInt(l), Const::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmSInt(l << r));
                            modified = true;
                        }
                        (Const::UInt(l), Const::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmUInt(l << r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::RightShift => match (left, right) {
                        (Const::Int(l), Const::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmInt(l >> r));
                            modified = true;
                        }
                        (Const::SInt(l), Const::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmSInt(l >> r));
                            modified = true;
                        }
                        (Const::UInt(l), Const::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmUInt(l >> r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::Eq => match (left, right) {
                        (Const::Int(l), Const::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l == r));
                            modified = true;
                        }
                        (Const::SInt(l), Const::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l == r));
                            modified = true;
                        }
                        (Const::UInt(l), Const::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l == r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::Ne => match (left, right) {
                        (Const::Int(l), Const::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l != r));
                            modified = true;
                        }
                        (Const::SInt(l), Const::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l != r));
                            modified = true;
                        }
                        (Const::UInt(l), Const::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l != r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::Lt => match (left, right) {
                        (Const::Int(l), Const::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l < r));
                            modified = true;
                        }
                        (Const::SInt(l), Const::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l < r));
                            modified = true;
                        }
                        (Const::UInt(l), Const::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l < r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::Gt => match (left, right) {
                        (Const::Int(l), Const::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l > r));
                            modified = true;
                        }
                        (Const::SInt(l), Const::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l > r));
                            modified = true;
                        }
                        (Const::UInt(l), Const::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l > r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::Le => match (left, right) {
                        (Const::Int(l), Const::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l <= r));
                            modified = true;
                        }
                        (Const::SInt(l), Const::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l <= r));
                            modified = true;
                        }
                        (Const::UInt(l), Const::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l <= r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::Ge => match (left, right) {
                        (Const::Int(l), Const::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l >= r));
                            modified = true;
                        }
                        (Const::SInt(l), Const::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l >= r));
                            modified = true;
                        }
                        (Const::UInt(l), Const::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l >= r));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::LogAnd => match (left, right) {
                        (Const::Bool(l), Const::Bool(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l && r));
                            modified = true;
                        }
                        (Const::Int(l), Const::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l != 0 && r != 0));
                            modified = true;
                        }
                        (Const::SInt(l), Const::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l != 0 && r != 0));
                            modified = true;
                        }
                        (Const::UInt(l), Const::UInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l != 0 && r != 0));
                            modified = true;
                        }
                        _ => {
                            instructions.insert(res, x);
                        }
                    },
                    IntrinsicBinaryOperation::LogOr => match (left, right) {
                        (Const::Bool(l), Const::Bool(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l || r));
                            modified = true;
                        }
                        (Const::Int(l), Const::Int(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l != 0 || r != 0));
                            modified = true;
                        }
                        (Const::SInt(l), Const::SInt(r)) => {
                            const_map.insert(res, BlockInstruction::ImmBool(l != 0 || r != 0));
                            modified = true;
                        }
                        (Const::UInt(l), Const::UInt(r)) => {
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

/// 同じconstant命令を若い番号のregisterにまとめる
pub fn unify_constants<'a, 's>(
    blocks: &mut [Block],
    mod_instructions: &mut HashMap<RegisterRef, BlockInstruction<'a, 's>>,
    const_map: &HashMap<RegisterRef, BlockInstruction<'a, 's>>,
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
                    register_rewrite_map.insert(*mapped, *r);
                } else {
                    register_rewrite_map.insert(*r, **e.get());
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

    let mut modified = false;
    for x in mod_instructions.values_mut() {
        let m1 = x.apply_register_alias(&register_rewrite_map);

        modified = modified || m1;
    }
    for b in blocks.iter_mut() {
        let m1 = b.apply_flow_register_alias(&register_rewrite_map);

        modified = modified || m1;
    }

    modified
}
