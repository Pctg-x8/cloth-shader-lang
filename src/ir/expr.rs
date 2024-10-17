use std::collections::HashSet;

use typed_arena::Arena;

use crate::{
    concrete_type::{
        BinaryOpTermConversion, BinaryOpTypeConversion, ConcreteType, IntrinsicScalarType,
        IntrinsicType, UserDefinedType,
    },
    parser::ExpressionNode,
    ref_path::RefPath,
    scope::{SymbolScope, VarId, VarLookupResult},
    source_ref::{SourceRef, SourceRefSliceEq},
    symbol::IntrinsicFunctionSymbol,
    utils::{swizzle_indices, PtrEq},
};

use super::{
    block::{
        transform_statement, Block, BlockFlowInstruction, BlockGenerationContext,
        BlockInstructionEmissionContext, BlockInstructionEmitter, BlockRef,
        IntrinsicBinaryOperation, IntrinsicUnaryOperation, RegisterRef,
    },
    ExprRef,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScopeCaptureSource {
    Expr(ExprRef),
    Capture(usize),
}

pub fn simplify_lefthand_expression<'a, 's>(
    expr: ExpressionNode<'s>,
    block_ctx: &mut BlockGenerationContext<'a, 's>,
    inst_ctx: &mut BlockInstructionEmissionContext<'a, 's>,
    scope: &'a SymbolScope<'a, 's>,
) -> SimplifyResult {
    match expr {
        ExpressionNode::Var(name) => {
            let Some((scope, v)) = scope.lookup(name.slice) else {
                panic!(
                    "Error: referencing undefined symbol '{}' {name:?}",
                    name.slice
                );
            };

            let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
            let result = match v {
                VarLookupResult::IntrinsicFunctions(_) => {
                    panic!("Error: cannot get pointer of IntrinsicFunctions");
                }
                VarLookupResult::IntrinsicTypeConstructor(_) => {
                    panic!("Error: cannot get pointer of IntrinsicTypeConstructor")
                }
                VarLookupResult::ScopeLocalVar(vid, _, false) => {
                    inst.scope_local_var_ref(scope, vid)
                }
                VarLookupResult::ScopeLocalVar(vid, _, true) => {
                    inst.scope_local_var_mutable_ref(scope, vid)
                }
                VarLookupResult::FunctionInputVar(vid, _, false) => {
                    inst.function_input_var_ref(scope, vid)
                }
                VarLookupResult::FunctionInputVar(vid, _, true) => {
                    inst.function_input_var_mutable_ref(scope, vid)
                }
                VarLookupResult::UserDefinedFunction(_) => {
                    panic!("Error: cannot store any value to UserDefinedFunction");
                }
            };

            let perform_block = inst.create_block(BlockFlowInstruction::Undetermined);
            SimplifyResult {
                result,
                start_block: perform_block,
                end_block: perform_block,
            }
        }
        ExpressionNode::MemberRef(base, _, name) => {
            let base = simplify_expression(*base, block_ctx, inst_ctx, scope);

            let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
            let result = match *base.result.ty(inst.instruction_emission_context) {
                ConcreteType::Ref(ref inner) => match &**inner {
                    ConcreteType::Intrinsic(x) => match (x.scalar_type(), x.vector_elements()) {
                        (None, _) | (_, None) => panic!("cannot member ref to complex data"),
                        (_, Some(1)) => panic!("scalar value cannot be swizzled"),
                        (Some(_), Some(count)) => match swizzle_indices(name.slice, count) {
                            Some([Some(a), None, None, None]) => {
                                inst.swizzle_ref(base.result, vec![a])
                            }
                            Some([Some(a), Some(b), None, None]) => {
                                inst.swizzle_ref(base.result, vec![a, b])
                            }
                            Some([Some(a), Some(b), Some(c), None]) => {
                                inst.swizzle_ref(base.result, vec![a, b, c])
                            }
                            Some([Some(a), Some(b), Some(c), Some(d)]) => {
                                inst.swizzle_ref(base.result, vec![a, b, c, d])
                            }
                            Some(_) => panic!("invalid swizzle ref"),
                            None => panic!("too many swizzle components"),
                        },
                    },
                    ConcreteType::Struct(members) => {
                        let target_member = members.iter().find(|x| x.name.0.slice == name.slice);

                        match target_member {
                            Some(x) => {
                                if x.mutable {
                                    inst.member_mutable_ref(
                                        base.result,
                                        SourceRef::from(&name),
                                        x.ty.clone(),
                                    )
                                } else {
                                    inst.member_ref(
                                        base.result,
                                        SourceRef::from(&name),
                                        x.ty.clone(),
                                    )
                                }
                            }
                            None => {
                                panic!("Struct has no member named '{}'", name.slice);
                            }
                        }
                    }
                    ref ty => {
                        panic!("unsupported member ref op for type {ty:?}");
                    }
                },
                ConcreteType::MutableRef(ref inner) => match &**inner {
                    ConcreteType::Intrinsic(x) => match (x.scalar_type(), x.vector_elements()) {
                        (None, _) | (_, None) => panic!("cannot member ref to complex data"),
                        (_, Some(1)) => panic!("scalar value cannot be swizzled"),
                        (Some(_), Some(count)) => match swizzle_indices(name.slice, count) {
                            Some([Some(a), None, None, None]) => {
                                inst.swizzle_mutable_ref(base.result, vec![a])
                            }
                            Some([Some(a), Some(b), None, None]) => {
                                inst.swizzle_mutable_ref(base.result, vec![a, b])
                            }
                            Some([Some(a), Some(b), Some(c), None]) => {
                                inst.swizzle_mutable_ref(base.result, vec![a, b, c])
                            }
                            Some([Some(a), Some(b), Some(c), Some(d)]) => {
                                inst.swizzle_mutable_ref(base.result, vec![a, b, c, d])
                            }
                            Some(_) => panic!("invalid swizzle ref"),
                            None => panic!("too many swizzle components"),
                        },
                    },
                    ConcreteType::Struct(members) => {
                        let target_member = members.iter().find(|x| x.name.0.slice == name.slice);

                        match target_member {
                            Some(x) => inst.member_mutable_ref(
                                base.result,
                                SourceRef::from(&name),
                                x.ty.clone(),
                            ),
                            None => {
                                panic!("Struct has no member named '{}'", name.slice);
                            }
                        }
                    }
                    ty => {
                        panic!("unsupported member ref op for type {ty:?}");
                    }
                },
                ref ty => {
                    panic!("unable to store to rhs value {ty:?}");
                }
            };

            let perform_block = inst.create_block(BlockFlowInstruction::Undetermined);
            assert!(
                block_ctx.try_chain(base.end_block, perform_block),
                "base multiple out?"
            );

            SimplifyResult {
                result,
                start_block: base.start_block,
                end_block: perform_block,
            }
        }
        ExpressionNode::ArrayIndex(base, _, ix, _) => {
            let base = simplify_expression(*base, block_ctx, inst_ctx, scope);
            let ix = simplify_expression(*ix, block_ctx, inst_ctx, scope);

            let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
            let ix_value = inst.loaded(ix.result);
            let result = match *base.result.ty(inst.instruction_emission_context) {
                ConcreteType::Ref(ref inner) => match &**inner {
                    ConcreteType::Array(elm, _) => {
                        inst.array_ref(base.result, ix_value, *elm.clone())
                    }
                    _ => panic!("Error: cannot indexing this type: {inner:?}"),
                },
                ConcreteType::MutableRef(ref inner) => match &**inner {
                    ConcreteType::Array(elm, _) => {
                        inst.array_mutable_ref(base.result, ix_value, *elm.clone())
                    }
                    _ => panic!("Error: cannot indexing this type: {inner:?}"),
                },
                ConcreteType::Array(_, _) => {
                    unimplemented!("rhs value direct indexing")
                }
                ref ty => panic!("Error: cannot indexing this type: {ty:?}"),
            };

            let perform_block = inst.create_block(BlockFlowInstruction::Undetermined);
            assert!(
                block_ctx.try_chain(base.end_block, ix.start_block),
                "base multiple out?"
            );
            assert!(
                block_ctx.try_chain(ix.end_block, perform_block),
                "index multiple out?"
            );

            SimplifyResult {
                result,
                start_block: base.start_block,
                end_block: perform_block,
            }
        }
        _ => panic!("Error: invalid lefthand expression: {expr:?}"),
    }
}

pub struct SimplifyResult {
    pub result: RegisterRef,
    pub start_block: BlockRef,
    pub end_block: BlockRef,
}

pub fn simplify_expression<'a, 's>(
    ast: ExpressionNode<'s>,
    block_ctx: &mut BlockGenerationContext<'a, 's>,
    inst_ctx: &mut BlockInstructionEmissionContext<'a, 's>,
    symbol_scope: &'a SymbolScope<'a, 's>,
) -> SimplifyResult {
    match ast {
        ExpressionNode::Binary(left, op, right) => {
            let left = simplify_expression(*left, block_ctx, inst_ctx, symbol_scope);
            let right = simplify_expression(*right, block_ctx, inst_ctx, symbol_scope);

            assert!(
                block_ctx.try_chain(left.end_block, right.start_block),
                "left multi branch?"
            );

            if op.slice.starts_with('`') && op.slice.ends_with('`') {
                // infix funcall

                let mut function_load_inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
                let f = emit_varref(
                    SourceRef {
                        slice: &op.slice[1..op.slice.len() - 1],
                        line: op.line,
                        col: op.col,
                    },
                    &mut function_load_inst,
                    symbol_scope,
                );
                let f = match *f.ty(function_load_inst.instruction_emission_context) {
                    ConcreteType::Ref(_) | ConcreteType::MutableRef(_) => {
                        function_load_inst.load_ref(f)
                    }
                    _ => f,
                };

                let function_load_block =
                    function_load_inst.create_block(BlockFlowInstruction::Undetermined);

                let (result, perform_block) =
                    funcall(f, vec![left.result, right.result], block_ctx, inst_ctx);
                assert!(
                    block_ctx.try_chain(right.end_block, function_load_block),
                    "right multi branch?"
                );
                assert!(block_ctx.try_chain(function_load_block, perform_block));

                return SimplifyResult {
                    result,
                    start_block: left.start_block,
                    end_block: perform_block,
                };
            }

            let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
            let left_value = inst.loaded(left.result);
            let right_value = inst.loaded(right.result);
            let result = binary_op(left_value, SourceRef::from(&op), right_value, &mut inst);
            let perform_block = inst.create_block(BlockFlowInstruction::Undetermined);
            assert!(
                block_ctx.try_chain(right.end_block, perform_block),
                "right multi branch?"
            );

            SimplifyResult {
                result,
                start_block: left.start_block,
                end_block: perform_block,
            }
        }
        ExpressionNode::Prefixed(op, expr) => {
            let expr = simplify_expression(*expr, block_ctx, inst_ctx, symbol_scope);

            let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
            let expr_value = inst.loaded(expr.result);

            let result = match op.slice {
                "+" if expr_value.ty(&inst).is_scalar_type() => expr_value,
                "-" => match expr_value.ty(&inst).scalar_type() {
                    Some(IntrinsicScalarType::Bool) | Some(IntrinsicScalarType::UInt) => {
                        let target_type: ConcreteType = IntrinsicScalarType::SInt
                            .of_vector(expr_value.ty(&inst).vector_elements().unwrap())
                            .unwrap()
                            .into();
                        let expr = inst.cast(expr_value, target_type.clone());

                        inst.intrinsic_unary_op(expr, IntrinsicUnaryOperation::Neg, target_type)
                    }
                    Some(IntrinsicScalarType::UnknownIntClass) => {
                        let expr =
                            inst.instantiate_intrinsic_type_class(expr_value, IntrinsicType::SInt);

                        inst.intrinsic_unary_op(
                            expr,
                            IntrinsicUnaryOperation::Neg,
                            IntrinsicType::SInt.into(),
                        )
                    }
                    Some(IntrinsicScalarType::UnknownNumberClass) => {
                        let expr =
                            inst.instantiate_intrinsic_type_class(expr_value, IntrinsicType::Float);

                        inst.intrinsic_unary_op(
                            expr,
                            IntrinsicUnaryOperation::Neg,
                            IntrinsicType::Float.into(),
                        )
                    }
                    Some(_) => inst.intrinsic_unary_op(
                        expr_value,
                        IntrinsicUnaryOperation::Neg,
                        expr_value.ty(&inst).clone(),
                    ),
                    None => panic!("Error: cannot apply prefixed - to the term"),
                },
                "!" => match expr_value.ty(&inst).scalar_type() {
                    Some(IntrinsicScalarType::SInt)
                    | Some(IntrinsicScalarType::UInt)
                    | Some(IntrinsicScalarType::Float) => {
                        let target_type: ConcreteType = IntrinsicScalarType::Bool
                            .of_vector(expr_value.ty(&inst).vector_elements().unwrap())
                            .unwrap()
                            .into();
                        let expr = inst.cast(expr_value, target_type.clone());

                        inst.intrinsic_unary_op(
                            expr,
                            IntrinsicUnaryOperation::LogNot,
                            target_type.clone(),
                        )
                    }
                    Some(IntrinsicScalarType::UnknownIntClass) => {
                        let expr =
                            inst.instantiate_intrinsic_type_class(expr_value, IntrinsicType::UInt);
                        let expr = inst.cast(expr, IntrinsicType::Bool.into());

                        inst.intrinsic_unary_op(
                            expr,
                            IntrinsicUnaryOperation::LogNot,
                            IntrinsicType::Bool.into(),
                        )
                    }
                    Some(IntrinsicScalarType::UnknownNumberClass) => {
                        let expr =
                            inst.instantiate_intrinsic_type_class(expr_value, IntrinsicType::Float);
                        let expr = inst.cast(expr, IntrinsicType::Bool.into());

                        inst.intrinsic_unary_op(
                            expr,
                            IntrinsicUnaryOperation::LogNot,
                            IntrinsicType::Bool.into(),
                        )
                    }
                    Some(_) => inst.intrinsic_unary_op(
                        expr_value,
                        IntrinsicUnaryOperation::LogNot,
                        expr_value.ty(&inst).clone(),
                    ),
                    None => panic!("Error: cannot apply prefixed ! to the term"),
                },
                "~" => match expr_value.ty(&inst).scalar_type() {
                    Some(IntrinsicScalarType::Bool) | Some(IntrinsicScalarType::SInt) => {
                        let target_type: ConcreteType = IntrinsicScalarType::UInt
                            .of_vector(expr_value.ty(&inst).vector_elements().unwrap())
                            .unwrap()
                            .into();
                        let expr = inst.cast(expr_value, target_type.clone());

                        inst.intrinsic_unary_op(expr, IntrinsicUnaryOperation::BitNot, target_type)
                    }
                    Some(IntrinsicScalarType::UnknownIntClass) => {
                        let expr =
                            inst.instantiate_intrinsic_type_class(expr_value, IntrinsicType::UInt);

                        inst.intrinsic_unary_op(
                            expr,
                            IntrinsicUnaryOperation::BitNot,
                            IntrinsicType::UInt.into(),
                        )
                    }
                    Some(IntrinsicScalarType::UInt) => inst.intrinsic_unary_op(
                        expr_value,
                        IntrinsicUnaryOperation::BitNot,
                        expr_value.ty(&inst).clone(),
                    ),
                    _ => panic!("Error: cannot apply prefixed ~ to the term"),
                },
                x => panic!("Error: cannot apply prefixed op {x} to the term"),
            };

            let perform_block = inst.create_block(BlockFlowInstruction::Undetermined);
            assert!(
                block_ctx.try_chain(expr.end_block, perform_block),
                "expr multiple out?"
            );

            SimplifyResult {
                result,
                start_block: expr.start_block,
                end_block: perform_block,
            }
        }
        ExpressionNode::Lifted(_, x, _) => {
            simplify_expression(*x, block_ctx, inst_ctx, symbol_scope)
        }
        ExpressionNode::Blocked(stmts, x) => {
            let new_symbol_scope = block_ctx.symbol_scope_arena.alloc(symbol_scope.new_child());
            let mut first_block = None;
            let mut last_block = None;

            for s in stmts {
                let (start_block, end_block) =
                    transform_statement(s, new_symbol_scope, block_ctx, inst_ctx);
                if first_block.is_none() {
                    first_block = Some(start_block);
                }
                if let Some(b) = last_block {
                    assert!(block_ctx.try_chain(b, start_block), "multi out?");
                }

                last_block = Some(end_block);
            }

            match x {
                Some(x) => {
                    let res = simplify_expression(*x, block_ctx, inst_ctx, new_symbol_scope);
                    if let Some(b) = last_block {
                        assert!(block_ctx.try_chain(b, res.start_block), "multi out?");
                    }

                    SimplifyResult {
                        result: res.result,
                        start_block: first_block.unwrap_or(res.start_block),
                        end_block: res.end_block,
                    }
                }
                None => {
                    // return unit
                    let result = inst_ctx.const_unit();
                    let block = block_ctx.add(Block::flow_only(BlockFlowInstruction::Undetermined));
                    if let Some(b) = last_block {
                        assert!(block_ctx.try_chain(b, block), "multi out?");
                    }

                    SimplifyResult {
                        result,
                        start_block: first_block.unwrap_or(block),
                        end_block: block,
                    }
                }
            }
        }
        ExpressionNode::MemberRef(base, _, name) => {
            let base = simplify_expression(*base, block_ctx, inst_ctx, symbol_scope);

            let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
            let result = match *base.result.ty(&inst) {
                ConcreteType::Intrinsic(x) => match (x.scalar_type(), x.vector_elements()) {
                    (None, _) | (_, None) => panic!("cannot member ref to complex data"),
                    (_, Some(1)) => panic!("scalar value cannot be swizzled"),
                    (Some(_), Some(count)) => match swizzle_indices(name.slice, count) {
                        Some([Some(a), None, None, None]) => inst.swizzle(base.result, vec![a]),
                        Some([Some(a), Some(b), None, None]) => {
                            inst.swizzle(base.result, vec![a, b])
                        }
                        Some([Some(a), Some(b), Some(c), None]) => {
                            inst.swizzle(base.result, vec![a, b, c])
                        }
                        Some([Some(a), Some(b), Some(c), Some(d)]) => {
                            inst.swizzle(base.result, vec![a, b, c, d])
                        }
                        Some(_) => panic!("invalid swizzle ref"),
                        None => panic!("too many swizzle components"),
                    },
                },
                ConcreteType::UserDefined {
                    name: ty_name,
                    ref generic_args,
                } => {
                    let (_, (_, ty)) = symbol_scope
                        .lookup_user_defined_type(ty_name)
                        .expect("No user defined type defined");

                    match ty {
                        UserDefinedType::Struct(members) => {
                            let target_member =
                                members.iter().find(|x| x.name.0.slice == name.slice);
                            match target_member {
                                Some(x) => {
                                    if x.mutable {
                                        inst.member_mutable_ref(
                                            base.result,
                                            SourceRef::from(&name),
                                            x.ty.clone(),
                                        )
                                    } else {
                                        inst.member_ref(
                                            base.result,
                                            SourceRef::from(&name),
                                            x.ty.clone(),
                                        )
                                    }
                                }
                                None => {
                                    panic!("Struct {ty_name} has no member named '{}'", name.slice);
                                }
                            }
                        }
                    }
                }
                ConcreteType::Ref(ref inner) => match &**inner {
                    ConcreteType::Intrinsic(x) => match (x.scalar_type(), x.vector_elements()) {
                        (None, _) | (_, None) => panic!("cannot member ref to complex data"),
                        (_, Some(1)) => panic!("scalar value cannot be swizzled"),
                        (Some(_), Some(count)) => match swizzle_indices(name.slice, count) {
                            Some([Some(a), None, None, None]) => {
                                let base = inst.load_ref(base.result);

                                inst.swizzle(base, vec![a])
                            }
                            Some([Some(a), Some(b), None, None]) => {
                                let base = inst.load_ref(base.result);

                                inst.swizzle(base, vec![a, b])
                            }
                            Some([Some(a), Some(b), Some(c), None]) => {
                                let base = inst.load_ref(base.result);

                                inst.swizzle(base, vec![a, b, c])
                            }
                            Some([Some(a), Some(b), Some(c), Some(d)]) => {
                                let base = inst.load_ref(base.result);

                                inst.swizzle(base, vec![a, b, c, d])
                            }
                            Some(_) => panic!("invalid swizzle ref"),
                            None => panic!("too many swizzle components"),
                        },
                    },
                    ConcreteType::Struct(members) => {
                        let target_member = members.iter().find(|x| x.name.0.slice == name.slice);

                        match target_member {
                            Some(x) => {
                                if x.mutable {
                                    inst.member_mutable_ref(
                                        base.result,
                                        SourceRef::from(&name),
                                        x.ty.clone(),
                                    )
                                } else {
                                    inst.member_ref(
                                        base.result,
                                        SourceRef::from(&name),
                                        x.ty.clone(),
                                    )
                                }
                            }
                            None => {
                                panic!("Struct has no member named '{}'", name.slice);
                            }
                        }
                    }
                    ty => {
                        panic!("Error: unsupported member ref op for type {ty:?}");
                    }
                },
                ConcreteType::MutableRef(ref inner) => match &**inner {
                    ConcreteType::Intrinsic(x) => match (x.scalar_type(), x.vector_elements()) {
                        (None, _) | (_, None) => panic!("cannot member ref to complex data"),
                        (_, Some(1)) => panic!("scalar value cannot be swizzled"),
                        (Some(_), Some(count)) => match swizzle_indices(name.slice, count) {
                            Some([Some(a), None, None, None]) => {
                                let base = inst.load_ref(base.result);

                                inst.swizzle(base, vec![a])
                            }
                            Some([Some(a), Some(b), None, None]) => {
                                let base = inst.load_ref(base.result);

                                inst.swizzle(base, vec![a, b])
                            }
                            Some([Some(a), Some(b), Some(c), None]) => {
                                let base = inst.load_ref(base.result);

                                inst.swizzle(base, vec![a, b, c])
                            }
                            Some([Some(a), Some(b), Some(c), Some(d)]) => {
                                let base = inst.load_ref(base.result);

                                inst.swizzle(base, vec![a, b, c, d])
                            }
                            Some(_) => panic!("invalid swizzle ref"),
                            None => panic!("too many swizzle components"),
                        },
                    },
                    ConcreteType::Struct(members) => {
                        let target_member = members.iter().find(|x| x.name.0.slice == name.slice);

                        match target_member {
                            Some(x) => inst.member_mutable_ref(
                                base.result,
                                SourceRef::from(&name),
                                x.ty.clone(),
                            ),
                            None => {
                                panic!("Struct has no member named '{}'", name.slice);
                            }
                        }
                    }
                    ty => {
                        panic!("Error: unsupported member ref op for type {ty:?}")
                    }
                },
                ref ty => {
                    panic!("Error: unsupported member ref op for type {ty:?}");
                }
            };

            let perform_blk = inst.create_block(BlockFlowInstruction::Undetermined);
            assert!(
                block_ctx.try_chain(base.end_block, perform_blk),
                "base mutliple out?"
            );

            SimplifyResult {
                result,
                start_block: base.start_block,
                end_block: perform_blk,
            }
        }
        ExpressionNode::ArrayIndex(base, _, index, _) => {
            let base = simplify_expression(*base, block_ctx, inst_ctx, symbol_scope);
            let index = simplify_expression(*index, block_ctx, inst_ctx, symbol_scope);

            let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
            let index_value = inst.loaded(index.result);
            let result = match *base.result.ty(&inst) {
                ConcreteType::Ref(ref inner) => match &**inner {
                    ConcreteType::Array(elm, _) => {
                        inst.array_ref(base.result, index_value, *elm.clone())
                    }
                    _ => panic!("Error: cannot indexing this type: {inner:?}"),
                },
                ConcreteType::MutableRef(ref inner) => match &**inner {
                    ConcreteType::Array(elm, _) => {
                        inst.array_mutable_ref(base.result, index_value, *elm.clone())
                    }
                    _ => panic!("Error: cannot indexing this type: {inner:?}"),
                },
                ConcreteType::Array(_, _) => {
                    unimplemented!("rhs value direct indexing")
                }
                ref ty => panic!("Error: cannot indexing this type: {ty:?}"),
            };

            let perform_block = inst.create_block(BlockFlowInstruction::Undetermined);
            assert!(
                block_ctx.try_chain(base.end_block, index.start_block),
                "base multiple out?"
            );
            assert!(
                block_ctx.try_chain(index.end_block, perform_block),
                "index multiple out?"
            );

            SimplifyResult {
                result,
                start_block: base.start_block,
                end_block: perform_block,
            }
        }
        ExpressionNode::Funcall {
            base_expr, args, ..
        } => {
            let base = simplify_expression(*base_expr, block_ctx, inst_ctx, symbol_scope);
            let (args, arg_block_range): (Vec<_>, Vec<_>) = args
                .into_iter()
                .map(|(x, _)| simplify_expression(x, block_ctx, inst_ctx, symbol_scope))
                .map(|r| (r.result, (r.start_block, r.end_block)))
                .unzip();

            let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
            let base_value = inst.loaded(base.result);
            let base_load_block = inst.create_block(BlockFlowInstruction::Undetermined);

            let (result, perform_block) = funcall(base_value, args, block_ctx, inst_ctx);
            if !arg_block_range.is_empty() {
                let mut before_block = base.end_block;
                for &(start, end) in &arg_block_range {
                    assert!(block_ctx.try_chain(before_block, start), "multiple out?");
                    before_block = end;
                }
                assert!(
                    block_ctx.try_chain(before_block, base_load_block),
                    "multiple out?"
                );
                assert!(block_ctx.try_chain(base_load_block, perform_block));
            } else {
                assert!(
                    block_ctx.try_chain(base.end_block, base_load_block),
                    "multiple out?"
                );
                assert!(block_ctx.try_chain(base_load_block, perform_block));
            }

            SimplifyResult {
                result,
                start_block: base.start_block,
                end_block: perform_block,
            }
        }
        ExpressionNode::FuncallSingle(base_expr, arg) => {
            let base = simplify_expression(*base_expr, block_ctx, inst_ctx, symbol_scope);
            let arg = simplify_expression(*arg, block_ctx, inst_ctx, symbol_scope);

            let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
            let base_value = inst.loaded(base.result);
            let base_load_block = inst.create_block(BlockFlowInstruction::Undetermined);

            let (result, perform_block) =
                funcall(base_value, vec![arg.result], block_ctx, inst_ctx);
            assert!(
                block_ctx.try_chain(base.end_block, arg.start_block),
                "multiple out?"
            );
            assert!(
                block_ctx.try_chain(arg.end_block, base_load_block),
                "multiple out?"
            );
            assert!(block_ctx.try_chain(base_load_block, perform_block));

            SimplifyResult {
                result,
                start_block: base.start_block,
                end_block: perform_block,
            }
        }
        ExpressionNode::Number(t) => {
            let has_hex_prefix = t.slice.starts_with("0x") || t.slice.starts_with("0X");
            let has_float_suffix = t.slice.ends_with(['f', 'F']);
            let has_fpart = t.slice.contains('.');

            let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
            let res = if has_hex_prefix {
                inst.const_int(SourceRef::from(&t))
            } else if has_float_suffix {
                inst.const_float(SourceRef::from(&t))
            } else if has_fpart {
                inst.const_number(SourceRef::from(&t))
            } else {
                inst.const_int(SourceRef::from(&t))
            };

            let perform_block = inst.create_block(BlockFlowInstruction::Undetermined);
            SimplifyResult {
                result: res,
                start_block: perform_block,
                end_block: perform_block,
            }
        }
        ExpressionNode::Var(x) => {
            let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);

            let res = emit_varref(SourceRef::from(&x), &mut inst, symbol_scope);

            let perform_block = inst.create_block(BlockFlowInstruction::Undetermined);

            SimplifyResult {
                result: res,
                start_block: perform_block,
                end_block: perform_block,
            }
        }
        ExpressionNode::Tuple(_, xs, _) => {
            let (xs, eval_block_ranges): (Vec<_>, Vec<_>) = xs
                .into_iter()
                .map(|(x, _)| simplify_expression(x, block_ctx, inst_ctx, symbol_scope))
                .map(|r| (r.result, (r.start_block, r.end_block)))
                .unzip();

            let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
            let res = inst.construct_tuple(xs);
            let perform_block = inst.create_block(BlockFlowInstruction::Undetermined);

            let start_block;
            if let Some(&(start, mut last_block)) = eval_block_ranges.first() {
                start_block = start;

                for &(start, end) in &eval_block_ranges[1..] {
                    assert!(block_ctx.try_chain(last_block, start), "multiple out?");
                    last_block = end;
                }
                assert!(
                    block_ctx.try_chain(last_block, perform_block),
                    "multiple out?"
                );
            } else {
                start_block = perform_block;
            }

            SimplifyResult {
                result: res,
                start_block,
                end_block: perform_block,
            }
        }
        ExpressionNode::StructValue {
            ty,
            mut initializers,
            ..
        } => {
            let ty = ConcreteType::build(symbol_scope, &HashSet::new(), *ty.clone())
                .instantiate(symbol_scope);
            let ConcreteType::Struct(ref members) = ty else {
                panic!("Error: cannot construct a structure of this type");
            };

            let (initializers, initializer_eval_block_ranges): (Vec<_>, Vec<_>) = members
                .iter()
                .map(|m| {
                    let initializer_pos = initializers
                        .iter()
                        .position(|i| i.0.slice == m.name.0.slice)
                        .expect("initializers have extra member");
                    let initializer = initializers.remove(initializer_pos);
                    let v = simplify_expression(initializer.2, block_ctx, inst_ctx, symbol_scope);
                    if *v.result.ty(inst_ctx) != m.ty {
                        panic!("initializer value type does not match with member type");
                    }

                    (v.result, (v.start_block, v.end_block))
                })
                .unzip();

            let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
            let res = inst.construct_struct(initializers, ty);
            let perform_block = inst.create_block(BlockFlowInstruction::Undetermined);

            let start_block;
            if let Some(&(start, mut last_block)) = initializer_eval_block_ranges.first() {
                start_block = start;

                for &(start, end) in &initializer_eval_block_ranges[1..] {
                    assert!(block_ctx.try_chain(last_block, start), "multiple out?");
                    last_block = end;
                }
                assert!(
                    block_ctx.try_chain(last_block, perform_block),
                    "multiple out?"
                );
            } else {
                start_block = perform_block;
            }

            SimplifyResult {
                result: res,
                start_block,
                end_block: perform_block,
            }
        }
        ExpressionNode::If {
            condition,
            then_expr,
            else_expr,
            ..
        } => {
            let condition = simplify_expression(*condition, block_ctx, inst_ctx, symbol_scope);
            let then_expr = simplify_expression(*then_expr, block_ctx, inst_ctx, symbol_scope);
            let else_expr = match else_expr {
                None => {
                    let r = inst_ctx.const_unit();
                    let perform_block =
                        block_ctx.add(Block::flow_only(BlockFlowInstruction::Undetermined));

                    SimplifyResult {
                        result: r,
                        start_block: perform_block,
                        end_block: perform_block,
                    }
                }
                Some(x) => simplify_expression(*x, block_ctx, inst_ctx, symbol_scope),
            };

            let condition = match *condition.result.ty(inst_ctx) {
                ConcreteType::Intrinsic(IntrinsicType::Bool) => condition,
                _ => {
                    let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
                    let result = inst.cast(condition.result, IntrinsicType::Bool.into());
                    let cast_block = inst.create_block(BlockFlowInstruction::Undetermined);
                    assert!(
                        block_ctx.try_chain(condition.end_block, cast_block),
                        "condition multiple out?"
                    );

                    SimplifyResult {
                        result,
                        start_block: condition.start_block,
                        end_block: cast_block,
                    }
                }
            };

            let (res_ty, then, r#else) = match (
                &*then_expr.result.ty(inst_ctx),
                &*else_expr.result.ty(inst_ctx),
            ) {
                (a, b) if a == b => (a.clone(), then_expr, else_expr),
                (ConcreteType::Intrinsic(IntrinsicType::Unit), _) => {
                    (IntrinsicType::Unit.into(), then_expr, else_expr)
                }
                (_, ConcreteType::Intrinsic(IntrinsicType::Unit)) => {
                    (IntrinsicType::Unit.into(), then_expr, else_expr)
                }
                // TODO: 他の変換は必要になったら書く
                (ConcreteType::UnknownIntClass, ConcreteType::Intrinsic(IntrinsicType::SInt)) => {
                    let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
                    let new_then = inst
                        .instantiate_intrinsic_type_class(then_expr.result, IntrinsicType::SInt);
                    let new_then_blk = inst.create_block(BlockFlowInstruction::Undetermined);
                    assert!(
                        block_ctx.try_chain(then_expr.end_block, new_then_blk),
                        "then multiple out?"
                    );

                    (
                        IntrinsicType::SInt.into(),
                        SimplifyResult {
                            result: new_then,
                            start_block: then_expr.start_block,
                            end_block: new_then_blk,
                        },
                        else_expr,
                    )
                }
                (ConcreteType::Intrinsic(IntrinsicType::SInt), ConcreteType::UnknownIntClass) => {
                    let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
                    let new_else = inst
                        .instantiate_intrinsic_type_class(else_expr.result, IntrinsicType::SInt);
                    let new_else_blk = inst.create_block(BlockFlowInstruction::Undetermined);
                    assert!(
                        block_ctx.try_chain(else_expr.end_block, new_else_blk),
                        "else multiple out?"
                    );

                    (
                        IntrinsicType::SInt.into(),
                        then_expr,
                        SimplifyResult {
                            result: new_else,
                            start_block: else_expr.start_block,
                            end_block: new_else_blk,
                        },
                    )
                }
                (ConcreteType::UnknownIntClass, ConcreteType::Intrinsic(IntrinsicType::UInt)) => {
                    let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
                    let new_then = inst
                        .instantiate_intrinsic_type_class(then_expr.result, IntrinsicType::UInt);
                    let new_then_blk = inst.create_block(BlockFlowInstruction::Undetermined);
                    assert!(
                        block_ctx.try_chain(then_expr.end_block, new_then_blk),
                        "then multiple out?"
                    );

                    (
                        IntrinsicType::UInt.into(),
                        SimplifyResult {
                            result: new_then,
                            start_block: then_expr.start_block,
                            end_block: new_then_blk,
                        },
                        else_expr,
                    )
                }
                (ConcreteType::Intrinsic(IntrinsicType::UInt), ConcreteType::UnknownIntClass) => {
                    let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
                    let new_else = inst
                        .instantiate_intrinsic_type_class(else_expr.result, IntrinsicType::UInt);
                    let new_else_blk = inst.create_block(BlockFlowInstruction::Undetermined);
                    assert!(
                        block_ctx.try_chain(else_expr.end_block, new_else_blk),
                        "else multiple out?"
                    );

                    (
                        IntrinsicType::UInt.into(),
                        then_expr,
                        SimplifyResult {
                            result: new_else,
                            start_block: else_expr.start_block,
                            end_block: new_else_blk,
                        },
                    )
                }
                (ConcreteType::UnknownIntClass, ConcreteType::Intrinsic(IntrinsicType::Float)) => {
                    let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
                    let new_then = inst
                        .instantiate_intrinsic_type_class(then_expr.result, IntrinsicType::Float);
                    let new_then_blk = inst.create_block(BlockFlowInstruction::Undetermined);
                    assert!(
                        block_ctx.try_chain(then_expr.end_block, new_then_blk),
                        "then multiple out?"
                    );

                    (
                        IntrinsicType::Float.into(),
                        SimplifyResult {
                            result: new_then,
                            start_block: then_expr.start_block,
                            end_block: new_then_blk,
                        },
                        else_expr,
                    )
                }
                (
                    ConcreteType::UnknownNumberClass,
                    ConcreteType::Intrinsic(IntrinsicType::Float),
                ) => {
                    let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
                    let new_then = inst
                        .instantiate_intrinsic_type_class(then_expr.result, IntrinsicType::Float);
                    let new_then_blk = inst.create_block(BlockFlowInstruction::Undetermined);
                    assert!(
                        block_ctx.try_chain(then_expr.end_block, new_then_blk),
                        "then multiple out?"
                    );

                    (
                        IntrinsicType::Float.into(),
                        SimplifyResult {
                            result: new_then,
                            start_block: then_expr.start_block,
                            end_block: new_then_blk,
                        },
                        else_expr,
                    )
                }
                (ConcreteType::Intrinsic(IntrinsicType::Float), ConcreteType::UnknownIntClass) => {
                    let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
                    let new_else = inst
                        .instantiate_intrinsic_type_class(else_expr.result, IntrinsicType::Float);
                    let new_else_blk = inst.create_block(BlockFlowInstruction::Undetermined);
                    assert!(
                        block_ctx.try_chain(else_expr.end_block, new_else_blk),
                        "else multiple out?"
                    );

                    (
                        IntrinsicType::Float.into(),
                        then_expr,
                        SimplifyResult {
                            result: new_else,
                            start_block: else_expr.start_block,
                            end_block: new_else_blk,
                        },
                    )
                }
                (
                    ConcreteType::Intrinsic(IntrinsicType::Float),
                    ConcreteType::UnknownNumberClass,
                ) => {
                    let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
                    let new_else = inst
                        .instantiate_intrinsic_type_class(else_expr.result, IntrinsicType::Float);
                    let new_else_blk = inst.create_block(BlockFlowInstruction::Undetermined);
                    assert!(
                        block_ctx.try_chain(else_expr.end_block, new_else_blk),
                        "else multiple out?"
                    );

                    (
                        IntrinsicType::Float.into(),
                        then_expr,
                        SimplifyResult {
                            result: new_else,
                            start_block: else_expr.start_block,
                            end_block: new_else_blk,
                        },
                    )
                }
                _ => {
                    panic!("Error: if then block and else block has different result type");
                }
            };

            let then_term_block =
                block_ctx.add(Block::flow_only(BlockFlowInstruction::ConditionalEnd));
            let else_term_block =
                block_ctx.add(Block::flow_only(BlockFlowInstruction::ConditionalEnd));
            assert!(
                block_ctx.try_chain(then.end_block, then_term_block),
                "then multiple out?"
            );
            assert!(
                block_ctx.try_chain(r#else.end_block, else_term_block),
                "else multiple out?"
            );

            let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
            let result = inst.phi(
                [
                    (then_term_block, then.result),
                    (r#else_term_block, r#else.result),
                ]
                .into_iter()
                .collect(),
                res_ty.clone(),
            );
            let merge_block = inst.create_block(BlockFlowInstruction::Undetermined);

            assert!(
                block_ctx.block_mut(condition.end_block).try_set_branch(
                    condition.result,
                    then.start_block,
                    r#else.start_block,
                    merge_block
                ),
                "already chained?"
            );

            SimplifyResult {
                result,
                start_block: condition.start_block,
                end_block: merge_block,
            }
        }
    }
}

fn emit_varref<'a, 's>(
    name: SourceRef<'s>,
    inst: &mut BlockInstructionEmitter<'_, 'a, 's>,
    symbol_scope: &'a SymbolScope<'a, 's>,
) -> RegisterRef {
    let Some((scope, v)) = symbol_scope.lookup(name.slice) else {
        panic!(
            "Error: referencing undefined symbol '{}' {name:?}",
            name.slice
        );
    };

    match v {
        VarLookupResult::IntrinsicFunctions(xs) => inst.intrinsic_function_ref(xs.to_vec()),
        VarLookupResult::IntrinsicTypeConstructor(t) => inst.intrinsic_type_constructor_ref(t),
        VarLookupResult::ScopeLocalVar(vid, _, false) => inst.scope_local_var_ref(scope, vid),
        VarLookupResult::ScopeLocalVar(vid, _, true) => {
            inst.scope_local_var_mutable_ref(scope, vid)
        }
        VarLookupResult::FunctionInputVar(vid, _, false) => inst.function_input_var_ref(scope, vid),
        VarLookupResult::FunctionInputVar(vid, _, true) => {
            inst.function_input_var_mutable_ref(scope, vid)
        }
        VarLookupResult::UserDefinedFunction(fs) => {
            inst.user_defined_function_ref(scope, fs.occurence.clone())
        }
    }
}

pub fn binary_op<'a, 's>(
    left: RegisterRef,
    op: SourceRef<'s>,
    right: RegisterRef,
    inst: &mut BlockInstructionEmitter<'_, 'a, 's>,
) -> RegisterRef {
    if matches!(
        op.slice,
        "^^" | "+" | "-" | "/" | "%" | "==" | "!=" | "<=" | ">=" | "<" | ">"
    ) {
        // gen2 conversion
        let conv = match op.slice {
            "^^" => left
                .ty(inst)
                .clone()
                .pow_op_type_conversion(right.ty(inst).clone()),
            "+" | "-" | "/" | "%" => left
                .ty(inst)
                .clone()
                .arithmetic_compare_op_type_conversion(right.ty(inst).clone()),
            // 比較演算の出力は必ずBoolになる
            "==" | "!=" | "<=" | ">=" | "<" | ">" => left
                .ty(inst)
                .clone()
                .arithmetic_compare_op_type_conversion(right.ty(inst).clone())
                .map(|(conv_left, conv_right, t)| {
                    (
                        conv_left,
                        conv_right,
                        IntrinsicType::Bool
                            .of_vector(t.vector_elements().unwrap())
                            .unwrap()
                            .into(),
                    )
                }),
            _ => unreachable!(),
        };
        let (left_conv, right_conv, result_ty) = match conv {
            Some(x) => x,
            None => {
                eprintln!(
                    "Error: cannot apply binary op {}({op:?}) between terms ({:?} and {:?})",
                    op.slice,
                    *left.ty(inst),
                    *right.ty(inst)
                );

                (
                    BinaryOpTermConversion::NoConversion,
                    BinaryOpTermConversion::NoConversion,
                    ConcreteType::Never,
                )
            }
        };

        let left = match left_conv {
            BinaryOpTermConversion::NoConversion => left,
            BinaryOpTermConversion::Cast(to) => inst.cast(left, to.into()),
            BinaryOpTermConversion::Instantiate(it) => {
                inst.instantiate_intrinsic_type_class(left, it)
            }
            BinaryOpTermConversion::Distribute(to) => {
                inst.construct_intrinsic_composite(to, vec![left])
            }
            BinaryOpTermConversion::CastAndDistribute(elm_type, count) => {
                let vec_type = elm_type.of_vector(count as _).unwrap();
                let left = inst.cast(left, elm_type.into());
                inst.construct_intrinsic_composite(vec_type, vec![left])
            }
            BinaryOpTermConversion::InstantiateAndDistribute(elm_type, count) => {
                let vec_type = elm_type.of_vector(count as _).unwrap();
                let left = inst.instantiate_intrinsic_type_class(left, elm_type);
                inst.construct_intrinsic_composite(vec_type, vec![left])
            }
            BinaryOpTermConversion::PromoteUnknownNumber => inst.promote_int_to_number(left),
        };
        let right = match right_conv {
            BinaryOpTermConversion::NoConversion => right,
            BinaryOpTermConversion::Cast(to) => inst.cast(right, to.into()),
            BinaryOpTermConversion::Instantiate(it) => {
                inst.instantiate_intrinsic_type_class(right, it)
            }
            BinaryOpTermConversion::Distribute(to) => {
                inst.construct_intrinsic_composite(to, vec![right])
            }
            BinaryOpTermConversion::CastAndDistribute(elm_type, count) => {
                let vec_type = elm_type.of_vector(count as _).unwrap();
                let right = inst.cast(right, elm_type.into());
                inst.construct_intrinsic_composite(vec_type, vec![right])
            }
            BinaryOpTermConversion::InstantiateAndDistribute(elm_type, count) => {
                let vec_type = elm_type.of_vector(count as _).unwrap();
                let right = inst.instantiate_intrinsic_type_class(right, elm_type);
                inst.construct_intrinsic_composite(vec_type, vec![right])
            }
            BinaryOpTermConversion::PromoteUnknownNumber => inst.promote_int_to_number(right),
        };

        let op = match op.slice {
            "^^" => IntrinsicBinaryOperation::Pow,
            "+" => IntrinsicBinaryOperation::Add,
            "-" => IntrinsicBinaryOperation::Sub,
            "*" => IntrinsicBinaryOperation::Mul,
            "/" => IntrinsicBinaryOperation::Div,
            "%" => IntrinsicBinaryOperation::Rem,
            "&" => IntrinsicBinaryOperation::BitAnd,
            "|" => IntrinsicBinaryOperation::BitOr,
            "^" => IntrinsicBinaryOperation::BitXor,
            "<<" => IntrinsicBinaryOperation::LeftShift,
            ">>" => IntrinsicBinaryOperation::RightShift,
            "==" => IntrinsicBinaryOperation::Eq,
            "!=" => IntrinsicBinaryOperation::Ne,
            ">=" => IntrinsicBinaryOperation::Ge,
            "<=" => IntrinsicBinaryOperation::Le,
            ">" => IntrinsicBinaryOperation::Gt,
            "<" => IntrinsicBinaryOperation::Lt,
            "&&" => IntrinsicBinaryOperation::LogAnd,
            "||" => IntrinsicBinaryOperation::LogOr,
            _ => unreachable!("unknown binary op"),
        };

        return inst.intrinsic_binary_op(left, op, right, result_ty);
    }

    let (left, right) =
        if op.slice == "*" && left.ty(inst).is_scalar_type() && right.ty(inst).is_vector_type() {
            // scalar times vectorの演算はないのでオペランドだけ逆にする（評価順は維持する）
            (right, left)
        } else {
            (left, right)
        };

    let r = match op.slice {
        // 行列とかの掛け算があるので特別扱い
        "*" => left
            .ty(inst)
            .clone()
            .multiply_op_type_conversion(right.ty(inst).clone()),
        "&" | "|" | "^" | ">>" | "<<" => left
            .ty(inst)
            .clone()
            .bitwise_op_type_conversion(right.ty(inst).clone()),
        "&&" | "||" => left
            .ty(inst)
            .clone()
            .logical_op_type_conversion(right.ty(inst).clone()),
        _ => None,
    };
    let (conv, res) = match r {
        Some(x) => x,
        None => {
            eprintln!(
                "Error: cannot apply binary op {} between terms ({:?} and {:?})",
                op.slice,
                *left.ty(inst),
                *right.ty(inst)
            );
            (BinaryOpTypeConversion::NoConversion, ConcreteType::Never)
        }
    };

    let (left, right) = match conv {
        BinaryOpTypeConversion::NoConversion => (left, right),
        BinaryOpTypeConversion::CastLeftHand(to) => {
            let left = inst.cast(left, to.into());

            (left, right)
        }
        BinaryOpTypeConversion::CastRightHand(to) => {
            let right = inst.cast(right, to.into());

            (left, right)
        }
        BinaryOpTypeConversion::InstantiateAndCastLeftHand(it) => {
            let left = inst.instantiate_intrinsic_type_class(left, it);
            let left = inst.cast(left, res.clone());

            (left, right)
        }
        BinaryOpTypeConversion::InstantiateAndCastRightHand(it) => {
            let right = inst.instantiate_intrinsic_type_class(right, it);
            let right = inst.cast(right, res.clone());

            (left, right)
        }
        BinaryOpTypeConversion::InstantiateLeftHand(it) => {
            let left = inst.instantiate_intrinsic_type_class(left, it);

            (left, right)
        }
        BinaryOpTypeConversion::InstantiateRightHand(it) => {
            let right = inst.instantiate_intrinsic_type_class(right, it);

            (left, right)
        }
        BinaryOpTypeConversion::InstantiateLeftAndCastRightHand(it) => {
            let left = inst.instantiate_intrinsic_type_class(left, it);
            let right = inst.cast(right, res.clone());

            (left, right)
        }
        BinaryOpTypeConversion::InstantiateRightAndCastLeftHand(it) => {
            let right = inst.instantiate_intrinsic_type_class(right, it);
            let left = inst.cast(left, res.clone());

            (left, right)
        }
    };

    let op = match op.slice {
        "^^" => IntrinsicBinaryOperation::Pow,
        "+" => IntrinsicBinaryOperation::Add,
        "-" => IntrinsicBinaryOperation::Sub,
        "*" => IntrinsicBinaryOperation::Mul,
        "/" => IntrinsicBinaryOperation::Div,
        "%" => IntrinsicBinaryOperation::Rem,
        "&" => IntrinsicBinaryOperation::BitAnd,
        "|" => IntrinsicBinaryOperation::BitOr,
        "^" => IntrinsicBinaryOperation::BitXor,
        "<<" => IntrinsicBinaryOperation::LeftShift,
        ">>" => IntrinsicBinaryOperation::RightShift,
        "==" => IntrinsicBinaryOperation::Eq,
        "!=" => IntrinsicBinaryOperation::Ne,
        ">=" => IntrinsicBinaryOperation::Ge,
        "<=" => IntrinsicBinaryOperation::Le,
        ">" => IntrinsicBinaryOperation::Gt,
        "<" => IntrinsicBinaryOperation::Lt,
        "&&" => IntrinsicBinaryOperation::LogAnd,
        "||" => IntrinsicBinaryOperation::LogOr,
        _ => unreachable!("unknown binary op"),
    };

    inst.intrinsic_binary_op(left, op, right, res)
}

fn funcall<'a, 's>(
    callee: RegisterRef,
    mut args: Vec<RegisterRef>,
    block_ctx: &mut BlockGenerationContext<'a, 's>,
    inst_ctx: &mut BlockInstructionEmissionContext<'a, 's>,
) -> (RegisterRef, BlockRef) {
    match *callee.ty(inst_ctx) {
        ConcreteType::IntrinsicTypeConstructor(t) => {
            let element_ty = t.scalar_type().unwrap();
            let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
            for a in args.iter_mut() {
                *a = inst.loaded(*a);
                match (&*a.ty(&inst), &element_ty) {
                    // TODO: 他の変換は必要になったら書く
                    (ConcreteType::UnknownIntClass, IntrinsicScalarType::Float) => {
                        *a = inst.instantiate_intrinsic_type_class(*a, IntrinsicType::Float);
                    }
                    (ConcreteType::UnknownNumberClass, IntrinsicScalarType::Float) => {
                        *a = inst.instantiate_intrinsic_type_class(*a, IntrinsicType::Float);
                    }
                    _ => (),
                }
            }

            let eval_registers = inst.into_eval_impure_registers();
            let result_register = RegisterRef::Impure(inst_ctx.alloc_impure_register(t.into()));
            let eval_block = block_ctx.add(Block {
                eval_impure_registers: eval_registers,
                flow: BlockFlowInstruction::Funcall {
                    callee,
                    args,
                    result: result_register,
                    after_return: None,
                },
            });

            (result_register, eval_block)
        }
        ConcreteType::Function {
            args: ref def_arg_types,
            ref output,
        } if def_arg_types.len() == args.len() => {
            let mut matches = true;
            let def_arg_types = def_arg_types.clone();
            let output = output.clone();
            let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
            for (dt, a) in def_arg_types.iter().zip(args.iter_mut()) {
                matches = matches
                    && match (&*a.ty(&inst), dt) {
                        (
                            ConcreteType::UnknownIntClass,
                            ConcreteType::Intrinsic(IntrinsicType::SInt),
                        ) => {
                            *a = inst.instantiate_intrinsic_type_class(*a, IntrinsicType::SInt);
                            true
                        }
                        (
                            ConcreteType::UnknownIntClass,
                            ConcreteType::Intrinsic(IntrinsicType::UInt),
                        ) => {
                            *a = inst.instantiate_intrinsic_type_class(*a, IntrinsicType::UInt);
                            true
                        }
                        (
                            ConcreteType::UnknownIntClass,
                            ConcreteType::Intrinsic(IntrinsicType::Float),
                        ) => {
                            *a = inst.instantiate_intrinsic_type_class(*a, IntrinsicType::Float);
                            true
                        }
                        (
                            ConcreteType::UnknownNumberClass,
                            ConcreteType::Intrinsic(IntrinsicType::Float),
                        ) => {
                            *a = inst.instantiate_intrinsic_type_class(*a, IntrinsicType::Float);
                            true
                        }
                        (ConcreteType::Ref(ref c), c2) if **c == *c2 => {
                            *a = inst.load_ref(*a);
                            true
                        }
                        (ConcreteType::MutableRef(ref c), c2) if **c == *c2 => {
                            *a = inst.load_ref(*a);
                            true
                        }
                        (t, _) => t == dt,
                    };
            }

            if !matches {
                panic!("Error: argument types mismatched");
            }

            let eval_registers = inst.into_eval_impure_registers();
            let result_register = RegisterRef::Impure(
                inst_ctx.alloc_impure_register(
                    output
                        .as_ref()
                        .map_or_else(|| IntrinsicType::Unit.into(), |x| *x.clone()),
                ),
            );
            let eval_block = block_ctx.add(Block {
                eval_impure_registers: eval_registers,
                flow: BlockFlowInstruction::Funcall {
                    callee,
                    args,
                    result: result_register,
                    after_return: None,
                },
            });

            (result_register, eval_block)
        }
        ConcreteType::Function {
            args: ref def_args, ..
        } => {
            panic!(
                "Error: argument types mismatched({def_args:?} and {:?})",
                args.iter().map(|a| a.ty(inst_ctx)).collect::<Vec<_>>()
            );
        }
        ConcreteType::OverloadedFunctions(ref xs) => {
            let arg_types = args
                .iter()
                .map(|a| a.ty(inst_ctx).clone())
                .collect::<Vec<_>>();
            let matching_overloads = resolve_overload(xs, &arg_types).collect::<Vec<_>>();
            println!("matching overloads: {matching_overloads:?}");
            let exact_matches = matching_overloads
                .iter()
                .filter_map(|(n, r)| r.is_exact_match().then_some(*n))
                .collect::<Vec<_>>();
            match &exact_matches[..] {
                &[] => {
                    let instantiated_matches = matching_overloads
                        .iter()
                        .filter(|(_, r)| !r.is_exact_match())
                        .collect::<Vec<_>>();
                    match &instantiated_matches[..] {
                        &[] => panic!("Error: no matching overloads found: args={arg_types:?}, candidates={xs:?}"),
                        &[(n, r)] => {
                            let output = *xs[*n].1.clone();
                            let mut inst = BlockInstructionEmitter::new(block_ctx, inst_ctx);
                            for (t, a) in r.args_promotion.iter().zip(args.iter_mut()) {
                                match t {
                                    None => (),
                                    &Some(OverloadArgumentPromotionType::InstantiateIntrinsicTypeClass(it)) => {
                                        *a = inst.instantiate_intrinsic_type_class(*a, it);
                                    },
                                    Some(OverloadArgumentPromotionType::Dereference) => {
                                        *a = inst.load_ref(*a);
                                    }
                                }
                            }

                            let eval_registers = inst.into_eval_impure_registers();
                            let result_register = RegisterRef::Impure(inst_ctx.alloc_impure_register(output));
                            let eval_block = block_ctx.add(Block {
                                eval_impure_registers: eval_registers,
                                flow: BlockFlowInstruction::Funcall {
                                    callee, args, result: result_register, after_return: None
                                }
                            });

                            (result_register, eval_block)
                        },
                        &[..] => panic!("Error: multiple candidates matches to the usage: args={arg_types:?}, candidates={xs:?}")
                    }
                },
                &[n] => {
                    let result_register = RegisterRef::Impure(inst_ctx.alloc_impure_register(*xs[n].1.clone()));
                    let eval_block = block_ctx.add(Block::flow_only(BlockFlowInstruction::Funcall {
                        callee, args, result: result_register, after_return: None
                    }));

                    (result_register, eval_block)
                },
                &[..] => panic!("Error: multiple candidates matches to the usage: args={arg_types:?}, candidates={xs:?}")
            }
        }
        _ => panic!("Error: not applyable type"),
    }
}

#[derive(Debug, Clone)]
pub enum OverloadArgumentPromotionType {
    InstantiateIntrinsicTypeClass(IntrinsicType),
    Dereference,
}

#[derive(Debug, Clone)]
pub struct OverloadMatchingRequirements {
    pub args_promotion: Vec<Option<OverloadArgumentPromotionType>>,
}
impl OverloadMatchingRequirements {
    #[inline(always)]
    pub fn is_exact_match(&self) -> bool {
        self.args_promotion.iter().all(Option::is_none)
    }
}
fn resolve_overload<'v, 's>(
    candidates: &'v [(Vec<ConcreteType<'s>>, Box<ConcreteType<'s>>)],
    inputs: &'v [ConcreteType<'s>],
) -> impl Iterator<Item = (usize, OverloadMatchingRequirements)> + 'v {
    candidates.iter().enumerate().filter_map(|(n, (args, _))| {
        let mut args_promotion = Vec::with_capacity(args.len());
        for (a, i) in args.iter().zip(inputs.iter()) {
            match (a, i) {
                (x, y) if x == y => args_promotion.push(None),
                (ConcreteType::Intrinsic(IntrinsicType::UInt), ConcreteType::UnknownIntClass) => {
                    args_promotion.push(Some(
                        OverloadArgumentPromotionType::InstantiateIntrinsicTypeClass(
                            IntrinsicType::UInt,
                        ),
                    ))
                }
                (ConcreteType::Intrinsic(IntrinsicType::SInt), ConcreteType::UnknownIntClass) => {
                    args_promotion.push(Some(
                        OverloadArgumentPromotionType::InstantiateIntrinsicTypeClass(
                            IntrinsicType::SInt,
                        ),
                    ))
                }
                (ConcreteType::Intrinsic(IntrinsicType::Float), ConcreteType::UnknownIntClass) => {
                    args_promotion.push(Some(
                        OverloadArgumentPromotionType::InstantiateIntrinsicTypeClass(
                            IntrinsicType::Float,
                        ),
                    ))
                }
                (
                    ConcreteType::Intrinsic(IntrinsicType::Float),
                    ConcreteType::UnknownNumberClass,
                ) => args_promotion.push(Some(
                    OverloadArgumentPromotionType::InstantiateIntrinsicTypeClass(
                        IntrinsicType::Float,
                    ),
                )),
                (c, ConcreteType::Ref(c2)) if c == &**c2 => {
                    args_promotion.push(Some(OverloadArgumentPromotionType::Dereference))
                }
                (c, ConcreteType::MutableRef(c2)) if c == &**c2 => {
                    args_promotion.push(Some(OverloadArgumentPromotionType::Dereference))
                }
                // not match
                _ => return None,
            }
        }

        Some((n, OverloadMatchingRequirements { args_promotion }))
    })
}
