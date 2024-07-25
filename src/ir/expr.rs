use typed_arena::Arena;

use crate::{
    concrete_type::{
        BinaryOpTypeConversion, ConcreteType, IntrinsicScalarType, IntrinsicType, UserDefinedType,
    },
    parser::{ExpressionNode, StatementNode},
    ref_path::RefPath,
    scope::{SymbolScope, VarId, VarLookupResult},
    source_ref::{SourceRef, SourceRefSliceEq},
    utils::{swizzle_indices, PtrEq},
};

use super::ExprRef;

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
    LoadVar(PtrEq<'a, SymbolScope<'a, 's>>, VarId),
    InitializeVar(PtrEq<'a, SymbolScope<'a, 's>>, VarId),
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
        symbol_scope: PtrEq<'a, SymbolScope<'a, 's>>,
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

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ConstModifiers: u8 {
        const NEGATE = 1 << 0;
        const BIT_NOT = 1 << 1;
        const LOGICAL_NOT = 1 << 2;
    }
}

pub struct SimplificationContext<'a, 's> {
    pub symbol_scope_arena: &'a Arena<SymbolScope<'a, 's>>,
    pub vars: Vec<(SimplifiedExpression<'a, 's>, ConcreteType<'s>)>,
}
impl<'a, 's> SimplificationContext<'a, 's> {
    pub fn add(&mut self, expr: SimplifiedExpression<'a, 's>, ty: ConcreteType<'s>) -> ExprRef {
        self.vars.push((expr, ty));

        ExprRef(self.vars.len() - 1)
    }
}
pub fn simplify_expression<'a, 's>(
    ast: ExpressionNode<'s>,
    ctx: &mut SimplificationContext<'a, 's>,
    symbol_scope: &'a SymbolScope<'a, 's>,
) -> (ExprRef, ConcreteType<'s>) {
    match ast {
        ExpressionNode::Binary(left, op, right) => {
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
        ExpressionNode::Prefixed(op, expr) => {
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
        ExpressionNode::Lifted(_, x, _) => simplify_expression(*x, ctx, symbol_scope),
        ExpressionNode::Blocked(stmts, x) => {
            let new_symbol_scope = ctx
                .symbol_scope_arena
                .alloc(SymbolScope::new(Some(symbol_scope), false));
            let mut new_ctx = SimplificationContext {
                symbol_scope_arena: ctx.symbol_scope_arena,
                vars: Vec::new(),
            };

            for s in stmts {
                match s {
                    StatementNode::Let {
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
        ExpressionNode::MemberRef(base, _, name) => {
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
        ExpressionNode::Funcall {
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
        ExpressionNode::FuncallSingle(base_expr, arg) => {
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
        ExpressionNode::Number(t) => {
            let has_hex_prefix = t.slice.starts_with("0x") || t.slice.starts_with("0X");
            let has_float_suffix = t.slice.ends_with(['f', 'F']);
            let has_fpart = t.slice.contains('.');

            let (expr, ty) = if has_hex_prefix {
                (
                    SimplifiedExpression::ConstInt(SourceRefSliceEq::from(&t)),
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
                    SimplifiedExpression::ConstNumber(SourceRefSliceEq::from(&t)),
                    ConcreteType::UnknownNumberClass,
                )
            } else {
                (
                    SimplifiedExpression::ConstInt(SourceRefSliceEq::from(&t)),
                    ConcreteType::UnknownIntClass,
                )
            };

            (ctx.add(expr, ty.clone()), ty)
        }
        ExpressionNode::Var(x) => {
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
                VarLookupResult::ScopeLocalVar(vid, ty) => {
                    let ty = ty.clone().instantiate(scope);

                    (
                        ctx.add(
                            SimplifiedExpression::LoadVar(PtrEq(scope), VarId::ScopeLocal(vid)),
                            ty.clone(),
                        ),
                        ty,
                    )
                }
                VarLookupResult::FunctionInputVar(vid, ty) => {
                    let ty = ty.clone().instantiate(scope);

                    (
                        ctx.add(
                            SimplifiedExpression::LoadVar(PtrEq(scope), VarId::FunctionInput(vid)),
                            ty.clone().instantiate(scope),
                        ),
                        ty.clone(),
                    )
                }
            }
        }
        ExpressionNode::Tuple(_, xs, _) => {
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
        ExpressionNode::If {
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
