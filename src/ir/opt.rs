use std::collections::{HashMap, HashSet};

use crate::{
    concrete_type::{ConcreteType, IntrinsicType},
    ref_path::RefPath,
    scope::{SymbolScope, VarId},
    source_ref::{SourceRef, SourceRefSliceEq},
    utils::PtrEq,
};

use super::{
    expr::{ConstModifiers, SimplifiedExpression},
    ExprRef,
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

pub fn optimize_pure_expr<'a, 's>(
    expressions: &mut Vec<(SimplifiedExpression<'a, 's>, ConcreteType<'s>)>,
    scope: &'a SymbolScope<'a, 's>,
    mut block_returning_ref: Option<&mut ExprRef>,
) -> bool {
    let mut least_one_tree_modified = false;
    let mut tree_modified = true;

    while tree_modified {
        tree_modified = false;

        // promote single scope
        if expressions.len() == 1 {
            match expressions.pop().unwrap() {
                (
                    SimplifiedExpression::ScopedBlock {
                        symbol_scope: child_scope,
                        expressions: mut scope_expr,
                        returning,
                    },
                    ty,
                ) => {
                    assert_eq!(ty, scope_expr[returning.0].1);
                    let parent_scope = child_scope.0.parent.unwrap();
                    let local_var_offset = parent_scope.merge_local_vars(child_scope.0);
                    println!("scopemerge {child_scope:?} -> {:?}", PtrEq(parent_scope));

                    for x in scope_expr.iter_mut() {
                        promote_local_var_scope(
                            &mut x.0,
                            child_scope.0,
                            parent_scope,
                            local_var_offset,
                        );
                    }

                    expressions.extend(scope_expr);
                    if let Some(ref mut b) = block_returning_ref {
                        **b = returning;
                    }

                    tree_modified = true;
                }
                (x, t) => expressions.push((x, t)),
            }
        }

        // unfold pure computation scope
        for n in 0..expressions.len() {
            match &mut expressions[n] {
                (
                    SimplifiedExpression::ScopedBlock {
                        expressions: scope_expr,
                        symbol_scope,
                        returning,
                    },
                    _,
                ) if !symbol_scope.0.has_local_vars()
                    && scope_expr.iter().all(|x| x.0.is_pure()) =>
                {
                    assert_eq!(returning.0, scope_expr.len() - 1);

                    for x in scope_expr.iter_mut() {
                        x.0.relocate_ref(|x| x + n);
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
                        expressions[m]
                            .0
                            .relocate_ref(|x| if x >= n { x + nth_shifts } else { x });
                    }

                    scope.relocate_local_var_init_expr(|r| {
                        if r.0 >= n {
                            ExprRef(r.0 + nth_shifts)
                        } else {
                            r
                        }
                    });

                    if let Some(ref mut ret) = block_returning_ref {
                        ret.0 += if ret.0 >= n { nth_shifts } else { 0 };
                    }

                    tree_modified = true;
                }
                _ => (),
            }
        }

        // inlining loadvar until dirtified
        let mut last_loadvar_expr_id = HashMap::new();
        let mut expr_id_alias = HashMap::new();
        let mut last_expr_ids = HashMap::new();
        for n in 0..expressions.len() {
            match &mut expressions[n].0 {
                &mut SimplifiedExpression::LoadVar(vscope, vid) => {
                    if let Some(x) =
                        last_loadvar_expr_id.get(&(vscope.0 as *const SymbolScope, vid))
                    {
                        expr_id_alias.insert(n, *x);
                        scope.relocate_local_var_init_expr(
                            |r| if r.0 == n { ExprRef(*x) } else { r },
                        );
                    } else {
                        last_loadvar_expr_id.insert((vscope.0 as *const SymbolScope, vid), n);
                    }
                }
                &mut SimplifiedExpression::InitializeVar(vscope, VarId::ScopeLocal(vx)) => {
                    let init_expr_id = vscope.0.init_expr_id(vx).unwrap();

                    expr_id_alias.insert(n, init_expr_id.0);
                    scope.relocate_local_var_init_expr(|r| if r.0 == n { init_expr_id } else { r });
                    last_loadvar_expr_id
                        .insert((vscope.as_ptr(), VarId::ScopeLocal(vx)), init_expr_id.0);
                }
                x => {
                    tree_modified |=
                        x.relocate_ref(|x| expr_id_alias.get(&x).copied().unwrap_or(x));
                    if let Some(x) = last_expr_ids.get(&*x) {
                        expr_id_alias.insert(n, *x);
                        scope.relocate_local_var_init_expr(
                            |r| if r.0 == n { ExprRef(*x) } else { r },
                        );
                    } else {
                        last_expr_ids.insert(x.clone(), n);
                    }
                }
            }
        }

        let mut referenced_expr = HashSet::new();
        let mut current_scope_var_usages = HashMap::new();
        referenced_expr.extend(block_returning_ref.as_ref().map(|x| **x));
        let expressions_head_ptr = expressions.as_ptr();
        for n in 0..expressions.len() {
            match &mut expressions[n].0 {
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
                &mut SimplifiedExpression::Select(c, t, e) => {
                    referenced_expr.extend([c, t, e]);
                }
                &mut SimplifiedExpression::Funcall(base, ref args) => {
                    let intrinsic_constructor =
                        match unsafe { &(&*expressions_head_ptr.add(base.0)).1 } {
                            ConcreteType::IntrinsicTypeConstructor(it) => match it {
                                IntrinsicType::Float2 => Some((IntrinsicType::Float2, 2)),
                                IntrinsicType::Float3 => Some((IntrinsicType::Float3, 3)),
                                IntrinsicType::Float4 => Some((IntrinsicType::Float4, 4)),
                                IntrinsicType::UInt2 => Some((IntrinsicType::UInt2, 2)),
                                IntrinsicType::UInt3 => Some((IntrinsicType::UInt3, 3)),
                                IntrinsicType::UInt4 => Some((IntrinsicType::UInt4, 4)),
                                IntrinsicType::SInt2 => Some((IntrinsicType::SInt2, 2)),
                                IntrinsicType::SInt3 => Some((IntrinsicType::SInt3, 3)),
                                IntrinsicType::SInt4 => Some((IntrinsicType::SInt4, 4)),
                                _ => None,
                            },
                            _ => None,
                        };

                    if let Some((it, count)) = intrinsic_constructor {
                        let mut args = args.clone();
                        referenced_expr.extend(args.iter().copied());

                        let org_arg_count = args.len();
                        let mut ins_count = 0;
                        while args.len() < count {
                            // extend by repeating
                            args.push(args[ins_count % org_arg_count]);
                            ins_count += 1;
                        }

                        expressions[n].0 =
                            SimplifiedExpression::ConstructIntrinsicComposite(it, args);
                        tree_modified = true;
                    } else {
                        let intrinsic_function =
                            match unsafe { &(&*expressions_head_ptr.add(base.0)).0 } {
                                &SimplifiedExpression::IntrinsicFunction(name, is_pure) => {
                                    Some((name, is_pure))
                                }
                                _ => None,
                            };

                        if let Some((instr, is_pure)) = intrinsic_function {
                            let args = args.clone();
                            referenced_expr.extend(args.iter().copied());

                            expressions[n].0 =
                                SimplifiedExpression::IntrinsicFuncall(instr, is_pure, args);
                            tree_modified = true;
                        } else {
                            referenced_expr.insert(base);
                            referenced_expr.extend(args.iter().copied());
                        }
                    }
                }
                &mut SimplifiedExpression::MemberRef(
                    base,
                    SourceRefSliceEq(SourceRef { slice: name, .. }),
                ) => match &expressions[base.0].0 {
                    SimplifiedExpression::LoadByCanonicalRefPath(rp) => {
                        expressions[n].0 = SimplifiedExpression::LoadByCanonicalRefPath(
                            RefPath::Member(Box::new(rp.clone()), name),
                        );
                        tree_modified = true;
                    }
                    _ => {
                        referenced_expr.insert(base);
                    }
                },
                &mut SimplifiedExpression::LoadVar(scope, VarId::FunctionInput(vx))
                    if scope.0.is_toplevel_function =>
                {
                    expressions[n].0 =
                        SimplifiedExpression::LoadByCanonicalRefPath(RefPath::FunctionInput(vx));
                    tree_modified = true;
                }
                &mut SimplifiedExpression::LoadVar(vscope, VarId::ScopeLocal(vx))
                    if vscope == PtrEq(scope) =>
                {
                    current_scope_var_usages
                        .entry(vx)
                        .or_insert(LocalVarUsage::Unaccessed)
                        .mark_read();
                }
                &mut SimplifiedExpression::LoadVar(_, _) => (),
                &mut SimplifiedExpression::InitializeVar(vscope, VarId::ScopeLocal(vx))
                    if vscope == PtrEq(scope) =>
                {
                    current_scope_var_usages
                        .entry(vx)
                        .or_insert(LocalVarUsage::Unaccessed)
                        .mark_write(ExprRef(n));
                }
                &mut SimplifiedExpression::InitializeVar(_, _) => (),
                &mut SimplifiedExpression::LoadByCanonicalRefPath(_) => (),
                &mut SimplifiedExpression::StoreLocal(_, v) => {
                    referenced_expr.insert(v);
                }
                &mut SimplifiedExpression::IntrinsicFunction(_, _) => (),
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
                &mut SimplifiedExpression::ConstUInt(_, _) => (),
                &mut SimplifiedExpression::ConstSInt(_, _) => (),
                &mut SimplifiedExpression::ConstFloat(_, _) => (),
                &mut SimplifiedExpression::ConstructTuple(ref xs) => {
                    referenced_expr.extend(xs.iter().copied());
                }
                &mut SimplifiedExpression::ConstructIntrinsicComposite(_, ref xs) => {
                    referenced_expr.extend(xs.iter().copied());
                }
                &mut SimplifiedExpression::StoreOutput(x, _) => {
                    referenced_expr.insert(x);
                }
                &mut SimplifiedExpression::DistributeOutputTuple(x, _) => {
                    referenced_expr.insert(x);
                }
                SimplifiedExpression::ScopedBlock {
                    ref mut expressions,
                    ref mut returning,
                    ref symbol_scope,
                } => {
                    tree_modified |=
                        optimize_pure_expr(expressions, symbol_scope.0, Some(returning));

                    for (n, x) in expressions.iter().enumerate() {
                        match x.0 {
                            SimplifiedExpression::LoadVar(vscope, VarId::ScopeLocal(vx))
                                if vscope == PtrEq(scope) =>
                            {
                                current_scope_var_usages
                                    .entry(vx)
                                    .or_insert(LocalVarUsage::Unaccessed)
                                    .mark_read();
                            }
                            SimplifiedExpression::InitializeVar(vscope, VarId::ScopeLocal(vx))
                                if vscope == PtrEq(scope) =>
                            {
                                current_scope_var_usages
                                    .entry(vx)
                                    .or_insert(LocalVarUsage::Unaccessed)
                                    .mark_write(ExprRef(n));
                            }
                            _ => (),
                        }
                    }
                }
            }
        }

        // collect stripped expression ids
        let mut stripped_ops = HashSet::new();
        for (_, t) in current_scope_var_usages.iter() {
            if let &LocalVarUsage::Write(last_write) = t {
                println!("striplastwrite: {last_write:?}");
                stripped_ops.insert(last_write.0);
            }
        }
        for n in 0..expressions.len() {
            if !referenced_expr.contains(&ExprRef(n)) && expressions[n].0.is_pure() {
                stripped_ops.insert(n);
            }
        }

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
                    &mut SimplifiedExpression::LoadVar(vscope, VarId::ScopeLocal(ref mut vx))
                        if vscope == PtrEq(scope) =>
                    {
                        *vx -= if *vx > n { 1 } else { 0 };
                    }
                    &mut SimplifiedExpression::InitializeVar(
                        vscope,
                        VarId::ScopeLocal(ref mut vx),
                    ) if vscope == PtrEq(scope) => {
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

        // println!("transformed(cont?{tree_modified}):");
        // for (n, (x, t)) in expressions.iter().enumerate() {
        //     print_simp_expr(x, t, n, 0);
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
        &mut SimplifiedExpression::LoadVar(scope, VarId::ScopeLocal(lv))
            if scope == PtrEq(old_scope) =>
        {
            *expr = SimplifiedExpression::LoadVar(
                PtrEq(new_scope),
                VarId::ScopeLocal(lv + local_var_offset),
            );
        }
        &mut SimplifiedExpression::InitializeVar(scope, VarId::ScopeLocal(lv))
            if scope == PtrEq(old_scope) =>
        {
            *expr = SimplifiedExpression::InitializeVar(
                PtrEq(new_scope),
                VarId::ScopeLocal(lv + local_var_offset),
            );
        }
        SimplifiedExpression::ScopedBlock { expressions, .. } => {
            for x in expressions.iter_mut() {
                promote_local_var_scope(&mut x.0, old_scope, new_scope, local_var_offset);
            }
        }
        _ => (),
    }
}
