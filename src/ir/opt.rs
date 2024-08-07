use std::{
    collections::{HashMap, HashSet},
    io::Write,
};

use typed_arena::Arena;

use crate::{
    concrete_type::{ConcreteType, IntrinsicType},
    ir::expr::print_simp_expr,
    ref_path::RefPath,
    scope::{self, SymbolScope, VarId},
    source_ref::{SourceRef, SourceRefSliceEq},
    utils::PtrEq,
};

use super::{
    expr::{ConstModifiers, ScopeCaptureSource, SimplifiedExpression},
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

pub fn inline_function1<'a, 's>(
    expressions: &mut [(SimplifiedExpression<'a, 's>, ConcreteType<'s>)],
    scope_arena: &'a Arena<SymbolScope<'a, 's>>,
) -> bool {
    unimplemented!("inline function 1");
    // let mut last_reffunction = HashMap::<ExprRef, (&'a SymbolScope<'a, 's>, &'s str)>::new();
    // let mut tree_modified = false;

    // for (n, (x, _)) in expressions.iter_mut().enumerate() {
    //     match x {
    //         &mut SimplifiedExpression::RefFunction(vscope, name) => {
    //             last_reffunction.insert(ExprRef(n), (vscope.0, name));
    //         }
    //         &mut SimplifiedExpression::Funcall(base, ref args) => {
    //             if let Some((fscope, fname)) = last_reffunction.get(&base) {
    //                 if let Some(fbody) = fscope.user_defined_function_body(fname) {
    //                     let mut expressions = fbody.borrow().expressions.clone();
    //                     let substitutions = (0..args.len()).collect::<Vec<_>>();
    //                     replace_inlined_function_input_refs(
    //                         &mut expressions,
    //                         fbody.borrow().symbol_scope,
    //                         &substitutions,
    //                     );

    //                     *x = SimplifiedExpression::ScopedBlock {
    //                         capturing: args.iter().map(|&x| ScopeCaptureSource::Expr(x)).collect(),
    //                         symbol_scope: PtrEq(
    //                             scope_arena
    //                                 .alloc(fbody.borrow().symbol_scope.new_function_inlined()),
    //                         ),
    //                         expressions,
    //                         returning: fbody.borrow().returning.id,
    //                     };

    //                     tree_modified = true;
    //                 }
    //             }
    //         }
    //         &mut SimplifiedExpression::ScopedBlock {
    //             ref mut expressions,
    //             ..
    //         } => {
    //             tree_modified |= inline_function1(expressions, scope_arena);
    //         }
    //         &mut SimplifiedExpression::LoopBlock {
    //             ref mut expressions,
    //             ..
    //         } => {
    //             tree_modified |= inline_function1(expressions, scope_arena);
    //         }
    //         _ => (),
    //     }
    // }

    // tree_modified
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

fn unfold_pure_computation_scopes<'a, 's>(
    expressions: &mut Vec<(SimplifiedExpression<'a, 's>, ConcreteType<'s>)>,
    scope: &'a SymbolScope<'a, 's>,
    block_returning_ref: &mut Option<&mut ExprRef>,
) -> bool {
    let mut tree_modified = false;

    println!("[unlift input]");
    let mut so = std::io::stdout().lock();
    for (n, (x, xt)) in expressions.iter().enumerate() {
        print_simp_expr(&mut so, x, xt, n, 0);
    }
    so.flush().unwrap();
    drop(so);

    let mut n = 0;
    while n < expressions.len() {
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
                && scope_expr
                    .iter()
                    .enumerate()
                    .all(|(n, _)| ExprRef(n).is_pure(&scope_expr)) =>
            {
                assert_eq!(returning.0, scope_expr.len() - 1);

                // relocate scope local ids and unlift scope capture refs
                for x in scope_expr.iter_mut() {
                    println!("[unlift reloc] {x:?}");

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
                        &mut SimplifiedExpression::ScopedBlock {
                            capturing: ref mut inner_capturing,
                            ..
                        } => {
                            println!("inner scope captures: {inner_capturing:?}");

                            for c in inner_capturing.iter_mut() {
                                match c {
                                    &mut ScopeCaptureSource::Capture(n) => {
                                        println!(
                                            "promotecapture: Capture({n}) -> {:?}",
                                            capturing[n]
                                        );

                                        *c = capturing[n];
                                    }
                                    _ => (),
                                }
                            }
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

                println!("[unlift scope]");
                let mut so = std::io::stdout().lock();
                for (n, (x, xt)) in expressions.iter().enumerate() {
                    print_simp_expr(&mut so, x, xt, n, 0);
                }
                so.flush().unwrap();
                drop(so);

                n += nth_shifts + 1;
                tree_modified = true;
            }
            _ => {
                n += 1;
            }
        }
    }

    tree_modified
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
        tree_modified |=
            unfold_pure_computation_scopes(expressions, scope, &mut block_returning_ref);
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

                    scope.relocate_local_var_init_expr(|r| {
                        if r.0 == n {
                            ExprRef(n + returning_rel)
                        } else if r.0 >= n {
                            ExprRef(r.0 + nth_shifts)
                        } else {
                            r
                        }
                    });

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
