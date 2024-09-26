use std::{cell::RefCell, collections::HashMap};

use crate::{
    concrete_type::{ConcreteType, IntrinsicType, UserDefinedStructMember, UserDefinedType},
    ir::{ExprRef, FunctionBody},
    source_ref::SourceRef,
    symbol::{
        FunctionInputVariable, IntrinsicFunctionSymbol, LocalVariable, UserDefinedFunctionSymbol,
    },
    utils::DebugPrintGuard,
};

#[derive(Debug, Clone)]
pub struct SymbolScope<'a, 's> {
    pub parent: Option<&'a SymbolScope<'a, 's>>,
    pub is_toplevel_function: bool,
    intrinsic_symbols: HashMap<&'s str, Vec<IntrinsicFunctionSymbol>>,
    user_defined_type_symbols: HashMap<&'s str, (SourceRef<'s>, UserDefinedType<'s>)>,
    user_defined_function_symbols: HashMap<&'s str, UserDefinedFunctionSymbol<'s>>,
    user_defined_function_body:
        DebugPrintGuard<RefCell<HashMap<&'s str, RefCell<FunctionBody<'a, 's>>>>>,
    pub function_input_vars: Vec<FunctionInputVariable<'s>>,
    pub local_vars: RefCell<Vec<LocalVariable<'s>>>,
    var_id_by_name: RefCell<HashMap<&'s str, VarId>>,
}
impl<'a, 's> SymbolScope<'a, 's> {
    pub fn new(parent: Option<&'a SymbolScope<'a, 's>>, is_toplevel_function: bool) -> Self {
        Self {
            parent,
            is_toplevel_function,
            intrinsic_symbols: HashMap::new(),
            user_defined_type_symbols: HashMap::new(),
            user_defined_function_symbols: HashMap::new(),
            user_defined_function_body: DebugPrintGuard(RefCell::new(HashMap::new())),
            function_input_vars: Vec::new(),
            local_vars: RefCell::new(Vec::new()),
            var_id_by_name: RefCell::new(HashMap::new()),
        }
    }

    #[inline(always)]
    pub fn new_child(&'a self) -> Self {
        Self::new(Some(self), false)
    }

    pub fn new_function_inlined(&self) -> Self {
        Self {
            function_input_vars: Vec::new(),
            ..self.clone()
        }
    }

    pub fn new_intrinsics() -> Self {
        let mut intrinsic_symbols = HashMap::new();
        let mut var_id_by_name = HashMap::new();

        intrinsic_symbols.insert(
            "subpassLoad",
            vec![IntrinsicFunctionSymbol {
                name: "Cloth.Intrinsic.SubpassLoad",
                args: vec![IntrinsicType::SubpassInput.into()],
                output: IntrinsicType::Float4.into(),
                is_pure: true,
            }],
        );
        intrinsic_symbols.insert(
            "normalize",
            vec![
                IntrinsicFunctionSymbol {
                    name: "Cloth.Intrinsic.Normalize#Float4",
                    args: vec![IntrinsicType::Float4.into()],
                    output: IntrinsicType::Float4.into(),
                    is_pure: true,
                },
                IntrinsicFunctionSymbol {
                    name: "Cloth.Intrinsic.Normalize#Float3",
                    args: vec![IntrinsicType::Float3.into()],
                    output: IntrinsicType::Float3.into(),
                    is_pure: true,
                },
                IntrinsicFunctionSymbol {
                    name: "Cloth.Intrinsic.Normalize#Float2",
                    args: vec![IntrinsicType::Float2.into()],
                    output: IntrinsicType::Float2.into(),
                    is_pure: true,
                },
            ],
        );
        intrinsic_symbols.insert(
            "dot",
            vec![
                IntrinsicFunctionSymbol {
                    name: "Cloth.Intrinsic.Dot#Float4",
                    args: vec![IntrinsicType::Float4.into(), IntrinsicType::Float4.into()],
                    output: IntrinsicType::Float.into(),
                    is_pure: true,
                },
                IntrinsicFunctionSymbol {
                    name: "Cloth.Intrinsic.Dot#Float3",
                    args: vec![IntrinsicType::Float3.into(), IntrinsicType::Float3.into()],
                    output: IntrinsicType::Float.into(),
                    is_pure: true,
                },
                IntrinsicFunctionSymbol {
                    name: "Cloth.Intrinsic.Dot#Float2",
                    args: vec![IntrinsicType::Float2.into(), IntrinsicType::Float2.into()],
                    output: IntrinsicType::Float.into(),
                    is_pure: true,
                },
            ],
        );
        intrinsic_symbols.insert(
            "transpose",
            vec![
                IntrinsicFunctionSymbol {
                    name: "Cloth.Intrinsic.Transpose#Float4x4",
                    args: vec![IntrinsicType::Float4x4.into()],
                    output: IntrinsicType::Float4x4.into(),
                    is_pure: true,
                },
                IntrinsicFunctionSymbol {
                    name: "Cloth.Intrinsic.Transpose#Float3x3",
                    args: vec![IntrinsicType::Float3x3.into()],
                    output: IntrinsicType::Float3x3.into(),
                    is_pure: true,
                },
                IntrinsicFunctionSymbol {
                    name: "Cloth.Intrinsic.Transpose#Float2x2",
                    args: vec![IntrinsicType::Float2x2.into()],
                    output: IntrinsicType::Float2x2.into(),
                    is_pure: true,
                },
            ],
        );
        intrinsic_symbols.insert(
            "sampleAt",
            vec![
                IntrinsicFunctionSymbol {
                    name: "Cloth.Intrinsic.SampleAt#Texture1D",
                    args: vec![IntrinsicType::Texture1D.into(), IntrinsicType::Float.into()],
                    output: IntrinsicType::Float4.into(),
                    is_pure: true,
                },
                IntrinsicFunctionSymbol {
                    name: "Cloth.Intrinsic.SampleAt#Texture2D",
                    args: vec![
                        IntrinsicType::Texture2D.into(),
                        IntrinsicType::Float2.into(),
                    ],
                    output: IntrinsicType::Float4.into(),
                    is_pure: true,
                },
                IntrinsicFunctionSymbol {
                    name: "Cloth.Intrinsic.SampleAt#Texture3D",
                    args: vec![
                        IntrinsicType::Texture3D.into(),
                        IntrinsicType::Float3.into(),
                    ],
                    output: IntrinsicType::Float4.into(),
                    is_pure: true,
                },
            ],
        );
        intrinsic_symbols.insert(
            "log2",
            vec![IntrinsicFunctionSymbol {
                name: "Cloth.Intrinsic.Math.Log2",
                args: vec![IntrinsicType::Float.into()],
                output: IntrinsicType::Float.into(),
                is_pure: true,
            }],
        );
        intrinsic_symbols.insert(
            "exp2",
            vec![IntrinsicFunctionSymbol {
                name: "Cloth.Intrinsic.Math.Exp2",
                args: vec![IntrinsicType::Float.into()],
                output: IntrinsicType::Float.into(),
                is_pure: true,
            }],
        );
        intrinsic_symbols.insert(
            "max",
            vec![IntrinsicFunctionSymbol {
                name: "Cloth.Intrinsic.Max#UInt",
                args: vec![IntrinsicType::UInt.into(), IntrinsicType::UInt.into()],
                output: IntrinsicType::UInt.into(),
                is_pure: true,
            }],
        );
        intrinsic_symbols.insert(
            "min",
            vec![
                IntrinsicFunctionSymbol {
                    name: "Cloth.Intrinsic.Min#UInt",
                    args: vec![IntrinsicType::UInt.into(), IntrinsicType::UInt.into()],
                    output: IntrinsicType::UInt.into(),
                    is_pure: true,
                },
                IntrinsicFunctionSymbol {
                    name: "Cloth.Intrinsic.Min#Float",
                    args: vec![IntrinsicType::Float.into(), IntrinsicType::Float.into()],
                    output: IntrinsicType::Float.into(),
                    is_pure: true,
                },
            ],
        );
        intrinsic_symbols.insert(
            "clamp",
            vec![IntrinsicFunctionSymbol {
                name: "Cloth.Intrinsic.Clamp",
                args: vec![
                    IntrinsicType::Float.into(),
                    IntrinsicType::Float.into(),
                    IntrinsicType::Float.into(),
                ],
                output: IntrinsicType::Float.into(),
                is_pure: true,
            }],
        );
        intrinsic_symbols.insert(
            "barrier",
            vec![IntrinsicFunctionSymbol {
                name: "Cloth.Intrinsic.ExecutionBarrier",
                args: vec![],
                output: IntrinsicType::Unit.into(),
                is_pure: false,
            }],
        );
        intrinsic_symbols.insert(
            "loadPixelAt",
            vec![IntrinsicFunctionSymbol {
                name: "Cloth.Intrinsic.Image.Load#Image2D",
                args: vec![IntrinsicType::Image2D.into(), IntrinsicType::UInt2.into()],
                output: IntrinsicType::Float4.into(),
                is_pure: true,
            }],
        );
        intrinsic_symbols.insert(
            "atomicAdd",
            vec![IntrinsicFunctionSymbol {
                name: "Cloth.Intrinsic.Atomic.Add#UInt",
                args: vec![
                    ConcreteType::from(IntrinsicType::UInt).mutable_ref(),
                    IntrinsicType::UInt.into(),
                ],
                output: IntrinsicType::Unit.into(),
                is_pure: false,
            }],
        );
        var_id_by_name.insert(
            "Float4",
            VarId::IntrinsicTypeConstructor(IntrinsicType::Float4),
        );
        var_id_by_name.insert(
            "Float3",
            VarId::IntrinsicTypeConstructor(IntrinsicType::Float3),
        );
        var_id_by_name.insert(
            "Float2",
            VarId::IntrinsicTypeConstructor(IntrinsicType::Float2),
        );
        var_id_by_name.insert("UInt", VarId::IntrinsicTypeConstructor(IntrinsicType::UInt));
        var_id_by_name.insert(
            "UInt2",
            VarId::IntrinsicTypeConstructor(IntrinsicType::UInt2),
        );

        Self {
            parent: None,
            is_toplevel_function: false,
            intrinsic_symbols,
            user_defined_type_symbols: HashMap::new(),
            user_defined_function_symbols: HashMap::new(),
            user_defined_function_body: DebugPrintGuard(RefCell::new(HashMap::new())),
            function_input_vars: Vec::new(),
            local_vars: RefCell::new(Vec::new()),
            var_id_by_name: RefCell::new(var_id_by_name),
        }
    }

    pub fn has_local_vars(&self) -> bool {
        !self.local_vars.borrow().is_empty()
    }

    pub fn merge_local_vars(&self, from: &'a Self) -> usize {
        let offset = self.local_vars.borrow().len();
        self.local_vars
            .borrow_mut()
            .extend(from.local_vars.borrow_mut().drain(..));
        self.var_id_by_name
            .borrow_mut()
            .extend(from.var_id_by_name.borrow_mut().drain().map(|(k, v)| {
                (
                    k,
                    match v {
                        VarId::ScopeLocal(x) => VarId::ScopeLocal(x + offset),
                        x => x,
                    },
                )
            }));

        offset
    }

    #[inline]
    pub fn all_local_var_ids(&self) -> impl Iterator<Item = usize> {
        0..self.local_vars.borrow().len()
    }

    pub fn remove_local_var_by_id(&self, id: usize) {
        self.local_vars.borrow_mut().remove(id);

        for lv in self.var_id_by_name.borrow_mut().values_mut() {
            match lv {
                &mut VarId::ScopeLocal(ref mut vx) => {
                    *vx -= if *vx > id { 1 } else { 0 };
                }
                _ => (),
            }
        }
    }

    pub fn declare_struct(
        &mut self,
        occurence: SourceRef<'s>,
        members: Vec<UserDefinedStructMember<'s>>,
    ) {
        self.user_defined_type_symbols.insert(
            occurence.slice,
            (occurence, UserDefinedType::Struct(members)),
        );
    }

    pub fn declare_function(&mut self, details: UserDefinedFunctionSymbol<'s>) {
        match self
            .user_defined_function_symbols
            .entry(details.occurence.slice)
        {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(details);
            }
            std::collections::hash_map::Entry::Occupied(v) => {
                panic!(
                    "Error: same name function was already declared at {}:{}",
                    v.get().occurence.line + 1,
                    v.get().occurence.col + 1
                );
            }
        }
    }

    #[inline]
    pub fn user_defined_function_symbol(
        &self,
        name: &str,
    ) -> Option<&UserDefinedFunctionSymbol<'s>> {
        self.user_defined_function_symbols.get(name)
    }

    #[inline]
    pub fn iter_user_defined_function_symbols<'e>(
        &'e self,
    ) -> impl Iterator<Item = &'e UserDefinedFunctionSymbol<'s>> + 'e {
        self.user_defined_function_symbols.values()
    }

    pub fn attach_function_body(&self, fname: &'s str, body: FunctionBody<'a, 's>) {
        match self.user_defined_function_body.0.borrow_mut().entry(fname) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(RefCell::new(body));
            }
            std::collections::hash_map::Entry::Occupied(_) => {
                panic!("Error: same name function body was already declared");
            }
        }
    }

    #[inline]
    pub fn user_defined_function_body<'e>(
        &'e self,
        name: &str,
    ) -> Option<core::cell::Ref<'e, RefCell<FunctionBody<'a, 's>>>> {
        core::cell::Ref::filter_map(self.user_defined_function_body.0.borrow(), |x| x.get(name))
            .ok()
    }

    #[inline]
    pub fn user_defined_function_body_mut<'e>(
        &'e self,
        name: &str,
    ) -> Option<core::cell::RefMut<'e, RefCell<FunctionBody<'a, 's>>>> {
        core::cell::RefMut::filter_map(self.user_defined_function_body.0.borrow_mut(), |x| {
            x.get_mut(name)
        })
        .ok()
    }

    pub fn declare_function_input(
        &mut self,
        name: SourceRef<'s>,
        ty: ConcreteType<'s>,
        mutable: bool,
    ) -> VarId {
        match self.var_id_by_name.borrow_mut().entry(name.slice) {
            std::collections::hash_map::Entry::Vacant(v) => {
                self.function_input_vars.push(FunctionInputVariable {
                    occurence: name.clone(),
                    ty,
                    mutable,
                });
                let vid = VarId::FunctionInput(self.function_input_vars.len() - 1);
                v.insert(vid);
                vid
            }
            std::collections::hash_map::Entry::Occupied(_) => {
                panic!("Function Input {} is already declared", name.slice);
            }
        }
    }

    pub fn declare_local_var(
        &self,
        name: SourceRef<'s>,
        ty: ConcreteType<'s>,
        mutable: bool,
    ) -> VarId {
        self.local_vars.borrow_mut().push(LocalVariable {
            occurence: name.clone(),
            ty,
            mutable,
        });
        let vid = VarId::ScopeLocal(self.local_vars.borrow().len() - 1);
        self.var_id_by_name.borrow_mut().insert(name.slice, vid);
        vid
    }

    pub fn declare_anon_local_var(&self, ty: ConcreteType<'s>, mutable: bool) -> VarId {
        self.local_vars.borrow_mut().push(LocalVariable {
            occurence: SourceRef {
                slice: "",
                line: 0,
                col: 0,
            },
            ty,
            mutable,
        });
        let vid = VarId::ScopeLocal(self.local_vars.borrow().len() - 1);
        vid
    }

    pub fn lookup<'x>(&'x self, name: &str) -> Option<(&Self, VarLookupResult<'x, 's>)> {
        if let Some(x) = self.intrinsic_symbols.get(name) {
            return Some((self, VarLookupResult::IntrinsicFunctions(x)));
        }

        if let Some(x) = self.user_defined_function_symbols.get(name) {
            return Some((self, VarLookupResult::UserDefinedFunction(x)));
        }

        match self.var_id_by_name.borrow().get(name) {
            Some(VarId::FunctionInput(x)) => Some((
                self,
                VarLookupResult::FunctionInputVar(
                    *x,
                    &self.function_input_vars[*x].ty,
                    self.function_input_vars[*x].mutable,
                ),
            )),
            Some(VarId::ScopeLocal(x)) => Some((
                self,
                VarLookupResult::ScopeLocalVar(
                    *x,
                    self.local_vars.borrow()[*x].ty.clone(),
                    self.local_vars.borrow()[*x].mutable,
                ),
            )),
            Some(VarId::IntrinsicTypeConstructor(x)) => {
                Some((self, VarLookupResult::IntrinsicTypeConstructor(*x)))
            }
            None => match self.parent {
                Some(ref p) => p.lookup(name),
                None => None,
            },
        }
    }

    pub fn lookup_user_defined_type(
        &self,
        name: &'s str,
    ) -> Option<(&Self, &(SourceRef<'s>, UserDefinedType<'s>))> {
        match self.user_defined_type_symbols.get(name) {
            Some(p) => Some((self, p)),
            None => match self.parent {
                Some(ref p) => p.lookup_user_defined_type(name),
                None => None,
            },
        }
    }

    pub fn nearest_function_scope(&self) -> Option<&Self> {
        if self.is_toplevel_function {
            Some(self)
        } else {
            self.parent.and_then(Self::nearest_function_scope)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VarId {
    FunctionInput(usize),
    ScopeLocal(usize),
    IntrinsicTypeConstructor(IntrinsicType),
}

#[derive(Debug, Clone)]
pub enum VarLookupResult<'x, 's> {
    FunctionInputVar(usize, &'x ConcreteType<'s>, bool),
    ScopeLocalVar(usize, ConcreteType<'s>, bool),
    IntrinsicFunctions(&'x [IntrinsicFunctionSymbol]),
    IntrinsicTypeConstructor(IntrinsicType),
    UserDefinedFunction(&'x UserDefinedFunctionSymbol<'s>),
}
