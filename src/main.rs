use std::{cell::RefCell, collections::HashMap};

use typed_arena::Arena;

fn main() {
    let src = std::fs::read_to_string("./sample_bloom_extract.csh").expect("Failed to load source");
    let mut tokenizer = Tokenizer {
        source: &src,
        line: 0,
        col: 0,
        current_line_indent: 0,
    };
    // populate line indent for first line
    tokenizer.populate_line_indent();
    let mut tokens = Vec::new();
    while let Some(t) = tokenizer.next_token().unwrap() {
        tokens.push(t);
    }

    let mut parse_state = ParseState {
        token_list: tokens,
        token_ptr: 0,
        indent_context_stack: Vec::new(),
    };
    let mut tlds = Vec::new();
    while parse_state.current_token().is_some() {
        let tld = parse_toplevel_declaration(&mut parse_state).unwrap();
        println!("tld: {tld:#?}");
        tlds.push(tld);
    }

    let intrinsic_symbol_scope = IntrinsicSymbolScope::new();
    let mut scope_stack = Vec::new();

    let mut partially_typed_types = HashMap::new();
    for tld in tlds.iter() {
        match tld {
            ToplevelDeclaration::Struct(s) => {
                partially_typed_types.insert(
                    s.name_token.slice,
                    (
                        SourceRef::from(&s.name_token),
                        UserDefinedTypePartiallyTyped::Struct(
                            s.member_list
                                .iter()
                                .map(|x| UserDefinedStructMemberPartiallyTyped {
                                    name: SourceRef::from(&x.name_token),
                                    ty: x.ty.clone(),
                                    attributes: x
                                        .attribute_lists
                                        .iter()
                                        .flat_map(|xs| {
                                            xs.attribute_list.iter().map(|x| x.0.clone())
                                        })
                                        .collect(),
                                })
                                .collect(),
                        ),
                    ),
                );
            }
            ToplevelDeclaration::Function(_) => (),
        }
    }

    let mut top_scope = SymbolScope {
        user_defined_type_symbols: HashMap::new(),
        function_inputs: Vec::new(),
        local_vars: Vec::new(),
        function_input_index_by_name: HashMap::new(),
        local_var_index_by_name: HashMap::new(),
    };
    top_scope
        .user_defined_type_symbols
        .extend(partially_typed_types.into_iter().map(|(k, (org, v))| {
            (
                k,
                (
                    org,
                    match v {
                        UserDefinedTypePartiallyTyped::Struct(s) => UserDefinedType::Struct(
                            s.into_iter()
                                .map(|x| UserDefinedStructMember {
                                    name: x.name,
                                    ty: ConcreteType::build(&scope_stack, x.ty),
                                    attributes: x.attributes,
                                })
                                .collect(),
                        ),
                    },
                ),
            )
        }));
    println!("TopScopeSymbols: {top_scope:#?}");
    scope_stack.push(top_scope);

    let ToplevelDeclaration::Function(f) = tlds.remove(1) else {
        panic!("not a function?");
    };
    let symbol_scope_arena = Arena::new();
    let global_symbol_scope = symbol_scope_arena.alloc(SymbolScope2::new_intrinsics());
    let function_symbol_scope =
        symbol_scope_arena.alloc(SymbolScope2::new(Some(global_symbol_scope)));
    match f.input_args {
        FunctionDeclarationInputArguments::Single {
            varname_token, ty, ..
        } => {
            function_symbol_scope.declare_function_input(
                SourceRef::from(&varname_token),
                ConcreteType::build(&scope_stack, ty),
            );
        }
        FunctionDeclarationInputArguments::Multiple { args, .. } => {
            for (_, n, _, ty, _) in args {
                function_symbol_scope.declare_function_input(
                    SourceRef::from(&n),
                    ConcreteType::build(&scope_stack, ty),
                );
            }
        }
    }
    let mut simplify_context = SimplificationContext {
        symbol_scope_arena: &symbol_scope_arena,
        vars: Vec::new(),
    };
    let last_var_id = simplify_expression(f.body, &mut simplify_context, &function_symbol_scope);
    println!("simplified exprs:");
    for (n, x) in simplify_context.vars.iter().enumerate() {
        print_simp_expr(x, n, 0);
    }
    println!("last id: {last_var_id}");

    // let ToplevelDeclaration::Function(f) = tlds.remove(1) else {
    //     panic!("not a function decl?");
    // };

    // let mut fscope = SymbolScope {
    //     user_defined_type_symbols: HashMap::new(),
    //     function_inputs: Vec::new(),
    //     local_vars: Vec::new(),
    //     function_input_index_by_name: HashMap::new(),
    //     local_var_index_by_name: HashMap::new(),
    // };
    // match f.input_args {
    //     FunctionDeclarationInputArguments::Single {
    //         varname_token, ty, ..
    //     } => {
    //         fscope.declare_function_input(&varname_token, ConcreteType::build(&scope_stack, ty));
    //     }
    //     FunctionDeclarationInputArguments::Multiple { args, .. } => {
    //         for a in args {
    //             fscope.declare_function_input(&a.1, ConcreteType::build(&scope_stack, a.3));
    //         }
    //     }
    // }
    // scope_stack.push(fscope);
    // let transformed = transform_expression(&intrinsic_symbol_scope, &mut scope_stack, f.body);
    // println!("transformed body: {transformed:#?}");
    // println!("scope: {scope_stack:#?}");
}

fn print_simp_expr(x: &SimplifiedExpression, vid: usize, nested: usize) {
    match x {
        SimplifiedExpression::ScopedBlock {
            expressions,
            returning,
            ..
        } => {
            println!("  {}%{vid} = Scope {{", "  ".repeat(nested));
            for (n, x) in expressions.iter().enumerate() {
                print_simp_expr(x, n, nested + 1);
            }
            println!("  {}returning %{returning}", "  ".repeat(nested + 1));
            println!("  {}}}", "  ".repeat(nested));
        }
        _ => println!("  {}%{vid} = {x:?}", "  ".repeat(nested)),
    }
}

#[derive(Debug)]
pub struct IntrinsicFunctionSymbol {
    pub name: &'static str,
    pub ty: ConcreteType<'static>,
}

#[derive(Debug)]
pub struct FunctionInputVariable<'s> {
    pub occurence: SourceRef<'s>,
    pub ty: ConcreteType<'s>,
}

#[derive(Debug)]
pub struct LocalVariable<'s> {
    pub occurence: SourceRef<'s>,
    pub ty: ConcreteType<'s>,
    pub init_expr_id: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum VarId {
    FunctionInput(usize),
    ScopeLocal(usize),
    IntrinsicFunction(&'static str),
    IntrinsicTypeConstructor(IntrinsicType),
}

#[derive(Debug)]
pub struct SymbolScope2<'a, 's> {
    parent: Option<&'a SymbolScope2<'a, 's>>,
    intrinsic_symbols: HashMap<&'s str, IntrinsicFunctionSymbol>,
    function_input_vars: Vec<FunctionInputVariable<'s>>,
    local_vars: RefCell<Vec<LocalVariable<'s>>>,
    var_id_by_name: RefCell<HashMap<&'s str, VarId>>,
}
impl<'a, 's> SymbolScope2<'a, 's> {
    pub fn new(parent: Option<&'a SymbolScope2<'a, 's>>) -> Self {
        Self {
            parent,
            intrinsic_symbols: HashMap::new(),
            function_input_vars: Vec::new(),
            local_vars: RefCell::new(Vec::new()),
            var_id_by_name: RefCell::new(HashMap::new()),
        }
    }

    pub fn new_intrinsics() -> Self {
        let mut intrinsic_symbols = HashMap::new();
        let mut var_id_by_name = HashMap::new();

        intrinsic_symbols.insert(
            "subpassLoad",
            IntrinsicFunctionSymbol {
                name: "OpImageRead#SubpassData",
                ty: ConcreteType::Function {
                    args: vec![ConcreteType::Intrinsic(IntrinsicType::SubpassInput)],
                    output: Some(Box::new(ConcreteType::Intrinsic(IntrinsicType::Float4))),
                },
            },
        );
        var_id_by_name.insert(
            "Float4",
            VarId::IntrinsicTypeConstructor(IntrinsicType::Float4),
        );

        Self {
            parent: None,
            intrinsic_symbols,
            function_input_vars: Vec::new(),
            local_vars: RefCell::new(Vec::new()),
            var_id_by_name: RefCell::new(var_id_by_name),
        }
    }

    pub fn declare_function_input(&mut self, name: SourceRef<'s>, ty: ConcreteType<'s>) -> VarId {
        match self.var_id_by_name.borrow_mut().entry(name.slice) {
            std::collections::hash_map::Entry::Vacant(v) => {
                self.function_input_vars.push(FunctionInputVariable {
                    occurence: name.clone(),
                    ty,
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
        init_expr: usize,
    ) -> VarId {
        self.local_vars.borrow_mut().push(LocalVariable {
            occurence: name.clone(),
            ty,
            init_expr_id: init_expr,
        });
        let vid = VarId::ScopeLocal(self.local_vars.borrow().len() - 1);
        self.var_id_by_name.borrow_mut().insert(name.slice, vid);
        vid
    }

    pub fn lookup(&self, name: &'s str) -> Option<(&Self, VarId)> {
        if let Some(x) = self.intrinsic_symbols.get(name) {
            return Some((self, VarId::IntrinsicFunction(x.name)));
        }

        match self.var_id_by_name.borrow().get(name) {
            Some(vid) => Some((self, *vid)),
            None => match self.parent {
                Some(ref p) => p.lookup(name),
                None => None,
            },
        }
    }
}

#[derive(Debug)]
pub enum SimplifiedExpression<'a, 's> {
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
    Rem(usize, usize),
    BitAnd(usize, usize),
    BitOr(usize, usize),
    BitXor(usize, usize),
    Eq(usize, usize),
    Ne(usize, usize),
    Gt(usize, usize),
    Ge(usize, usize),
    Lt(usize, usize),
    Le(usize, usize),
    LogAnd(usize, usize),
    LogOr(usize, usize),
    Neg(usize),
    BitNot(usize),
    LogNot(usize),
    Funcall(usize, Vec<usize>),
    MemberRef(usize, SourceRef<'s>),
    LoadVar(&'a SymbolScope2<'a, 's>, VarId),
    InitializeVar(&'a SymbolScope2<'a, 's>, VarId),
    StoreLocal(SourceRef<'s>, usize),
    IntrinsicFunction(&'static str),
    IntrinsicTypeConstructor(IntrinsicType),
    Select(usize, usize, usize),
    Cast(usize, ConcreteType<'s>),
    ConstInt(SourceRef<'s>),
    ConstFloat(SourceRef<'s>),
    ConstNumber(SourceRef<'s>),
    ConstUnit,
    ConstructTuple(Vec<usize>),
    ScopedBlock {
        symbol_scope: &'a SymbolScope2<'a, 's>,
        expressions: Vec<SimplifiedExpression<'a, 's>>,
        returning: usize,
    },
}
pub struct SimplificationContext<'a, 's> {
    pub symbol_scope_arena: &'a Arena<SymbolScope2<'a, 's>>,
    pub vars: Vec<SimplifiedExpression<'a, 's>>,
}
impl<'a, 's> SimplificationContext<'a, 's> {
    pub fn add(&mut self, expr: SimplifiedExpression<'a, 's>) -> usize {
        self.vars.push(expr);

        self.vars.len() - 1
    }
}
fn simplify_expression<'a, 's>(
    ast: Expression<'s>,
    ctx: &mut SimplificationContext<'a, 's>,
    symbol_scope: &'a SymbolScope2<'a, 's>,
) -> usize {
    match ast {
        Expression::Binary(left, op, right) => {
            let left = simplify_expression(*left, ctx, symbol_scope);
            let right = simplify_expression(*right, ctx, symbol_scope);

            ctx.add(match op.slice {
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
            })
        }
        Expression::Prefixed(op, expr) => {
            let expr = simplify_expression(*expr, ctx, symbol_scope);

            match op.slice {
                "+" => expr,
                "-" => ctx.add(SimplifiedExpression::Neg(expr)),
                "!" => ctx.add(SimplifiedExpression::LogNot(expr)),
                "~" => ctx.add(SimplifiedExpression::BitNot(expr)),
                _ => unreachable!("unknown prefixed op"),
            }
        }
        Expression::Lifted(_, x, _) => simplify_expression(*x, ctx, symbol_scope),
        Expression::Blocked(stmts, x) => {
            let new_symbol_scope = ctx
                .symbol_scope_arena
                .alloc(SymbolScope2::new(Some(symbol_scope)));
            let mut new_ctx = SimplificationContext {
                symbol_scope_arena: ctx.symbol_scope_arena,
                vars: Vec::new(),
            };

            for s in stmts {
                match s {
                    Statement::Let {
                        varname_token,
                        expr,
                        ..
                    } => {
                        let res = simplify_expression(expr, &mut new_ctx, new_symbol_scope);
                        let vid = new_symbol_scope.declare_local_var(
                            SourceRef::from(&varname_token),
                            ConcreteType::Inferred,
                            res,
                        );
                        new_ctx.add(SimplifiedExpression::InitializeVar(new_symbol_scope, vid));
                    }
                }
            }

            let last_id = simplify_expression(*x, &mut new_ctx, new_symbol_scope);
            ctx.add(SimplifiedExpression::ScopedBlock {
                symbol_scope: new_symbol_scope,
                expressions: new_ctx.vars,
                returning: last_id,
            })
        }
        Expression::MemberRef(base, _, name) => {
            let base = simplify_expression(*base, ctx, symbol_scope);

            ctx.add(SimplifiedExpression::MemberRef(
                base,
                SourceRef::from(&name),
            ))
        }
        Expression::Funcall {
            base_expr, args, ..
        } => {
            let base_expr = simplify_expression(*base_expr, ctx, symbol_scope);
            let args = args
                .into_iter()
                .map(|(x, _)| simplify_expression(x, ctx, symbol_scope))
                .collect();

            ctx.add(SimplifiedExpression::Funcall(base_expr, args))
        }
        Expression::FuncallSingle(base_expr, arg) => {
            let base_expr = simplify_expression(*base_expr, ctx, symbol_scope);
            let arg = simplify_expression(*arg, ctx, symbol_scope);

            ctx.add(SimplifiedExpression::Funcall(base_expr, vec![arg]))
        }
        Expression::Number(t) => {
            let has_hex_prefix = t.slice.starts_with("0x") || t.slice.starts_with("0X");
            let has_float_suffix = t.slice.ends_with(['f', 'F']);
            let has_fpart = t.slice.contains('.');

            ctx.add(if has_hex_prefix {
                SimplifiedExpression::ConstInt(SourceRef::from(&t))
            } else if has_float_suffix {
                SimplifiedExpression::ConstFloat(SourceRef::from(&t))
            } else if has_fpart {
                SimplifiedExpression::ConstNumber(SourceRef::from(&t))
            } else {
                SimplifiedExpression::ConstInt(SourceRef::from(&t))
            })
        }
        Expression::Var(x) => {
            let Some((scope, vid)) = symbol_scope.lookup(x.slice) else {
                panic!("Error: referencing undefined symbol '{}' {x:?}", x.slice);
            };

            match vid {
                VarId::IntrinsicFunction(name) => {
                    ctx.add(SimplifiedExpression::IntrinsicFunction(name))
                }
                VarId::IntrinsicTypeConstructor(t) => {
                    ctx.add(SimplifiedExpression::IntrinsicTypeConstructor(t))
                }
                _ => ctx.add(SimplifiedExpression::LoadVar(scope, vid)),
            }
        }
        Expression::Tuple(_, xs, _) => {
            let xs = xs
                .into_iter()
                .map(|(x, _)| simplify_expression(x, ctx, symbol_scope))
                .collect();

            ctx.add(SimplifiedExpression::ConstructTuple(xs))
        }
        Expression::If {
            condition,
            then_expr,
            else_expr,
            ..
        } => {
            let condition = simplify_expression(*condition, ctx, symbol_scope);
            let then_expr = simplify_expression(*then_expr, ctx, symbol_scope);
            let else_expr = match else_expr {
                None => ctx.add(SimplifiedExpression::ConstUnit),
                Some(x) => simplify_expression(*x, ctx, symbol_scope),
            };

            ctx.add(SimplifiedExpression::Select(
                condition, then_expr, else_expr,
            ))
        }
    }
}

#[derive(Debug, Clone)]
pub struct SourceRef<'s> {
    pub slice: &'s str,
    pub line: usize,
    pub col: usize,
}
impl<'s> From<&'_ Token<'s>> for SourceRef<'s> {
    #[inline]
    fn from(value: &'_ Token<'s>) -> Self {
        Self {
            slice: value.slice,
            line: value.line,
            col: value.col,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum IntrinsicType {
    Bool,
    UInt,
    UInt2,
    UInt3,
    UInt4,
    SInt,
    SInt2,
    SInt3,
    SInt4,
    Float,
    Float2,
    Float3,
    Float4,
    Float2x2,
    Float2x3,
    Float2x4,
    Float3x2,
    Float3x3,
    Float3x4,
    Float4x2,
    Float4x3,
    Float4x4,
    Sampler1D,
    Sampler2D,
    Sampler3D,
    Texture1D,
    Texture2D,
    Texture3D,
    SubpassInput,
}

#[derive(Debug, Clone)]
pub enum ConcreteType<'s> {
    Inferred,
    Intrinsic(IntrinsicType),
    UnknownIntClass,
    UnknownNumberClass,
    UserDefined {
        name: &'s str,
        generic_args: Vec<ConcreteType<'s>>,
    },
    Tuple(Vec<ConcreteType<'s>>),
    Function {
        args: Vec<ConcreteType<'s>>,
        output: Option<Box<ConcreteType<'s>>>,
    },
    IntrinsicTypeConstructor(IntrinsicType),
}
impl<'s> ConcreteType<'s> {
    pub fn build(scope_stack: &[SymbolScope], t: Type<'s>) -> Self {
        fn rec_lookup<'s>(t: Type<'s>, scope_stack: &[SymbolScope]) -> ConcreteType<'s> {
            match scope_stack.last() {
                None => panic!("Error: referencing undefined type: {}", t.name_token.slice),
                Some(x) => match x.user_defined_type_symbols.get(t.name_token.slice) {
                    Some(_) => ConcreteType::UserDefined {
                        name: t.name_token.slice,
                        generic_args: t
                            .generic_args
                            .map_or_else(Vec::new, |x| x.args)
                            .into_iter()
                            .map(|x| ConcreteType::build(scope_stack, x.0))
                            .collect(),
                    },
                    None => rec_lookup(t, &scope_stack[..scope_stack.len() - 1]),
                },
            }
        }

        match t.name_token.slice {
            "UInt" => Self::Intrinsic(IntrinsicType::UInt),
            "UInt2" => Self::Intrinsic(IntrinsicType::UInt2),
            "UInt3" => Self::Intrinsic(IntrinsicType::UInt3),
            "UInt4" => Self::Intrinsic(IntrinsicType::UInt4),
            "SInt" => Self::Intrinsic(IntrinsicType::SInt),
            "SInt2" => Self::Intrinsic(IntrinsicType::SInt2),
            "SInt3" => Self::Intrinsic(IntrinsicType::SInt3),
            "SInt4" => Self::Intrinsic(IntrinsicType::SInt4),
            "Float" => Self::Intrinsic(IntrinsicType::Float),
            "Float2" => Self::Intrinsic(IntrinsicType::Float2),
            "Float3" => Self::Intrinsic(IntrinsicType::Float3),
            "Float4" => Self::Intrinsic(IntrinsicType::Float4),
            "Float2x2" => Self::Intrinsic(IntrinsicType::Float2x2),
            "Float2x3" => Self::Intrinsic(IntrinsicType::Float2x3),
            "Float2x4" => Self::Intrinsic(IntrinsicType::Float2x4),
            "Float3x2" => Self::Intrinsic(IntrinsicType::Float3x2),
            "Float3x3" => Self::Intrinsic(IntrinsicType::Float3x3),
            "Float3x4" => Self::Intrinsic(IntrinsicType::Float3x4),
            "Float4x2" => Self::Intrinsic(IntrinsicType::Float4x2),
            "Float4x3" => Self::Intrinsic(IntrinsicType::Float4x3),
            "Float4x4" => Self::Intrinsic(IntrinsicType::Float4x4),
            "Sampler1D" => Self::Intrinsic(IntrinsicType::Sampler1D),
            "Sampler2D" => Self::Intrinsic(IntrinsicType::Sampler2D),
            "Sampler3D" => Self::Intrinsic(IntrinsicType::Sampler3D),
            "Texture1D" => Self::Intrinsic(IntrinsicType::Texture1D),
            "Texture2D" => Self::Intrinsic(IntrinsicType::Texture2D),
            "Texture3D" => Self::Intrinsic(IntrinsicType::Texture3D),
            "SubpassInput" => Self::Intrinsic(IntrinsicType::SubpassInput),
            _ => rec_lookup(t, scope_stack),
        }
    }
}

#[derive(Debug)]
pub struct Variable<'s> {
    pub declaration_ref: SourceRef<'s>,
    pub ty: ConcreteType<'s>,
    pub init_expr: Option<TypedExpression<'s>>,
}

#[derive(Debug)]
pub struct IntrinsicSymbolScope {
    symbol_type_table: HashMap<&'static str, ConcreteType<'static>>,
    type_constructors: HashMap<&'static str, IntrinsicType>,
}
impl IntrinsicSymbolScope {
    pub fn new() -> Self {
        let mut symbol_type_table = HashMap::new();
        symbol_type_table.insert(
            "subpassLoad",
            ConcreteType::Function {
                args: vec![ConcreteType::Intrinsic(IntrinsicType::SubpassInput)],
                output: Some(Box::new(ConcreteType::Intrinsic(IntrinsicType::Float4))),
            },
        );
        let mut type_constructors = HashMap::new();
        type_constructors.insert("Float4", IntrinsicType::Float4);

        Self {
            symbol_type_table,
            type_constructors,
        }
    }
}

#[derive(Debug)]
pub struct UserDefinedStructMemberPartiallyTyped<'s> {
    pub name: SourceRef<'s>,
    pub ty: Type<'s>,
    pub attributes: Vec<Attribute<'s>>,
}

#[derive(Debug)]
pub enum UserDefinedTypePartiallyTyped<'s> {
    Struct(Vec<UserDefinedStructMemberPartiallyTyped<'s>>),
}

#[derive(Debug)]
pub struct UserDefinedStructMember<'s> {
    pub name: SourceRef<'s>,
    pub ty: ConcreteType<'s>,
    pub attributes: Vec<Attribute<'s>>,
}

#[derive(Debug)]
pub enum UserDefinedType<'s> {
    Struct(Vec<UserDefinedStructMember<'s>>),
}

#[derive(Debug)]
pub struct SymbolScope<'s> {
    user_defined_type_symbols: HashMap<&'s str, (SourceRef<'s>, UserDefinedType<'s>)>,
    function_inputs: Vec<Variable<'s>>,
    local_vars: Vec<Variable<'s>>,
    function_input_index_by_name: HashMap<&'s str, usize>,
    local_var_index_by_name: HashMap<&'s str, usize>,
}
impl<'s> SymbolScope<'s> {
    pub fn declare_function_input(
        &mut self,
        name_token: &Token<'s>,
        ty: ConcreteType<'s>,
    ) -> usize {
        let var_index = self.function_inputs.len();
        self.function_inputs.push(Variable {
            declaration_ref: SourceRef {
                slice: name_token.slice,
                line: name_token.line,
                col: name_token.col,
            },
            ty,
            init_expr: None,
        });
        self.function_input_index_by_name
            .insert(name_token.slice, var_index);

        var_index
    }

    pub fn declare_local_var(
        &mut self,
        name_token: &Token<'s>,
        ty: ConcreteType<'s>,
        init_expr: TypedExpression<'s>,
    ) -> usize {
        let var_index = self.local_vars.len();
        self.local_vars.push(Variable {
            declaration_ref: SourceRef {
                slice: name_token.slice,
                line: name_token.line,
                col: name_token.col,
            },
            ty,
            init_expr: Some(init_expr),
        });
        self.local_var_index_by_name
            .insert(name_token.slice, var_index);

        var_index
    }
}

#[derive(Debug)]
pub enum SimplifiedStatement {
    InitializeLocalVar { var_index: usize },
}

#[derive(Debug)]
pub enum TypedExpression<'s> {
    Block(Vec<SimplifiedStatement>, Box<TypedExpression<'s>>),
    Tuple(Vec<TypedExpression<'s>>),
    Add(Box<TypedExpression<'s>>, Box<TypedExpression<'s>>),
    Sub(Box<TypedExpression<'s>>, Box<TypedExpression<'s>>),
    Mul(Box<TypedExpression<'s>>, Box<TypedExpression<'s>>),
    Div(Box<TypedExpression<'s>>, Box<TypedExpression<'s>>),
    Mod(Box<TypedExpression<'s>>, Box<TypedExpression<'s>>),
    BitAnd(Box<TypedExpression<'s>>, Box<TypedExpression<'s>>),
    BitOr(Box<TypedExpression<'s>>, Box<TypedExpression<'s>>),
    BitXor(Box<TypedExpression<'s>>, Box<TypedExpression<'s>>),
    Eq(Box<TypedExpression<'s>>, Box<TypedExpression<'s>>),
    Ne(Box<TypedExpression<'s>>, Box<TypedExpression<'s>>),
    Lt(Box<TypedExpression<'s>>, Box<TypedExpression<'s>>),
    Le(Box<TypedExpression<'s>>, Box<TypedExpression<'s>>),
    Gt(Box<TypedExpression<'s>>, Box<TypedExpression<'s>>),
    Ge(Box<TypedExpression<'s>>, Box<TypedExpression<'s>>),
    LogAnd(Box<TypedExpression<'s>>, Box<TypedExpression<'s>>),
    LogOr(Box<TypedExpression<'s>>, Box<TypedExpression<'s>>),
    If(
        Box<TypedExpression<'s>>,
        Box<TypedExpression<'s>>,
        Option<Box<TypedExpression<'s>>>,
    ),
    Neg(Box<TypedExpression<'s>>),
    BitNot(Box<TypedExpression<'s>>),
    LogNot(Box<TypedExpression<'s>>),
    MemberRef(Box<TypedExpression<'s>>, SourceRef<'s>),
    Funcall(Box<TypedExpression<'s>>, Vec<TypedExpression<'s>>),
    LocalVar {
        var_index: usize,
        scope_ascending: usize,
    },
    InputVar(usize),
    IntrinsicRef(&'s str),
    IntrinsicTyConRef(IntrinsicType),
    IntLiteral(&'s str),
    FloatLiteral(&'s str),
    NumberLiteral(&'s str),
    Cast(Box<TypedExpression<'s>>, ConcreteType<'s>),
}
pub fn simplify_statement<'s>(
    intrinsic_symbol_scope: &IntrinsicSymbolScope,
    scope_stack: &mut [SymbolScope<'s>],
    stmt: Statement<'s>,
) -> SimplifiedStatement {
    match stmt {
        Statement::Let {
            varname_token,
            expr,
            ..
        } => {
            let expr = transform_expression(intrinsic_symbol_scope, scope_stack, expr);

            let Some(scope) = scope_stack.last_mut() else {
                panic!("cannot declare variables outside of a function");
            };

            let var_index = scope.declare_local_var(&varname_token, ConcreteType::Inferred, expr);
            SimplifiedStatement::InitializeLocalVar { var_index }
        }
    }
}
pub fn transform_expression<'s>(
    intrinsic_symbol_scope: &IntrinsicSymbolScope,
    scope_stack: &mut [SymbolScope<'s>],
    expr: Expression<'s>,
) -> TypedExpression<'s> {
    match expr {
        Expression::Lifted(_, x, _) => {
            transform_expression(intrinsic_symbol_scope, scope_stack, *x)
        }
        Expression::Binary(left, op, right) => match op.slice {
            "+" => TypedExpression::Add(
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *left,
                )),
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *right,
                )),
            ),
            "-" => TypedExpression::Sub(
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *left,
                )),
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *right,
                )),
            ),
            "*" => TypedExpression::Mul(
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *left,
                )),
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *right,
                )),
            ),
            "/" => TypedExpression::Div(
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *left,
                )),
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *right,
                )),
            ),
            "%" => TypedExpression::Mod(
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *left,
                )),
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *right,
                )),
            ),
            "&" => TypedExpression::BitAnd(
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *left,
                )),
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *right,
                )),
            ),
            "|" => TypedExpression::BitOr(
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *left,
                )),
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *right,
                )),
            ),
            "^" => TypedExpression::BitXor(
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *left,
                )),
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *right,
                )),
            ),
            "==" => TypedExpression::Eq(
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *left,
                )),
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *right,
                )),
            ),
            "!=" => TypedExpression::Ne(
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *left,
                )),
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *right,
                )),
            ),
            ">=" => TypedExpression::Ge(
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *left,
                )),
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *right,
                )),
            ),
            "<=" => TypedExpression::Le(
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *left,
                )),
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *right,
                )),
            ),
            ">" => TypedExpression::Gt(
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *left,
                )),
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *right,
                )),
            ),
            "<" => TypedExpression::Lt(
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *left,
                )),
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *right,
                )),
            ),
            "&&" => TypedExpression::LogAnd(
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *left,
                )),
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *right,
                )),
            ),
            "||" => TypedExpression::LogOr(
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *left,
                )),
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *right,
                )),
            ),
            _ => unreachable!("unknown op"),
        },
        Expression::Prefixed(op, x) => match op.slice {
            "+" => transform_expression(intrinsic_symbol_scope, scope_stack, *x),
            "-" => TypedExpression::Neg(Box::new(transform_expression(
                intrinsic_symbol_scope,
                scope_stack,
                *x,
            ))),
            "!" => TypedExpression::LogNot(Box::new(transform_expression(
                intrinsic_symbol_scope,
                scope_stack,
                *x,
            ))),
            "~" => TypedExpression::BitNot(Box::new(transform_expression(
                intrinsic_symbol_scope,
                scope_stack,
                *x,
            ))),
            _ => unreachable!("unknown op"),
        },
        Expression::Blocked(stmts, final_expr) => TypedExpression::Block(
            stmts
                .into_iter()
                .map(|x| simplify_statement(intrinsic_symbol_scope, scope_stack, x))
                .collect(),
            Box::new(transform_expression(
                intrinsic_symbol_scope,
                scope_stack,
                *final_expr,
            )),
        ),
        Expression::If {
            condition,
            then_expr,
            else_expr,
            ..
        } => TypedExpression::If(
            Box::new(transform_expression(
                intrinsic_symbol_scope,
                scope_stack,
                *condition,
            )),
            Box::new(transform_expression(
                intrinsic_symbol_scope,
                scope_stack,
                *then_expr,
            )),
            else_expr.map(|x| {
                Box::new(transform_expression(
                    intrinsic_symbol_scope,
                    scope_stack,
                    *x,
                ))
            }),
        ),
        Expression::MemberRef(src, _, name) => TypedExpression::MemberRef(
            Box::new(transform_expression(
                intrinsic_symbol_scope,
                scope_stack,
                *src,
            )),
            SourceRef {
                slice: name.slice,
                line: name.line,
                col: name.col,
            },
        ),
        Expression::Funcall {
            base_expr, args, ..
        } => TypedExpression::Funcall(
            Box::new(transform_expression(
                intrinsic_symbol_scope,
                scope_stack,
                *base_expr,
            )),
            args.into_iter()
                .map(|x| transform_expression(intrinsic_symbol_scope, scope_stack, x.0))
                .collect(),
        ),
        Expression::FuncallSingle(base_expr, arg) => TypedExpression::Funcall(
            Box::new(transform_expression(
                intrinsic_symbol_scope,
                scope_stack,
                *base_expr,
            )),
            vec![transform_expression(
                intrinsic_symbol_scope,
                scope_stack,
                *arg,
            )],
        ),
        Expression::Number(t) => {
            let has_fpart = t.slice.contains('.');
            let has_float_suffix = t.slice.ends_with(['f', 'F']);
            let has_hex_prefix = t.slice.starts_with("0x") || t.slice.starts_with("0X");

            if has_hex_prefix {
                TypedExpression::IntLiteral(t.slice)
            } else if has_float_suffix {
                TypedExpression::FloatLiteral(t.slice)
            } else if has_fpart {
                TypedExpression::NumberLiteral(t.slice)
            } else {
                TypedExpression::IntLiteral(t.slice)
            }
        }
        Expression::Var(v) => {
            match lookup_local_var(v.slice, scope_stack, intrinsic_symbol_scope) {
                Some(LookupLocalVarResult::LocalVar(index, ascendants)) => {
                    TypedExpression::LocalVar {
                        var_index: index,
                        scope_ascending: ascendants,
                    }
                }
                Some(LookupLocalVarResult::Input(x)) => TypedExpression::InputVar(x),
                Some(LookupLocalVarResult::Intrinsic) => TypedExpression::IntrinsicRef(v.slice),
                Some(LookupLocalVarResult::IntrinsicTypeConstructor(t)) => {
                    TypedExpression::IntrinsicTyConRef(t)
                }
                None => panic!("Error: referencing undefined variable: {} {v:?}", v.slice),
            }
        }
        Expression::Tuple(_, elements, _) => TypedExpression::Tuple(
            elements
                .into_iter()
                .map(|x| transform_expression(intrinsic_symbol_scope, scope_stack, x.0))
                .collect(),
        ),
    }
}

enum LookupLocalVarResult {
    LocalVar(usize, usize),
    Input(usize),
    Intrinsic,
    IntrinsicTypeConstructor(IntrinsicType),
}

fn lookup_local_var(
    name: &str,
    scope_stack: &[SymbolScope],
    intrinsic_symbol_scope: &IntrinsicSymbolScope,
) -> Option<LookupLocalVarResult> {
    fn rec(
        name: &str,
        scope_stack: &[SymbolScope],
        ascendants: usize,
    ) -> Option<LookupLocalVarResult> {
        let Some(scope) = scope_stack.last() else {
            return None;
        };

        if let Some(&x) = scope.local_var_index_by_name.get(name) {
            return Some(LookupLocalVarResult::LocalVar(x, ascendants));
        }

        if let Some(&x) = scope.function_input_index_by_name.get(name) {
            return Some(LookupLocalVarResult::Input(x));
        }

        rec(name, &scope_stack[..scope_stack.len() - 1], ascendants + 1)
    }

    if intrinsic_symbol_scope.symbol_type_table.contains_key(name) {
        return Some(LookupLocalVarResult::Intrinsic);
    }

    if let Some(t) = intrinsic_symbol_scope.type_constructors.get(name) {
        return Some(LookupLocalVarResult::IntrinsicTypeConstructor(t.clone()));
    }

    rec(name, scope_stack, 0)
}

#[derive(Debug)]
pub enum ParseErrorKind {
    ExpectedKeyword(&'static str),
    ExpectedKind(TokenKind),
    ExpectedConstExpression,
    ExpectedExpression,
    ExpectedFunctionBodyStarter,
    Outdent(IndentContext),
    ListNotPunctuated(TokenKind),
}

#[derive(Debug)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub line: usize,
    pub col: usize,
}
pub type ParseResult<T> = Result<T, ParseError>;

#[derive(Debug, Clone, Copy)]
pub enum IndentContext {
    Free,
    Exclusive(usize),
    Inclusive(usize),
}
impl IndentContext {
    #[inline]
    pub fn satisfies(self, target: usize) -> bool {
        match self {
            Self::Free => true,
            Self::Exclusive(x) => x < target,
            Self::Inclusive(x) => x <= target,
        }
    }
}

struct ParseState<'s> {
    pub token_list: Vec<Token<'s>>,
    pub token_ptr: usize,
    pub indent_context_stack: Vec<IndentContext>,
}
impl<'s> ParseState<'s> {
    #[inline]
    pub fn current_token(&self) -> Option<&Token<'s>> {
        self.token_list.get(self.token_ptr)
    }

    #[inline]
    pub fn consume_token(&mut self) {
        self.token_ptr += 1;
    }

    #[inline]
    pub fn current_indent_context(&self) -> IndentContext {
        self.indent_context_stack
            .last()
            .copied()
            .unwrap_or(IndentContext::Free)
    }

    #[inline]
    pub fn check_indent_requirements(&self) -> bool {
        self.current_token()
            .map_or(true, |t| self.current_indent_context().satisfies(t.col))
    }

    #[inline]
    pub fn push_indent_context(&mut self, ctx: IndentContext) {
        self.indent_context_stack.push(ctx)
    }

    #[inline]
    pub fn pop_indent_context(&mut self) {
        self.indent_context_stack.pop();
    }

    #[inline]
    pub fn require_in_block_next(&self) -> Result<(), ParseError> {
        if !self.check_indent_requirements() {
            return Err(self.err(ParseErrorKind::Outdent(self.current_indent_context())));
        }

        Ok(())
    }

    #[inline]
    pub fn consume_keyword(&mut self, kw: &'static str) -> Result<&Token<'s>, ParseError> {
        match self.token_list.get(self.token_ptr) {
            Some(x) if x.kind == TokenKind::Keyword && x.slice == kw => {
                self.token_ptr += 1;
                Ok(x)
            }
            t => Err(self.err_on(ParseErrorKind::ExpectedKeyword(kw), t)),
        }
    }

    #[inline]
    pub fn consume_by_kind(&mut self, kind: TokenKind) -> Result<&Token<'s>, ParseError> {
        match self.token_list.get(self.token_ptr) {
            Some(x) if x.kind == kind => {
                self.token_ptr += 1;
                Ok(x)
            }
            t => Err(self.err_on(ParseErrorKind::ExpectedKind(kind), t)),
        }
    }

    #[inline]
    pub fn consume_in_block_keyword(&mut self, kw: &'static str) -> Result<&Token<'s>, ParseError> {
        if !self.check_indent_requirements() {
            return Err(self.err(ParseErrorKind::Outdent(self.current_indent_context())));
        }

        match self.token_list.get(self.token_ptr) {
            Some(x) if x.kind == TokenKind::Keyword && x.slice == kw => {
                self.token_ptr += 1;
                Ok(x)
            }
            t => Err(self.err_on(ParseErrorKind::ExpectedKeyword(kw), t)),
        }
    }

    #[inline]
    pub fn consume_in_block_by_kind(&mut self, kind: TokenKind) -> Result<&Token<'s>, ParseError> {
        if !self.check_indent_requirements() {
            return Err(self.err(ParseErrorKind::Outdent(self.current_indent_context())));
        }

        match self.token_list.get(self.token_ptr) {
            Some(x) if x.kind == kind => {
                self.token_ptr += 1;
                Ok(x)
            }
            t => Err(self.err_on(ParseErrorKind::ExpectedKind(kind), t)),
        }
    }

    #[inline]
    pub fn err_on(&self, kind: ParseErrorKind, tref: Option<&Token<'s>>) -> ParseError {
        ParseError {
            kind,
            line: tref
                .or_else(|| self.token_list.last())
                .map_or(0, |t| t.line),
            col: tref.or_else(|| self.token_list.last()).map_or(0, |t| t.col),
        }
    }

    #[inline]
    pub fn err(&self, kind: ParseErrorKind) -> ParseError {
        self.err_on(kind, self.current_token())
    }
}

#[derive(Debug)]
pub enum ToplevelDeclaration<'s> {
    Struct(StructDeclaration<'s>),
    Function(FunctionDeclaration<'s>),
}
fn parse_toplevel_declaration<'s>(
    state: &mut ParseState<'s>,
) -> ParseResult<ToplevelDeclaration<'s>> {
    match state.current_token() {
        Some(t) if t.kind == TokenKind::Keyword && t.slice == "struct" => {
            parse_struct_declaration(state).map(ToplevelDeclaration::Struct)
        }
        _ => parse_function_declaration(state).map(ToplevelDeclaration::Function),
    }
}

#[derive(Debug, Clone)]
pub struct TypeGenericArgs<'s> {
    pub open_angle_bracket_token: Token<'s>,
    pub args: Vec<(Type<'s>, Option<Token<'s>>)>,
    pub close_angle_bracket_token: Token<'s>,
}
fn parse_type_generic_args<'s>(state: &mut ParseState<'s>) -> ParseResult<TypeGenericArgs<'s>> {
    let open_angle_bracket_token = state.consume_by_kind(TokenKind::OpenAngleBracket)?.clone();

    let mut args = Vec::new();
    let mut can_continue = true;
    while state
        .current_token()
        .is_some_and(|t| t.kind != TokenKind::CloseAngleBracket)
    {
        if !can_continue {
            return Err(state.err(ParseErrorKind::ListNotPunctuated(TokenKind::Comma)));
        }

        let t = parse_type(state)?;
        let opt_comma_token = state.consume_by_kind(TokenKind::Comma).ok().cloned();

        can_continue = opt_comma_token.is_some();
        args.push((t, opt_comma_token));
    }

    let close_angle_bracket_token = state.consume_by_kind(TokenKind::CloseAngleBracket)?.clone();

    Ok(TypeGenericArgs {
        open_angle_bracket_token,
        args,
        close_angle_bracket_token,
    })
}

#[derive(Debug, Clone)]
pub struct Type<'s> {
    pub name_token: Token<'s>,
    pub generic_args: Option<TypeGenericArgs<'s>>,
}
fn parse_type<'s>(state: &mut ParseState<'s>) -> ParseResult<Type<'s>> {
    let name_token = state.consume_by_kind(TokenKind::Identifier)?.clone();
    let generic_args = match state.current_token() {
        Some(t) if t.kind == TokenKind::OpenAngleBracket => Some(parse_type_generic_args(state)?),
        _ => None,
    };

    Ok(Type {
        name_token,
        generic_args,
    })
}

#[derive(Debug, Clone)]
pub enum ConstExpression<'s> {
    Number(Token<'s>),
}
fn parse_const_expression<'s>(state: &mut ParseState<'s>) -> ParseResult<ConstExpression<'s>> {
    match state.current_token() {
        Some(t) if t.kind == TokenKind::Number => {
            let t = t.clone();
            state.token_ptr += 1;
            Ok(ConstExpression::Number(t))
        }
        t => Err(state.err_on(ParseErrorKind::ExpectedConstExpression, t)),
    }
}
fn lookahead_const_expression(state: &ParseState) -> bool {
    state
        .current_token()
        .is_some_and(|t| t.kind == TokenKind::Number)
}

#[derive(Debug, Clone)]
pub enum AttributeArg<'s> {
    Single(ConstExpression<'s>),
    Multiple {
        open_parenthese_token: Token<'s>,
        arg_list: Vec<(ConstExpression<'s>, Option<Token<'s>>)>,
        close_parenthese_token: Token<'s>,
    },
}
fn parse_attribute_arg<'s>(state: &mut ParseState<'s>) -> ParseResult<AttributeArg<'s>> {
    match state.current_token() {
        Some(t) if t.kind == TokenKind::OpenParenthese => {
            let open_parenthese_token = state.consume_by_kind(TokenKind::OpenParenthese)?.clone();
            let mut arg_list = Vec::new();
            let mut can_continue = true;
            while state
                .current_token()
                .is_some_and(|t| t.kind != TokenKind::CloseParenthese)
            {
                if !can_continue {
                    return Err(state.err(ParseErrorKind::ListNotPunctuated(TokenKind::Comma)));
                }

                let arg = parse_const_expression(state)?;
                let opt_comma_token = state.consume_by_kind(TokenKind::Comma).ok().cloned();
                can_continue = opt_comma_token.is_some();
                arg_list.push((arg, opt_comma_token));
            }
            let close_parenthese_token = state.consume_by_kind(TokenKind::CloseParenthese)?.clone();

            Ok(AttributeArg::Multiple {
                open_parenthese_token,
                arg_list,
                close_parenthese_token,
            })
        }
        _ => parse_const_expression(state).map(AttributeArg::Single),
    }
}
fn lookahead_attribute_arg(state: &ParseState) -> bool {
    state
        .current_token()
        .is_some_and(|t| t.kind == TokenKind::OpenParenthese)
        || lookahead_const_expression(state)
}

#[derive(Debug, Clone)]
pub struct Attribute<'s> {
    pub name_token: Token<'s>,
    pub arg: Option<AttributeArg<'s>>,
}
fn parse_attribute<'s>(state: &mut ParseState<'s>) -> ParseResult<Attribute<'s>> {
    let name_token = state.consume_by_kind(TokenKind::Identifier)?.clone();
    let arg = if lookahead_attribute_arg(state) {
        Some(parse_attribute_arg(state)?)
    } else {
        None
    };

    Ok(Attribute { name_token, arg })
}

#[derive(Debug)]
pub struct AttributeList<'s> {
    pub open_bracket_token: Token<'s>,
    pub attribute_list: Vec<(Attribute<'s>, Option<Token<'s>>)>,
    pub close_bracket_token: Token<'s>,
}
fn parse_attribute_list<'s>(state: &mut ParseState<'s>) -> ParseResult<AttributeList<'s>> {
    let open_bracket_token = state.consume_by_kind(TokenKind::OpenBracket)?.clone();
    let mut attribute_list = Vec::new();
    let mut can_continue = true;
    while state
        .current_token()
        .is_some_and(|t| t.kind != TokenKind::CloseBracket)
    {
        if !can_continue {
            return Err(state.err(ParseErrorKind::ListNotPunctuated(TokenKind::Comma)));
        }

        let a = parse_attribute(state)?;
        let opt_comma_token = state.consume_by_kind(TokenKind::Comma).ok().cloned();
        can_continue = opt_comma_token.is_some();
        attribute_list.push((a, opt_comma_token));
    }
    let close_bracket_token = state.consume_by_kind(TokenKind::CloseBracket)?.clone();

    Ok(AttributeList {
        open_bracket_token,
        attribute_list,
        close_bracket_token,
    })
}

#[derive(Debug)]
pub struct StructMember<'s> {
    pub attribute_lists: Vec<AttributeList<'s>>,
    pub name_token: Token<'s>,
    pub colon_token: Token<'s>,
    pub ty: Type<'s>,
}
fn parse_struct_member<'s>(state: &mut ParseState<'s>) -> ParseResult<StructMember<'s>> {
    let mut attribute_lists = Vec::new();
    while state.check_indent_requirements()
        && state
            .current_token()
            .is_some_and(|t| t.kind == TokenKind::OpenBracket)
    {
        attribute_lists.push(parse_attribute_list(state)?);
    }
    let name_token = state
        .consume_in_block_by_kind(TokenKind::Identifier)?
        .clone();
    let colon_token = state.consume_by_kind(TokenKind::Colon)?.clone();
    let ty = parse_type(state)?;

    Ok(StructMember {
        attribute_lists,
        name_token,
        colon_token,
        ty,
    })
}

#[derive(Debug)]
pub struct StructDeclaration<'s> {
    pub decl_token: Token<'s>,
    pub name_token: Token<'s>,
    pub with_token: Option<Token<'s>>,
    pub member_list: Vec<StructMember<'s>>,
}
fn parse_struct_declaration<'s>(
    state: &mut ParseState<'s>,
) -> Result<StructDeclaration<'s>, ParseError> {
    let decl_token = state.consume_keyword("struct")?.clone();
    let name_token = state.consume_by_kind(TokenKind::Identifier)?.clone();
    let Some(with_token) = state.consume_keyword("with").ok().cloned() else {
        return Ok(StructDeclaration {
            decl_token,
            name_token,
            with_token: None,
            member_list: Vec::new(),
        });
    };

    state.push_indent_context(IndentContext::Exclusive(with_token.line_indent));
    let mut member_list = Vec::new();
    while state.check_indent_requirements() {
        member_list.push(parse_struct_member(state)?);
    }
    state.pop_indent_context();

    Ok(StructDeclaration {
        decl_token,
        name_token,
        with_token: Some(with_token),
        member_list,
    })
}

#[derive(Debug)]
pub enum FunctionDeclarationInputArguments<'s> {
    Single {
        attribute_lists: Vec<AttributeList<'s>>,
        varname_token: Token<'s>,
        colon_token: Token<'s>,
        ty: Type<'s>,
    },
    Multiple {
        open_parenthese_token: Token<'s>,
        args: Vec<(
            Vec<AttributeList<'s>>,
            Token<'s>,
            Token<'s>,
            Type<'s>,
            Option<Token<'s>>,
        )>,
        close_parenthese_token: Token<'s>,
    },
}
fn parse_function_declaration_input_arguments<'s>(
    state: &mut ParseState<'s>,
) -> ParseResult<FunctionDeclarationInputArguments<'s>> {
    match state.current_token() {
        Some(t) if t.kind == TokenKind::OpenParenthese => {
            let open_parenthese_token = t.clone();
            state.consume_token();

            let mut args = Vec::new();
            let mut can_continue = true;
            while state
                .current_token()
                .is_some_and(|t| t.kind != TokenKind::CloseParenthese)
            {
                if !can_continue {
                    return Err(state.err(ParseErrorKind::ListNotPunctuated(TokenKind::Comma)));
                }

                let mut attribute_lists = Vec::new();
                while state
                    .current_token()
                    .is_some_and(|t| t.kind == TokenKind::OpenBracket)
                {
                    attribute_lists.push(parse_attribute_list(state)?);
                }

                let name_token = state.consume_by_kind(TokenKind::Identifier)?.clone();
                let colon_token = state.consume_by_kind(TokenKind::Colon)?.clone();
                let ty = parse_type(state)?;
                let opt_comma_token = state.consume_by_kind(TokenKind::Comma).ok().cloned();

                can_continue = opt_comma_token.is_some();
                args.push((
                    attribute_lists,
                    name_token,
                    colon_token,
                    ty,
                    opt_comma_token,
                ));
            }

            let close_parenthese_token = state.consume_by_kind(TokenKind::CloseParenthese)?.clone();

            Ok(FunctionDeclarationInputArguments::Multiple {
                open_parenthese_token,
                args,
                close_parenthese_token,
            })
        }
        _ => {
            let mut attribute_lists = Vec::new();
            while state
                .current_token()
                .is_some_and(|t| t.kind == TokenKind::OpenBracket)
            {
                attribute_lists.push(parse_attribute_list(state)?);
            }

            let varname_token = state.consume_by_kind(TokenKind::Identifier)?.clone();
            let colon_token = state.consume_by_kind(TokenKind::Colon)?.clone();
            let ty = parse_type(state)?;

            Ok(FunctionDeclarationInputArguments::Single {
                attribute_lists,
                varname_token,
                colon_token,
                ty,
            })
        }
    }
}

#[derive(Debug)]
pub enum FunctionDeclarationOutput<'s> {
    Single {
        attribute_lists: Vec<AttributeList<'s>>,
        ty: Type<'s>,
    },
    Tupled {
        open_parenthese_token: Token<'s>,
        elements: Vec<(Vec<AttributeList<'s>>, Type<'s>, Option<Token<'s>>)>,
        close_parenthese_token: Token<'s>,
    },
}
fn parse_function_declaration_output<'s>(
    state: &mut ParseState<'s>,
) -> ParseResult<FunctionDeclarationOutput<'s>> {
    match state.current_token() {
        Some(t) if t.kind == TokenKind::OpenParenthese => {
            let open_parenthese_token = t.clone();
            state.consume_token();

            let mut elements = Vec::new();
            let mut can_continue = true;
            while state
                .current_token()
                .is_some_and(|t| t.kind != TokenKind::CloseParenthese)
            {
                if !can_continue {
                    return Err(state.err(ParseErrorKind::ListNotPunctuated(TokenKind::Comma)));
                }

                let mut attribute_lists = Vec::new();
                while state
                    .current_token()
                    .is_some_and(|t| t.kind == TokenKind::OpenBracket)
                {
                    attribute_lists.push(parse_attribute_list(state)?);
                }

                let ty = parse_type(state)?;
                let opt_comma_token = state.consume_by_kind(TokenKind::Comma).ok().cloned();

                can_continue = opt_comma_token.is_some();
                elements.push((attribute_lists, ty, opt_comma_token));
            }

            let close_parenthese_token = state.consume_by_kind(TokenKind::CloseParenthese)?.clone();

            Ok(FunctionDeclarationOutput::Tupled {
                open_parenthese_token,
                elements,
                close_parenthese_token,
            })
        }
        _ => {
            let mut attribute_lists = Vec::new();
            while state
                .current_token()
                .is_some_and(|t| t.kind == TokenKind::OpenBracket)
            {
                attribute_lists.push(parse_attribute_list(state)?);
            }

            let ty = parse_type(state)?;

            Ok(FunctionDeclarationOutput::Single {
                attribute_lists,
                ty,
            })
        }
    }
}

#[derive(Debug)]
pub struct FunctionDeclaration<'s> {
    pub attribute_lists: Vec<AttributeList<'s>>,
    pub fname_token: Token<'s>,
    pub input_args: FunctionDeclarationInputArguments<'s>,
    pub arrow_to_right_token: Option<Token<'s>>,
    pub output: Option<FunctionDeclarationOutput<'s>>,
    pub body_starter_token: Token<'s>,
    pub body: Expression<'s>,
}
fn parse_function_declaration<'s>(
    state: &mut ParseState<'s>,
) -> ParseResult<FunctionDeclaration<'s>> {
    let mut attribute_lists = Vec::new();
    while state
        .current_token()
        .is_some_and(|t| t.kind == TokenKind::OpenBracket)
    {
        attribute_lists.push(parse_attribute_list(state)?);
    }

    let fname_token = state.consume_by_kind(TokenKind::Identifier)?.clone();
    let input_args = parse_function_declaration_input_arguments(state)?;
    let arrow_to_right_token = state.consume_by_kind(TokenKind::ArrowToRight).ok().cloned();
    let output = if arrow_to_right_token.is_some() {
        Some(parse_function_declaration_output(state)?)
    } else {
        None
    };
    let body_starter_token = match state.current_token() {
        Some(t) if t.kind == TokenKind::Eq || (t.kind == TokenKind::Keyword && t.slice == "do") => {
            let tok = t.clone();
            state.consume_token();

            tok
        }
        t => return Err(state.err_on(ParseErrorKind::ExpectedFunctionBodyStarter, t)),
    };

    state.push_indent_context(IndentContext::Exclusive(body_starter_token.line_indent));
    let body = parse_block(state)?;
    state.pop_indent_context();

    Ok(FunctionDeclaration {
        attribute_lists,
        fname_token,
        input_args,
        arrow_to_right_token,
        output,
        body_starter_token,
        body,
    })
}

#[derive(Debug)]
pub enum Statement<'s> {
    Let {
        let_token: Token<'s>,
        varname_token: Token<'s>,
        eq_token: Token<'s>,
        expr: Expression<'s>,
    },
}

#[derive(Debug)]
pub enum Expression<'s> {
    Blocked(Vec<Statement<'s>>, Box<Expression<'s>>),
    Lifted(Token<'s>, Box<Expression<'s>>, Token<'s>),
    Binary(Box<Expression<'s>>, Token<'s>, Box<Expression<'s>>),
    Prefixed(Token<'s>, Box<Expression<'s>>),
    MemberRef(Box<Expression<'s>>, Token<'s>, Token<'s>),
    Funcall {
        base_expr: Box<Expression<'s>>,
        open_parenthese_token: Token<'s>,
        args: Vec<(Expression<'s>, Option<Token<'s>>)>,
        close_parenthese_token: Token<'s>,
    },
    FuncallSingle(Box<Expression<'s>>, Box<Expression<'s>>),
    Number(Token<'s>),
    Var(Token<'s>),
    Tuple(
        Token<'s>,
        Vec<(Expression<'s>, Option<Token<'s>>)>,
        Token<'s>,
    ),
    If {
        if_token: Token<'s>,
        condition: Box<Expression<'s>>,
        then_token: Token<'s>,
        then_expr: Box<Expression<'s>>,
        else_token: Option<Token<'s>>,
        else_expr: Option<Box<Expression<'s>>>,
    },
}
fn parse_block<'s>(state: &mut ParseState<'s>) -> ParseResult<Expression<'s>> {
    let mut statements = Vec::new();
    loop {
        state.require_in_block_next()?;

        match state.current_token() {
            Some(t) if t.kind == TokenKind::Keyword && t.slice == "let" => {
                let let_token = t.clone();
                state.consume_token();
                let varname_token = state.consume_by_kind(TokenKind::Identifier)?.clone();
                let eq_token = state.consume_by_kind(TokenKind::Eq)?.clone();
                state.push_indent_context(IndentContext::Exclusive(eq_token.line_indent));
                let expr = parse_block(state)?;
                state.pop_indent_context();

                statements.push(Statement::Let {
                    let_token,
                    varname_token,
                    eq_token,
                    expr,
                })
            }
            _ => break,
        }
    }

    let final_expr = parse_expression(state)?;
    Ok(Expression::Blocked(statements, Box::new(final_expr)))
}
fn parse_expression<'s>(state: &mut ParseState<'s>) -> ParseResult<Expression<'s>> {
    parse_expression_if(state)
}
fn parse_expression_if<'s>(state: &mut ParseState<'s>) -> ParseResult<Expression<'s>> {
    let Some(if_token) = state.consume_keyword("if").ok().cloned() else {
        return parse_expression_logical_ops(state);
    };
    state.push_indent_context(IndentContext::Inclusive(if_token.line_indent));

    let condition = parse_expression_logical_ops(state)?;
    let then_token = state.consume_in_block_keyword("then")?.clone();
    state.push_indent_context(IndentContext::Exclusive(then_token.line_indent));
    let then_expr = parse_block(state)?;
    state.pop_indent_context();
    let else_token = state.consume_in_block_keyword("else").ok().cloned();
    let else_expr = if let Some(ref e) = else_token {
        state.push_indent_context(IndentContext::Exclusive(e.line_indent));
        let x = parse_block(state)?;
        state.pop_indent_context();
        Some(x)
    } else {
        None
    };

    state.pop_indent_context();

    Ok(Expression::If {
        if_token,
        condition: Box::new(condition),
        then_token,
        then_expr: Box::new(then_expr),
        else_token,
        else_expr: else_expr.map(Box::new),
    })
}
fn parse_expression_logical_ops<'s>(state: &mut ParseState<'s>) -> ParseResult<Expression<'s>> {
    let mut expr = parse_expression_compare_ops(state)?;

    loop {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Op && t.slice == "||" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_compare_ops(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "&&" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_compare_ops(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            _ => break Ok(expr),
        }
    }
}
fn parse_expression_compare_ops<'s>(state: &mut ParseState<'s>) -> ParseResult<Expression<'s>> {
    let mut expr = parse_expression_bitwise_ops(state)?;

    loop {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Op && t.slice == "==" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_bitwise_ops(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "!=" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_bitwise_ops(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "<=" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_bitwise_ops(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == ">=" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_bitwise_ops(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::OpenAngleBracket => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_bitwise_ops(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::CloseAngleBracket => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_bitwise_ops(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            _ => break Ok(expr),
        }
    }
}
fn parse_expression_bitwise_ops<'s>(state: &mut ParseState<'s>) -> ParseResult<Expression<'s>> {
    let mut expr = parse_expression_arithmetic_ops_1(state)?;

    loop {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Op && t.slice == "|" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_arithmetic_ops_1(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "&" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_arithmetic_ops_1(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "^" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_arithmetic_ops_1(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            _ => break Ok(expr),
        }
    }
}
fn parse_expression_arithmetic_ops_1<'s>(
    state: &mut ParseState<'s>,
) -> ParseResult<Expression<'s>> {
    let mut expr = parse_expression_arithmetic_ops_2(state)?;

    loop {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Op && t.slice == "+" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_arithmetic_ops_2(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "-" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_arithmetic_ops_2(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            _ => break Ok(expr),
        }
    }
}
fn parse_expression_arithmetic_ops_2<'s>(
    state: &mut ParseState<'s>,
) -> ParseResult<Expression<'s>> {
    let mut expr = parse_expression_prefixed_ops(state)?;

    loop {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Op && t.slice == "*" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_prefixed_ops(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "/" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_prefixed_ops(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "%" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_prefixed_ops(state)?;
                expr = Expression::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            _ => break Ok(expr),
        }
    }
}
fn parse_expression_prefixed_ops<'s>(state: &mut ParseState<'s>) -> ParseResult<Expression<'s>> {
    match state.current_token() {
        Some(t) if t.kind == TokenKind::Op && t.slice == "+" => {
            let op_token = t.clone();
            state.consume_token();
            let expr = parse_expression_prefixed_ops(state)?;

            Ok(Expression::Prefixed(op_token, Box::new(expr)))
        }
        Some(t) if t.kind == TokenKind::Op && t.slice == "-" => {
            let op_token = t.clone();
            state.consume_token();
            let expr = parse_expression_prefixed_ops(state)?;

            Ok(Expression::Prefixed(op_token, Box::new(expr)))
        }
        Some(t) if t.kind == TokenKind::Op && t.slice == "~" => {
            let op_token = t.clone();
            state.consume_token();
            let expr = parse_expression_prefixed_ops(state)?;

            Ok(Expression::Prefixed(op_token, Box::new(expr)))
        }
        Some(t) if t.kind == TokenKind::Op && t.slice == "!" => {
            let op_token = t.clone();
            state.consume_token();
            let expr = parse_expression_prefixed_ops(state)?;

            Ok(Expression::Prefixed(op_token, Box::new(expr)))
        }
        _ => parse_expression_suffixed_ops(state),
    }
}
fn parse_expression_suffixed_ops<'s>(state: &mut ParseState<'s>) -> ParseResult<Expression<'s>> {
    let mut expr = parse_expression_prime(state)?;

    loop {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Op && t.slice == "." => {
                let dot_token = t.clone();
                state.consume_token();
                let name_token = state.consume_by_kind(TokenKind::Identifier)?.clone();

                expr = Expression::MemberRef(Box::new(expr), dot_token, name_token);
            }
            Some(t) if t.kind == TokenKind::OpenParenthese => {
                let open_parenthese_token = t.clone();
                state.consume_token();
                let mut args = Vec::new();
                let mut can_continue = true;
                while state
                    .current_token()
                    .is_some_and(|t| t.kind != TokenKind::CloseParenthese)
                {
                    if !can_continue {
                        return Err(state.err(ParseErrorKind::ListNotPunctuated(TokenKind::Comma)));
                    }

                    let arg = parse_expression(state)?;
                    let opt_comma_token = state.consume_by_kind(TokenKind::Comma).ok().cloned();
                    can_continue = opt_comma_token.is_some();
                    args.push((arg, opt_comma_token));
                }
                let close_parenthese_token =
                    state.consume_by_kind(TokenKind::CloseParenthese)?.clone();

                expr = Expression::Funcall {
                    base_expr: Box::new(expr),
                    open_parenthese_token,
                    args,
                    close_parenthese_token,
                }
            }
            _ => {
                break Ok(
                    if let Ok(arg) = parse_expression_funcall_single_arg(state) {
                        Expression::FuncallSingle(Box::new(expr), Box::new(arg))
                    } else {
                        expr
                    },
                );
            }
        }
    }
}
fn parse_expression_funcall_single_arg<'s>(
    state: &mut ParseState<'s>,
) -> ParseResult<Expression<'s>> {
    let mut expr = parse_expression_prime(state)?;

    loop {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Op && t.slice == "." => {
                let dot_token = t.clone();
                state.consume_token();
                let name_token = state.consume_by_kind(TokenKind::Identifier)?.clone();

                expr = Expression::MemberRef(Box::new(expr), dot_token, name_token);
            }
            Some(t) if t.kind == TokenKind::OpenParenthese => {
                let open_parenthese_token = t.clone();
                state.consume_token();
                let mut args = Vec::new();
                let mut can_continue = true;
                while state
                    .current_token()
                    .is_some_and(|t| t.kind != TokenKind::CloseParenthese)
                {
                    if !can_continue {
                        return Err(state.err(ParseErrorKind::ListNotPunctuated(TokenKind::Comma)));
                    }

                    let arg = parse_expression(state)?;
                    let opt_comma_token = state.consume_by_kind(TokenKind::Comma).ok().cloned();
                    can_continue = opt_comma_token.is_some();
                    args.push((arg, opt_comma_token));
                }
                let close_parenthese_token =
                    state.consume_by_kind(TokenKind::CloseParenthese)?.clone();

                expr = Expression::Funcall {
                    base_expr: Box::new(expr),
                    open_parenthese_token,
                    args,
                    close_parenthese_token,
                }
            }
            _ => break Ok(expr),
        }
    }
}
fn parse_expression_prime<'s>(state: &mut ParseState<'s>) -> ParseResult<Expression<'s>> {
    match state.current_token() {
        Some(t) if t.kind == TokenKind::Number => {
            let tok = t.clone();
            state.consume_token();

            Ok(Expression::Number(tok))
        }
        Some(t) if t.kind == TokenKind::Identifier => {
            let tok = t.clone();
            state.consume_token();

            Ok(Expression::Var(tok))
        }
        Some(t) if t.kind == TokenKind::OpenParenthese => {
            let open_parenthese_token = t.clone();
            state.consume_token();

            let mut expressions = Vec::new();
            let mut can_continue = true;
            while state
                .current_token()
                .is_some_and(|t| t.kind != TokenKind::CloseParenthese)
            {
                if !can_continue {
                    return Err(state.err(ParseErrorKind::ListNotPunctuated(TokenKind::Comma)));
                }

                let e = parse_expression(state)?;
                let opt_comma_token = state.consume_by_kind(TokenKind::Comma).ok().cloned();
                can_continue = opt_comma_token.is_some();
                expressions.push((e, opt_comma_token));
            }

            let close_parenthese_token = state.consume_by_kind(TokenKind::CloseParenthese)?.clone();

            if expressions.len() == 1 && expressions.last().is_some_and(|(_, t)| t.is_none()) {
                // single and not terminated by ",": lifted expression
                Ok(Expression::Lifted(
                    open_parenthese_token,
                    Box::new(expressions.pop().unwrap().0),
                    close_parenthese_token,
                ))
            } else {
                Ok(Expression::Tuple(
                    open_parenthese_token,
                    expressions,
                    close_parenthese_token,
                ))
            }
        }
        t => Err(state.err_on(ParseErrorKind::ExpectedExpression, t)),
    }
}

#[derive(Debug)]
pub enum TokenizerErrorKind {
    IncompleteHexLiteral,
}

#[derive(Debug)]
pub struct TokenizerError {
    pub kind: TokenizerErrorKind,
    pub line: usize,
    pub col: usize,
}

struct Tokenizer<'s> {
    pub source: &'s str,
    pub line: usize,
    pub col: usize,
    pub current_line_indent: usize,
}
impl<'s> Tokenizer<'s> {
    pub fn populate_line_indent(&mut self) {
        let (line_indent_chars, line_indent_bytes) = self
            .source
            .chars()
            .take_while(|&c| c.is_whitespace() && c != '\n')
            .fold((0, 0), |(a, b), c| (a + 1, b + c.len_utf8()));
        self.current_line_indent = line_indent_chars;
        self.col = line_indent_chars;
        self.source = &self.source[line_indent_bytes..];
    }

    pub fn next_token(&mut self) -> Result<Option<Token<'s>>, TokenizerError> {
        let (head_space_chars, head_space_bytes) = self
            .source
            .chars()
            .take_while(|&c| c.is_whitespace() && c != '\n')
            .fold((0, 0), |(a, b), c| (a + 1, b + c.len_utf8()));
        self.col += head_space_chars;
        self.source = &self.source[head_space_bytes..];

        while self.source.starts_with('\n') {
            let lf_count = self
                .source
                .chars()
                .take_while(|&c| c == '\n' || c == '\r')
                .fold(0, |a, c| a + (if c == '\n' { 1 } else { 0 }));
            self.line += lf_count;
            self.source = self.source.trim_start_matches(|c| c == '\n' || c == '\r');
            self.populate_line_indent();
        }

        if self.source.is_empty() {
            return Ok(None);
        }

        if self.source.starts_with('#') {
            // line comment
            self.source = self.source.trim_start_matches(|c| c != '\n');
            return self.next_token();
        }

        let double_byte_tok = if self.source.as_bytes().len() >= 2 {
            match &self.source.as_bytes()[..2] {
                b"->" => Some(TokenKind::ArrowToRight),
                b"==" | b"!=" | b"<=" | b">=" | b"&&" | b"||" => Some(TokenKind::Op),
                _ => None,
            }
        } else {
            None
        };

        if let Some(k) = double_byte_tok {
            let tk = Token {
                slice: &self.source[..2],
                kind: k,
                line: self.line,
                col: self.col,
                line_indent: self.current_line_indent,
            };
            self.source = &self.source[2..];
            self.col += 2;
            return Ok(Some(tk));
        }

        let single_tok = match self.source.as_bytes()[0] {
            b'[' => Some(TokenKind::OpenBracket),
            b']' => Some(TokenKind::CloseBracket),
            b'(' => Some(TokenKind::OpenParenthese),
            b')' => Some(TokenKind::CloseParenthese),
            b'<' => Some(TokenKind::OpenAngleBracket),
            b'>' => Some(TokenKind::CloseAngleBracket),
            b',' => Some(TokenKind::Comma),
            b':' => Some(TokenKind::Colon),
            b'=' => Some(TokenKind::Eq),
            b'+' | b'-' | b'*' | b'/' | b'%' | b'&' | b'|' | b'^' | b'~' | b'!' | b'.' => {
                Some(TokenKind::Op)
            }
            _ => None,
        };
        if let Some(k) = single_tok {
            let tk = Token {
                slice: &self.source[..1],
                kind: k,
                line: self.line,
                col: self.col,
                line_indent: self.current_line_indent,
            };
            self.source = &self.source[1..];
            self.col += 1;
            return Ok(Some(tk));
        }

        if self.source.starts_with("0x") || self.source.starts_with("0X") {
            // hexlit
            let hexpart_count = self.source[2..]
                .chars()
                .take_while(|c| {
                    ('0'..='9').contains(c) || ('a'..='f').contains(c) || ('A'..='F').contains(c)
                })
                .count();
            if hexpart_count == 0 {
                return Err(TokenizerError {
                    kind: TokenizerErrorKind::IncompleteHexLiteral,
                    line: self.line,
                    col: self.col,
                });
            }

            let tk = Token {
                slice: &self.source[..2 + hexpart_count],
                kind: TokenKind::Number,
                line: self.line,
                col: self.col,
                line_indent: self.current_line_indent,
            };
            self.source = &self.source[2 + hexpart_count..];
            self.col += 2 + hexpart_count;
            return Ok(Some(tk));
        }

        if self.source.starts_with(|c: char| ('0'..='9').contains(&c)) {
            let ipart_count = self
                .source
                .chars()
                .take_while(|c| ('0'..='9').contains(c))
                .count();
            let extended_count = if self.source[ipart_count..].starts_with('.') {
                1 + self.source[ipart_count + 1..]
                    .chars()
                    .take_while(|c| ('0'..='9').contains(c))
                    .count()
            } else {
                0
            };

            let has_float_suffix =
                self.source[ipart_count + extended_count..].starts_with(['f', 'F']);
            let total_length = ipart_count + extended_count + if has_float_suffix { 1 } else { 0 };

            let tk = Token {
                slice: &self.source[..total_length],
                kind: TokenKind::Number,
                line: self.line,
                col: self.col,
                line_indent: self.current_line_indent,
            };
            self.source = &self.source[total_length..];
            self.col += total_length;
            return Ok(Some(tk));
        }

        let (ident_char_count, ident_byte_count) = self
            .source
            .chars()
            .take_while(|&c| !"=!\"#%&'()[]?><.,;:@=~-^|\\ \t\r\n".contains(c))
            .fold((0, 0), |(a, b), c| (a + 1, b + c.len_utf8()));
        assert!(
            ident_byte_count > 0,
            "empty identifier token(src: {}...)",
            &self.source[..8]
        );
        let tk = Token {
            slice: &self.source[..ident_byte_count],
            kind: match &self.source[..ident_byte_count] {
                "struct" | "with" | "if" | "else" | "then" | "do" | "let" => TokenKind::Keyword,
                _ => TokenKind::Identifier,
            },
            line: self.line,
            col: self.col,
            line_indent: self.current_line_indent,
        };
        self.source = &self.source[ident_byte_count..];
        self.col += ident_char_count;
        Ok(Some(tk))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Identifier,
    Keyword,
    Op,
    Number,
    OpenBracket,
    CloseBracket,
    OpenParenthese,
    CloseParenthese,
    OpenAngleBracket,
    CloseAngleBracket,
    Comma,
    Colon,
    ArrowToRight,
    Eq,
}

#[derive(Debug, Clone)]
pub struct Token<'s> {
    pub slice: &'s str,
    pub kind: TokenKind,
    pub line: usize,
    pub col: usize,
    pub line_indent: usize,
}
