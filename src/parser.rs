#![allow(dead_code)]

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

pub struct ParseStateSavepoint {
    token_ptr: usize,
    indent_context_stack: Vec<IndentContext>,
}

pub struct ParseState<'s> {
    pub token_list: Vec<Token<'s>>,
    pub token_ptr: usize,
    pub indent_context_stack: Vec<IndentContext>,
}
impl<'s> ParseState<'s> {
    pub fn new(token_list: Vec<Token<'s>>) -> Self {
        Self {
            token_list,
            token_ptr: 0,
            indent_context_stack: Vec::new(),
        }
    }

    #[inline]
    pub fn save(&self) -> ParseStateSavepoint {
        ParseStateSavepoint {
            token_ptr: self.token_ptr,
            indent_context_stack: self.indent_context_stack.clone(),
        }
    }

    #[inline]
    pub fn restore(&mut self, savepoint: ParseStateSavepoint) {
        self.token_ptr = savepoint.token_ptr;
        self.indent_context_stack = savepoint.indent_context_stack;
    }

    #[inline]
    pub fn current_token(&self) -> Option<&Token<'s>> {
        self.token_list.get(self.token_ptr)
    }

    #[inline]
    pub fn current_token_in_block(&self) -> Option<&Token<'s>> {
        if !self.check_indent_requirements() {
            return None;
        }

        self.current_token()
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
pub struct CompilationUnit<'s> {
    pub module: Option<ModuleDeclaration<'s>>,
    pub declarations: Vec<ToplevelDeclaration<'s>>,
}
impl<'s> CompilationUnit<'s> {
    pub fn parse(mut state: ParseState<'s>) -> ParseResult<Self> {
        let module = if ModuleDeclaration::lookahead(&mut state) {
            Some(ModuleDeclaration::parse(&mut state)?)
        } else {
            None
        };

        let mut declarations = Vec::new();
        while state.current_token().is_some() {
            declarations.push(ToplevelDeclaration::parse(&mut state)?);
        }

        Ok(Self {
            module,
            declarations,
        })
    }
}

#[derive(Debug)]
pub struct PathSyntax<'s> {
    pub root_token: Token<'s>,
    pub tails: Vec<(Token<'s>, Token<'s>)>,
}
impl<'s> PathSyntax<'s> {
    fn parse(state: &mut ParseState<'s>) -> ParseResult<Self> {
        let root_token = state.consume_by_kind(TokenKind::Identifier)?.clone();

        let mut tails = Vec::new();
        while let Some(t) = state
            .current_token()
            .filter(|t| t.kind == TokenKind::Op && t.slice == ".")
        {
            let dot_token = t.clone();
            state.consume_token();
            let ns_token = state.consume_by_kind(TokenKind::Identifier)?.clone();

            tails.push((dot_token, ns_token));
        }

        Ok(Self { root_token, tails })
    }
}

#[derive(Debug)]
pub struct ModuleDeclaration<'s> {
    module_token: Token<'s>,
    path: PathSyntax<'s>,
}
impl<'s> ModuleDeclaration<'s> {
    fn parse(state: &mut ParseState<'s>) -> ParseResult<Self> {
        let module_token = state.consume_keyword("module")?.clone();
        let path = PathSyntax::parse(state)?;

        Ok(Self { module_token, path })
    }

    fn lookahead(state: &ParseState<'s>) -> bool {
        state
            .current_token()
            .is_some_and(|t| t.kind == TokenKind::Keyword && t.slice == "module")
    }
}

#[derive(Debug)]
pub enum ToplevelDeclaration<'s> {
    Struct(StructDeclaration<'s>),
    Function(FunctionDeclaration<'s>),
}
impl<'s> ToplevelDeclaration<'s> {
    fn parse(state: &mut ParseState<'s>) -> ParseResult<Self> {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Keyword && t.slice == "struct" => {
                parse_struct_declaration(state).map(ToplevelDeclaration::Struct)
            }
            _ => parse_function_declaration(state).map(ToplevelDeclaration::Function),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TypeGenericArgsSyntax<'s> {
    pub open_angle_bracket_token: Token<'s>,
    pub args: Vec<(TypeSyntax<'s>, Option<Token<'s>>)>,
    pub close_angle_bracket_token: Token<'s>,
}
fn parse_type_generic_args<'s>(
    state: &mut ParseState<'s>,
) -> ParseResult<TypeGenericArgsSyntax<'s>> {
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

    Ok(TypeGenericArgsSyntax {
        open_angle_bracket_token,
        args,
        close_angle_bracket_token,
    })
}

#[derive(Debug, Clone)]
pub enum TypeSyntax<'s> {
    Simple {
        name_token: Token<'s>,
        generic_args: Option<TypeGenericArgsSyntax<'s>>,
    },
    Array(
        Box<TypeSyntax<'s>>,
        Token<'s>,
        ExpressionNode<'s>,
        Token<'s>,
    ),
    Ref {
        ampasand_token: Token<'s>,
        mut_token: Option<Token<'s>>,
        decorator_token: Option<Token<'s>>,
        pointee_type: Box<TypeSyntax<'s>>,
    },
}
fn parse_type<'s>(state: &mut ParseState<'s>) -> ParseResult<TypeSyntax<'s>> {
    if let Some(t) = state
        .current_token()
        .filter(|t| t.kind == TokenKind::Op && t.slice == "&")
    {
        // ref
        let ampasand_token = t.clone();
        state.consume_token();

        let mut_token = state.consume_keyword("mut").ok().cloned();
        let decorator_token = match state.current_token_in_block() {
            Some(t) if t.kind == TokenKind::Keyword && t.slice == "uniform" => {
                let tok = t.clone();
                state.consume_token();
                Some(tok)
            }
            _ => None,
        };
        let pointee_type = parse_type(state)?;

        return Ok(TypeSyntax::Ref {
            ampasand_token,
            mut_token,
            decorator_token,
            pointee_type: Box::new(pointee_type),
        });
    }

    let name_token = state.consume_by_kind(TokenKind::Identifier)?.clone();
    let generic_args = match state.current_token() {
        Some(t) if t.kind == TokenKind::OpenAngleBracket => Some(parse_type_generic_args(state)?),
        _ => None,
    };

    let mut node = TypeSyntax::Simple {
        name_token,
        generic_args,
    };
    loop {
        match state.current_token_in_block() {
            Some(t) if t.kind == TokenKind::OpenBracket => {
                let open_bracket_token = t.clone();
                state.consume_token();

                let length = parse_expression(state)?;

                let close_bracket_token = state.consume_by_kind(TokenKind::CloseBracket)?.clone();

                node = TypeSyntax::Array(
                    Box::new(node),
                    open_bracket_token,
                    length,
                    close_bracket_token,
                );
            }
            _ => break Ok(node),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ConstExpressionNode<'s> {
    Number(Token<'s>),
}
fn parse_const_expression<'s>(state: &mut ParseState<'s>) -> ParseResult<ConstExpressionNode<'s>> {
    match state.current_token() {
        Some(t) if t.kind == TokenKind::Number => {
            let t = t.clone();
            state.token_ptr += 1;
            Ok(ConstExpressionNode::Number(t))
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
pub enum AttributeArgSyntax<'s> {
    Single(ExpressionNode<'s>),
    Multiple {
        open_parenthese_token: Token<'s>,
        arg_list: Vec<(ExpressionNode<'s>, Option<Token<'s>>)>,
        close_parenthese_token: Token<'s>,
    },
}
fn parse_attribute_arg<'s>(state: &mut ParseState<'s>) -> ParseResult<AttributeArgSyntax<'s>> {
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

                let arg = parse_expression(state)?;
                let opt_comma_token = state.consume_by_kind(TokenKind::Comma).ok().cloned();
                can_continue = opt_comma_token.is_some();
                arg_list.push((arg, opt_comma_token));
            }
            let close_parenthese_token = state.consume_by_kind(TokenKind::CloseParenthese)?.clone();

            Ok(AttributeArgSyntax::Multiple {
                open_parenthese_token,
                arg_list,
                close_parenthese_token,
            })
        }
        _ => parse_expression(state).map(AttributeArgSyntax::Single),
    }
}
fn lookahead_attribute_arg(state: &ParseState) -> bool {
    state
        .current_token()
        .is_some_and(|t| t.kind == TokenKind::OpenParenthese)
        || lookahead_const_expression(state)
}

#[derive(Debug, Clone)]
pub struct AttributeSyntax<'s> {
    pub name_token: Token<'s>,
    pub arg: Option<AttributeArgSyntax<'s>>,
}
fn parse_attribute<'s>(state: &mut ParseState<'s>) -> ParseResult<AttributeSyntax<'s>> {
    let name_token = state.consume_by_kind(TokenKind::Identifier)?.clone();
    let arg = if state
        .current_token()
        .is_some_and(|t| t.kind != TokenKind::CloseBracket && t.kind != TokenKind::Comma)
    {
        Some(parse_attribute_arg(state)?)
    } else {
        None
    };

    Ok(AttributeSyntax { name_token, arg })
}

#[derive(Debug)]
pub struct AttributeList<'s> {
    pub open_bracket_token: Token<'s>,
    pub attribute_list: Vec<(AttributeSyntax<'s>, Option<Token<'s>>)>,
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
    pub mut_token: Option<Token<'s>>,
    pub name_token: Token<'s>,
    pub colon_token: Token<'s>,
    pub ty: TypeSyntax<'s>,
}
impl<'s> StructMember<'s> {
    #[inline]
    pub fn iter_attributes<'e>(&'e self) -> impl Iterator<Item = &'e AttributeSyntax<'s>> + 'e {
        self.attribute_lists
            .iter()
            .flat_map(|xs| xs.attribute_list.iter().map(|x| &x.0))
    }
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
    let mut_token = state.consume_keyword("mut").ok().cloned();
    let name_token = state
        .consume_in_block_by_kind(TokenKind::Identifier)?
        .clone();
    let colon_token = state.consume_by_kind(TokenKind::Colon)?.clone();
    state.push_indent_context(IndentContext::Exclusive(name_token.line_indent));
    let ty = parse_type(state)?;
    state.pop_indent_context();

    Ok(StructMember {
        attribute_lists,
        mut_token,
        name_token,
        colon_token,
        ty,
    })
}

#[derive(Debug)]
pub struct StructDeclaration<'s> {
    pub decl_token: Token<'s>,
    pub name_token: Token<'s>,
    pub members_starter_token: Option<Token<'s>>,
    pub member_list: Vec<StructMember<'s>>,
}
fn parse_struct_declaration<'s>(
    state: &mut ParseState<'s>,
) -> Result<StructDeclaration<'s>, ParseError> {
    let decl_token = state.consume_keyword("struct")?.clone();
    let name_token = state.consume_by_kind(TokenKind::Identifier)?.clone();
    let members_starter_token;
    match state.current_token() {
        Some(t) if t.kind == TokenKind::Keyword && t.slice == "with" => {
            members_starter_token = t.clone();
            state.consume_token();
        }
        Some(t) if t.kind == TokenKind::Colon => {
            members_starter_token = t.clone();
            state.consume_token();
        }
        _ => {
            return Ok(StructDeclaration {
                decl_token,
                name_token,
                members_starter_token: None,
                member_list: Vec::new(),
            })
        }
    };

    state.push_indent_context(IndentContext::Exclusive(members_starter_token.line_indent));
    let mut member_list = Vec::new();
    while state.check_indent_requirements() {
        member_list.push(parse_struct_member(state)?);
    }
    state.pop_indent_context();

    Ok(StructDeclaration {
        decl_token,
        name_token,
        members_starter_token: Some(members_starter_token),
        member_list,
    })
}

#[derive(Debug)]
pub enum FunctionDeclarationInputArguments<'s> {
    Single {
        attribute_lists: Vec<AttributeList<'s>>,
        mut_token: Option<Token<'s>>,
        varname_token: Token<'s>,
        colon_token: Token<'s>,
        ty: TypeSyntax<'s>,
    },
    Multiple {
        open_parenthese_token: Token<'s>,
        args: Vec<(
            Vec<AttributeList<'s>>,
            Option<Token<'s>>,
            Token<'s>,
            Token<'s>,
            TypeSyntax<'s>,
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

                let mut_token = state.consume_keyword("mut").ok().cloned();
                let name_token = state.consume_by_kind(TokenKind::Identifier)?.clone();
                let colon_token = state.consume_by_kind(TokenKind::Colon)?.clone();
                let ty = parse_type(state)?;
                let opt_comma_token = state.consume_by_kind(TokenKind::Comma).ok().cloned();

                can_continue = opt_comma_token.is_some();
                args.push((
                    attribute_lists,
                    mut_token,
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

            let mut_token = state.consume_keyword("mut").ok().cloned();
            let varname_token = state.consume_by_kind(TokenKind::Identifier)?.clone();
            let colon_token = state.consume_by_kind(TokenKind::Colon)?.clone();
            let ty = parse_type(state)?;

            Ok(FunctionDeclarationInputArguments::Single {
                attribute_lists,
                mut_token,
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
        ty: TypeSyntax<'s>,
    },
    Tupled {
        open_parenthese_token: Token<'s>,
        elements: Vec<(Vec<AttributeList<'s>>, TypeSyntax<'s>, Option<Token<'s>>)>,
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
    pub body: ExpressionNode<'s>,
}
impl<'s> FunctionDeclaration<'s> {
    #[inline]
    pub fn iter_attributes<'e>(&'e self) -> impl Iterator<Item = &'e AttributeSyntax<'s>> + 'e {
        self.attribute_lists
            .iter()
            .flat_map(|xs| xs.attribute_list.iter().map(|x| &x.0))
    }
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
        Some(t)
            if t.kind == TokenKind::Eq
                || (t.kind == TokenKind::Keyword && t.slice == "do")
                || (t.kind == TokenKind::Keyword && t.slice == "does") =>
        {
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

#[derive(Debug, Clone)]
pub enum StatementNode<'s> {
    Let {
        let_token: Token<'s>,
        mut_token: Option<Token<'s>>,
        varname_token: Token<'s>,
        eq_token: Token<'s>,
        expr: ExpressionNode<'s>,
    },
    OpEq {
        left_expr: ExpressionNode<'s>,
        opeq_token: Token<'s>,
        expr: ExpressionNode<'s>,
    },
    While {
        while_token: Token<'s>,
        condition: ExpressionNode<'s>,
        block_starter_token: Token<'s>,
        runs: ExpressionNode<'s>,
    },
    Expression(ExpressionNode<'s>),
}

#[derive(Debug, Clone)]
pub enum ExpressionNode<'s> {
    Blocked(Vec<StatementNode<'s>>, Option<Box<ExpressionNode<'s>>>),
    Lifted(Token<'s>, Box<ExpressionNode<'s>>, Token<'s>),
    Binary(Box<ExpressionNode<'s>>, Token<'s>, Box<ExpressionNode<'s>>),
    Prefixed(Token<'s>, Box<ExpressionNode<'s>>),
    MemberRef(Box<ExpressionNode<'s>>, Token<'s>, Token<'s>),
    ArrayIndex(
        Box<ExpressionNode<'s>>,
        Token<'s>,
        Box<ExpressionNode<'s>>,
        Token<'s>,
    ),
    Funcall {
        base_expr: Box<ExpressionNode<'s>>,
        open_parenthese_token: Token<'s>,
        args: Vec<(ExpressionNode<'s>, Option<Token<'s>>)>,
        close_parenthese_token: Token<'s>,
    },
    FuncallSingle(Box<ExpressionNode<'s>>, Box<ExpressionNode<'s>>),
    Number(Token<'s>),
    Var(Token<'s>),
    StructValue {
        ty: Box<TypeSyntax<'s>>,
        open_brace_token: Token<'s>,
        initializers: Vec<(Token<'s>, Token<'s>, ExpressionNode<'s>, Option<Token<'s>>)>,
        close_brace_token: Token<'s>,
    },
    Tuple(
        Token<'s>,
        Vec<(ExpressionNode<'s>, Option<Token<'s>>)>,
        Token<'s>,
    ),
    If {
        if_token: Token<'s>,
        condition: Box<ExpressionNode<'s>>,
        then_token: Token<'s>,
        then_expr: Box<ExpressionNode<'s>>,
        else_token: Option<Token<'s>>,
        else_expr: Option<Box<ExpressionNode<'s>>>,
    },
}
fn parse_block<'s>(state: &mut ParseState<'s>) -> ParseResult<ExpressionNode<'s>> {
    let mut statements = Vec::new();
    while state.check_indent_requirements() {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Keyword && t.slice == "let" => {
                let let_token = t.clone();
                state.consume_token();
                let mut_token = state.consume_keyword("mut").ok().cloned();
                let varname_token = state.consume_by_kind(TokenKind::Identifier)?.clone();
                let eq_token = state.consume_by_kind(TokenKind::Eq)?.clone();
                state.push_indent_context(IndentContext::Exclusive(eq_token.line_indent));
                let expr = parse_block(state)?;
                state.pop_indent_context();

                statements.push(StatementNode::Let {
                    let_token,
                    mut_token,
                    varname_token,
                    eq_token,
                    expr,
                })
            }
            Some(t) if t.kind == TokenKind::Keyword && t.slice == "while" => {
                let while_token = t.clone();
                state.consume_token();
                state.push_indent_context(IndentContext::Inclusive(while_token.line_indent));
                state.push_indent_context(IndentContext::Exclusive(while_token.line_indent));
                let condition = parse_block(state)?;
                state.pop_indent_context();
                let block_starter_token;
                match state.current_token_in_block() {
                    Some(t)
                        if t.kind == TokenKind::Keyword
                            && (t.slice == "do" || t.slice == "does") =>
                    {
                        block_starter_token = t.clone();
                        state.consume_token();
                    }
                    _ => return Err(state.err(ParseErrorKind::ExpectedKeyword("do"))),
                }
                state.pop_indent_context();
                state.push_indent_context(IndentContext::Exclusive(while_token.line_indent));
                let runs = parse_block(state)?;
                state.pop_indent_context();

                statements.push(StatementNode::While {
                    while_token,
                    condition,
                    block_starter_token,
                    runs,
                });
            }
            _ => {
                let save = state.save();

                let Ok(left_expr) = parse_expression(state) else {
                    state.restore(save);
                    break;
                };
                let opeq_token;
                match state.current_token() {
                    Some(t) if t.kind == TokenKind::Eq || t.kind == TokenKind::OpEq => {
                        opeq_token = t.clone();
                        state.consume_token();
                    }
                    _ => {
                        statements.push(StatementNode::Expression(left_expr));
                        continue;
                    }
                }

                state.push_indent_context(IndentContext::Exclusive(opeq_token.line_indent));
                let expr = parse_block(state)?;
                state.pop_indent_context();

                statements.push(StatementNode::OpEq {
                    left_expr,
                    opeq_token,
                    expr,
                });
            }
        }
    }

    let final_expr = match statements.pop() {
        Some(StatementNode::Expression(x)) => Some(x),
        Some(s) => {
            statements.push(s);
            None
        }
        None => None,
    };
    Ok(ExpressionNode::Blocked(
        statements,
        final_expr.map(Box::new),
    ))
}
fn parse_expression<'s>(state: &mut ParseState<'s>) -> ParseResult<ExpressionNode<'s>> {
    state.push_indent_context(IndentContext::Exclusive(
        state.current_token().map_or(0, |t| t.line_indent),
    ));
    let res = parse_expression_if(state);
    state.pop_indent_context();
    res
}
fn parse_expression_if<'s>(state: &mut ParseState<'s>) -> ParseResult<ExpressionNode<'s>> {
    let Some(if_token) = state.consume_keyword("if").ok().cloned() else {
        return parse_expression_logical_ops(state);
    };
    state.push_indent_context(IndentContext::Inclusive(if_token.line_indent));

    state.require_in_block_next()?;
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

    Ok(ExpressionNode::If {
        if_token,
        condition: Box::new(condition),
        then_token,
        then_expr: Box::new(then_expr),
        else_token,
        else_expr: else_expr.map(Box::new),
    })
}
fn parse_expression_logical_ops<'s>(state: &mut ParseState<'s>) -> ParseResult<ExpressionNode<'s>> {
    let mut expr = parse_expression_compare_ops(state)?;

    while state.check_indent_requirements() {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Op && t.slice == "||" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_compare_ops(state)?;
                expr = ExpressionNode::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "&&" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_compare_ops(state)?;
                expr = ExpressionNode::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            _ => break,
        }
    }

    Ok(expr)
}
fn parse_expression_compare_ops<'s>(state: &mut ParseState<'s>) -> ParseResult<ExpressionNode<'s>> {
    let mut expr = parse_expression_bitwise_ops(state)?;

    while state.check_indent_requirements() {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Op && t.slice == "==" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_bitwise_ops(state)?;
                expr = ExpressionNode::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "!=" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_bitwise_ops(state)?;
                expr = ExpressionNode::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "<=" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_bitwise_ops(state)?;
                expr = ExpressionNode::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == ">=" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_bitwise_ops(state)?;
                expr = ExpressionNode::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::OpenAngleBracket => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_bitwise_ops(state)?;
                expr = ExpressionNode::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::CloseAngleBracket => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_bitwise_ops(state)?;
                expr = ExpressionNode::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            _ => break,
        }
    }

    Ok(expr)
}
fn parse_expression_bitwise_ops<'s>(state: &mut ParseState<'s>) -> ParseResult<ExpressionNode<'s>> {
    let mut expr = parse_expression_arithmetic_ops_1(state)?;

    while state.check_indent_requirements() {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Op && t.slice == "|" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_arithmetic_ops_1(state)?;
                expr = ExpressionNode::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "&" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_arithmetic_ops_1(state)?;
                expr = ExpressionNode::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "^" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_arithmetic_ops_1(state)?;
                expr = ExpressionNode::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == ">>" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_arithmetic_ops_1(state)?;
                expr = ExpressionNode::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "<<" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_arithmetic_ops_1(state)?;
                expr = ExpressionNode::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            _ => break,
        }
    }

    Ok(expr)
}
fn parse_expression_arithmetic_ops_1<'s>(
    state: &mut ParseState<'s>,
) -> ParseResult<ExpressionNode<'s>> {
    let mut expr = parse_expression_arithmetic_ops_2(state)?;

    while state.check_indent_requirements() {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Op && t.slice == "+" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_arithmetic_ops_2(state)?;
                expr = ExpressionNode::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "-" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_arithmetic_ops_2(state)?;
                expr = ExpressionNode::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            _ => break,
        }
    }

    Ok(expr)
}
fn parse_expression_arithmetic_ops_2<'s>(
    state: &mut ParseState<'s>,
) -> ParseResult<ExpressionNode<'s>> {
    let mut expr = parse_expression_arithmetic_ops_3(state)?;

    while state.check_indent_requirements() {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Op && t.slice == "*" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_arithmetic_ops_3(state)?;
                expr = ExpressionNode::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "/" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_arithmetic_ops_3(state)?;
                expr = ExpressionNode::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            Some(t) if t.kind == TokenKind::Op && t.slice == "%" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_arithmetic_ops_3(state)?;
                expr = ExpressionNode::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            _ => break,
        }
    }

    Ok(expr)
}
fn parse_expression_arithmetic_ops_3<'s>(
    state: &mut ParseState<'s>,
) -> ParseResult<ExpressionNode<'s>> {
    let mut expr = parse_expression_infix_funcall_ops(state)?;

    while state.check_indent_requirements() {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Op && t.slice == "^^" => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_infix_funcall_ops(state)?;
                expr = ExpressionNode::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            _ => break,
        }
    }

    Ok(expr)
}
fn parse_expression_infix_funcall_ops<'s>(
    state: &mut ParseState<'s>,
) -> ParseResult<ExpressionNode<'s>> {
    let mut expr = parse_expression_prefixed_ops(state)?;

    while state.check_indent_requirements() {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::InfixIdentifier => {
                let op_token = t.clone();
                state.consume_token();
                let right_expr = parse_expression_prefixed_ops(state)?;
                expr = ExpressionNode::Binary(Box::new(expr), op_token, Box::new(right_expr));
            }
            _ => break,
        }
    }

    Ok(expr)
}
fn parse_expression_prefixed_ops<'s>(
    state: &mut ParseState<'s>,
) -> ParseResult<ExpressionNode<'s>> {
    match state.current_token() {
        Some(t) if t.kind == TokenKind::Op && t.slice == "+" => {
            let op_token = t.clone();
            state.consume_token();
            let expr = parse_expression_prefixed_ops(state)?;

            Ok(ExpressionNode::Prefixed(op_token, Box::new(expr)))
        }
        Some(t) if t.kind == TokenKind::Op && t.slice == "-" => {
            let op_token = t.clone();
            state.consume_token();
            let expr = parse_expression_prefixed_ops(state)?;

            Ok(ExpressionNode::Prefixed(op_token, Box::new(expr)))
        }
        Some(t) if t.kind == TokenKind::Op && t.slice == "~" => {
            let op_token = t.clone();
            state.consume_token();
            let expr = parse_expression_prefixed_ops(state)?;

            Ok(ExpressionNode::Prefixed(op_token, Box::new(expr)))
        }
        Some(t) if t.kind == TokenKind::Op && t.slice == "!" => {
            let op_token = t.clone();
            state.consume_token();
            let expr = parse_expression_prefixed_ops(state)?;

            Ok(ExpressionNode::Prefixed(op_token, Box::new(expr)))
        }
        _ => parse_expression_suffixed_ops(state),
    }
}
fn parse_expression_suffixed_ops<'s>(
    state: &mut ParseState<'s>,
) -> ParseResult<ExpressionNode<'s>> {
    let mut expr = parse_expression_prime(state)?;

    while state.check_indent_requirements() {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Op && t.slice == "." => {
                let dot_token = t.clone();
                state.consume_token();
                let name_token = state.consume_by_kind(TokenKind::Identifier)?.clone();

                expr = ExpressionNode::MemberRef(Box::new(expr), dot_token, name_token);
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

                expr = ExpressionNode::Funcall {
                    base_expr: Box::new(expr),
                    open_parenthese_token,
                    args,
                    close_parenthese_token,
                }
            }
            Some(t) if t.kind == TokenKind::OpenBracket => {
                let open_bracket_token = t.clone();
                state.consume_token();
                let index = parse_expression(state)?;
                let close_bracket_token = state.consume_by_kind(TokenKind::CloseBracket)?.clone();

                expr = ExpressionNode::ArrayIndex(
                    Box::new(expr),
                    open_bracket_token,
                    Box::new(index),
                    close_bracket_token,
                );
            }
            _ if state.check_indent_requirements() => {
                let save = state.save();
                if let Ok(arg) = parse_expression_funcall_single_arg(state) {
                    expr = ExpressionNode::FuncallSingle(Box::new(expr), Box::new(arg));
                } else {
                    state.restore(save);
                }
                break;
            }
            _ => break,
        }
    }

    Ok(expr)
}
fn parse_expression_funcall_single_arg<'s>(
    state: &mut ParseState<'s>,
) -> ParseResult<ExpressionNode<'s>> {
    let mut expr = parse_expression_prime(state)?;

    while state.check_indent_requirements() {
        match state.current_token() {
            Some(t) if t.kind == TokenKind::Op && t.slice == "." => {
                let dot_token = t.clone();
                state.consume_token();
                let name_token = state.consume_by_kind(TokenKind::Identifier)?.clone();

                expr = ExpressionNode::MemberRef(Box::new(expr), dot_token, name_token);
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

                expr = ExpressionNode::Funcall {
                    base_expr: Box::new(expr),
                    open_parenthese_token,
                    args,
                    close_parenthese_token,
                }
            }
            _ => break,
        }
    }

    Ok(expr)
}
fn parse_expression_prime<'s>(state: &mut ParseState<'s>) -> ParseResult<ExpressionNode<'s>> {
    match state.current_token() {
        Some(t) if t.kind == TokenKind::Number => {
            let tok = t.clone();
            state.consume_token();

            Ok(ExpressionNode::Number(tok))
        }
        Some(t) if t.kind == TokenKind::Identifier => {
            let tok = t.clone();
            'alt: {
                let savepoint = state.save();

                let Ok(ty) = parse_type(state) else {
                    state.restore(savepoint);
                    break 'alt;
                };
                let Ok(open_brace_token) = state.consume_by_kind(TokenKind::OpenBrace) else {
                    state.restore(savepoint);
                    break 'alt;
                };
                let open_brace_token = open_brace_token.clone();

                let mut initializers = Vec::new();
                let mut can_continue = true;
                while state
                    .current_token()
                    .is_some_and(|t| t.kind != TokenKind::CloseBrace)
                {
                    if !can_continue {
                        return Err(state.err(ParseErrorKind::ListNotPunctuated(TokenKind::Comma)));
                    }

                    let member_name_token = state.consume_by_kind(TokenKind::Identifier)?.clone();
                    let equal_token = state.consume_by_kind(TokenKind::Eq)?.clone();
                    let init_value = parse_expression(state)?;
                    let opt_comma_expr = state.consume_by_kind(TokenKind::Comma).ok().cloned();

                    can_continue = opt_comma_expr.is_some();
                    initializers.push((member_name_token, equal_token, init_value, opt_comma_expr));
                }

                let close_brace_token = state.consume_by_kind(TokenKind::CloseBrace)?.clone();

                return Ok(ExpressionNode::StructValue {
                    ty: Box::new(ty),
                    open_brace_token,
                    initializers,
                    close_brace_token,
                });
            }
            state.consume_token();

            Ok(ExpressionNode::Var(tok))
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
                Ok(ExpressionNode::Lifted(
                    open_parenthese_token,
                    Box::new(expressions.pop().unwrap().0),
                    close_parenthese_token,
                ))
            } else {
                Ok(ExpressionNode::Tuple(
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
    UnclosedInfixIdentifier,
}

#[derive(Debug)]
pub struct TokenizerError {
    pub kind: TokenizerErrorKind,
    pub line: usize,
    pub col: usize,
}

pub struct Tokenizer<'s> {
    pub source: &'s str,
    pub line: usize,
    pub col: usize,
    pub current_line_indent: usize,
}
impl<'s> Tokenizer<'s> {
    pub fn new(source: &'s str) -> Self {
        let mut this = Self {
            source,
            line: 0,
            col: 0,
            current_line_indent: 0,
        };
        // initial indent generation for first line
        this.populate_line_indent();

        this
    }

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

        let triple_byte_tok = if self.source.as_bytes().len() >= 3 {
            match &self.source.as_bytes()[..3] {
                b"&&=" | b"||=" | b"^^=" | b">>=" | b"<<=" => Some(TokenKind::OpEq),
                _ => None,
            }
        } else {
            None
        };

        if let Some(k) = triple_byte_tok {
            let tk = Token {
                slice: &self.source[..3],
                kind: k,
                line: self.line,
                col: self.col,
                line_indent: self.current_line_indent,
            };
            self.source = &self.source[3..];
            self.col += 3;
            return Ok(Some(tk));
        }

        let double_byte_tok = if self.source.as_bytes().len() >= 2 {
            match &self.source.as_bytes()[..2] {
                b"->" => Some(TokenKind::ArrowToRight),
                b"==" | b"!=" | b"<=" | b">=" | b"&&" | b"||" | b"^^" | b">>" | b"<<=" => {
                    Some(TokenKind::Op)
                }
                b"+=" | b"-=" | b"*=" | b"/=" | b"%=" | b"&=" | b"|=" | b"^=" => {
                    Some(TokenKind::OpEq)
                }
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
            b'{' => Some(TokenKind::OpenBrace),
            b'}' => Some(TokenKind::CloseBrace),
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

        if self.source.starts_with('`') {
            let (ident_char_count, ident_byte_count) = self.source[1..]
                .chars()
                .take_while(|&c| !"=!\"#%&'`()[]?><.,;:@=~-^|\\ \t\r\n".contains(c))
                .fold((0, 0), |(a, b), c| (a + 1, b + c.len_utf8()));
            assert!(
                ident_byte_count > 0,
                "empty identifier token(src: {}...)",
                &self.source[..8]
            );
            if !self.source[1 + ident_byte_count..].starts_with('`') {
                return Err(TokenizerError {
                    kind: TokenizerErrorKind::UnclosedInfixIdentifier,
                    line: self.line,
                    col: self.col,
                });
            }
            let tk = Token {
                slice: &self.source[..ident_byte_count + 2],
                kind: TokenKind::InfixIdentifier,
                line: self.line,
                col: self.col,
                line_indent: self.current_line_indent,
            };
            self.source = &self.source[ident_byte_count + 2..];
            self.col += ident_char_count + 2;
            return Ok(Some(tk));
        }

        let (ident_char_count, ident_byte_count) = self
            .source
            .chars()
            .take_while(|&c| !"=!\"#%&'`()[]?><.,;:@=~-^|\\ \t\r\n".contains(c))
            .fold((0, 0), |(a, b), c| (a + 1, b + c.len_utf8()));
        assert!(
            ident_byte_count > 0,
            "empty identifier token(src: {}...)",
            &self.source[..8]
        );
        let tk = Token {
            slice: &self.source[..ident_byte_count],
            kind: match &self.source[..ident_byte_count] {
                "struct" | "with" | "if" | "else" | "then" | "do" | "does" | "let" | "mut"
                | "module" | "while" | "uniform" => TokenKind::Keyword,
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
    InfixIdentifier,
    Keyword,
    Op,
    OpEq,
    Number,
    OpenBracket,
    CloseBracket,
    OpenParenthese,
    CloseParenthese,
    OpenAngleBracket,
    CloseAngleBracket,
    OpenBrace,
    CloseBrace,
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
