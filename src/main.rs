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
    while parse_state.current_token().is_some() {
        let tld = parse_toplevel_declaration(&mut parse_state).unwrap();
        println!("tld: {tld:#?}");
    }
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

#[derive(Debug)]
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

#[derive(Debug)]
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

#[derive(Debug)]
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

#[derive(Debug)]
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

#[derive(Debug)]
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
