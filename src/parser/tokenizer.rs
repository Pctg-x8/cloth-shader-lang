use crate::utils::{BoolToErrorHelper, Located};

#[derive(Debug)]
pub enum TokenizerError {
    IncompleteHexLiteral,
    UnclosedInfixIdentifier,
    EmptyInfixIdentifier,
    InvalidCharacter,
}
impl TokenizerError {
    #[inline(always)]
    pub const fn at(self, line: usize, col: usize) -> Located<Self> {
        Located { t: self, line, col }
    }
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

    /// トークン頭のスペースを読み飛ばす
    fn skip_head_spaces(&mut self) {
        let (chars, bytes) = self
            .source
            .chars()
            .take_while(|&c| c.is_whitespace() && c != '\n')
            .fold((0, 0), |(a, b), c| (a + 1, b + c.len_utf8()));
        self.col += chars;
        self.source = &self.source[bytes..];
    }

    /// 繋がった改行を一気に読み飛ばす
    fn skip_chained_newlines(&mut self) {
        if self.source.starts_with('\n') {
            let (lf_count, skip_bytes) = self
                .source
                .chars()
                .take_while(|&c| c == '\n' || c == '\r')
                .fold((0, 0), |(c, b), ch| {
                    (c + (if ch == '\n' { 1 } else { 0 }), b + ch.len_utf8())
                });
            self.line += lf_count;
            self.source = &self.source[skip_bytes..];
            self.populate_line_indent();
        }
    }

    pub fn next_token(&mut self) -> Result<Option<Token<'s>>, Located<TokenizerError>> {
        self.skip_head_spaces();
        self.skip_chained_newlines();

        if self.source.is_empty() {
            // 読み切った
            return Ok(None);
        }

        if self.source.starts_with('#') {
            // line comment
            self.source = self.source.trim_start_matches(|c| c != '\n');
            return self.next_token();
        }

        // 3バイトのトークンを読む
        'try3: {
            if self.source.as_bytes().len() >= 3 {
                let tk = Token {
                    slice: &self.source[..3],
                    kind: match &self.source.as_bytes()[..3] {
                        b"&&=" | b"||=" | b"^^=" | b">>=" | b"<<=" => TokenKind::OpEq,
                        _ => break 'try3,
                    },
                    line: self.line,
                    col: self.col,
                    line_indent: self.current_line_indent,
                };
                self.source = &self.source[3..];
                self.col += 3;

                return Ok(Some(tk));
            }
        }

        // 2バイトのトークンを読む
        'try2: {
            if self.source.as_bytes().len() >= 2 {
                let tk = Token {
                    slice: &self.source[..2],
                    kind: match &self.source.as_bytes()[..2] {
                        b"->" => TokenKind::ArrowToRight,
                        b"==" | b"!=" | b"<=" | b">=" | b"&&" | b"||" | b"^^" | b">>" | b"<<=" => {
                            TokenKind::Op
                        }
                        b"+=" | b"-=" | b"*=" | b"/=" | b"%=" | b"&=" | b"|=" | b"^=" => {
                            TokenKind::OpEq
                        }
                        _ => break 'try2,
                    },
                    line: self.line,
                    col: self.col,
                    line_indent: self.current_line_indent,
                };
                self.source = &self.source[2..];
                self.col += 2;

                return Ok(Some(tk));
            }
        }

        // 1バイトのトークンを読む
        'try1: {
            let tk = Token {
                slice: &self.source[..1],
                kind: match self.source.as_bytes()[0] {
                    b'[' => TokenKind::OpenBracket,
                    b']' => TokenKind::CloseBracket,
                    b'(' => TokenKind::OpenParenthese,
                    b')' => TokenKind::CloseParenthese,
                    b'{' => TokenKind::OpenBrace,
                    b'}' => TokenKind::CloseBrace,
                    b'<' => TokenKind::OpenAngleBracket,
                    b'>' => TokenKind::CloseAngleBracket,
                    b',' => TokenKind::Comma,
                    b':' => TokenKind::Colon,
                    b'=' => TokenKind::Eq,
                    b'+' | b'-' | b'*' | b'/' | b'%' | b'&' | b'|' | b'^' | b'~' | b'!' | b'.' => {
                        TokenKind::Op
                    }
                    _ => break 'try1,
                },
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
            (hexpart_count > 0)
                .or_err(|| TokenizerError::IncompleteHexLiteral.at(self.line, self.col))?;

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
            (ident_byte_count > 0)
                .or_err(|| TokenizerError::EmptyInfixIdentifier.at(self.line, self.col))?;
            self.source[1 + ident_byte_count..]
                .starts_with('`')
                .or_err(|| TokenizerError::UnclosedInfixIdentifier.at(self.line, self.col))?;

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
        (ident_byte_count > 0)
            .or_err(|| TokenizerError::InvalidCharacter.at(self.line, self.col))?;
        let tk = Token {
            slice: &self.source[..ident_byte_count],
            kind: match &self.source[..ident_byte_count] {
                "struct" | "with" | "if" | "else" | "then" | "do" | "does" | "let" | "mut"
                | "module" | "while" => TokenKind::Keyword,
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
