use std::borrow::Cow;
use std::iter::Filter;

use lazy_static::lazy_static;
use regex::Regex;

use crate::utils::split_in_three;

pub(crate) type Tokens<'a> = Filter<Tokenizer<'a>, fn(&Token) -> bool>;

pub(crate) fn tokenize(phrase: &str) -> Tokens {
    let tokens = Tokenizer {
        string: phrase,
        next_token: None,
    };
    tokens.filter(|t| !t.is_empty())
}

lazy_static! {
    static ref WORD_SEPARATORS: Regex = Regex::new(r#"[,.?!:;()«»„“”"—\s]+"#).unwrap();
}

pub(crate) struct Tokenizer<'a> {
    next_token: Option<Token<'a>>,
    string: &'a str,
}

impl<'a> Iterator for Tokenizer<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(token) = self.next_token.take() {
            return Some(token);
        }

        if self.string.is_empty() {
            return None;
        }

        if let Some(mat) = WORD_SEPARATORS.find(self.string) {
            let (before, matching_part, rest) = split_in_three(self.string, mat.start(), mat.end());
            self.string = rest;
            self.next_token = Some(Token::new(matching_part, TokenKind::Separator));
            return Some(Token::new(before, TokenKind::Word));
        } else {
            let rest = self.string;
            self.string = "";
            return Some(Token::new(rest, TokenKind::Word));
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) enum TokenKind {
    Word,
    Separator,
}

#[derive(Debug, Clone)]
pub(crate) struct Token<'a> {
    pub original: &'a str,
    pub normalized: Cow<'a, str>,
    pub kind: TokenKind,
}

impl<'a> Token<'a> {
    pub fn new(original: &'a str, kind: TokenKind) -> Self {
        let normalized = if kind == TokenKind::Word && original.contains(char::is_uppercase) {
            Cow::Owned(original.to_lowercase())
        } else {
            Cow::Borrowed(original)
        };

        Self {
            original,
            normalized,
            kind,
        }
    }

    pub fn is_meaningful(&self) -> bool {
        self.kind == TokenKind::Word
    }

    pub fn is_empty(&self) -> bool {
        self.original.is_empty()
    }

    // these functions are only necessary because closures can't be cloned and
    // `join_with_spaces` requires a cloneable iterator
    pub fn get_normalized(&self) -> &str {
        self.normalized.as_ref()
    }

    pub fn get_original(&self) -> &str {
        self.original
    }
}
