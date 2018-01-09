extern crate failure;
extern crate itertools;
#[macro_use]
extern crate lazy_static;
extern crate regex;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::iter::{once, Chain, Filter, Once};

use failure::Error;
use itertools::Itertools;
use regex::Regex;

/// Trainer for new truecasing models
#[derive(Default)]
pub struct ModelTrainer {
    unigram_stats: CaseStats,
    bigram_stats: CaseStats,
    trigram_stats: CaseStats,
}

impl ModelTrainer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_sentences_from_file(&mut self, filename: &str) -> Result<&mut Self, Error> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            self.add_sentence(&line?);
        }

        Ok(self)
    }

    pub fn add_sentences_from_iter<I>(&mut self, iter: I) -> &mut Self
    where
        I: Iterator,
        I::Item: AsRef<str>,
    {
        for sentence in iter {
            self.add_sentence(sentence.as_ref());
        }

        self
    }

    pub fn add_sentence(&mut self, sentence: &str) -> &mut Self {
        if is_sentence_sane(sentence) {
            let tokens: Vec<_> = tokenize(sentence.as_ref())
                .filter(Token::is_meaningful)
                .collect();

            for token in tokens.iter().cloned() {
                self.unigram_stats.add(token);
            }

            for ngram in tokens.windows(2).map(Token::ngram) {
                self.bigram_stats.add(ngram);
            }

            for ngram in tokens.windows(3).map(Token::ngram) {
                self.trigram_stats.add(ngram);
            }
        }

        self
    }

    pub fn into_model(self) -> Model {
        Model {
            unigrams: self.unigram_stats.into_most_common(1),
            bigrams: self.bigram_stats.into_most_common(10),
            trigrams: self.trigram_stats.into_most_common(10),
        }
    }
}

/// Simple statistical truecaser for sentences
#[derive(Serialize, Deserialize, Debug)]
pub struct Model {
    unigrams: CaseMap,
    bigrams: CaseMap,
    trigrams: CaseMap,
}

impl Model {
    pub fn load_from_file(filename: &str) -> Result<Self, Error> {
        let mut string = String::new();
        File::open(filename)?.read_to_string(&mut string)?;
        let model = serde_json::from_str(&string)?;

        Ok(model)
    }

    pub fn save_to_file(&self, filename: &str) -> Result<(), Error> {
        let serialized = serde_json::to_string(&self)?;
        File::create(filename)?.write_all(serialized.as_bytes())?;

        Ok(())
    }

    pub fn truecase(&self, sentence: &str) -> String {
        let tokens: Vec<_> = tokenize(sentence).collect();

        let words_with_indexes: Vec<_> = tokens
            .iter()
            .enumerate()
            .filter(|x| x.1.is_meaningful())
            .collect();

        let mut truecase_tokens: Vec<_> = tokens.iter().map(|t| t.normalized.clone()).collect();

        for &(index, token) in &words_with_indexes {
            if let Some(truecased_word) = self.unigrams.get(&token.normalized) {
                truecase_tokens[index] = truecased_word.to_owned();
            }
        }

        let select_true_case_from_ngrams = |size, source: &CaseMap, result: &mut Vec<String>| {
            for slice in words_with_indexes.windows(size) {
                let indexes = slice.iter().map(|x| x.0);
                let normalized_ngram = slice.iter().map(|x| &x.1.normalized).join(" ");
                if let Some(truecased_ngram) = source.get(&normalized_ngram) {
                    for (word, index) in truecased_ngram.split(' ').zip(indexes) {
                        result[index] = word.to_owned();
                    }
                }
            }
        };

        select_true_case_from_ngrams(2, &self.bigrams, &mut truecase_tokens);
        select_true_case_from_ngrams(3, &self.trigrams, &mut truecase_tokens);

        truecase_tokens.join("")
    }
}

#[derive(Default)]
struct CaseStats {
    stats: HashMap<String, HashMap<String, u32>>,
}

impl CaseStats {
    fn add(&mut self, token: Token) {
        let count = self.stats
            .entry(token.normalized)
            .or_insert_with(HashMap::new)
            .entry(token.original)
            .or_insert(0);

        *count += 1;
    }

    fn into_most_common(self, min_frequency: u32) -> CaseMap {
        self.stats
            .into_iter()
            .flat_map(|(normalized, possible_cases)| {
                possible_cases
                    .into_iter()
                    .filter(|&(_, frequency)| frequency >= min_frequency)
                    .max_by_key(|&(_, frequency)| frequency)
                    .map(|(most_common_case, _)| (normalized, most_common_case))
            })
            .collect()
    }
}

type CaseMap = HashMap<String, String>;

// Wow, this type is just horrible, isn't it.
type Tokens<'a> = Chain<Chain<Once<Token>, Filter<Tokenizer<'a>, fn(&Token) -> bool>>, Once<Token>>;

fn tokenize(sentence: &str) -> Tokens {
    // The fact that I have to do this is just unfortunate.
    let token_isnt_empty = (|token| !token.is_empty()) as fn(&Token) -> bool;

    let beginning = once(Token::padding());
    let tokens = Tokenizer {
        string: sentence,
        next_token: None,
    };
    let end = once(Token::padding());

    beginning.chain(tokens.filter(token_isnt_empty)).chain(end)
}

lazy_static!{
    static ref WORD_SEPARATORS: Regex = Regex::new(r"[,.?!:;\s]+").unwrap();
}

struct Tokenizer<'a> {
    next_token: Option<Token>,
    string: &'a str,
}

impl<'a> Iterator for Tokenizer<'a> {
    type Item = Token;

    fn next(&mut self) -> Option<Token> {
        if let Some(token) = self.next_token.take() {
            return Some(token);
        }

        if self.string.is_empty() {
            return None;
        }

        if let Some(mat) = WORD_SEPARATORS.find(self.string) {
            let (before, matching_part, rest) = split_in_three(self.string, mat.start(), mat.end());
            self.string = rest;
            self.next_token = Some(Token::separator(matching_part));
            return Some(Token::word(before));
        } else {
            let rest = self.string;
            self.string = "";
            return Some(Token::word(rest));
        }
    }
}

#[derive(Debug, Clone)]
enum TokenKind {
    Word,
    Ngram,
    Separator,
    Padding,
}

#[derive(Debug, Clone)]
struct Token {
    pub original: String,
    pub normalized: String,
    pub kind: TokenKind,
}

impl Token {
    fn new(string: &str, kind: TokenKind) -> Self {
        let original = string.to_owned();
        let normalized = normalize(&original);
        Self {
            original,
            normalized,
            kind,
        }
    }

    fn padding() -> Self {
        Self::new("", TokenKind::Padding)
    }

    fn word(string: &str) -> Self {
        Self::new(string, TokenKind::Word)
    }

    fn ngram(tokens: &[Token]) -> Self {
        let original = tokens.iter().map(|t| &t.original).join(" ");
        let normalized = tokens.iter().map(|t| &t.normalized).join(" ");
        Self {
            original,
            normalized,
            kind: TokenKind::Ngram,
        }
    }

    fn separator(string: &str) -> Self {
        Self::new(string, TokenKind::Separator)
    }

    fn is_meaningful(&self) -> bool {
        match self.kind {
            TokenKind::Separator => false,
            _ => true,
        }
    }

    fn is_empty(&self) -> bool {
        self.original.is_empty()
    }
}

fn split_in_three(string: &str, index1: usize, index2: usize) -> (&str, &str, &str) {
    let (first, rest) = string.split_at(index1);
    let (second, third) = rest.split_at(index2 - index1);
    (first, second, third)
}

fn is_sentence_sane(sentence: &str) -> bool {
    !sentence.chars().all(char::is_uppercase) && !sentence.chars().all(char::is_lowercase)
}

fn normalize(token: &str) -> String {
    token.to_lowercase()
}
