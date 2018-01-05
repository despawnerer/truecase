extern crate failure;
extern crate itertools;
#[macro_use]
extern crate lazy_static;
extern crate regex;
extern crate serde;
#[macro_use]
extern crate serde_derive;

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::iter::{Chain, Take, Repeat, repeat};

use failure::Error;
use itertools::Itertools;
use regex::Regex;

lazy_static!{
    static ref WORD_SEPARATORS: Regex = Regex::new(r"[,.?!:;\s]+").unwrap();
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Model {
    unigrams: CaseMap,
    bigrams: CaseMap,
    trigrams: CaseMap,
}

impl Model {
    pub fn train_on_file(filename: &str) -> Result<Self, Error> {
        let mut text = String::new();
        File::open(filename)?.read_to_string(&mut text)?;
        Self::train_on_text(&text)
    }

    pub fn train_on_text(text: &str) -> Result<Self, Error> {
        let mut unigram_stats = CaseStats::new();
        let mut bigram_stats = CaseStats::new();
        let mut trigram_stats = CaseStats::new();

        for sentence in text.lines().filter(|s| is_sentence_sane(s)) {
            let tokens: Vec<_> = tokenize(sentence).pad(1, Token::padding()).filter(Token::is_meaningful).collect();

            for token in tokens.iter().cloned() {
                unigram_stats.add(token);
            }

            for ngram in tokens.windows(2).map(Token::ngram) {
                bigram_stats.add(ngram);
            }

            for ngram in tokens.windows(3).map(Token::ngram) {
                trigram_stats.add(ngram);
            }
        }

        Ok(Model {
            unigrams: unigram_stats.into_most_common(1),
            bigrams: bigram_stats.into_most_common(10),
            trigrams: trigram_stats.into_most_common(10),
        })
    }

    pub fn truecase(&self, sentence: &str) -> String {
        let tokens: Vec<_> = tokenize(sentence).pad(1, Token::padding()).collect();
        let total_tokens = tokens.len();

        let words_with_indexes: Vec<_> = tokens
            .iter()
            .enumerate()
            .filter(|x| x.1.is_meaningful())
            .collect();

        let mut truecase_tokens: Vec<Option<String>> = Vec::with_capacity(total_tokens);
        truecase_tokens.resize(total_tokens, None);

        let select_true_case_from_ngrams =
            |size, source: &CaseMap, result: &mut Vec<Option<String>>| {
                for slice in words_with_indexes.windows(size) {
                    let indexes = slice.iter().map(|x| x.0);
                    let normalized_ngram = slice.iter().map(|x| &x.1.normalized).join(" ");
                    if let Some(truecased_ngram) = source.get(&normalized_ngram) {
                        for (word, index) in truecased_ngram.split(" ").zip(indexes) {
                            result[index].get_or_insert_with(|| word.to_owned());
                        }
                    }
                }
            };

        select_true_case_from_ngrams(3, &self.trigrams, &mut truecase_tokens);
        select_true_case_from_ngrams(2, &self.bigrams, &mut truecase_tokens);

        for (truecase, token) in truecase_tokens.iter_mut().zip(tokens.iter()) {
            if token.is_meaningful() && truecase.is_none() {
                *truecase = self.unigrams.get(&token.normalized).map(String::clone);
            }

            if truecase.is_none() {
                *truecase = Some(token.original.clone());
            }
        }

        truecase_tokens.into_iter().map(Option::unwrap).join("")
    }
}


/// `CaseStats` keeps track of how often we see each casing for different tokens
#[derive(Default)]
struct CaseStats {
    stats: HashMap<String, HashMap<String, u32>>,
}

impl CaseStats {
    fn new() -> Self {
        Self::default()
    }

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

struct Tokens<'a> {
    next_token: Option<Token>,
    string: Option<&'a str>,
}

impl<'a> Iterator for Tokens<'a> {
    type Item = Token;

    fn next(&mut self) -> Option<Token> {
        if let Some(string) = self.next_token.take() {
            if !string.is_empty() {
                return Some(string);
            }
        }

        if let Some(string) = self.string.take() {
            if let Some(mat) = WORD_SEPARATORS.find(string) {
                let (before, matching_part, rest) = split_in_three(string, mat.start(), mat.end());
                self.string = Some(rest);

                let separator = Token::separator(matching_part);
                if before.is_empty() {
                    return Some(separator);
                } else {
                    self.next_token = Some(separator);
                    return Some(Token::word(before));
                }
            } else if !string.is_empty() {
                return Some(Token::word(string));
            }
        }

        return None;
    }
}

fn tokenize(sentence: &str) -> Tokens {
    Tokens {
        string: Some(sentence),
        next_token: None,
    }
}

fn split_in_three(string: &str, index1: usize, index2: usize) -> (&str, &str, &str) {
    let (first, rest) = string.split_at(index1);
    let (second, third) = rest.split_at(index2 - index1);
    (first, second, third)
}

fn is_sentence_sane(sentence: &str) -> bool {
    !sentence.chars().all(char::is_uppercase)
}

fn normalize(token: &str) -> String {
    token.to_lowercase()
}

type Padded<I: Iterator> = Chain<Chain<Take<Repeat<I::Item>>, I>, Take<Repeat<I::Item>>>;

trait Paddable: Iterator {
    fn pad(self, n: usize, padding: Self::Item) -> Padded<Self>
        where Self: Sized,
              Self::Item: Clone,
    {
        let before = repeat(padding.clone()).take(n);
        let after = repeat(padding).take(n);
        before.chain(self).chain(after)
    }
}

impl<T: ?Sized> Paddable for T where T: Iterator { }
