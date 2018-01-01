extern crate failure;
extern crate itertools;
extern crate regex;
#[macro_use]
extern crate serde_derive;
extern crate serde;
#[macro_use]
extern crate lazy_static;

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

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
            let mut tokens = tokenize(sentence);
            tokens.retain(|t| t.is_meaningful);

            for token in tokens.iter().cloned() {
                unigram_stats.add_token(token);
            }

            bigram_stats.add_ngrams(&tokens, 2);
            trigram_stats.add_ngrams(&tokens, 3);
        }

        Ok(Model {
            unigrams: unigram_stats.into_most_common(1),
            bigrams: bigram_stats.into_most_common(10),
            trigrams: trigram_stats.into_most_common(10),
        })
    }

    pub fn truecase(&self, sentence: &str) -> String {
        let tokens = tokenize(sentence);
        let total_tokens = tokens.len();

        let words_with_indexes: Vec<_> = tokens.iter().enumerate().filter(|x| x.1.is_meaningful).collect();

        let mut truecase_tokens: Vec<Option<String>> = Vec::with_capacity(total_tokens);
        truecase_tokens.resize(total_tokens, None);

        let select_true_case_from_ngrams = |size, source: &CaseMap, result: &mut Vec<Option<String>>| {
            for slice in words_with_indexes.windows(size) {
                let indexes = slice.iter().map(|x| x.0);
                let ngram = slice.iter().map(|x| &x.1.normalized).join(" ");
                if let Some(truecased_ngram) = source.get(&ngram) {
                    for (word, index) in truecased_ngram.split(" ").zip(indexes) {
                        result[index].get_or_insert_with(|| word.to_owned());
                    }
                }
            }
        };

        select_true_case_from_ngrams(3, &self.trigrams, &mut truecase_tokens);
        select_true_case_from_ngrams(2, &self.bigrams, &mut truecase_tokens);

        for (truecase, token) in truecase_tokens.iter_mut().zip(tokens.iter()) {
            if token.is_meaningful && truecase.is_none() {
                *truecase = self.unigrams.get(&token.normalized).map(String::clone);
            }

            if truecase.is_none() {
                *truecase = Some(token.original.clone());
            }
        }

        truecase_tokens.into_iter().map(Option::unwrap).join("")
    }
}


type FreqDist = HashMap<String, u32>;

/// `CaseStats` keeps track of how often we see each casing for different tokens
#[derive(Default)]
struct CaseStats {
    stats: HashMap<String, FreqDist>,
}

impl CaseStats {
    fn new() -> Self {
        Self::default()
    }

    fn add_token(&mut self, token: Token) {
        let count = self.stats
            .entry(token.normalized)
            .or_insert_with(FreqDist::new)
            .entry(token.original)
            .or_insert(0);

        *count += 1;
    }

    fn add_ngrams(&mut self, tokens: &[Token], size: usize) {
        for ngram in tokens.windows(size) {
            let original = ngram.iter().map(|t| &t.original).join(" ");
            let normalized = ngram.iter().map(|t| &t.normalized).join(" ");
            self.add_token(Token {
                original,
                normalized,
                is_meaningful: true
            });
        }
    }

    fn into_most_common(self, min_frequency: u32) -> CaseMap {
        self.stats
            .into_iter()
            .flat_map(|(token, possible_cases)| {
                possible_cases
                    .into_iter()
                    .filter(|&(_, frequency)| frequency >= min_frequency)
                    .max_by_key(|&(_, frequency)| frequency)
                    .map(|(most_common_case, _)| (token, most_common_case))
            })
            .collect()
    }
}

type CaseMap = HashMap<String, String>;
type Tokens = Vec<Token>;

#[derive(Debug, Clone)]
struct Token {
    pub original: String,
    pub normalized: String,
    pub is_meaningful: bool,
}

impl Token {
    fn new(original: &str, is_meaningful: bool) -> Self {
        let original = original.to_owned();
        let normalized = normalize(&original);
        Self {
            original,
            normalized,
            is_meaningful,
        }
    }
}

fn is_sentence_sane(sentence: &str) -> bool {
    !sentence.chars().all(char::is_uppercase)
}

fn tokenize(sentence: &str) -> Tokens {
    let mut tokens = Vec::new();

    let mut string = sentence;
    while let Some(mat) = WORD_SEPARATORS.find(string) {
        let (before, matching_part, rest) = split_in_three(string, mat.start(), mat.end());
        tokens.push(Token::new(before, true));
        tokens.push(Token::new(matching_part, false));
        string = rest;
    }

    tokens.push(Token::new(string, true));
    tokens.retain(|token| !token.original.is_empty());

    tokens
}

fn split_in_three(string: &str, index1: usize, index2: usize) -> (&str, &str, &str) {
    let (first, rest) = string.split_at(index1);
    let (second, third) = rest.split_at(index2 - index1);
    (first, second, third)
}

fn normalize(token: &str) -> String {
    token.to_lowercase()
}
