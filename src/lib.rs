extern crate failure;
extern crate itertools;
extern crate regex;
#[macro_use]
extern crate serde_derive;
extern crate serde;


use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

use failure::Error;
use itertools::Itertools;
use regex::Regex;

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
            let tokens: Tokens = tokenize(sentence).into_iter().map(Token::from).collect();

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
        let words: Vec<_> = tokenize(sentence).into_iter().map(|s| normalize(&s)).collect();

        let mut result: Vec<Option<String>> = Vec::with_capacity(words.len());
        for _ in 0..words.len() {
            result.push(None);
        }

        for (index, trigram) in words.windows(3).enumerate() {
            let trigram_string = trigram.iter().join(" ");
            if let Some(truecased) = self.trigrams.get(&trigram_string) {
                for (pos, word) in truecased.split(" ").enumerate() {
                    if result[index+pos].is_none() {
                        result[index+pos] = Some(word.to_owned());
                    }
                }
            }
        }

        for (index, bigram) in words.windows(2).enumerate() {
            let bigram_string = bigram.iter().join(" ");
            if let Some(truecased) = self.bigrams.get(&bigram_string) {
                for (pos, word) in truecased.split(" ").enumerate() {
                    if result[index+pos].is_none() {
                        result[index+pos] = Some(word.to_owned());
                    }
                }
            }
        }

        for (index, word) in words.iter().enumerate() {
            if result[index].is_none() {
                let truecased = self.unigrams.get(word).unwrap_or(word).to_owned();
                result[index] = Some(truecased);
            }
        }

        result.into_iter().map(Option::unwrap).join(" ")
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
}

impl From<String> for Token {
    fn from(original: String) -> Token {
        let normalized = normalize(&original);
        Token {
            original,
            normalized,
        }
    }
}

fn is_sentence_sane(sentence: &str) -> bool {
    !sentence.chars().all(char::is_uppercase)
}

fn tokenize(sentence: &str) -> Vec<String> {
    // TODO: split punctuation into separate tokens
    sentence
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}

fn normalize(token: &str) -> String {
    token.to_lowercase()
}
