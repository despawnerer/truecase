//! This is a simple statistical truecasing library.
//!
//! _Truecasing_ is restoration of original letter cases in text:
//! for example, turning all-uppercase, or all-lowercase text into
//! one that has proper sentence casing (capital first letter,
//! capitalized names etc).
//!
//! This crate attempts to solve this problem by gathering statistics
//! from a set of training sentences, then using those statistics
//! to truecase sentences with broken casings. It comes with a
//! command-line utility that makes training a model easy.
//!
//! # Training a model using the CLI tool
//!
//! 1. Create a file containing training sentences. Each sentence
//!    must be on its own line and have proper casing. The bigger
//!    the training set, the better and more accurate the model will be.
//!
//! 2. Use `truecase` CLI tool to build a model. This may take some time,
//!    depending on the size of the training set. The following command will
//!    read training data from `training_sentences.txt` file and write
//!    the model into `model.json` file.
//!
//!    ```bash
//!    truecase train -i training_sentences.txt -o model.json
//!    ```
//!
//!    Run `truecase train --help` for more details.
//!
//! # Training a model from Rust
//!
//! ```
//! use truecase::ModelTrainer;
//!
//! let mut trainer = ModelTrainer::new();
//! trainer.add_sentence("Here's a sample training sentence for truecasing");
//! trainer.add_sentences_from_file("training_data.txt")?;
//!
//! let model = trainer.into_model();
//! model.save_to_file("model.json")?;
//! ```
//!
//! See also [`ModelTrainer`](struct.ModelTrainer.html).
//!
//! # Using a model to truecase text
//!
//! ```
//! use truecase::Model;
//!
//! let model = Model::load_from_file("model.json")?;
//! let truecased_text = model.truecase("i don't think shakespeare would approve of this sample text");
//! assert_eq!(truecase_text, "I don't think Shakespeare would approve of this sample text");
//! ```
//!
//! See also [`Model`](struct.Model.html).
//!
//! For truecasing using the CLI tool, see `truecase truecase --help`.

extern crate failure;
#[macro_use]
extern crate lazy_static;
extern crate regex;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::io::{BufRead, BufReader, Read, Write};
use std::iter::Filter;

use failure::Error;
use regex::Regex;

/// Trainer for new truecasing models.
///
/// Use this to create your own models from a set of training sentences.
/// See [crate documentation](index.html) for examples.
#[derive(Debug, Default)]
pub struct ModelTrainer {
    unigram_stats: CaseStats,
    bigram_stats: CaseStats,
    trigram_stats: CaseStats,
}

impl ModelTrainer {
    /// Create a new model trainer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add sentences to the training set from a file.
    ///
    /// The file is assumed to have one sentence per line.
    pub fn add_sentences_from_file(&mut self, filename: &str) -> io::Result<&mut Self> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            self.add_sentence(&line?);
        }

        Ok(self)
    }

    /// Add multiple sentences to the training set from an iterator.
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

    /// Add one sentence to the training set.
    pub fn add_sentence(&mut self, sentence: &str) -> &mut Self {
        if !is_sentence_sane(sentence) {
            return self;
        }

        let tokens: Vec<_> = tokenize(sentence)
            .filter(Token::is_meaningful)
            // skip the first word of the sentence because certain words are more
            // likely to start a sentence than to be in the middle of it,
            // but when they are indeed in the middle, they are not capitalized
            // which leads to statistical data that doesn't make much sense
            .skip(1)
            .collect();

        for token in &tokens {
            self.unigram_stats.add_token(token);
        }

        for ngram in tokens.windows(2) {
            self.bigram_stats.add_ngram(ngram);
        }

        for ngram in tokens.windows(3) {
            self.trigram_stats.add_ngram(ngram);
        }

        self
    }

    /// Build a model from all gathered statistics.
    pub fn into_model(self) -> Model {
        let mut unigrams = self.unigram_stats.into_most_common(1);
        unigrams.retain(|k, v| k != v);
        Model {
            unigrams,
            bigrams: self.bigram_stats.into_most_common(10),
            trigrams: self.trigram_stats.into_most_common(10),
        }
    }
}

/// Truecasing model itself.
///
/// See [crate documentation](index.html) for examples.
#[derive(Serialize, Deserialize, Debug)]
pub struct Model {
    unigrams: CaseMap,
    bigrams: CaseMap,
    trigrams: CaseMap,
}

enum TruecasingMode {
    Sentence,
    Phrase,
}

impl Model {
    /// Save this model into a file with the given filename.
    /// The format is simple JSON right now.
    pub fn save_to_file(&self, filename: &str) -> Result<(), Error> {
        let serialized = serde_json::to_string(&self)?;
        File::create(filename)?.write_all(serialized.as_bytes())?;

        Ok(())
    }

    /// Load a previously saved model from a file
    pub fn load_from_file(filename: &str) -> Result<Self, Error> {
        let mut string = String::new();
        File::open(filename)?.read_to_string(&mut string)?;
        let model = serde_json::from_str(&string)?;

        Ok(model)
    }

    /// Restore word casings in a sentence
    pub fn truecase(&self, sentence: &str) -> String {
        self.truecase_tokens(tokenize(sentence), TruecasingMode::Sentence)
    }

    /// Restore word casings in a phrase (sentence fragment)
    pub fn truecase_phrase(&self, phrase: &str) -> String {
        self.truecase_tokens(tokenize(phrase), TruecasingMode::Phrase)
    }

    fn truecase_tokens<'a, I>(&self, tokens: I, mode: TruecasingMode) -> String
    where
        I: Iterator<Item = Token<'a>>,
    {
        let mut truecase_tokens = Vec::with_capacity(3);
        let mut normalized_words_with_indexes = Vec::with_capacity(3);

        let mut special_case_first_word = match mode {
            TruecasingMode::Sentence => true,
            TruecasingMode::Phrase => false,
        };

        for (index, token) in tokens.enumerate() {
            let truecased;

            if token.is_meaningful() {
                let from_unigrams = self.unigrams
                    .get(token.normalized.as_ref())
                    .map(|string| string.as_str());

                if special_case_first_word {
                    special_case_first_word = false;
                    truecased = match from_unigrams {
                        Some(s) => s.to_owned(),
                        None => uppercase_first_letter(&token.normalized),
                    };
                } else {
                    truecased = match from_unigrams {
                        Some(s) => s.to_owned(),
                        None => token.normalized.as_ref().to_owned(),
                    };

                    normalized_words_with_indexes.push((index, token.normalized));
                }
            } else {
                truecased = token.original.to_owned();
            }

            truecase_tokens.push(truecased);
        }

        let update_true_cases_from_ngrams = |size, source: &CaseMap, result: &mut Vec<String>| {
            for slice in normalized_words_with_indexes.windows(size) {
                let indexes = slice.iter().map(|x| x.0);
                let ngram = join_with_spaces(slice.iter().map(get_second_item));
                if let Some(truecased_ngram) = source.get(&ngram) {
                    for (word, index) in truecased_ngram.split(' ').zip(indexes) {
                        result[index] = word.to_owned();
                    }
                }
            }
        };

        update_true_cases_from_ngrams(2, &self.bigrams, &mut truecase_tokens);
        update_true_cases_from_ngrams(3, &self.trigrams, &mut truecase_tokens);

        truecase_tokens.join("")
    }
}

#[derive(Debug, Default)]
struct CaseStats {
    stats: HashMap<String, HashMap<String, u32>>,
}

impl CaseStats {
    fn add_token(&mut self, token: &Token) {
        self.add_string(token.original, &token.normalized)
    }

    fn add_ngram(&mut self, ngram: &[Token]) {
        let original = join_with_spaces(ngram.iter().map(Token::get_original));
        let normalized = join_with_spaces(ngram.iter().map(Token::get_normalized));
        self.add_string(&original, &normalized)
    }

    fn add_string(&mut self, original: &str, normalized: &str) {
        // currently it's impossible to add things to a hashmap ergonomically
        // using the .entry() API without needlessly cloning all of the source strings every time
        if let Some(counts) = self.stats.get_mut(normalized) {
            if let Some(count) = counts.get_mut(original) {
                *count += 1;
                return;
            }

            counts.insert(original.to_owned(), 1);
            return;
        }

        let mut counts = HashMap::with_capacity(1);
        counts.insert(original.to_owned(), 1);

        self.stats.insert(normalized.to_owned(), counts);
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

type Tokens<'a> = Filter<Tokenizer<'a>, fn(&Token) -> bool>;

fn tokenize(phrase: &str) -> Tokens {
    let tokens = Tokenizer {
        string: phrase,
        next_token: None,
    };
    tokens.filter(|t| !t.is_empty())
}

lazy_static! {
    static ref WORD_SEPARATORS: Regex = Regex::new(r#"[,.?!:;()«»„“”"—\s]+"#).unwrap();
}

struct Tokenizer<'a> {
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
enum TokenKind {
    Word,
    Separator,
}

#[derive(Debug, Clone)]
struct Token<'a> {
    pub original: &'a str,
    pub normalized: Cow<'a, str>,
    pub kind: TokenKind,
}

impl<'a> Token<'a> {
    fn new(original: &'a str, kind: TokenKind) -> Self {
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

    fn is_meaningful(&self) -> bool {
        self.kind == TokenKind::Word
    }

    fn is_empty(&self) -> bool {
        self.original.is_empty()
    }

    // these functions are only necessary because closures can't be cloned and
    // `join_with_spaces` requires a cloneable iterator
    fn get_normalized(&self) -> &str {
        self.normalized.as_ref()
    }

    fn get_original(&self) -> &str {
        self.original
    }
}

fn split_in_three(string: &str, index1: usize, index2: usize) -> (&str, &str, &str) {
    let (first, rest) = string.split_at(index1);
    let (second, third) = rest.split_at(index2 - index1);
    (first, second, third)
}

fn join_with_spaces<I>(mut iter: I) -> String
where
    I: Iterator + Clone,
    I::Item: AsRef<str>,
{
    let length: usize = iter.clone()
        .map(|item| item.as_ref().len() + 1)
        .sum::<usize>() - 1;
    let mut string = String::with_capacity(length);

    match iter.next() {
        Some(item) => string.push_str(item.as_ref()),
        None => return string,
    };

    for item in iter {
        string.push(' ');
        string.push_str(item.as_ref());
    }
    string
}

fn uppercase_first_letter(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

fn is_sentence_sane(sentence: &str) -> bool {
    !sentence.chars().all(char::is_uppercase) && !sentence.chars().all(char::is_lowercase)
        && sentence.trim().len() > 0
}

// this is only necessary because closures can't be cloned and
// `join_with_spaces` requires a cloneable iterator
fn get_second_item<A, B>(tuple: &(A, B)) -> &B {
    &tuple.1
}
