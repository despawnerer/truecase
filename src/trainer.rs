use std::io;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::collections::BTreeMap;
use std::fs::File;
use std::iter::once;

use indexmap::IndexMap;

use tokenizer::{tokenize, Token};
use truecase::{CaseMap, Model};
use utils::join_with_spaces;

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
    pub fn add_sentences_from_file<P: AsRef<Path>>(&mut self, path: P) -> io::Result<&mut Self> {
        let file = File::open(path)?;
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
        let mut unigrams = self.unigram_stats.into_most_frequent(1);
        let mut bigrams = self.bigram_stats.into_most_frequent(10);
        let mut trigrams = self.trigram_stats.into_most_frequent(10);

        trigrams.retain(|k, v| {
            let normalized_words = k.split(' ').collect::<Vec<_>>();
            let truecased_words = v.split(' ').collect::<Vec<_>>();

            let normalized_bigrams = normalized_words
                .windows(2)
                .map(|whatever| join_with_spaces(whatever.iter()));
            let truecased_bigrams = truecased_words
                .windows(2)
                .map(|whatever| join_with_spaces(whatever.iter()));

            normalized_bigrams
                .zip(truecased_bigrams)
                .any(|(k, v)| bigrams[&k] != v)
        });

        bigrams.retain(|k, v| {
            let normalized_words = k.split(' ');
            let truecased_words = v.split(' ');
            normalized_words
                .zip(truecased_words)
                .any(|(k, v)| unigrams[k] != v)
        });

        unigrams.retain(|k, v| k != v);

        Model {
            unigrams,
            bigrams,
            trigrams,
        }
    }
}

#[derive(Debug, Default)]
struct CaseStats {
    stats: IndexMap<String, CaseCounts>,
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
            counts.add(original, normalized);
            return;
        }

        let mut counts = CaseCounts::default();
        counts.add(original, normalized);

        self.stats.insert(normalized.to_owned(), counts);
    }

    fn into_most_frequent(self, min_frequency: u32) -> CaseMap {
        self.stats
            .into_iter()
            .flat_map(|(normalized, word_case_counts)| {
                word_case_counts
                    .into_most_frequent_kind(min_frequency)
                    .map(|kind| kind.into_to_string_from(&normalized))
                    .map(|truecased| (normalized, truecased))
            })
            .collect()
    }
}

#[derive(Debug, Default)]
struct CaseCounts {
    normalized: u32,
    other: BTreeMap<String, u32>,
}

impl CaseCounts {
    fn add(&mut self, string: &str, normalized: &str) {
        if string == normalized {
            self.normalized += 1;
        } else {
            if let Some(other_count) = self.other.get_mut(string) {
                *other_count += 1;
                return;
            }

            self.other.insert(string.to_owned(), 1);
        }
    }

    fn into_most_frequent_kind(self, min_frequency: u32) -> Option<CaseKind> {
        let normalized = (CaseKind::Normalized, self.normalized);
        let other_options = self.other
            .into_iter()
            .map(|(string, count)| (CaseKind::Other(string), count));

        once(normalized)
            .chain(other_options)
            .filter(|&(_, frequency)| frequency >= min_frequency)
            .max_by_key(|&(_, frequency)| frequency)
            .map(|(option, _)| option)
    }
}

enum CaseKind {
    Normalized,
    Other(String),
}

impl CaseKind {
    fn into_to_string_from(self, normalized: &str) -> String {
        match self {
            CaseKind::Normalized => normalized.to_owned(),
            CaseKind::Other(string) => string,
        }
    }
}

fn is_sentence_sane(sentence: &str) -> bool {
    !sentence.chars().all(char::is_uppercase) && !sentence.chars().all(char::is_lowercase)
        && !sentence.trim().is_empty()
}
