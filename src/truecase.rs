use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::errors::{ModelLoadingError, ModelSavingError};
use crate::tokenizer::{tokenize, Token};
use crate::utils::{join_with_spaces, uppercase_first_letter};

pub(crate) type CaseMap = HashMap<String, String>;

/// Truecasing model.
///
/// See [crate documentation](index.html) for examples.
#[derive(Serialize, Deserialize, Debug)]
pub struct Model {
    pub(crate) unigrams: CaseMap,
    pub(crate) bigrams: CaseMap,
    pub(crate) trigrams: CaseMap,
}

#[derive(Copy, Clone, Debug)]
enum Mode {
    Sentence,
    Phrase,
}

impl Model {
    /// Load a previously serialized model from a byte slice
    ///
    /// Use this method if you want to bundle the model into your binary instead
    /// of supplying it as a separate resource.
    pub fn deserialize(bytes: &[u8]) -> Result<Self, ModelLoadingError> {
        Ok(serde_json::from_slice(bytes)?)
    }

    /// Serialize the model into a vector of bytes
    pub fn serialize(&self) -> Result<Vec<u8>, ModelSavingError> {
        Ok(serde_json::to_vec(self)?)
    }

    /// Load a previously saved model from a file
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, ModelLoadingError> {
        let mut vec = Vec::new();
        File::open(path)?.read_to_end(&mut vec)?;
        Model::deserialize(&vec)
    }

    /// Save this model into a file with the given filename.
    /// The format is simple JSON right now.
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), ModelSavingError> {
        let serialized = Model::serialize(&self)?;
        File::create(path)?.write_all(&serialized)?;

        Ok(())
    }

    /// Restore word casings in a sentence.
    ///
    /// This will capitalize the first letter of the sentence if necessary.
    ///
    /// ```ignore
    /// assert_eq!(model.truecase("sample shakespeare text"), "Sample Shakespeare text");
    /// assert_eq!(model.truecase("iphones get sold quickly"), "iPhones get sold quickly");
    /// ```
    pub fn truecase(&self, sentence: &str) -> String {
        self.truecase_tokens(tokenize(sentence), Mode::Sentence)
    }

    /// Restore word casings in a phrase (sentence fragment).
    ///
    /// ```ignore
    /// assert_eq!(model.truecase_phrase("sample shakespeare text"), "sample Shakespeare text");
    /// assert_eq!(model.truecase_phrase("iphones get sold quickly"), "iPhones get sold quickly");
    /// ```
    pub fn truecase_phrase(&self, phrase: &str) -> String {
        self.truecase_tokens(tokenize(phrase), Mode::Phrase)
    }

    fn truecase_tokens<'a, I>(&self, tokens: I, mode: Mode) -> String
    where
        I: Iterator<Item = Token<'a>>,
    {
        let mut truecase_tokens = Vec::with_capacity(3);
        let mut normalized_words_with_indexes = Vec::with_capacity(3);

        let mut special_case_first_word = match mode {
            Mode::Sentence => true,
            Mode::Phrase => false,
        };

        for (index, token) in tokens.enumerate() {
            let truecased;

            if token.is_meaningful() {
                let from_unigrams = self
                    .unigrams
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

// this is only necessary because closures can't be cloned and
// `join_with_spaces` requires a cloneable iterator
fn get_second_item<A, B>(tuple: &(A, B)) -> &B {
    &tuple.1
}
