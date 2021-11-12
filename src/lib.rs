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
//! # Quick usage example
//!
//! ```
//! use truecase::{Model, ModelTrainer};
//!
//! // build a statistical model from sample sentences
//! let mut trainer = ModelTrainer::new();
//! trainer.add_sentence("There are very few writers as good as Shakespeare");
//! trainer.add_sentence("You and I will have to disagree about this");
//! trainer.add_sentence("She never came back from USSR");
//! let model = trainer.into_model();
//!
//! // use gathered statistics to restore case in caseless text
//! let truecased_text = model.truecase("i don't think shakespeare was born in ussr");
//! assert_eq!(truecased_text, "I don't think Shakespeare was born in USSR");
//! ```
//!
//! # Building a model a model using the CLI tool
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

mod tokenizer;
mod utils;
mod trainer;
mod truecase;

pub use trainer::ModelTrainer;
pub use truecase::Model;
pub use failure::Error;
