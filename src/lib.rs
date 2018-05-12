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
extern crate indexmap;
#[macro_use]
extern crate lazy_static;
extern crate regex;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;

mod tokenizer;
mod utils;
mod trainer;
mod truecase;

pub use trainer::ModelTrainer;
pub use truecase::Model;
