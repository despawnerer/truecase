truecase.rs
===========

[![Latest Version](https://img.shields.io/crates/v/truecase.svg)](https://crates.io/crates/truecase)
[![docs](https://docs.rs/truecase/badge.svg)](https://docs.rs/truecase)

`truecase.rs` is a simple statistical truecaser written in Rust.

_Truecasing_ is restoration of original letter cases in text: for example, turning all-uppercase, or all-lowercase text into one that has proper sentence casing (capital first letter, capitalized names etc).

This crate attempts to solve this problem by gathering statistics from a set of training sentences, then using those statistics to restore correct casings in broken sentences. It comes with a command-line utility that makes training the statistical model easy.

Quick usage example
-------------------

```rust
use truecase::{Model, ModelTrainer};

// build a statistical model from sample sentences
let mut trainer = ModelTrainer::new();
trainer.add_sentence("There are very few writers as good as Shakespeare");
trainer.add_sentence("You and I will have to disagree about this");
trainer.add_sentence("She never came back from USSR");
let model = trainer.into_model();

// use gathered statistics to restore case in caseless text
let truecased_text = model.truecase("i don't think shakespeare was born in ussr");
assert_eq!(truecased_text, "I don't think Shakespeare was born in USSR");
```

See [documentation](https://docs.rs/truecase) for more details.

License
-------

truecase.rs is licensed under the terms of the MIT License or the Apache License 2.0, at your choosing.
