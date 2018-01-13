truecase.rs
===========

[![Latest Version](https://img.shields.io/crates/v/truecase.svg)](https://crates.io/crates/truecase)
[![docs](https://docs.rs/truecase/badge.svg)](https://docs.rs/truecase)

`truecase.rs` is a simple statistical truecaser written in Rust.

_Truecasing_ is restoration of original letter cases in text: for example, turning all-uppercase, or all-lowercase text into one that has proper sentence casing (capital first letter, capitalized names etc).

This crate attempts to solve this problem by gathering statistics from a set of training sentences, then using those statistics to restore correct casings in broken sentences. It comes with a command-line utility that makes training the statistical model easy.

Usage
-----

```rust
use truecase::Model;

let model = Model::load_from_file("my_pretrained_model.json")?;
let truecased_text = model.truecase("i don't think shakespeare would approve of this sample text");
assert_eq!(truecase_text, "I don't think Shakespeare would approve of this sample text");
```

See [documentation](https://docs.rs/truecase) for more details about training a model

License
-------

truecase.rs is licensed under the terms of the MIT License or the Apache License 2.0, at your choosing.
