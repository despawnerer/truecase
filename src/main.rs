extern crate failure;
extern crate serde_json;
extern crate truecase;

use std::fs::File;
use std::io::{Read, Write};

use truecase::Model;
use failure::Error;

const TESTING_TRAINING_DATA: &'static str = include_str!("train.txt");

fn main() {
    let model = Model::train_on_text(TESTING_TRAINING_DATA).unwrap();
    // write_model(&model, "model.json").unwrap();
    // let model = read_model("model.json").unwrap();
    let shitty_text =
        "tHis Is a pretty Ridiculous testing text, but william sHakespeare wouldn'T APPROVE";
    let truecased_text = model.truecase(shitty_text);
    println!("{}", truecased_text);
}

fn read_model(filename: &str) -> Result<Model, Error> {
    let mut string = String::new();
    File::open(filename)?.read_to_string(&mut string)?;
    Ok(serde_json::from_str(&string)?)
}

fn write_model(model: &Model, filename: &str) -> Result<(), Error> {
    let serialized = serde_json::to_string(&model)?;
    File::create(filename)?.write_all(serialized.as_bytes())?;
    Ok(())
}
