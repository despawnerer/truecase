use thiserror::Error;

#[derive(Error, Debug)]
pub enum ModelLoadingError {
    #[error("cannot read model from file: {0}")]
    ReadFile(#[from] std::io::Error),
    #[error("malformed model file: {0}")]
    Deserialize(#[from] serde_json::Error),
}

#[derive(Error, Debug)]
pub enum ModelSavingError {
    #[error("cannot write model into file: {0}")]
    WriteFile(#[from] std::io::Error),
    #[error("can't serialize model: {0}")]
    Serialize(#[from] serde_json::Error),
}
