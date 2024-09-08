use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
pub struct CliArguments {
    pub embedding_model: PathBuf,
    pub detection_model: PathBuf,
    pub image: PathBuf,
}
