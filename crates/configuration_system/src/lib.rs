use std::path::{Path, PathBuf};

use dirs::config_dir;
use serde::de::DeserializeOwned;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("TOML parsing error: {0}")]
    Parsing(#[from] toml::de::Error),
}

pub trait Configuration: Default + DeserializeOwned + Merge {
    const DEFAULT_FILENAME: &'static str;

    fn default_path() -> PathBuf {
        let mut result = config_dir().unwrap();
        result.push("hulks");
        result.push(Self::DEFAULT_FILENAME);

        result
    }

    fn load() -> Result<Self, Error> {
        Self::load_from_file(Self::default_path())
    }

    fn load_from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        match std::fs::read_to_string(&path) {
            Ok(config_file) => {
                let mut configuration = Self::default();
                let user_configuration: Self = toml::from_str(&config_file)?;

                configuration.merge(user_configuration);

                Ok(configuration)
            }
            Err(error) => {
                log::info!(
                    "Could not load config file at {}: {error}",
                    path.as_ref().display()
                );

                Ok(Self::default())
            }
        }
    }
}

pub trait Merge {
    fn merge(&mut self, other: Self);
}

impl<T> Merge for Option<T> {
    fn merge(&mut self, other: Self) {
        if let Some(value) = other {
            *self = Some(value);
        }
    }
}

macro_rules! impl_merge {
    ($ty: ty) => {
        impl Merge for $ty {
            fn merge(&mut self, other: Self) {
                *self = other;
            }
        }
    };
}

impl_merge!(u8);
impl_merge!(i8);
impl_merge!(u16);
impl_merge!(i16);
impl_merge!(u32);
impl_merge!(i32);
impl_merge!(u64);
impl_merge!(i64);
impl_merge!(usize);
impl_merge!(isize);
impl_merge!(f32);
impl_merge!(f64);
impl_merge!(String);
// Add more primitive types here if you need them
