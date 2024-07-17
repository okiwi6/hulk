use nalgebra::{SMatrix, SVector};
use path_serde::{PathDeserialize, PathIntrospect, PathSerialize};
use serde::{Deserialize, Serialize};

#[derive(
    Clone,
    Copy,
    Debug,
    Serialize,
    Deserialize,
    PathSerialize,
    PathDeserialize,
    PathIntrospect,
    PartialEq,
)]
pub struct MultivariateNormalDistribution<const DIMENSION: usize> {
    pub mean: SVector<f32, DIMENSION>,
    #[path_serde(leaf)]
    pub covariance: SMatrix<f32, DIMENSION, DIMENSION>,
}

impl<const N: usize> Default for MultivariateNormalDistribution<N> {
    fn default() -> Self {
        Self {
            mean: SVector::zeros(),
            covariance: SMatrix::identity(),
        }
    }
}
