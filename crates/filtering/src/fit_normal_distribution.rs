use nalgebra::{Const, Dyn, Matrix, Point, VecStorage, Vector};
use types::multivariate_normal_distribution::MultivariateNormalDistribution;

pub trait Fit<const N: usize> {
    type SampleMatrix;
    type Sample;

    fn fit(samples: Vec<Self::Sample>) -> Self;
}

impl<const N: usize> Fit<N> for MultivariateNormalDistribution<N> {
    type SampleMatrix = Matrix<f32, Dyn, Const<N>, VecStorage<f32, Dyn, Const<N>>>;
    type Sample = Point<f32, N>;

    fn fit(samples: Vec<Self::Sample>) -> Self {
        let mean: Vector<_, _, _> = samples.iter().map(|point| point.coords).sum();
        let mean = mean / samples.len() as f32;

        let demeaned_samples: Vec<_> = samples
            .iter()
            .map(|sample| (sample.coords - mean).transpose())
            .collect();
        let demeaned_samples = Self::SampleMatrix::from_rows(&demeaned_samples);
        let covariance = demeaned_samples.transpose() * demeaned_samples;

        MultivariateNormalDistribution {
            mean: mean,
            covariance,
        }
    }
}
