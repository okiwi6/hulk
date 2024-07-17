use linear_algebra::Point2;
use path_serde::{PathDeserialize, PathIntrospect, PathSerialize};
use serde::{Deserialize, Serialize};

use coordinate_systems::{Ground, Pixel};

use crate::multivariate_normal_distribution::MultivariateNormalDistribution;

#[derive(
    Default, Clone, Debug, Deserialize, Serialize, PathSerialize, PathDeserialize, PathIntrospect,
)]
pub struct DetectedFeet {
    pub detection_in_ground: MultivariateNormalDistribution<2>,
}

#[derive(
    Default, Clone, Debug, Deserialize, Serialize, PathSerialize, PathDeserialize, PathIntrospect,
)]
pub struct ClusterPoint {
    pub pixel_coordinates: Point2<Pixel, u16>,
    pub position_in_ground: Point2<Ground>,
}

#[derive(
    Default, Clone, Debug, Deserialize, Serialize, PathSerialize, PathDeserialize, PathIntrospect,
)]
pub struct CountedCluster {
    pub leftmost_point: Point2<Ground>,
    pub rightmost_point: Point2<Ground>,
    pub running_mean_in_ground: Point2<Ground>,
    pub samples: Vec<Point2<Pixel, u16>>,
}

impl From<ClusterPoint> for CountedCluster {
    fn from(other: ClusterPoint) -> Self {
        CountedCluster {
            running_mean_in_ground: other.position_in_ground,
            samples: vec![other.pixel_coordinates],
            leftmost_point: other.position_in_ground,
            rightmost_point: other.position_in_ground,
        }
    }
}
