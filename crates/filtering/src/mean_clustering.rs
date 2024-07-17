use types::detected_feet::{ClusterPoint, CountedCluster};

pub trait MeanClustering {
    fn push(&mut self, other: ClusterPoint);
}

impl MeanClustering for CountedCluster {
    fn push(&mut self, other: ClusterPoint) {
        self.samples.push(other.pixel_coordinates);
        let in_ground = other.position_in_ground;
        self.running_mean_in_ground = (self.running_mean_in_ground * self.samples.len() as f32
            + in_ground.coords())
            / (self.samples.len() + 1) as f32;

        if in_ground.x() < self.leftmost_point.x() {
            self.leftmost_point = in_ground;
        } else if in_ground.x() > self.rightmost_point.x() {
            self.rightmost_point = in_ground;
        }
    }
}
