use serde::{Deserialize, Serialize};

use linear_algebra::{point, Point2};
use path_serde::{PathDeserialize, PathIntrospect, PathSerialize};

use coordinate_systems::Field;

#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    Deserialize,
    Serialize,
    PathSerialize,
    PathDeserialize,
    PathIntrospect,
)]
pub struct FieldDimensions {
    pub ball_radius: f32,
    pub length: f32,
    pub width: f32,
    pub line_width: f32,
    pub penalty_marker_size: f32,
    pub goal_box_area_length: f32,
    pub goal_box_area_width: f32,
    pub penalty_area_length: f32,
    pub penalty_area_width: f32,
    pub penalty_marker_distance: f32,
    pub center_circle_diameter: f32,
    pub border_strip_width: f32,
    pub goal_inner_width: f32,
    pub goal_post_diameter: f32,
    pub goal_depth: f32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Half {
    Own,
    Opponent,
}

impl Half {
    fn sign(self) -> f32 {
        match self {
            Half::Own => -1.0,
            Half::Opponent => 1.0,
        }
    }

    pub fn mirror(self) -> Self {
        match self {
            Half::Own => Half::Opponent,
            Half::Opponent => Half::Own,
        }
    }
}

pub enum Side {
    Left,
    Right,
}

impl Side {
    fn sign(self) -> f32 {
        match self {
            Side::Left => 1.0,
            Side::Right => -1.0,
        }
    }
}

impl FieldDimensions {
    pub fn is_inside_field(&self, position: Point2<Field>) -> bool {
        position.x().abs() < self.length / 2.0 && position.y().abs() < self.width / 2.0
    }

    pub fn is_inside_any_goal_box(&self, position: Point2<Field>) -> bool {
        position.x().abs() > self.length / 2.0 - self.goal_box_area_length
            && position.y().abs() < self.goal_box_area_width / 2.0
    }

    pub fn penalty_spot(&self, half: Half) -> Point2<Field> {
        let unsigned_x = self.length / 2.0 - self.penalty_marker_distance;
        point![unsigned_x * half.sign(), 0.0]
    }

    pub fn corner(&self, half: Half, side: Side) -> Point2<Field> {
        let unsigned_x = self.length / 2.0;
        let unsigned_y = self.width / 2.0;
        point![unsigned_x * half.sign(), unsigned_y * side.sign()]
    }

    pub fn goal_box_corner(&self, half: Half, side: Side) -> Point2<Field> {
        let unsigned_x = self.length / 2.0 - self.goal_box_area_length;
        let unsigned_y = self.goal_box_area_width / 2.0;
        point![unsigned_x * half.sign(), unsigned_y * side.sign()]
    }

    pub fn center(&self) -> Point2<Field> {
        Point2::origin()
    }

    pub fn left_goalpost(&self, half: Half) -> Point2<Field> {
        let sign = match half {
            Half::Own => -1.0,
            Half::Opponent => 1.0,
        };
        point![
            sign * self.length / 2.0,
            -sign * self.goal_inner_width / 2.0
        ]
    }

    pub fn right_goalpost(&self, half: Half) -> Point2<Field> {
        let mut left_goalpost = self.left_goalpost(half);
        left_goalpost.inner.y = -left_goalpost.inner.y;
        left_goalpost
    }
}
