use std::ops::Range;

use coordinate_systems::{Field, Ground};
use framework::AdditionalOutput;
use geometry::{
    is_inside_polygon::is_inside_polygon, line::Line, line_segment::LineSegment, look_at::LookAt,
};
use linear_algebra::{distance, point, Orientation2, Point2, Pose2};
use ordered_float::NotNan;
use spl_network_messages::{GamePhase, SubState, Team};
use types::{
    field_dimensions::{FieldDimensions, Half},
    filtered_game_controller_state::FilteredGameControllerState,
    motion_command::MotionCommand,
    parameters::RolePositionsParameters,
    path_obstacles::PathObstacle,
    support_foot::Side,
    world_state::{BallState, WorldState},
};

use super::{head::LookAction, walk_to_pose::WalkAndStand};

pub struct Defend<'cycle> {
    world_state: &'cycle WorldState,
    field_dimensions: &'cycle FieldDimensions,
    role_positions: &'cycle RolePositionsParameters,
    walk_and_stand: &'cycle WalkAndStand<'cycle>,
    look_action: &'cycle LookAction<'cycle>,
}

impl<'cycle> Defend<'cycle> {
    pub fn new(
        world_state: &'cycle WorldState,
        field_dimensions: &'cycle FieldDimensions,
        role_positions: &'cycle RolePositionsParameters,
        walk_and_stand: &'cycle WalkAndStand,
        look_action: &'cycle LookAction,
    ) -> Self {
        Self {
            world_state,
            field_dimensions,
            role_positions,
            walk_and_stand,
            look_action,
        }
    }

    fn with_orient_to(
        &self,
        pose: Pose2<Ground>,
        orientation: Orientation2<Ground>,
        path_obstacles_output: &mut AdditionalOutput<Vec<PathObstacle>>,
    ) -> Option<MotionCommand> {
        self.walk_and_stand.return_to_defend(
            pose,
            self.look_action.execute(),
            orientation,
            path_obstacles_output,
        )
    }

    fn with_pose(
        &self,
        pose: Pose2<Ground>,
        path_obstacles_output: &mut AdditionalOutput<Vec<PathObstacle>>,
    ) -> Option<MotionCommand> {
        self.walk_and_stand
            .execute(pose, self.look_action.execute(), path_obstacles_output)
    }

    pub fn left(
        &self,
        path_obstacles_output: &mut AdditionalOutput<Vec<PathObstacle>>,
    ) -> Option<MotionCommand> {
        let pose = defend_pose(
            self.world_state,
            self.field_dimensions,
            self.role_positions,
            Side::Left,
        )?;
        self.with_orient_to(pose, pose.orientation(), path_obstacles_output)
    }

    pub fn right(
        &self,
        path_obstacles_output: &mut AdditionalOutput<Vec<PathObstacle>>,
    ) -> Option<MotionCommand> {
        let pose = defend_pose(
            self.world_state,
            self.field_dimensions,
            self.role_positions,
            Side::Right,
        )?;
        self.with_orient_to(pose, pose.orientation(), path_obstacles_output)
    }

    pub fn penalty_kick(
        &self,
        path_obstacles_output: &mut AdditionalOutput<Vec<PathObstacle>>,
    ) -> Option<MotionCommand> {
        let pose =
            defend_penalty_kick(self.world_state, self.field_dimensions, self.role_positions)?;
        self.with_pose(pose, path_obstacles_output)
    }

    pub fn goal(
        &self,
        path_obstacles_output: &mut AdditionalOutput<Vec<PathObstacle>>,
    ) -> Option<MotionCommand> {
        let pose = defend_goal_pose(self.world_state, self.field_dimensions, self.role_positions)?;
        self.with_pose(pose, path_obstacles_output)
    }

    pub fn kick_off(
        &self,
        path_obstacles_output: &mut AdditionalOutput<Vec<PathObstacle>>,
    ) -> Option<MotionCommand> {
        let pose =
            defend_kick_off_pose(self.world_state, self.field_dimensions, self.role_positions)?;
        self.with_pose(pose, path_obstacles_output)
    }
}

fn defend_pose(
    world_state: &WorldState,
    field_dimensions: &FieldDimensions,
    role_positions: &RolePositionsParameters,
    side: Side,
) -> Option<Pose2<Ground>> {
    let ground_to_field = world_state.robot.ground_to_field?;
    let sign = match side {
        Side::Left => 1.0,
        Side::Right => -1.0,
    };

    let default_defender_position = point![
        -field_dimensions.length / 2.0 + role_positions.defender_y_offset,
        sign * role_positions.defender_y_offset,
    ];

    let Some(ball) = world_state.rule_ball.or(world_state.ball) else {
        return Some(
            ground_to_field.inverse() * Pose2::new(default_defender_position.coords(), 0.0),
        );
    };

    // let position_to_defend = point![
    //     -field_dimensions.length / 2.0,
    //     match side {
    //         Side::Left => role_positions.defender_y_offset,
    //         Side::Right => -role_positions.defender_y_offset,
    //     }
    // ];

    // let distance_to_target = if side == ball.field_side {
    //     role_positions.defender_aggressive_ring_radius
    // } else {
    //     role_positions.defender_passive_ring_radius
    // };
    // let distance_to_target = penalty_kick_defender_radius(
    //     distance_to_target,
    //     world_state.filtered_game_controller_state,
    //     field_dimensions,
    // );

    // let defend_pose = block_on_circle(ball.ball_in_field, position_to_defend, distance_to_target);
    // Some(ground_to_field.inverse() * defend_pose)

    let defender_radius = role_positions.defender_base_radius;
    let defender_polygon = [
        point![-field_dimensions.length / 2.0, sign * 2.0 * defender_radius],
        point![
            -field_dimensions.length / 2.0 + field_dimensions.penalty_area_length,
            sign * 2.0 * defender_radius,
        ],
        point![
            -field_dimensions.length / 2.0 + field_dimensions.penalty_area_length,
            sign * field_dimensions.goal_box_area_width / 2.0,
        ],
        point![
            -field_dimensions.length / 2.0,
            sign * field_dimensions.width / 2.0,
        ],
    ];

    Some(
        ground_to_field.inverse()
            * defender_pose(
                ball.ball_in_field,
                field_dimensions,
                role_positions.defender_base_radius,
                &defender_polygon,
            ),
    )
}

fn defender_pose(
    ball_position: Point2<Field>,
    field_dimensions: &FieldDimensions,
    defender_radius: f32,
    defender_polygon: &[Point2<Field>],
) -> Pose2<Field> {
    let left_goalpost = field_dimensions.left_goalpost(Half::Own);
    let right_goalpost = field_dimensions.right_goalpost(Half::Own);

    let ball_to_left_goalpost = left_goalpost - ball_position;
    let ball_to_right_goalpost = right_goalpost - ball_position;

    let half_opening_angle = ball_to_left_goalpost.angle(ball_to_right_goalpost) / 2.0;
    let maximum_goal_closing_distance = defender_radius / half_opening_angle.tan();

    let ball_opening_center_line = (ball_to_right_goalpost + ball_to_left_goalpost).normalize();

    let goal_closing_defender_position =
        ball_position + ball_opening_center_line * maximum_goal_closing_distance;

    if is_inside_polygon(defender_polygon, goal_closing_defender_position) {
        return Pose2::new(
            goal_closing_defender_position.coords(),
            goal_closing_defender_position
                .look_at(&ball_position)
                .angle(),
        );
    }

    let segments = defender_polygon
        .iter()
        .zip(defender_polygon.iter().cycle().skip(1))
        .map(|(start, end)| LineSegment::new(*start, *end));

    let clipped_position = segments
        .map(|segment| segment.closest_point(goal_closing_defender_position))
        .min_by_key(|&clipped_defender| {
            NotNan::new(distance(clipped_defender, goal_closing_defender_position))
                .expect("expected distance to be finite")
        })
        .expect("expected at least one segment");

    Pose2::new(
        clipped_position.coords(),
        clipped_position.look_at(&ball_position).angle(),
    )
}

fn defend_penalty_kick(
    world_state: &WorldState,
    field_dimensions: &FieldDimensions,
    role_positions: &RolePositionsParameters,
) -> Option<Pose2<Ground>> {
    let ground_to_field = world_state.robot.ground_to_field?;
    let ball = world_state
        .rule_ball
        .or(world_state.ball)
        .unwrap_or_else(|| BallState::new_at_center(ground_to_field));

    let position_to_defend = point![
        (-field_dimensions.length + field_dimensions.penalty_area_length) / 2.0,
        0.0
    ];
    let mut distance_to_target = if ball.field_side == Side::Left {
        role_positions.defender_aggressive_ring_radius
    } else {
        role_positions.defender_passive_ring_radius
    };
    distance_to_target = penalty_kick_defender_radius(
        distance_to_target,
        world_state.filtered_game_controller_state,
        field_dimensions,
    );

    let defend_pose = block_on_circle(ball.ball_in_field, position_to_defend, distance_to_target);
    Some(ground_to_field.inverse() * defend_pose)
}

fn defend_goal_pose(
    world_state: &WorldState,
    field_dimensions: &FieldDimensions,
    role_positions: &RolePositionsParameters,
) -> Option<Pose2<Ground>> {
    let ground_to_field = world_state.robot.ground_to_field?;
    let ball = world_state
        .rule_ball
        .or(world_state.ball)
        .unwrap_or_else(|| BallState::new_at_center(ground_to_field));

    let keeper_x_offset = match world_state.filtered_game_controller_state {
        Some(
            FilteredGameControllerState {
                game_phase:
                    GamePhase::PenaltyShootout {
                        kicking_team: Team::Opponent,
                    },
                ..
            }
            | FilteredGameControllerState {
                sub_state: Some(SubState::PenaltyKick),
                kicking_team: Team::Opponent,
                ..
            },
        ) => 0.0,
        _ => role_positions.keeper_x_offset,
    };

    let position_to_defend = point![-field_dimensions.length / 2.0 - 1.0, 0.0];
    let defend_pose = block_on_line(
        ball.ball_in_field,
        position_to_defend,
        -field_dimensions.length / 2.0 + keeper_x_offset,
        -0.7..0.7,
    );
    Some(ground_to_field.inverse() * defend_pose)
}

fn defend_kick_off_pose(
    world_state: &WorldState,
    field_dimensions: &FieldDimensions,
    role_positions: &RolePositionsParameters,
) -> Option<Pose2<Ground>> {
    let ground_to_field = world_state.robot.ground_to_field?;
    let absolute_ball_position = match world_state.ball {
        Some(ball) => ball.ball_in_field,
        None => Point2::origin(),
    };
    let position_to_defend = point![-field_dimensions.length / 2.0, 0.0];
    let center_circle_radius = field_dimensions.center_circle_diameter / 2.0;
    let distance_to_target = distance(position_to_defend, absolute_ball_position)
        - center_circle_radius
        - role_positions.striker_distance_to_non_free_center_circle;
    let defend_pose = block_on_circle(
        absolute_ball_position,
        position_to_defend,
        distance_to_target,
    );
    Some(ground_to_field.inverse() * defend_pose)
}

pub fn block_on_circle(
    ball_position: Point2<Field>,
    target: Point2<Field>,
    distance_to_target: f32,
) -> Pose2<Field> {
    let target_to_ball = ball_position - target;
    let block_position = target + (target_to_ball.normalize() * distance_to_target);
    Pose2::new(
        block_position.coords(),
        block_position.look_at(&ball_position).angle(),
    )
}

fn block_on_line(
    ball_position: Point2<Field>,
    target: Point2<Field>,
    defense_line_x: f32,
    defense_line_y_range: Range<f32>,
) -> Pose2<Field> {
    let is_ball_in_front_of_defense_line = defense_line_x < ball_position.x();
    if is_ball_in_front_of_defense_line {
        let defense_line = Line(
            point![defense_line_x, defense_line_y_range.start],
            point![defense_line_x, defense_line_y_range.end],
        );
        let ball_target_line = Line(ball_position, target);
        let intersection_point = defense_line.intersection(&ball_target_line);
        let defense_position = point![
            intersection_point.x(),
            intersection_point
                .y()
                .clamp(defense_line_y_range.start, defense_line_y_range.end)
        ];
        Pose2::new(
            defense_position.coords(),
            defense_position.look_at(&ball_position).angle(),
        )
    } else {
        let defense_position = point![
            defense_line_x,
            (defense_line_y_range.start + defense_line_y_range.end) / 2.0
        ];
        Pose2::new(
            defense_position.coords(),
            defense_position.look_at(&ball_position).angle(),
        )
    }
}

fn penalty_kick_defender_radius(
    distance_to_target: f32,
    filtered_game_controller_state: Option<FilteredGameControllerState>,
    field_dimensions: &FieldDimensions,
) -> f32 {
    if let Some(FilteredGameControllerState {
        kicking_team: Team::Opponent,
        sub_state: Some(SubState::PenaltyKick),
        ..
    }) = filtered_game_controller_state
    {
        let half_penalty_width = field_dimensions.penalty_area_width / 2.0;
        let minimum_penalty_defender_radius =
            nalgebra::vector![field_dimensions.penalty_area_length, half_penalty_width].norm();
        distance_to_target.max(minimum_penalty_defender_radius)
    } else {
        distance_to_target
    }
}
