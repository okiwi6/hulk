use std::{str::FromStr, sync::Arc};

use color_eyre::Result;
use communication::client::CyclerOutput;
use coordinate_systems::Transform;
use eframe::epaint::{Color32, Stroke};
use nalgebra::Isometry2;
use types::{
    coordinate_systems::{Field, Ground},
    field_dimensions::FieldDimensions,
    obstacles::Obstacle,
};

use crate::{
    nao::Nao, panels::map::layer::Layer, twix_painter::TwixPainter, value_buffer::ValueBuffer,
};

pub struct Obstacles {
    ground_to_field: ValueBuffer,
    obstacles: ValueBuffer,
}

impl Layer for Obstacles {
    const NAME: &'static str = "Obstacles";

    fn new(nao: Arc<Nao>) -> Self {
        let ground_to_field =
            nao.subscribe_output(CyclerOutput::from_str("Control.main.ground_to_field").unwrap());
        let obstacles =
            nao.subscribe_output(CyclerOutput::from_str("Control.main.obstacles").unwrap());
        Self {
            ground_to_field,
            obstacles,
        }
    }

    fn paint(
        &self,
        painter: &TwixPainter<Field>,
        _field_dimensions: &FieldDimensions,
    ) -> Result<()> {
        let ground_to_field: Transform<Ground, Field, Isometry2<f32>> =
            self.ground_to_field.parse_latest().unwrap_or_default();
        let obstacles: Vec<Obstacle> = self.obstacles.require_latest()?;

        let hip_height_stroke = Stroke {
            width: 0.025,
            color: Color32::RED,
        };
        let foot_height_stroke = Stroke {
            width: 0.025,
            color: Color32::BLUE,
        };
        for obstacle in obstacles {
            painter.circle_stroke(
                ground_to_field * obstacle.position,
                obstacle.radius_at_hip_height,
                hip_height_stroke,
            );
            painter.circle_stroke(
                ground_to_field * obstacle.position,
                obstacle.radius_at_foot_height,
                foot_height_stroke,
            );
        }
        Ok(())
    }
}
