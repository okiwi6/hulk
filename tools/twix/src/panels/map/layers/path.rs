use std::{str::FromStr, sync::Arc};

use color_eyre::Result;
use eframe::epaint::Color32;

use communication::client::CyclerOutput;
use coordinate_systems::Ground;

use types::{field_dimensions::FieldDimensions, motion_command::MotionCommand};

use crate::{
    nao::Nao, panels::map::layer::Layer, twix_painter::TwixPainter, value_buffer::ValueBuffer,
};

pub struct Path {
    motion_command: ValueBuffer,
}

impl Layer<Ground> for Path {
    const NAME: &'static str = "Path";

    fn new(nao: Arc<Nao>) -> Self {
        let motion_command =
            nao.subscribe_output(CyclerOutput::from_str("Control.main.motion_command").unwrap());
        Self { motion_command }
    }

    fn paint(
        &self,
        painter: &TwixPainter<Ground>,
        _field_dimensions: &FieldDimensions,
    ) -> Result<()> {
        let motion_command: MotionCommand = self.motion_command.require_latest()?;

        if let MotionCommand::Walk { path, .. } = motion_command {
            painter.path(path, Color32::BLUE, Color32::LIGHT_BLUE, 0.025);
        }
        Ok(())
    }
}
