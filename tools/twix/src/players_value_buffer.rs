use std::sync::Arc;

use color_eyre::Result;

use types::players::Players;

use crate::{nao::Nao, value_buffer::ValueBuffer};

pub struct PlayersValueBuffer(pub Players<ValueBuffer>);

impl PlayersValueBuffer {
    pub fn try_new(nao: Arc<Nao>, prefix: &str, output: &str) -> Result<Self> {
        let buffers = Players {
            one: nao.subscribe_output(format!("{prefix}.one.{output}")),
            two: nao.subscribe_output(format!("{prefix}.two.{output}")),
            three: nao.subscribe_output(format!("{prefix}.three.{output}")),
            four: nao.subscribe_output(format!("{prefix}.four.{output}")),
            five: nao.subscribe_output(format!("{prefix}.five.{output}")),
            six: nao.subscribe_output(format!("{prefix}.six.{output}")),
            seven: nao.subscribe_output(format!("{prefix}.seven.{output}")),
        };

        Ok(Self(buffers))
    }
}
