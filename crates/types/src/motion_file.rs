use std::{fs::File, path::Path, time::Duration};

use color_eyre::eyre::{Result, WrapErr};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::from_reader;
use splines::{Spline, Key, Interpolation, impl_Interpolate};

use crate::Joints;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct MotionFile {
    pub initial_positions: Joints,
    pub frames: Vec<MotionFileFrame>,
}

impl MotionFile {
    pub fn from_path(motion_file_path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(&motion_file_path).wrap_err_with(|| {
            format!("failed to open motion file {:?}", motion_file_path.as_ref())
        })?;
        from_reader(file).wrap_err_with(|| {
            format!(
                "failed to parse motion file {:?}",
                motion_file_path.as_ref()
            )
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
pub struct MotionFileFrame {
    #[serde(
        serialize_with = "serialize_float_seconds",
        deserialize_with = "deserialize_float_seconds"
    )]
    pub duration: Duration,
    pub positions: Joints,
}

fn serialize_float_seconds<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_f32(duration.as_secs_f32())
}

fn deserialize_float_seconds<'de, D>(deserializer: D) -> Result<Duration, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Duration::from_secs_f32(f32::deserialize(deserializer)?))
}


impl_Interpolate!(f32, Joints, std::f32::consts::PI);
pub struct MotionFileInterpolator {
    interpolators: Spline<f32, Joints>,
    current_time: Duration
}

impl From<MotionFile> for MotionFileInterpolator {
    fn from(motion_file: MotionFile) -> Self {
        assert!(!motion_file.frames.is_empty());

        let mut current_time = Duration::ZERO;
        let mut keys = vec![Key::new(current_time.as_secs_f32(), motion_file.initial_positions, Interpolation::Linear)];

        keys.extend(motion_file.frames.into_iter().map(|frame| {
            current_time += frame.duration;
            Key::new(current_time.as_secs_f32(), frame.positions, Interpolation::Linear)
        }));

        Self {
            interpolators: Spline::from_vec(keys),
            current_time: Duration::ZERO,
        }
    }
}

impl MotionFileInterpolator {
    pub fn reset(&mut self) {
        self.current_time = Duration::ZERO;
    }

    pub fn step(&mut self, time_step: Duration) -> Joints {
        self.current_time += time_step;
        self.value()
    }

    pub fn value(&self) -> Joints {
        let arg: f32 = self.current_time.as_secs_f32();
        self.interpolators.clamped_sample(arg).unwrap()
    }

    pub fn is_finished(&self) -> bool {
        self.interpolators.keys().last().unwrap().t <= self.current_time.as_secs_f32()
    }
}

