use std::{
    path::Path,
    time::{Duration, Instant},
};

use color_eyre::{eyre::Context, Result};
use eframe::egui::{pos2, Color32, ColorImage, Pos2, Rect, Vec2};
use image::{imageops::FilterType, DynamicImage, GenericImageView, ImageReader};
use ndarray::{s, Array, Array1, Array3, ArrayD, ArrayView1, Axis};
use ort::{GraphOptimizationLevel, Session, SessionInputs, Tensor};

const VOC_CLASSES: [&'static str; 20] = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
];

const CLASS_COLORS: [Color32; 20] = [
    Color32::RED,
    Color32::GOLD,
    Color32::GREEN,
    Color32::BLUE,
    Color32::LIGHT_BLUE,
    Color32::LIGHT_RED,
    Color32::LIGHT_GRAY,
    Color32::DARK_GRAY,
    Color32::YELLOW,
    Color32::KHAKI,
    Color32::WHITE,
    Color32::DARK_BLUE,
    Color32::BLACK,
    Color32::BLACK,
    Color32::BLACK,
    Color32::BLACK,
    Color32::BLACK,
    Color32::BLACK,
    Color32::BLACK,
    Color32::BLACK,
];

pub struct ImageProcessor {
    _embedding_model: Session,
    detection_model: Session,
    embedding: ArrayD<f32>,
}

#[derive(Debug, Copy, Clone)]
pub struct DetectionResult {
    pub bounding_box: Rect,
    pub class_index: usize,
}

impl DetectionResult {
    pub fn scale(self, scale: Vec2) -> Self {
        let min = self.bounding_box.min;
        let bounding_box = Rect::from_min_size(
            pos2(min.x * scale.x, min.y * scale.y),
            self.bounding_box.size() * scale,
        );
        DetectionResult {
            bounding_box,
            class_index: self.class_index,
        }
    }

    pub fn class(&self) -> &'static str {
        match self.class_index {
            0 => "background",
            1.. => VOC_CLASSES[self.class_index - 1],
        }
    }

    pub fn color(&self) -> Color32 {
        match self.class_index {
            0 => Color32::TRANSPARENT,
            1.. => CLASS_COLORS[self.class_index - 1],
        }
    }
}

fn load_model(model: impl AsRef<Path>) -> Result<Session> {
    Session::builder()
        .wrap_err("failed to build session")?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .wrap_err("failed to set optimization level")?
        .with_intra_threads(4)
        .wrap_err("failed to set intra threads")?
        .commit_from_file(model)
        .wrap_err("failed to load model")
}

fn compute_embedding(model: &Session, image: &DynamicImage) -> Result<ArrayD<f32>> {
    let image = image.resize_exact(224, 224, FilterType::Nearest);

    let tensor = Array::from_shape_fn((1, 3, 224, 224), |(_, channel, height, width)| {
        let pixel = image.get_pixel(width as u32, height as u32);
        let normalized_color = (pixel[channel] as f32) / 255.0;
        2.0 * normalized_color - 1.0
    });
    let tensor = Tensor::from_array(tensor).wrap_err("failed to create input tensor")?;
    let input = SessionInputs::from(vec![("input", tensor)]);

    let output = model.run(input).wrap_err("failed model inference")?;
    let embedding = output["output"]
        .try_extract_tensor::<f32>()
        .wrap_err("failed to extract tensor")?
        .into_owned();

    Ok(embedding)
}

fn softmax(logits: ArrayView1<f32>) -> Array1<f32> {
    let exp = logits.mapv(|x| x.exp());
    let sum: f32 = logits.iter().sum();
    exp.mapv(|x| x / sum)
}

fn argmax(softmax_values: Array1<f32>) -> (usize, f32) {
    let (index, score) = softmax_values
        .indexed_iter()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap();
    (index, *score)
}

impl ImageProcessor {
    pub fn new(
        embedding_model: impl AsRef<Path>,
        detection_model: impl AsRef<Path>,
        image: impl AsRef<Path>,
    ) -> Result<Self> {
        let embedding_model =
            load_model(embedding_model).wrap_err("failed to load embedding model")?;
        let detection_model =
            load_model(detection_model).wrap_err("failed to load detection model")?;

        let image = ImageReader::open(image)
            .wrap_err("failed to load image")?
            .decode()
            .wrap_err("failed to decode image")?;

        let embedding =
            compute_embedding(&embedding_model, &image).wrap_err("failed to compute embedding")?;

        Ok(Self {
            _embedding_model: embedding_model,
            detection_model,
            embedding,
        })
    }

    fn predict(&self, query: Tensor<f32>) -> Result<Vec<DetectionResult>> {
        let embedding = Tensor::from_array(self.embedding.view())
            .wrap_err("failed to create embedding tensor")?;
        let input = SessionInputs::from(vec![("embedding", embedding), ("queries", query)]);
        let output = self.detection_model.run(input)?;

        let prediction = output["output"].try_extract_tensor::<f32>()?.into_owned();

        let results = prediction
            .lanes(Axis(2))
            .into_iter()
            .map(|lane| {
                let bounding_box = lane.slice(s![..4]);
                let logits = lane.slice(s![4..]);
                let softmax = softmax(logits);
                let (class_index, _score) = argmax(softmax);

                let bounding_box = Rect::from_min_max(
                    pos2(bounding_box[0], bounding_box[1]),
                    pos2(bounding_box[2], bounding_box[3]),
                );
                DetectionResult {
                    bounding_box,
                    class_index,
                }
            })
            .collect::<Vec<_>>();

        Ok(results)
    }

    pub fn query(
        &self,
        normalized_position: Vec<Pos2>,
    ) -> Result<(Duration, Vec<DetectionResult>)> {
        let sample_points =
            Array3::from_shape_fn((1, normalized_position.len(), 2), |(_, p, c)| {
                let point = normalized_position[p];
                match c {
                    0 => point.x,
                    1 => point.y,
                    _ => unreachable!(),
                }
            });
        let sample_points =
            Tensor::from_array(sample_points).wrap_err("failed to create sample points tensor")?;

        let now = Instant::now();
        let result = self.predict(sample_points)?;
        let elapsed = now.elapsed();

        Ok((elapsed, result))
    }

    pub fn query_all(&self) -> Result<ColorImage> {
        let sample_points = Array3::from_shape_fn((1, 224 * 224, 2), |(_, i, c)| {
            let x = i % 224;
            let y = i / 224;
            match c {
                0 => x as f32 / 224.0,
                1 => y as f32 / 224.0,
                _ => unreachable!(),
            }
        });
        let sample_points = Tensor::from_array(sample_points)
            .wrap_err("failed to create sample points tensor")
            .unwrap();
        let result = self.predict(sample_points)?;

        let mut image = ColorImage::new([224, 224], Color32::TRANSPARENT);
        for (i, detection) in result.iter().enumerate() {
            let x = i % 224;
            let y = i / 224;
            let color = detection.color();
            image[(x, y)] = color;
        }

        Ok(image)
    }
}
