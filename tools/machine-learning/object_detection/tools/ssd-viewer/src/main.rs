pub mod cli;
pub mod model;

use std::path::PathBuf;

use clap::Parser;
use cli::CliArguments;
use eframe::{
    egui::{self, pos2, Align2, CentralPanel, Color32, Image, Sense, Stroke, TextStyle},
    App, CreationContext, NativeOptions,
};
use model::ImageProcessor;

fn main() -> eframe::Result {
    let native_options = NativeOptions::default();
    eframe::run_native(
        "SSD Viewer",
        native_options,
        Box::new(|cc| Ok(Box::new(SsdViewer::new(cc)))),
    )
}

struct SsdViewer {
    image_path: PathBuf,
    image_processor: ImageProcessor,
}

impl SsdViewer {
    fn new(cc: &CreationContext<'_>) -> Self {
        egui_extras::install_image_loaders(&cc.egui_ctx);
        let arguments = CliArguments::parse();

        let image_processor = ImageProcessor::new(
            arguments.embedding_model,
            arguments.detection_model,
            &arguments.image,
        )
        .expect("failed to load processor");

        let all_queries = image_processor.query_all().expect("failed");

        SsdViewer {
            image_path: arguments.image,
            image_processor,
        }
    }
}

impl App for SsdViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        CentralPanel::default().show(ctx, |ui| {
            let uri = format!("file://{}", self.image_path.display());
            let response = ui.add(Image::new(uri).sense(Sense::click()));
            if let Some(position) = response.hover_pos() {
                let normalized_x = position.x / response.rect.width();
                let normalized_y = position.y / response.rect.height();
                ui.label(format!("x: {:.2}, y: {:.2}", normalized_x, normalized_y));
                // if response.clicked() {
                let output = self
                    .image_processor
                    .query(vec![pos2(normalized_x, normalized_y)]);
                let Ok((elapsed, results)) = output else {
                    let err = output.unwrap_err();
                    ui.colored_label(Color32::RED, err.to_string());
                    return;
                };
                ui.label(format!("{}ms", 1000.0 * elapsed.as_secs_f32()));
                for detection in results {
                    let bounding_box = detection.scale(response.rect.size()).bounding_box;
                    ui.painter_at(response.rect).rect_stroke(
                        bounding_box,
                        1.0,
                        Stroke::new(1.0, Color32::GREEN),
                    );
                    ui.painter_at(response.rect).text(
                        bounding_box.left_top(),
                        Align2::LEFT_BOTTOM,
                        detection.class(),
                        Default::default(),
                        Color32::GREEN,
                    );
                }
                // }
            }
        });
    }
}
