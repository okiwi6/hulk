pub mod control_pane;
mod export;
pub mod github_integration;
pub mod polygon;
mod preview_widget;
pub mod segmentator_widget;

use std::{io::Cursor, path::PathBuf};

use color_eyre::{eyre::eyre, Result};
use configuration_system::{Configuration, Merge};
use control_pane::ControlPane;
use eframe::{
    egui::{vec2, CentralPanel, Context, IconData, SidePanel, TopBottomPanel, ViewportBuilder},
    run_native, App, Frame, NativeOptions,
};
use github_integration::GithubAccount;
use image::ImageReader;
use preview_widget::PreviewWidget;
use segmentator_widget::{SegmentationState, Segmentator};
use serde::Deserialize;

#[derive(Default, Deserialize, Debug)]
struct SegmentatorConfiguration {
    github_username: Option<String>,
}

impl Configuration for SegmentatorConfiguration {
    const DEFAULT_FILENAME: &'static str = "segmentator.toml";
}

impl Merge for SegmentatorConfiguration {
    fn merge(&mut self, other: Self) {
        self.github_username.merge(other.github_username)
    }
}

fn app_icon() -> IconData {
    let app_icon = include_bytes!("../labello.png");
    let image = ImageReader::new(Cursor::new(app_icon))
        .with_guessed_format()
        .expect("failed to guess app icon format")
        .decode()
        .expect("failed to decode app icon");
    let (width, height) = (image.width(), image.height());

    let rgba = image.into_rgba8();
    IconData {
        width,
        height,
        rgba: rgba.into_raw(),
    }
}

fn main() -> Result<()> {
    let viewport = ViewportBuilder::default()
        .with_icon(app_icon())
        .with_title("Labello");
    let native_options = NativeOptions {
        viewport,
        ..Default::default()
    };

    run_native(
        "Labello",
        native_options,
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);

            let configuration = SegmentatorConfiguration::load()?;

            let app = SegmentatorApp::new(configuration);

            Ok(Box::new(app))
        }),
    )
    .map_err(|error| eyre!(error.to_string()))
}

#[derive(Debug, Default)]
pub struct SegmentatorApp {
    configuration: SegmentatorConfiguration,
    image_paths: Vec<PathBuf>,
    current_index: Option<usize>,
    state: SegmentationState,
}

impl SegmentatorApp {
    fn new(configuration: SegmentatorConfiguration) -> Self {
        let image_paths = glob::glob("/home/rasmus/Downloads/Felix/*.png")
            .expect("fail")
            .flatten()
            .collect();

        Self {
            configuration,
            image_paths,
            current_index: None,
            state: SegmentationState::default(),
        }
    }
}

impl App for SegmentatorApp {
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        TopBottomPanel::bottom("Preview")
            .min_height(200.0)
            .show(ctx, |ui| {
                ui.add(PreviewWidget::new(
                    &self.image_paths,
                    &mut self.current_index,
                ));
            });
        SidePanel::right("Tools")
            .resizable(false)
            .min_width(200.0)
            .show(ctx, |ui| {
                let width = ui.available_width();
                ui.allocate_ui(vec2(width, 50.0), |ui| {
                    ui.add(GithubAccount::new("oleflb").hover_text("Test 123"));
                });

                ui.add(ControlPane)
            });
        CentralPanel::default().show(ctx, |ui| {
            if let Some(image_path) = self
                .current_index
                .and_then(|index| self.image_paths.get(index))
            {
                ui.add(Segmentator::new(image_path, &mut self.state));
            }
        });
    }
}
