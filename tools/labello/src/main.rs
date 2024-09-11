pub mod control_pane;
mod export;
pub mod github_integration;
pub mod polygon;
mod preview_widget;
pub mod segmentator_widget;
pub mod segmented_control;

use std::env;
use std::{io::Cursor, path::PathBuf};

use color_eyre::{eyre::eyre, Result};
use configuration_system::Configuration;
use configuration_system_derive::Merge;
use control_pane::ControlPane;
use eframe::{
    egui::{vec2, CentralPanel, Context, IconData, SidePanel, TopBottomPanel, ViewportBuilder},
    run_native, App, Frame, NativeOptions,
};
use github_integration::GithubAccount;
use image::ImageReader;
use preview_widget::PreviewWidget;
use segmentator_widget::{SegmentationState, Segmentator};
use segmented_control::SegmentedControl;
use serde::Deserialize;

#[derive(Debug, Default, Deserialize, Merge)]
struct LabelloConfiguration {
    github_username: Option<String>,
}

impl Configuration for LabelloConfiguration {
    const DEFAULT_FILENAME: &'static str = "segmentator.toml";
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
            let configuration = LabelloConfiguration::load()?;

            let app = LabelloApp::new(configuration);

            Ok(Box::new(app))
        }),
    )
    .map_err(|error| eyre!(error.to_string()))
}

#[derive(Debug, Default)]
pub struct LabelloApp {
    configuration: LabelloConfiguration,
    image_paths: Vec<PathBuf>,
    current_index: Option<usize>,
    state: SegmentationState,
}

impl LabelloApp {
    fn new(configuration: LabelloConfiguration) -> Self {
        let path = env::args().nth(1).expect("failed to get path");
        let image_paths = glob::glob(&path).expect("fail").flatten().collect();

        Self {
            configuration,
            image_paths,
            current_index: None,
            state: SegmentationState::default(),
        }
    }
}

impl App for LabelloApp {
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
            .min_width(300.0)
            .show(ctx, |ui| {
                let width = ui.available_width();
                ui.add(
                    SegmentedControl::new("segmented_control")
                        .rounding(5.0)
                        .add_segment("Segmentation")
                        .add_segment("Detection")
                        .add_segment("Classification"),
                );
                if let Some(name) = &self.configuration.github_username {
                    ui.allocate_ui(vec2(width, 50.0), |ui| {
                        ui.add(GithubAccount::new(name));
                    });
                }

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
