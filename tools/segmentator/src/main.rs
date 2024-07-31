pub mod control_pane;
mod export;
pub mod polygon;
mod preview_widget;
pub mod segmentator_widget;

use std::path::PathBuf;

use color_eyre::{eyre::eyre, Result};
use control_pane::ControlPane;
use eframe::{
    egui::{CentralPanel, Context, SidePanel, TopBottomPanel},
    run_native, App, Frame, NativeOptions,
};
use preview_widget::PreviewWidget;
use segmentator_widget::{SegmentationState, Segmentator};

fn main() -> Result<()> {
    run_native(
        "Segmentator",
        NativeOptions::default(),
        Box::new(|_cc| Box::new(SegmentatorApp::new())),
    )
    .map_err(|error| eyre!(error.to_string()))
}

#[derive(Debug, Default)]
pub struct SegmentatorApp {
    image_paths: Vec<PathBuf>,
    current_index: Option<usize>,
    state: SegmentationState,
}

impl SegmentatorApp {
    pub fn new() -> Self {
        let image_paths = glob::glob("/home/ole/Documents/Programs/programs/ultralytics/datasets/SPLObjDetectDatasetV2/test/images/*.png").expect("fail").flatten().collect();
        Self {
            image_paths,
            current_index: None,
            state: SegmentationState::default(),
        }
    }
}

impl App for SegmentatorApp {
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        egui_extras::install_image_loaders(ctx);

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
            .show(ctx, |ui| ui.add(ControlPane));
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
