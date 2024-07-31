pub mod cli;

use std::path::PathBuf;

use clap::Parser;
use cli::CliArguments;
use eframe::{
    egui::{self, CentralPanel},
    App, CreationContext, NativeOptions,
};

fn main() {
    let native_options = NativeOptions::default();
    eframe::run_native(
        "SSD Viewer",
        native_options,
        Box::new(|cc| Ok(Box::new(SsdViewer::new(cc)))),
    );
}

#[derive(Default)]
struct SsdViewer {
    image_path: PathBuf,
}

impl SsdViewer {
    fn new(cc: &CreationContext<'_>) -> Self {
        egui_extras::install_image_loaders(&cc.egui_ctx);
        let arguments = CliArguments::parse();
        SsdViewer {
            image_path: arguments.image,
        }
    }
}

impl App for SsdViewer {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        CentralPanel::default().show(ctx, |ui| {
            let uri = format!("file://{}", self.image_path.display());
            ui.image(uri)
        });
    }
}
