use eframe::egui::Widget;

pub enum Action {
    FinishPolygon,
    CancelPolygon,
    DeletePolygon,
}

pub struct ControlPane;

impl Widget for ControlPane {
    fn ui(self, ui: &mut eframe::egui::Ui) -> eframe::egui::Response {
        ui.vertical(|ui| {
            ui.label("Finish [f]");
            ui.label("Cancel [c]");
            ui.label("Delete [d]");
        })
        .response
    }
}
