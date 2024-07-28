use std::{collections::BTreeMap, mem::take, path::PathBuf};

use eframe::{
    egui::{
        emath::RectTransform, Color32, Key, PointerButton, Pos2, Rect, Sense, TextureOptions,
        Widget,
    },
    epaint::{Shape, Stroke, Vec2},
};
use serde::Serialize;

use crate::polygon::{paint_polygon, FixedPolygon, FixedPolygonBuilder};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum Class {
    Field,
    Line,
}

#[derive(Debug, Default)]
pub struct SegmentationState {
    pub polygons: BTreeMap<Class, Vec<FixedPolygon>>,
    pub current_vertices: Vec<Pos2>,
}

impl SegmentationState {
    fn finish(&mut self, class: Class) {
        let border = take(&mut self.current_vertices);
        let polygon = FixedPolygonBuilder::new(border)
            .color(Color32::RED.gamma_multiply(0.4))
            .build();
        self.polygons.entry(class).or_default().push(polygon);
    }

    fn cancel(&mut self) {
        self.current_vertices.clear();
    }
}

pub struct Segmentator<'ui> {
    image_path: &'ui PathBuf,
    polygons: &'ui mut SegmentationState,
}

impl<'ui> Segmentator<'ui> {
    pub fn new(image_path: &'ui PathBuf, state: &'ui mut SegmentationState) -> Self {
        Segmentator {
            image_path,
            polygons: state,
        }
    }
}

impl<'ui> Widget for Segmentator<'ui> {
    fn ui(self, ui: &mut eframe::egui::Ui) -> eframe::egui::Response {
        let (response, painter) = ui.allocate_painter(Vec2::new(300.0, 300.0), Sense::click());
        painter.rect_stroke(painter.clip_rect(), 0.0, Stroke::new(2.0, Color32::RED));

        let uri = format!("file://{}", self.image_path.display());
        // let image = Image::new(&uri)
        //     .show_loading_spinner(true)
        //     .maintain_aspect_ratio(true)
        //     .shrink_to_fit()
        //     .sense(Sense::click());
        let texture_id = ui
            .ctx()
            .try_load_texture(
                &uri,
                TextureOptions::default(),
                eframe::egui::SizeHint::Scale(1.0.into()),
            )
            .unwrap();
        painter.add(Shape::image(
            texture_id.texture_id().unwrap(),
            painter.clip_rect(),
            Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        ));

        let transform = RectTransform::from_to(
            response.rect,
            Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
        );

        for polygon in self.polygons.polygons.values().flatten() {
            polygon.paint(ui.painter(), transform.inverse());
        }

        let mouse = response
            .hover_pos()
            .map(|position| transform.transform_pos(position));

        if response.clicked_by(PointerButton::Primary) {
            self.polygons.current_vertices.extend(mouse);
        }

        paint_polygon(
            ui.painter(),
            self.polygons.current_vertices.iter().copied().chain(mouse),
            transform.inverse(),
            Color32::RED,
        );

        if ui.input(|input| input.key_pressed(Key::Escape)) {
            self.polygons.cancel();
        }
        if ui.input(|input| input.key_pressed(Key::F)) {
            self.polygons.finish(Class::Field);
        }

        response
    }
}
