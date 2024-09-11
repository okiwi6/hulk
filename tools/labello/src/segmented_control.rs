use eframe::egui::{
    vec2, Align2, Color32, Context, Id, Rect, Response, Rounding, Sense, TextStyle, Ui, Widget,
};

pub struct SegmentedControl {
    selectables: Vec<String>,
    id: Id,
    rounding: Rounding,
    background: Color32,
    text_style: TextStyle,
}

#[derive(Debug, Default, Clone)]
struct SegmentedControlState {
    selected: usize,
}

impl SegmentedControl {
    pub fn new(id: impl Into<Id>) -> Self {
        SegmentedControl {
            id: id.into(),
            selectables: Vec::new(),
            rounding: Rounding::default(),
            background: Color32::BLACK,
            text_style: TextStyle::Body,
        }
    }

    pub fn rounding(mut self, rounding: impl Into<Rounding>) -> Self {
        self.rounding = rounding.into();
        self
    }

    pub fn add_segment(mut self, selectable: impl ToString) -> Self {
        self.selectables.push(selectable.to_string());
        self
    }
}

impl Widget for SegmentedControl {
    fn ui(self, ui: &mut Ui) -> Response {
        let mut state = load_state(ui.ctx(), self.id);
        let width = ui.available_width();
        let text_style = ui
            .style()
            .text_styles
            .get(&self.text_style)
            .expect("failed to get text style")
            .clone();
        let text_size = text_style.size * ui.ctx().pixels_per_point();

        let (response, painter) = ui.allocate_painter(vec2(width, 2.0 * text_size), Sense::click());
        painter.rect_filled(response.rect, self.rounding, self.background);

        let text_rects = text_rects(response.rect, self.selectables.len());
        let offset = text_rects[0].width();

        let translation = animate_to(self.id, ui.ctx(), offset * state.selected as f32);
        let selector_rect = text_rects[0].translate(vec2(translation, 0.0)).shrink(2.0);
        painter.rect_filled(selector_rect, self.rounding, Color32::GRAY);

        for (idx, (&rect, text)) in text_rects.iter().zip(self.selectables.iter()).enumerate() {
            let response = ui.interact(rect, self.id.with(idx), Sense::click());
            if response.clicked() {
                state.selected = idx;
            }
            // ui.ctx().animate_bool_with_time_and_easing(id, target_value, animation_time, easing)
            painter.text(
                rect.center(),
                Align2::CENTER_CENTER,
                text,
                text_style.clone(),
                Color32::WHITE,
            );
        }

        save_state(ui.ctx(), self.id, state);
        response
    }
}

fn load_state(ctx: &Context, id: Id) -> SegmentedControlState {
    let persisted = ctx.data_mut(|reader| reader.get_temp(id));
    persisted.unwrap_or_default()
}

fn save_state(ctx: &Context, id: Id, state: SegmentedControlState) {
    ctx.data_mut(|writer| writer.insert_temp(id, state))
}

fn animate_to(source: Id, context: &Context, target: f32) -> f32 {
    context.animate_value_with_time(source, target, 0.1)
}

fn text_rects(mut rect: Rect, number_of_texts: usize) -> Vec<Rect> {
    let base_width = rect.width() / number_of_texts as f32;
    let base_rect = {
        rect.set_width(base_width);
        rect
    };
    (0..number_of_texts)
        .map(|idx| {
            let rect = base_rect;
            rect.translate(vec2(base_width * idx as f32, 0.0))
        })
        .collect()
}
