use eframe::egui::{Image, Response, Sense, Ui, Widget};

fn github_avatar_url(username: &str, size: Option<u32>) -> String {
    match size {
        Some(size) => format!("https://avatars.githubusercontent.com/{username}?size={size}",),
        None => format!("https://avatars.githubusercontent.com/{username}"),
    }
}

pub struct GithubAccount<'a> {
    username: &'a str,
    hover_text: Option<&'a str>,
}

pub struct SingleGithubAvatar<'a> {
    username: &'a str,
}

impl<'a> Widget for SingleGithubAvatar<'a> {
    fn ui(self, ui: &mut Ui) -> Response {
        let height = ui.available_height();
        let image = Image::new(github_avatar_url(
            self.username,
            Some(height.round() as u32),
        ))
        .maintain_aspect_ratio(true)
        .rounding(height / 2.)
        .sense(Sense::hover());
        ui.add(image)
    }
}

impl<'a> GithubAccount<'a> {
    pub fn new(username: &'a str) -> Self {
        Self {
            username,
            hover_text: None,
        }
    }

    pub fn hover_text(mut self, hover_text: &'a str) -> Self {
        self.hover_text = Some(hover_text);
        self
    }
}

impl<'a> Widget for GithubAccount<'a> {
    fn ui(self, ui: &mut Ui) -> Response {
        let response = ui.add(SingleGithubAvatar {
            username: self.username,
        });
        if let Some(hover_text) = self.hover_text {
            response.on_hover_text(hover_text)
        } else {
            response
        }
    }
}
