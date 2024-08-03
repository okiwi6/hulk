use std::path::Path;

use eframe::egui::{Response, Ui, Widget};

pub struct GithubAccount<'a> {
    username: &'a str,
}

impl<'a> GithubAccount<'a> {
    pub fn new(username: &'a str) -> Self {
        Self { username }
    }
}

impl<'a> Widget for GithubAccount<'a> {
    fn ui(self, ui: &mut Ui) -> Response {
        let avatar_url = format!("https://avatars.githubusercontent.com/{}", self.username);
        ui.image(avatar_url);
        ui.label("Github Account")
    }
}
