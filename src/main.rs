#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod blue_noise;
mod files;
mod imaging;
mod model;
mod render;
#[cfg(test)]
mod test_support;

fn main() -> eframe::Result<()> {
    app::run()
}
