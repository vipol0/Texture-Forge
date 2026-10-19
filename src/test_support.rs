use image::{DynamicImage, RgbaImage};
use std::path::{Path, PathBuf};

pub(crate) fn temp_dir(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("texture_forge_{tag}_{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

pub(crate) fn write_test_png(path: &Path) {
    let mut img = RgbaImage::new(8, 8);
    for (x, y, p) in img.enumerate_pixels_mut() {
        p.0 = [(x * 24) as u8, (y * 24) as u8, 40, 255];
    }
    DynamicImage::ImageRgba8(img).save(path).unwrap();
}
