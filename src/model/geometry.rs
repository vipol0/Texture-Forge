use image::RgbaImage;
use serde::{Deserialize, Serialize};

/// A rectangular crop expressed in normalized source coordinates (`0..=1`).
///
/// Keeping the crop normalized makes it independent of the source resolution and lets presets and
/// batch jobs reuse the same region across differently sized images.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub(crate) struct CropRect {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) w: f32,
    pub(crate) h: f32,
}

impl Default for CropRect {
    fn default() -> Self {
        Self::FULL
    }
}

impl CropRect {
    pub(crate) const FULL: Self = Self {
        x: 0.0,
        y: 0.0,
        w: 1.0,
        h: 1.0,
    };
    /// Smallest allowed crop side in normalized units (prevents zero-sized output).
    pub(crate) const MIN_SIDE: f32 = 1e-4;

    pub(crate) fn is_full(self) -> bool {
        (self.x - Self::FULL.x).abs() < 1e-5
            && (self.y - Self::FULL.y).abs() < 1e-5
            && (self.w - Self::FULL.w).abs() < 1e-5
            && (self.h - Self::FULL.h).abs() < 1e-5
    }

    /// Clamp the crop to the image and guarantee a non-empty rectangle.
    pub(crate) fn sanitized(self) -> Self {
        let x = self.x.clamp(0.0, 1.0 - Self::MIN_SIDE);
        let y = self.y.clamp(0.0, 1.0 - Self::MIN_SIDE);
        let w = self.w.clamp(Self::MIN_SIDE, 1.0 - x);
        let h = self.h.clamp(Self::MIN_SIDE, 1.0 - y);
        Self { x, y, w, h }
    }
}

/// Filter used for the final crop/resize stage.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub(crate) enum ResizeFilter {
    #[default]
    Nearest,
    Lanczos,
}

impl ResizeFilter {
    pub(crate) fn image_filter(self) -> image::imageops::FilterType {
        match self {
            ResizeFilter::Nearest => image::imageops::FilterType::Nearest,
            ResizeFilter::Lanczos => image::imageops::FilterType::Lanczos3,
        }
    }
}

/// How an image is fitted into an explicit output size.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub(crate) enum ResizeMode {
    /// Preserve aspect ratio and letterbox with transparent pixels.
    Fit,
    /// Preserve aspect ratio and center-crop to fill the output.
    Fill,
    /// Ignore aspect ratio and stretch to the exact output size.
    #[default]
    Stretch,
}

/// Compute the intermediate (downscaled) dimensions for the given target resolution.
///
/// `max_side` is the desired length of the longest side; `None` keeps the source size. The
/// result never exceeds the source, so presets larger than the image are a no-op.
pub(crate) fn target_dims(max_side: Option<u32>, w: u32, h: u32) -> (u32, u32) {
    match max_side {
        Some(max_side) => {
            let scale = (max_side as f32 / w.max(h) as f32).min(1.0);
            (
                ((w as f32 * scale).round() as u32).max(1),
                ((h as f32 * scale).round() as u32).max(1),
            )
        }
        None => (w, h),
    }
}

/// Convert a normalized crop to pixel coordinates `(x, y, width, height)`.
///
/// The rectangle is rounded to whole pixels, clamped to the image, and is always at least 1×1.
pub(crate) fn crop_pixel_rect(crop: CropRect, w: u32, h: u32) -> (u32, u32, u32, u32) {
    let w = w.max(1);
    let h = h.max(1);
    let crop = crop.sanitized();
    let x = ((crop.x * w as f32).round() as u32).min(w - 1);
    let y = ((crop.y * h as f32).round() as u32).min(h - 1);
    let cw = ((crop.w * w as f32).round() as u32).clamp(1, w - x);
    let ch = ((crop.h * h as f32).round() as u32).clamp(1, h - y);
    (x, y, cw, ch)
}

/// Apply a normalized crop to an RGBA image.
pub(crate) fn crop_rgba(image: &RgbaImage, crop: CropRect) -> RgbaImage {
    if crop.is_full() {
        return image.clone();
    }
    let (x, y, w, h) = crop_pixel_rect(crop, image.width(), image.height());
    image::imageops::crop_imm(image, x, y, w, h).to_image()
}

/// Dimensions produced by fitting `w`×`h` into `target` while preserving aspect ratio.
pub(crate) fn fit_dims(w: u32, h: u32, target_w: u32, target_h: u32) -> (u32, u32) {
    let tw = target_w.max(1);
    let th = target_h.max(1);
    let scale = (tw as f64 / w.max(1) as f64).min(th as f64 / h.max(1) as f64);
    (
        ((w as f64 * scale).round() as u32).clamp(1, tw),
        ((h as f64 * scale).round() as u32).clamp(1, th),
    )
}

/// Resize `image` to an explicit target using the selected mode and filter.
pub(crate) fn resize_with_mode(
    image: &RgbaImage,
    target_w: u32,
    target_h: u32,
    mode: ResizeMode,
    filter: ResizeFilter,
) -> RgbaImage {
    let tw = target_w.max(1);
    let th = target_h.max(1);
    let filter = filter.image_filter();
    let (w, h) = image.dimensions();
    if (w, h) == (tw, th) {
        return image.clone();
    }

    match mode {
        ResizeMode::Stretch => image::imageops::resize(image, tw, th, filter),
        ResizeMode::Fit => {
            let (rw, rh) = fit_dims(w, h, tw, th);
            let scaled = image::imageops::resize(image, rw, rh, filter);
            // RgbaImage::new already yields transparent black, which is the desired letterbox.
            let mut out = RgbaImage::new(tw, th);
            image::imageops::replace(
                &mut out,
                &scaled,
                (tw - rw) as i64 / 2,
                (th - rh) as i64 / 2,
            );
            out
        }
        ResizeMode::Fill => {
            // Crop the source to the target aspect ratio first, then scale that region directly.
            // This avoids allocating a potentially huge intermediate image for extreme aspects.
            let target_aspect = tw as f64 / th as f64;
            let source_aspect = w as f64 / h as f64;
            let (x, y, cw, ch) = if source_aspect > target_aspect {
                let cw = ((h as f64 * target_aspect).round() as u32).clamp(1, w);
                ((w - cw) / 2, 0, cw, h)
            } else {
                let ch = ((w as f64 / target_aspect).round() as u32).clamp(1, h);
                (0, (h - ch) / 2, w, ch)
            };
            let cropped = image::imageops::crop_imm(image, x, y, cw, ch).to_image();
            image::imageops::resize(&cropped, tw, th, filter)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use image::RgbaImage;

    #[test]
    fn target_dims_preserves_aspect_ratio_and_longest_side() {
        // Landscape: longest side matches the preset, aspect ratio preserved.
        assert_eq!(target_dims(Some(100), 400, 200), (100, 50));
        // Portrait: longest side is the height.
        assert_eq!(target_dims(Some(100), 200, 400), (50, 100));
        // Square.
        assert_eq!(target_dims(Some(128), 512, 512), (128, 128));
    }

    #[test]
    fn target_dims_never_upscales_and_none_keeps_source() {
        assert_eq!(target_dims(Some(2048), 40, 20), (40, 20));
        assert_eq!(target_dims(None, 40, 20), (40, 20));
    }

    #[test]
    fn crop_pixel_rect_rounds_clamps_and_keeps_minimum_size() {
        assert_eq!(crop_pixel_rect(CropRect::FULL, 400, 200), (0, 0, 400, 200));
        assert_eq!(
            crop_pixel_rect(
                CropRect {
                    x: 0.25,
                    y: 0.5,
                    w: 0.5,
                    h: 0.5,
                },
                400,
                200
            ),
            (100, 100, 200, 100)
        );
        // Out-of-bounds crops are clamped to the image, never producing empty output.
        assert_eq!(
            crop_pixel_rect(
                CropRect {
                    x: 0.9,
                    y: 0.9,
                    w: 0.5,
                    h: 0.5,
                },
                100,
                100
            ),
            (90, 90, 10, 10)
        );
        assert_eq!(
            crop_pixel_rect(
                CropRect {
                    x: 0.0,
                    y: 0.0,
                    w: 0.0,
                    h: 0.0,
                },
                10,
                20
            ),
            (0, 0, 1, 1)
        );
    }

    #[test]
    fn crop_rgba_extracts_the_normalized_region() {
        let mut img = RgbaImage::new(4, 2);
        for (x, y, p) in img.enumerate_pixels_mut() {
            p.0 = [x as u8, y as u8, 0, 255];
        }
        let cropped = crop_rgba(
            &img,
            CropRect {
                x: 0.5,
                y: 0.0,
                w: 0.5,
                h: 1.0,
            },
        );
        assert_eq!(cropped.dimensions(), (2, 2));
        assert_eq!(cropped.get_pixel(0, 0).0, [2, 0, 0, 255]);
        assert_eq!(cropped.get_pixel(1, 1).0, [3, 1, 0, 255]);
    }

    #[test]
    fn fit_dims_preserve_aspect_ratio() {
        assert_eq!(fit_dims(400, 200, 100, 100), (100, 50));
        assert_eq!(fit_dims(200, 400, 100, 100), (50, 100));
        assert_eq!(fit_dims(10, 10, 1, 1), (1, 1));
    }

    #[test]
    fn resize_modes_fit_fill_and_stretch() {
        let mut img = RgbaImage::new(4, 2);
        for (x, _y, p) in img.enumerate_pixels_mut() {
            p.0 = match x {
                0 => [255, 0, 0, 255],
                1 => [0, 255, 0, 255],
                2 => [0, 0, 255, 255],
                _ => [255, 255, 0, 255],
            };
        }

        let fit = resize_with_mode(&img, 4, 4, ResizeMode::Fit, ResizeFilter::Nearest);
        assert_eq!(fit.dimensions(), (4, 4));
        assert_eq!(
            fit.get_pixel(0, 0).0[3],
            0,
            "top letterbox must be transparent"
        );
        assert_eq!(
            fit.get_pixel(0, 3).0[3],
            0,
            "bottom letterbox must be transparent"
        );
        assert_eq!(fit.get_pixel(0, 1).0[3], 255);

        let fill = resize_with_mode(&img, 2, 2, ResizeMode::Fill, ResizeFilter::Nearest);
        assert_eq!(fill.dimensions(), (2, 2));
        assert_eq!(fill.get_pixel(0, 0).0, [0, 255, 0, 255]);
        assert_eq!(fill.get_pixel(1, 0).0, [0, 0, 255, 255]);

        let stretch = resize_with_mode(&img, 8, 6, ResizeMode::Stretch, ResizeFilter::Nearest);
        assert_eq!(stretch.dimensions(), (8, 6));
    }

    #[test]
    fn nearest_keeps_hard_blocks_while_lanczos_blends() {
        let mut img = RgbaImage::new(2, 1);
        img.put_pixel(0, 0, image::Rgba([255, 0, 0, 255]));
        img.put_pixel(1, 0, image::Rgba([0, 0, 255, 255]));

        let nearest = resize_with_mode(&img, 4, 2, ResizeMode::Stretch, ResizeFilter::Nearest);
        assert_eq!(nearest.get_pixel(0, 0).0, [255, 0, 0, 255]);
        assert_eq!(nearest.get_pixel(1, 0).0, [255, 0, 0, 255]);
        assert_eq!(nearest.get_pixel(2, 0).0, [0, 0, 255, 255]);

        let lanczos = resize_with_mode(&img, 4, 2, ResizeMode::Stretch, ResizeFilter::Lanczos);
        assert_ne!(lanczos.as_raw(), nearest.as_raw());
    }
}
