use crate::imaging::color::{auto_levels, luma_histogram, ColorGrade};
use crate::imaging::dither::DitherEngine;
use crate::model::geometry::{crop_rgba, resize_with_mode, target_dims, CropRect};
use crate::model::settings::{AlphaMode, PaletteMethod, Settings};
use image::{DynamicImage, RgbImage, RgbaImage};
use quantette::deps::palette::Srgb;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Parameters an adaptive palette depends on, used to invalidate the cached palette.
#[derive(Clone, Copy, PartialEq, Debug)]
struct PaletteKey {
    width: u32,
    height: u32,
    palette_size: u32,
    method: PaletteMethod,
}

/// Cached intermediate results shared across consecutive stylization requests.
///
/// Quantization and dithering run at the pixelized (downscaled) resolution, so the expensive
/// adaptive palette only has to be recomputed when the source, the target resolution, or the
/// palette parameters change. Switching presets that only alter dithering reuses everything.
#[derive(Default)]
pub(crate) struct StylizeCache {
    /// Source the cached layers were derived from (compared by pointer identity).
    source: Option<Arc<DynamicImage>>,
    /// Decoded source pixels at full resolution.
    rgba: Option<Arc<RgbaImage>>,
    /// Crop the cached `rgba` layer was last derived from.
    crop: Option<CropRect>,
    /// Source pixels after applying the manual crop. Shares `rgba` when the crop is full.
    cropped: Option<Arc<RgbaImage>>,
    /// Pixelized source layer at the target resolution.
    small: Option<RgbaImage>,
    /// Luma histogram of `small`, used by auto-contrast.
    histogram: Option<[u32; 256]>,
    /// RGB samples of `small`, fed to the quantizer.
    input: Option<quantette::ImageBuf<Srgb<u8>>>,
    /// Adaptive palette computed from `input`.
    palette: Option<quantette::PaletteBuf<Srgb<u8>>>,
    /// Parameters `palette` was computed from.
    palette_key: Option<PaletteKey>,
}

impl StylizeCache {
    /// Stylize `source` with `settings`, reusing cached layers wherever their inputs are unchanged.
    pub(crate) fn stylize(
        &mut self,
        source: Arc<DynamicImage>,
        settings: &Settings,
    ) -> Option<DynamicImage> {
        self.stylize_cancellable(source, settings, None)
    }

    /// Same as [`StylizeCache::stylize`], but aborts when `cancel` is set.
    ///
    /// Cancellation is checked between stages and inside the per-pixel grading loop, so a
    /// superseded render stops early instead of competing with the newer one for CPU time.
    pub(crate) fn stylize_cancellable(
        &mut self,
        source: Arc<DynamicImage>,
        settings: &Settings,
        cancel: Option<&AtomicBool>,
    ) -> Option<DynamicImage> {
        if Self::is_cancelled(cancel) {
            return None;
        }
        self.ensure_source(source);
        self.ensure_crop(settings.crop.sanitized())?;

        let cropped = self.cropped.clone()?;
        let (cropped_w, cropped_h) = cropped.dimensions();
        let (small_w, small_h) = target_dims(settings.resolution, cropped_w, cropped_h);
        if Self::is_cancelled(cancel) {
            return None;
        }

        self.ensure_pixelized(&cropped, small_w, small_h)?;
        self.ensure_palette(settings, small_w, small_h)?;
        let small = self.small.as_ref()?;
        let input = self.input.as_ref()?;
        let palette = self.palette.as_ref()?;
        if Self::is_cancelled(cancel) {
            return None;
        }

        let engine = DitherEngine::new(
            settings.dither,
            settings.dither_strength,
            settings.dither_serpentine,
        );
        let quantized = engine.remap(input.as_slice(), small_w, small_h, palette, cancel)?;
        if Self::is_cancelled(cancel) {
            return None;
        }

        let histogram = self.histogram.as_ref();
        let composed = Self::compose_output(
            settings,
            &quantized,
            small,
            (small_w, small_h),
            (cropped_w, cropped_h),
            histogram,
            cancel,
        )?;
        Some(DynamicImage::ImageRgba8(composed))
    }

    fn is_cancelled(cancel: Option<&AtomicBool>) -> bool {
        cancel.is_some_and(|flag| flag.load(Ordering::Relaxed))
    }

    /// Refresh the decoded source and invalidate every layer derived from it.
    fn ensure_source(&mut self, source: Arc<DynamicImage>) {
        if self
            .source
            .as_ref()
            .is_none_or(|cached| !Arc::ptr_eq(cached, &source))
        {
            self.rgba = Some(Arc::new(source.to_rgba8()));
            self.source = Some(source);
            self.reset_layers();
        }
    }

    /// Apply the manual crop once; every later stage works on the cropped image.
    fn ensure_crop(&mut self, crop: CropRect) -> Option<()> {
        if self.crop == Some(crop) {
            return Some(());
        }
        let rgba = self.rgba.as_ref()?;
        let cropped = if crop.is_full() {
            rgba.clone()
        } else {
            Arc::new(crop_rgba(rgba, crop))
        };
        self.cropped = Some(cropped);
        self.crop = Some(crop);
        self.reset_downstream();
        Some(())
    }

    /// Pixelize to the target resolution: downscale, and scale back up only at the very end.
    fn ensure_pixelized(&mut self, cropped: &RgbaImage, small_w: u32, small_h: u32) -> Option<()> {
        if self
            .small
            .as_ref()
            .is_some_and(|s| s.dimensions() == (small_w, small_h))
        {
            return Some(());
        }
        let small = image::imageops::resize(
            cropped,
            small_w,
            small_h,
            image::imageops::FilterType::Triangle,
        );
        let mut rgb = RgbImage::new(small_w, small_h);
        for (dst, src) in rgb.pixels_mut().zip(small.pixels()) {
            dst.0 = [src[0], src[1], src[2]];
        }
        self.input = Some(quantette::ImageBuf::try_from(rgb).ok()?);
        self.histogram = Some(luma_histogram(&small));
        self.small = Some(small);
        self.palette = None;
        self.palette_key = None;
        Some(())
    }

    /// Recompute the adaptive palette only when its inputs change; dithering does not affect it.
    fn ensure_palette(&mut self, settings: &Settings, small_w: u32, small_h: u32) -> Option<()> {
        let key = PaletteKey {
            width: small_w,
            height: small_h,
            palette_size: settings.palette_size,
            method: settings.palette_method,
        };
        if self.palette.is_some() && self.palette_key == Some(key) {
            return Some(());
        }
        let input = self.input.as_ref()?;
        let size = quantette::PaletteSize::from_u16_clamped(settings.palette_size as u16);
        let method = match settings.palette_method {
            // Wu's greedy orthogonal bipartitioning: fast, decent quality.
            PaletteMethod::Wu => quantette::QuantizeMethod::Wu,
            // K-means clustering: slower but more accurate.
            PaletteMethod::KMeans => quantette::QuantizeMethod::kmeans(),
        };
        self.palette = Some(
            quantette::Pipeline::new()
                .palette_size(size)
                .quantize_method(method)
                .ditherer(None)
                .parallel(true)
                .input_image(input.as_ref())
                .output_srgb8_palette()?,
        );
        self.palette_key = Some(key);
        Some(())
    }

    /// Merge quantized colors with pixelized alpha, grade them, then apply the output geometry.
    ///
    /// Grading happens after quantization so the adaptive palette cache stays valid while the
    /// sliders move and the palette color count is preserved.
    fn compose_output(
        settings: &Settings,
        quantized: &[Srgb<u8>],
        small: &RgbaImage,
        (small_w, small_h): (u32, u32),
        (cropped_w, cropped_h): (u32, u32),
        histogram: Option<&[u32; 256]>,
        cancel: Option<&AtomicBool>,
    ) -> Option<RgbaImage> {
        let auto_levels = histogram.map(auto_levels);
        let grade = ColorGrade::from_settings(settings, auto_levels);
        let mut small_out = RgbaImage::new(small_w, small_h);
        for (index, (dst, (color, src))) in small_out
            .pixels_mut()
            .zip(quantized.iter().zip(small.pixels()))
            .enumerate()
        {
            // Poll the cancellation flag periodically rather than for every pixel.
            if index % 4096 == 0 && Self::is_cancelled(cancel) {
                return None;
            }
            let [r, g, b] = grade.apply([color.red, color.green, color.blue]);
            dst.0 = [r, g, b, src[3]];
        }
        let result = match settings.output_size {
            Some([target_w, target_h]) => resize_with_mode(
                &small_out,
                target_w,
                target_h,
                settings.resize_mode,
                settings.resize_filter,
            ),
            None if (small_w, small_h) == (cropped_w, cropped_h) => small_out,
            None => image::imageops::resize(
                &small_out,
                cropped_w,
                cropped_h,
                settings.resize_filter.image_filter(),
            ),
        };
        Some(apply_alpha_mode(result, settings))
    }

    /// Drop every cached layer downstream of the decoded source.
    fn reset_layers(&mut self) {
        self.crop = None;
        self.cropped = None;
        self.reset_downstream();
    }

    /// Drop every cached layer downstream of the cropped image.
    fn reset_downstream(&mut self) {
        self.small = None;
        self.histogram = None;
        self.input = None;
        self.palette = None;
        self.palette_key = None;
    }
}

/// Composite RGBA pixels over an opaque background in gamma space.
pub(crate) fn flatten_rgba(image: &RgbaImage, background: [u8; 3]) -> RgbImage {
    let mut out = RgbImage::new(image.width(), image.height());
    let bg = background.map(|c| c as u32);
    for (dst, src) in out.pixels_mut().zip(image.pixels()) {
        let alpha = src[3] as u32;
        let inverse = 255 - alpha;
        dst.0 = [
            ((src[0] as u32 * alpha + bg[0] * inverse + 127) / 255) as u8,
            ((src[1] as u32 * alpha + bg[1] * inverse + 127) / 255) as u8,
            ((src[2] as u32 * alpha + bg[2] * inverse + 127) / 255) as u8,
        ];
    }
    out
}

/// Apply the selected alpha mode to a fully resized result.
pub(crate) fn apply_alpha_mode(mut image: RgbaImage, settings: &Settings) -> RgbaImage {
    match settings.alpha_mode {
        AlphaMode::Save => image,
        AlphaMode::Threshold => {
            let threshold = settings.alpha_threshold;
            for pixel in image.pixels_mut() {
                pixel[3] = if pixel[3] >= threshold { 255 } else { 0 };
            }
            image
        }
        AlphaMode::Fill => {
            let rgb = flatten_rgba(&image, settings.background);
            for (dst, src) in image.pixels_mut().zip(rgb.pixels()) {
                dst.0 = [src[0], src[1], src[2], 255];
            }
            image
        }
    }
}

/// Run the full stylization pipeline with a fresh cache. Pure and thread-safe; used by tests.
#[cfg(test)]
pub(crate) fn stylize_image(source: &DynamicImage, settings: &Settings) -> Option<DynamicImage> {
    StylizeCache::default().stylize(Arc::new(source.clone()), settings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imaging::dither::DitherMode;
    use crate::model::geometry::CropRect;
    use crate::model::geometry::{ResizeFilter, ResizeMode};
    use crate::model::settings::{AlphaMode, PaletteMethod, Settings};
    use image::{DynamicImage, RgbaImage};
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    #[test]
    fn stylize_respects_palette_size_for_every_mode() {
        use std::collections::HashSet;

        let mut img = RgbaImage::new(16, 16);
        for (x, y, p) in img.enumerate_pixels_mut() {
            p.0 = [(x * 16) as u8, (y * 16) as u8, ((x + y) * 8) as u8, 255];
        }

        let source = DynamicImage::ImageRgba8(img);
        let mut settings = Settings {
            resolution: None,
            palette_size: 8,
            dither_strength: 80,
            ..Default::default()
        };

        for method in [PaletteMethod::Wu, PaletteMethod::KMeans] {
            for dither in DitherMode::ALL {
                settings.palette_method = method;
                settings.dither = dither;
                let out = stylize_image(&source, &settings)
                    .expect("stylize produced output")
                    .to_rgb8();
                let unique: HashSet<[u8; 3]> = out.pixels().map(|p| p.0).collect();
                assert!(
                    unique.len() <= 8,
                    "too many colors for {method:?}/{dither:?}: {}",
                    unique.len()
                );
            }
        }
    }

    #[test]
    fn settings_auto_contrast_uses_histogram_levels() {
        let settings = Settings {
            auto_contrast: true,
            ..Default::default()
        };
        let grade = ColorGrade::from_settings(&settings, Some((100, 150)));
        assert_eq!(grade.apply([100, 100, 100]), [0, 0, 0]);
        assert_eq!(grade.apply([150, 150, 150]), [255, 255, 255]);
    }

    #[test]
    fn stylize_applies_color_adjustments_and_keeps_palette_size() {
        use std::collections::HashSet;

        let mut img = RgbaImage::new(16, 16);
        for (x, y, p) in img.enumerate_pixels_mut() {
            p.0 = [(x * 16) as u8, (y * 16) as u8, ((x + y) * 8) as u8, 255];
        }
        let source = DynamicImage::ImageRgba8(img);
        let base = Settings {
            resolution: None,
            palette_size: 8,
            dither: DitherMode::None,
            ..Default::default()
        };
        let graded = Settings {
            brightness: 25,
            saturation: 150,
            contrast: 120,
            ..base.clone()
        };

        let before = stylize_image(&source, &base)
            .expect("stylize produced output")
            .to_rgb8();
        let after = stylize_image(&source, &graded)
            .expect("stylize produced output")
            .to_rgb8();
        assert_ne!(before.as_raw(), after.as_raw(), "grading had no effect");

        let unique: HashSet<[u8; 3]> = after.pixels().map(|p| p.0).collect();
        assert!(unique.len() <= 8, "too many colors: {}", unique.len());
    }

    #[test]
    fn stylize_cache_matches_fresh_pipeline_across_changes() {
        let mut img = RgbaImage::new(32, 24);
        for (x, y, p) in img.enumerate_pixels_mut() {
            p.0 = [(x * 8) as u8, (y * 10) as u8, ((x + y) * 5) as u8, 255];
        }
        let source = Arc::new(DynamicImage::ImageRgba8(img));

        // A sequence that exercises every invalidation path: new source/resolution, palette
        // parameter changes, and dither changes that reuse the cached palette and pixel layer.
        let cases = [
            Settings {
                resolution: Some(16),
                palette_size: 8,
                dither: DitherMode::None,
                ..Default::default()
            },
            Settings {
                resolution: Some(16),
                palette_size: 8,
                dither: DitherMode::Ordered,
                dither_strength: 60,
                ..Default::default()
            },
            Settings {
                resolution: Some(8),
                palette_size: 8,
                dither: DitherMode::FloydSteinberg,
                ..Default::default()
            },
            Settings {
                resolution: Some(8),
                palette_size: 8,
                dither: DitherMode::Atkinson,
                dither_serpentine: false,
                ..Default::default()
            },
            Settings {
                resolution: Some(8),
                palette_size: 8,
                dither: DitherMode::BlueNoise,
                dither_strength: 70,
                ..Default::default()
            },
            Settings {
                resolution: Some(8),
                palette_size: 32,
                palette_method: PaletteMethod::KMeans,
                dither: DitherMode::Ordered,
                ..Default::default()
            },
            Settings {
                resolution: Some(8),
                palette_size: 32,
                palette_method: PaletteMethod::KMeans,
                dither: DitherMode::Ordered,
                brightness: 30,
                saturation: 140,
                contrast: 80,
                ..Default::default()
            },
            Settings {
                resolution: Some(8),
                palette_size: 32,
                dither: DitherMode::FloydSteinberg,
                gamma: 150,
                temperature: 35,
                tint: -20,
                black_point: 10,
                white_point: 245,
                curves: vec![[0.0, 0.05], [0.5, 0.7], [1.0, 0.95]],
                invert: true,
                ..Default::default()
            },
            Settings {
                resolution: Some(8),
                palette_size: 32,
                dither: DitherMode::Ordered,
                auto_contrast: true,
                grayscale: true,
                sepia: true,
                ..Default::default()
            },
            Settings {
                resolution: Some(16),
                crop: CropRect {
                    x: 0.25,
                    y: 0.1,
                    w: 0.5,
                    h: 0.8,
                },
                palette_size: 8,
                dither: DitherMode::None,
                ..Default::default()
            },
            Settings {
                resolution: Some(16),
                crop: CropRect {
                    x: 0.25,
                    y: 0.1,
                    w: 0.5,
                    h: 0.8,
                },
                output_size: Some([20, 12]),
                resize_mode: ResizeMode::Fill,
                resize_filter: ResizeFilter::Lanczos,
                palette_size: 8,
                dither: DitherMode::None,
                ..Default::default()
            },
        ];

        let mut cache = StylizeCache::default();
        for settings in cases {
            let cached = cache
                .stylize(source.clone(), &settings)
                .expect("cached stylize produced output")
                .to_rgba8();
            let fresh = stylize_image(source.as_ref(), &settings)
                .expect("fresh stylize produced output")
                .to_rgba8();
            assert_eq!(
                cached.as_raw(),
                fresh.as_raw(),
                "cached output diverged for {settings:?}"
            );
        }
    }

    #[test]
    fn alpha_modes_transform_alpha_as_selected() {
        let mut image = RgbaImage::new(4, 1);
        image.put_pixel(0, 0, image::Rgba([10, 20, 30, 0]));
        image.put_pixel(1, 0, image::Rgba([100, 150, 200, 127]));
        image.put_pixel(2, 0, image::Rgba([100, 150, 200, 128]));
        image.put_pixel(3, 0, image::Rgba([10, 20, 30, 255]));

        let settings = Settings {
            alpha_mode: AlphaMode::Save,
            ..Default::default()
        };
        assert_eq!(
            apply_alpha_mode(image.clone(), &settings).as_raw(),
            image.as_raw()
        );

        let settings = Settings {
            alpha_mode: AlphaMode::Threshold,
            alpha_threshold: 128,
            ..Default::default()
        };
        let thresholded = apply_alpha_mode(image.clone(), &settings);
        let alphas: Vec<u8> = thresholded.pixels().map(|p| p[3]).collect();
        assert_eq!(alphas, vec![0, 0, 255, 255]);

        let settings = Settings {
            alpha_mode: AlphaMode::Fill,
            background: [255, 255, 255],
            ..Default::default()
        };
        let filled = apply_alpha_mode(image, &settings);
        assert_eq!(filled.get_pixel(0, 0).0, [255, 255, 255, 255]);
        assert_eq!(filled.get_pixel(1, 0).0, [178, 203, 228, 255]);
        assert_eq!(filled.get_pixel(3, 0).0, [10, 20, 30, 255]);
        assert!(filled.pixels().all(|p| p[3] == 255));
    }

    #[test]
    fn stylize_applies_alpha_mode_after_resize() {
        let source =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(8, 8, image::Rgba([10, 20, 30, 0])));

        let settings = Settings {
            resolution: None,
            alpha_mode: AlphaMode::Threshold,
            alpha_threshold: 128,
            ..Default::default()
        };
        let out = stylize_image(&source, &settings)
            .expect("stylize produced output")
            .to_rgba8();
        assert!(out.pixels().all(|p| p[3] == 0));

        let settings = Settings {
            resolution: None,
            alpha_mode: AlphaMode::Fill,
            background: [0, 255, 0],
            ..Default::default()
        };
        let out = stylize_image(&source, &settings)
            .expect("stylize produced output")
            .to_rgba8();
        assert_eq!(out.get_pixel(0, 0).0, [0, 255, 0, 255]);
    }

    #[test]
    fn stylize_applies_crop_and_output_geometry() {
        let mut img = RgbaImage::new(400, 200);
        for (x, y, p) in img.enumerate_pixels_mut() {
            p.0 = [(x % 256) as u8, (y % 256) as u8, 80, 255];
        }
        let source = DynamicImage::ImageRgba8(img);

        // Cropping without an explicit output size yields the cropped dimensions.
        let settings = Settings {
            resolution: None,
            crop: CropRect {
                x: 0.25,
                y: 0.25,
                w: 0.5,
                h: 0.5,
            },
            palette_size: 16,
            dither: DitherMode::None,
            ..Default::default()
        };
        let out = stylize_image(&source, &settings).expect("stylize produced output");
        assert_eq!((out.width(), out.height()), (200, 100));

        // Fit into a square output adds transparent letterbox bands.
        let settings = Settings {
            output_size: Some([50, 50]),
            resize_mode: ResizeMode::Fit,
            resize_filter: ResizeFilter::Nearest,
            ..settings.clone()
        };
        let out = stylize_image(&source, &settings).expect("stylize produced output");
        assert_eq!((out.width(), out.height()), (50, 50));
        let out = out.to_rgba8();
        assert_eq!(out.get_pixel(0, 0).0[3], 0);
        assert_eq!(out.get_pixel(25, 25).0[3], 255);

        // Fill and stretch both honor the explicit output dimensions.
        for mode in [ResizeMode::Fill, ResizeMode::Stretch] {
            let settings = Settings {
                output_size: Some([64, 32]),
                resize_mode: mode,
                ..settings.clone()
            };
            let out = stylize_image(&source, &settings).expect("stylize produced output");
            assert_eq!((out.width(), out.height()), (64, 32));
        }
    }

    #[test]
    fn stylize_keeps_source_dimensions() {
        let mut img = RgbaImage::new(400, 200);
        for (x, y, p) in img.enumerate_pixels_mut() {
            p.0 = [(x % 256) as u8, (y % 256) as u8, 0, 255];
        }
        let source = DynamicImage::ImageRgba8(img);
        let settings = Settings {
            resolution: Some(100),
            ..Default::default()
        };
        let out = stylize_image(&source, &settings).expect("stylize produced output");
        assert_eq!((out.width(), out.height()), (400, 200));
    }

    #[test]
    fn stylize_cancellable_aborts_when_flagged() {
        let source = DynamicImage::ImageRgba8(RgbaImage::new(8, 8));
        let cancel = AtomicBool::new(true);
        let mut cache = StylizeCache::default();
        assert!(cache
            .stylize_cancellable(Arc::new(source), &Settings::default(), Some(&cancel))
            .is_none());
    }
}
