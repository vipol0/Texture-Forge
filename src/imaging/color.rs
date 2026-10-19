use crate::model::settings::{sanitize_curves, Settings};
use image::RgbaImage;

/// Build a 256-entry tone curve using monotone cubic (Fritsch–Carlson) interpolation.
///
/// Monotone interpolation avoids overshoot between control points, so dragging one point never
/// introduces bright or dark halos elsewhere along the curve.
pub(crate) fn build_curve_lut(points: &[[f32; 2]]) -> [u8; 256] {
    let pts = sanitize_curves(points);
    let n = pts.len();

    // Secant slopes between consecutive points.
    let mut secants = vec![0.0f32; n - 1];
    for i in 0..n - 1 {
        let dx = pts[i + 1][0] - pts[i][0];
        secants[i] = if dx > 0.0 {
            (pts[i + 1][1] - pts[i][1]) / dx
        } else {
            0.0
        };
    }

    // Initial tangents, then the Fritsch–Carlson limiter to preserve monotonicity.
    let mut tangents = vec![0.0f32; n];
    tangents[0] = secants[0];
    tangents[n - 1] = secants[n - 2];
    for i in 1..n - 1 {
        tangents[i] = if secants[i - 1] * secants[i] <= 0.0 {
            0.0
        } else {
            (secants[i - 1] + secants[i]) / 2.0
        };
    }
    for i in 0..n - 1 {
        if secants[i] == 0.0 {
            tangents[i] = 0.0;
            tangents[i + 1] = 0.0;
            continue;
        }
        let a = tangents[i] / secants[i];
        let b = tangents[i + 1] / secants[i];
        let sum = a * a + b * b;
        if sum > 9.0 {
            let scale = 3.0 / sum.sqrt();
            tangents[i] = scale * a * secants[i];
            tangents[i + 1] = scale * b * secants[i];
        }
    }

    let mut lut = [0u8; 256];
    for (value, slot) in lut.iter_mut().enumerate() {
        let x = value as f32 / 255.0;
        let y = if x <= pts[0][0] {
            pts[0][1]
        } else if x >= pts[n - 1][0] {
            pts[n - 1][1]
        } else {
            let mut segment = n - 2;
            for i in 0..n - 1 {
                if x <= pts[i + 1][0] {
                    segment = i;
                    break;
                }
            }
            let x0 = pts[segment][0];
            let x1 = pts[segment + 1][0];
            let h = (x1 - x0).max(1e-6);
            let t = (x - x0) / h;
            let t2 = t * t;
            let t3 = t2 * t;
            let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
            let h10 = t3 - 2.0 * t2 + t;
            let h01 = -2.0 * t3 + 3.0 * t2;
            let h11 = t3 - t2;
            h00 * pts[segment][1]
                + h10 * h * tangents[segment]
                + h01 * pts[segment + 1][1]
                + h11 * h * tangents[segment + 1]
        };
        *slot = (y * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    lut
}

/// Per-render color grading: a per-channel tone LUT plus the operations that mix channels.
///
/// The LUT chain is brightness → contrast → levels → gamma → curve → invert. Saturation,
/// temperature/tint, grayscale and sepia are applied afterwards because they depend on more than
/// one channel. Building the LUT once per render keeps the per-pixel cost low.
pub(crate) struct ColorGrade {
    lut: [u8; 256],
    saturation: f32,
    /// `-1..=1`, positive warms the image (more red, less blue).
    temperature: f32,
    /// `-1..=1`, positive pushes toward magenta (less green).
    tint: f32,
    grayscale: bool,
    sepia: bool,
    neutral: bool,
}

impl ColorGrade {
    pub(crate) fn from_settings(settings: &Settings, auto_levels: Option<(u8, u8)>) -> Self {
        let neutral = settings.color_is_neutral();
        let (black, white) = if settings.auto_contrast {
            auto_levels.unwrap_or((settings.black_point, settings.white_point))
        } else {
            (settings.black_point, settings.white_point)
        };
        let black = black.min(254) as f32;
        let white = (white.max(black as u8 + 1)) as f32;

        let offset = settings.brightness.clamp(-100, 100) as f32 * 2.55;
        let contrast = settings.contrast.clamp(0, 200) as f32 / 100.0;
        let gamma = settings.gamma.clamp(10, 300) as f32 / 100.0;
        let inv_gamma = 1.0 / gamma;
        let curve = build_curve_lut(&settings.curves);
        let invert = settings.invert;
        let levels_span = (white - black).max(1.0);

        let mut lut = [0u8; 256];
        for (value, slot) in lut.iter_mut().enumerate() {
            let mut v = value as f32;
            v = (v + offset - 128.0) * contrast + 128.0;
            v = (v - black) * 255.0 / levels_span;
            v = v.clamp(0.0, 255.0);
            v = 255.0 * (v / 255.0).powf(inv_gamma);
            v = curve[v.round().clamp(0.0, 255.0) as usize] as f32;
            if invert {
                v = 255.0 - v;
            }
            *slot = v.round().clamp(0.0, 255.0) as u8;
        }

        Self {
            lut,
            saturation: settings.saturation.clamp(0, 200) as f32 / 100.0,
            temperature: settings.temperature.clamp(-100, 100) as f32 / 100.0,
            tint: settings.tint.clamp(-100, 100) as f32 / 100.0,
            grayscale: settings.grayscale,
            sepia: settings.sepia,
            neutral,
        }
    }

    pub(crate) fn apply(&self, rgb: [u8; 3]) -> [u8; 3] {
        if self.neutral {
            return rgb;
        }
        let mut r = self.lut[rgb[0] as usize] as f32;
        let mut g = self.lut[rgb[1] as usize] as f32;
        let mut b = self.lut[rgb[2] as usize] as f32;

        if self.saturation != 1.0 {
            let luma = 0.299 * r + 0.587 * g + 0.114 * b;
            r = luma + (r - luma) * self.saturation;
            g = luma + (g - luma) * self.saturation;
            b = luma + (b - luma) * self.saturation;
        }
        if self.temperature != 0.0 {
            r += self.temperature * 40.0;
            b -= self.temperature * 40.0;
        }
        if self.tint != 0.0 {
            g -= self.tint * 30.0;
        }

        let mut out = [
            r.round().clamp(0.0, 255.0) as u8,
            g.round().clamp(0.0, 255.0) as u8,
            b.round().clamp(0.0, 255.0) as u8,
        ];

        if self.grayscale {
            let luma = (0.299 * out[0] as f32 + 0.587 * out[1] as f32 + 0.114 * out[2] as f32)
                .round()
                .clamp(0.0, 255.0) as u8;
            out = [luma, luma, luma];
        }
        if self.sepia {
            let [r, g, b] = out.map(|c| c as f32);
            let tone = |value: f32| value.round().clamp(0.0, 255.0) as u8;
            out = [
                tone(0.393 * r + 0.769 * g + 0.189 * b),
                tone(0.349 * r + 0.686 * g + 0.168 * b),
                tone(0.272 * r + 0.534 * g + 0.131 * b),
            ];
        }
        out
    }
}

/// Luma histogram of an image, ignoring fully transparent pixels.
pub(crate) fn luma_histogram(image: &RgbaImage) -> [u32; 256] {
    let mut histogram = [0u32; 256];
    for pixel in image.pixels() {
        if pixel[3] == 0 {
            continue;
        }
        let luma = (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32)
            .round()
            .clamp(0.0, 255.0) as usize;
        histogram[luma] += 1;
    }
    histogram
}

/// Black and white input levels that stretch the central 99% of the histogram to the full range.
pub(crate) fn auto_levels(histogram: &[u32; 256]) -> (u8, u8) {
    let total: u64 = histogram.iter().map(|&count| count as u64).sum();
    if total == 0 {
        return (0, 255);
    }
    let low_target = ((total as f64) * 0.005).ceil().max(1.0) as u64;
    let high_target = ((total as f64) * 0.995).floor().max(low_target as f64) as u64;

    let mut accumulated = 0u64;
    let mut black = 0usize;
    for (value, &count) in histogram.iter().enumerate() {
        accumulated += count as u64;
        if accumulated >= low_target {
            black = value;
            break;
        }
    }

    accumulated = 0;
    let mut white = 255usize;
    for (value, &count) in histogram.iter().enumerate() {
        accumulated += count as u64;
        if accumulated >= high_target {
            white = value;
            break;
        }
    }

    if white <= black {
        (black.min(254) as u8, 255)
    } else {
        (black as u8, white as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::settings::Settings;

    #[test]
    fn color_adjustments_neutral_are_identity() {
        let settings = Settings::default();
        assert!(settings.color_is_neutral());
        let grade = ColorGrade::from_settings(&settings, None);
        for rgb in [[0, 0, 0], [255, 255, 255], [37, 128, 219], [200, 10, 90]] {
            assert_eq!(grade.apply(rgb), rgb);
        }
    }

    #[test]
    fn color_adjustments_move_channels_in_expected_direction() {
        let grade = |settings: Settings| ColorGrade::from_settings(&settings, None);

        // Brightness is additive.
        let brighter = grade(Settings {
            brightness: 20,
            ..Default::default()
        });
        assert!(brighter.apply([100, 100, 100]).iter().all(|&c| c > 100));
        let darker = grade(Settings {
            brightness: -20,
            ..Default::default()
        });
        assert!(darker.apply([100, 100, 100]).iter().all(|&c| c < 100));

        // Contrast pushes away from mid-gray.
        let high = grade(Settings {
            contrast: 150,
            ..Default::default()
        });
        assert!(high.apply([40, 40, 40]).iter().all(|&c| c < 40));
        let low = grade(Settings {
            contrast: 50,
            ..Default::default()
        });
        assert!(low.apply([40, 40, 40]).iter().all(|&c| c > 40));

        // Zero saturation collapses to luminance (a gray).
        let gray = grade(Settings {
            saturation: 0,
            ..Default::default()
        });
        let gray = gray.apply([200, 50, 25]);
        assert_eq!(gray[0], gray[1]);
        assert_eq!(gray[1], gray[2]);

        // Clamping keeps channels within range at the extremes.
        let clamp_high = grade(Settings {
            brightness: 100,
            ..Default::default()
        });
        assert_eq!(clamp_high.apply([250, 250, 250]), [255, 255, 255]);
        let clamp_low = grade(Settings {
            brightness: -100,
            ..Default::default()
        });
        assert_eq!(clamp_low.apply([0, 0, 0]), [0, 0, 0]);
    }

    #[test]
    fn gamma_lifts_or_darkens_midtones() {
        let neutral = ColorGrade::from_settings(&Settings::default(), None);
        assert_eq!(neutral.apply([128, 128, 128]), [128, 128, 128]);

        let lighter = ColorGrade::from_settings(
            &Settings {
                gamma: 200,
                ..Default::default()
            },
            None,
        );
        assert!(lighter.apply([128, 128, 128])[0] > 128);

        let darker = ColorGrade::from_settings(
            &Settings {
                gamma: 50,
                ..Default::default()
            },
            None,
        );
        assert!(darker.apply([128, 128, 128])[0] < 128);
    }

    #[test]
    fn temperature_and_tint_shift_expected_channels() {
        let warm = ColorGrade::from_settings(
            &Settings {
                temperature: 50,
                ..Default::default()
            },
            None,
        )
        .apply([128, 128, 128]);
        assert!(warm[0] > 128 && warm[2] < 128, "warm shift: {warm:?}");

        let cool = ColorGrade::from_settings(
            &Settings {
                temperature: -50,
                ..Default::default()
            },
            None,
        )
        .apply([128, 128, 128]);
        assert!(cool[0] < 128 && cool[2] > 128, "cool shift: {cool:?}");

        let magenta = ColorGrade::from_settings(
            &Settings {
                tint: 50,
                ..Default::default()
            },
            None,
        )
        .apply([128, 128, 128]);
        assert!(magenta[1] < 128, "magenta shift: {magenta:?}");
    }

    #[test]
    fn levels_clip_black_and_white_points() {
        let grade = ColorGrade::from_settings(
            &Settings {
                black_point: 64,
                white_point: 192,
                ..Default::default()
            },
            None,
        );
        assert_eq!(grade.apply([64, 64, 64]), [0, 0, 0]);
        assert_eq!(grade.apply([192, 192, 192]), [255, 255, 255]);
        assert_eq!(grade.apply([128, 128, 128]), [128, 128, 128]);
    }

    #[test]
    fn invert_grayscale_and_sepia_behave_as_expected() {
        let invert = ColorGrade::from_settings(
            &Settings {
                invert: true,
                ..Default::default()
            },
            None,
        );
        assert_eq!(invert.apply([10, 20, 30]), [245, 235, 225]);

        let gray = ColorGrade::from_settings(
            &Settings {
                grayscale: true,
                ..Default::default()
            },
            None,
        )
        .apply([200, 50, 25]);
        assert_eq!(gray[0], gray[1]);
        assert_eq!(gray[1], gray[2]);

        let sepia = ColorGrade::from_settings(
            &Settings {
                sepia: true,
                ..Default::default()
            },
            None,
        )
        .apply([128, 128, 128]);
        assert!(
            sepia[0] > sepia[1] && sepia[1] > sepia[2],
            "sepia: {sepia:?}"
        );
    }

    #[test]
    fn auto_levels_stretch_low_contrast_histogram() {
        let mut histogram = [0u32; 256];
        for count in histogram.iter_mut().take(151).skip(100) {
            *count = 10;
        }
        let (black, white) = auto_levels(&histogram);
        assert_eq!(black, 100);
        assert_eq!(white, 150);

        // Empty histogram falls back to the full range.
        assert_eq!(auto_levels(&[0u32; 256]), (0, 255));
    }
}
