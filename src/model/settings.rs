use crate::imaging::dither::DitherMode;
use crate::model::geometry::{CropRect, ResizeFilter, ResizeMode};
use serde::{Deserialize, Serialize};

/// Selectable resolutions for the longest side.
pub(crate) const RESOLUTION_PRESETS: [u32; 6] = [64, 128, 256, 512, 1024, 2048];

/// How the alpha channel of the stylized result is handled.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub(crate) enum AlphaMode {
    /// Keep the alpha channel unchanged (PNG/WebP only; JPEG is flattened separately).
    #[default]
    Save,
    /// Make alpha binary: values at or above the threshold become opaque, the rest transparent.
    Threshold,
    /// Composite the image over the background color and make the result fully opaque.
    Fill,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub(crate) struct Settings {
    /// Target resolution of the longest side, or `None` to keep the source resolution.
    pub(crate) resolution: Option<u32>,
    /// Manual crop in normalized source coordinates.
    #[serde(default)]
    pub(crate) crop: CropRect,
    /// Explicit output size, or `None` to keep the cropped source size.
    #[serde(default)]
    pub(crate) output_size: Option<[u32; 2]>,
    /// How the stylized image is fitted into `output_size`.
    #[serde(default)]
    pub(crate) resize_mode: ResizeMode,
    /// Filter used by the final resize stage.
    #[serde(default)]
    pub(crate) resize_filter: ResizeFilter,
    pub(crate) palette_size: u32,
    pub(crate) palette_method: PaletteMethod,
    pub(crate) dither: DitherMode,
    pub(crate) dither_strength: u8,
    /// Traverse error-diffusion rows in alternating directions to break up directional artifacts.
    #[serde(default = "default_serpentine")]
    pub(crate) dither_serpentine: bool,
    /// Additive brightness offset in `-100..=100`, where `0` is neutral.
    #[serde(default)]
    pub(crate) brightness: i32,
    /// Saturation percentage in `0..=200`, where `100` is neutral.
    #[serde(default = "neutral_percent")]
    pub(crate) saturation: i32,
    /// Contrast percentage in `0..=200`, where `100` is neutral.
    #[serde(default = "neutral_percent")]
    pub(crate) contrast: i32,
    /// Gamma percentage in `10..=300`, where `100` is neutral. Values above 100 lift midtones.
    #[serde(default = "neutral_percent")]
    pub(crate) gamma: i32,
    /// White-balance shift in `-100..=100`: positive is warmer (more red, less blue).
    #[serde(default)]
    pub(crate) temperature: i32,
    /// Green–magenta shift in `-100..=100`: positive is magenta, negative is green.
    #[serde(default)]
    pub(crate) tint: i32,
    /// Input black level in `0..=254`; values at or below it become black.
    #[serde(default = "default_black_point")]
    pub(crate) black_point: u8,
    /// Input white level in `1..=255`; values at or above it become white.
    #[serde(default = "default_white_point")]
    pub(crate) white_point: u8,
    /// Stretch levels automatically from each image's histogram, ignoring the manual points.
    #[serde(default)]
    pub(crate) auto_contrast: bool,
    /// Tone curve control points in normalized `0..=1` coordinates, sorted by x.
    #[serde(default = "default_curves")]
    pub(crate) curves: Vec<[f32; 2]>,
    #[serde(default)]
    pub(crate) invert: bool,
    #[serde(default)]
    pub(crate) grayscale: bool,
    #[serde(default)]
    pub(crate) sepia: bool,
    /// Alpha handling applied to the stylized result.
    #[serde(default)]
    pub(crate) alpha_mode: AlphaMode,
    /// Alpha cutoff for [`AlphaMode::Threshold`].
    #[serde(default = "default_alpha_threshold")]
    pub(crate) alpha_threshold: u8,
    /// Opaque color used by [`AlphaMode::Fill`] and when encoding JPEG.
    #[serde(default = "default_background")]
    pub(crate) background: [u8; 3],
}

/// Neutral value shared by the percentage-based color controls (saturation, contrast, gamma).
pub(crate) fn neutral_percent() -> i32 {
    100
}

/// Error diffusion uses serpentine traversal unless explicitly disabled.
pub(crate) fn default_serpentine() -> bool {
    true
}

pub(crate) fn default_black_point() -> u8 {
    0
}

pub(crate) fn default_white_point() -> u8 {
    255
}

/// Identity tone curve: a straight line from black to white.
pub(crate) fn default_curves() -> Vec<[f32; 2]> {
    vec![[0.0, 0.0], [1.0, 1.0]]
}

pub(crate) fn default_alpha_threshold() -> u8 {
    128
}

pub(crate) fn default_background() -> [u8; 3] {
    [255, 255, 255]
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            resolution: Some(256),
            crop: CropRect::FULL,
            output_size: None,
            resize_mode: ResizeMode::Stretch,
            resize_filter: ResizeFilter::Nearest,
            palette_size: 32,
            palette_method: PaletteMethod::Wu,
            dither: DitherMode::FloydSteinberg,
            dither_strength: 100,
            dither_serpentine: default_serpentine(),
            brightness: 0,
            saturation: neutral_percent(),
            contrast: neutral_percent(),
            gamma: neutral_percent(),
            temperature: 0,
            tint: 0,
            black_point: default_black_point(),
            white_point: default_white_point(),
            auto_contrast: false,
            curves: default_curves(),
            invert: false,
            grayscale: false,
            sepia: false,
            alpha_mode: AlphaMode::Save,
            alpha_threshold: default_alpha_threshold(),
            background: default_background(),
        }
    }
}

/// Whether the color controls are all at their neutral values.
fn color_is_neutral(brightness: i32, saturation: i32, contrast: i32) -> bool {
    brightness == 0 && saturation == neutral_percent() && contrast == neutral_percent()
}

impl Settings {
    /// Whether every color-grading control is at its neutral value.
    pub(crate) fn color_is_neutral(&self) -> bool {
        color_is_neutral(self.brightness, self.saturation, self.contrast)
            && self.gamma == neutral_percent()
            && self.temperature == 0
            && self.tint == 0
            && !self.auto_contrast
            && self.black_point == default_black_point()
            && self.white_point == default_white_point()
            && !self.invert
            && !self.grayscale
            && !self.sepia
            && sanitize_curves(&self.curves) == default_curves()
    }

    /// Return all color controls to their neutral values.
    pub(crate) fn reset_color(&mut self) {
        self.brightness = 0;
        self.saturation = neutral_percent();
        self.contrast = neutral_percent();
        self.gamma = neutral_percent();
        self.temperature = 0;
        self.tint = 0;
        self.black_point = default_black_point();
        self.white_point = default_white_point();
        self.auto_contrast = false;
        self.curves = default_curves();
        self.invert = false;
        self.grayscale = false;
        self.sepia = false;
    }

    /// Whether the crop and resize controls are all at their neutral values.
    pub(crate) fn geometry_is_neutral(&self) -> bool {
        self.crop.is_full()
            && self.output_size.is_none()
            && self.resize_mode == ResizeMode::Stretch
            && self.resize_filter == ResizeFilter::Nearest
    }

    /// Restore the full frame and disable explicit output sizing.
    pub(crate) fn reset_geometry(&mut self) {
        self.crop = CropRect::FULL;
        self.output_size = None;
        self.resize_mode = ResizeMode::Stretch;
        self.resize_filter = ResizeFilter::Nearest;
    }
}

/// Palette quantization algorithm selected in the settings.
#[derive(PartialEq, Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) enum PaletteMethod {
    Wu,
    KMeans,
}

/// Normalize tone-curve control points: clamp to the unit square, sort by x, merge duplicates,
/// and guarantee an endpoint at x = 0 and x = 1.
pub(crate) fn sanitize_curves(points: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let mut sorted: Vec<[f32; 2]> = points
        .iter()
        .map(|p| [p[0].clamp(0.0, 1.0), p[1].clamp(0.0, 1.0)])
        .collect();
    sorted.sort_by(|a, b| a[0].total_cmp(&b[0]));

    let mut out: Vec<[f32; 2]> = Vec::with_capacity(sorted.len());
    for point in sorted {
        match out.last_mut() {
            Some(last) if (point[0] - last[0]).abs() < 1e-4 => *last = point,
            _ => out.push(point),
        }
    }
    if out.is_empty() {
        return default_curves();
    }
    if out[0][0] > 1e-4 {
        out.insert(0, [0.0, out[0][1]]);
    } else {
        out[0][0] = 0.0;
    }
    let last = out.len() - 1;
    if out[last][0] < 1.0 - 1e-4 {
        out.push([1.0, out[last][1]]);
    } else {
        out[last][0] = 1.0;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imaging::color::build_curve_lut;
    use crate::model::settings::default_curves;

    #[test]
    fn curve_lut_default_is_identity() {
        let lut = build_curve_lut(&default_curves());
        for (value, mapped) in lut.iter().enumerate() {
            assert_eq!(*mapped as usize, value);
        }
    }

    #[test]
    fn curve_lut_bends_midtones_and_keeps_endpoints() {
        let lut = build_curve_lut(&[[0.0, 0.0], [0.5, 0.8], [1.0, 1.0]]);
        assert_eq!(lut[0], 0);
        assert_eq!(lut[255], 255);
        assert!(lut[128] > 128, "midpoint was not lifted: {}", lut[128]);
    }

    #[test]
    fn sanitize_curves_sorts_clamps_and_adds_endpoints() {
        let points = sanitize_curves(&[[0.8, 1.5], [0.2, -1.0], [0.2, 0.3]]);
        assert_eq!(points.first().unwrap()[0], 0.0);
        assert_eq!(points.last().unwrap()[0], 1.0);
        assert!(points.windows(2).all(|w| w[0][0] < w[1][0]));
        assert!(points.iter().all(|p| (0.0..=1.0).contains(&p[1])));
    }

    #[test]
    fn settings_without_color_fields_deserialize_to_neutral_defaults() {
        // Legacy presets written before the color controls existed must still load.
        let legacy = r#"{
            "resolution": 128,
            "palette_size": 16,
            "palette_method": "Wu",
            "dither": "Ordered",
            "dither_strength": 75
        }"#;
        let settings: Settings = serde_json::from_str(legacy).expect("deserialize legacy settings");
        assert!(settings.dither_serpentine);
        assert_eq!(settings.brightness, 0);
        assert_eq!(settings.saturation, 100);
        assert_eq!(settings.contrast, 100);
        assert_eq!(settings.gamma, 100);
        assert_eq!(settings.temperature, 0);
        assert_eq!(settings.tint, 0);
        assert_eq!(settings.black_point, 0);
        assert_eq!(settings.white_point, 255);
        assert!(!settings.auto_contrast);
        assert_eq!(settings.curves, default_curves());
        assert!(!settings.invert);
        assert!(!settings.grayscale);
        assert!(!settings.sepia);
        assert!(settings.color_is_neutral());
        assert_eq!(settings.crop, CropRect::FULL);
        assert_eq!(settings.output_size, None);
        assert_eq!(settings.resize_mode, ResizeMode::Stretch);
        assert_eq!(settings.resize_filter, ResizeFilter::Nearest);
        assert!(settings.geometry_is_neutral());
        assert_eq!(settings.alpha_mode, AlphaMode::Save);
        assert_eq!(settings.alpha_threshold, 128);
        assert_eq!(settings.background, [255, 255, 255]);
    }
}
