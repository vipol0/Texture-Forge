use crate::blue_noise;
use quantette::color_map::{
    IndexedColorMap, NearestNeighborColorMap, NearestNeighborParallelColorMap,
};
use quantette::deps::palette::Srgb;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};

/// Standard 8x8 Bayer threshold matrix (values 0..=63).
pub(crate) const BAYER8: [[u8; 8]; 8] = [
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
];

#[derive(PartialEq, Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) enum DitherMode {
    None,
    Ordered,
    BlueNoise,
    FloydSteinberg,
    Atkinson,
    JarvisJudiceNinke,
    Stucki,
    Burkes,
    Sierra,
}

impl DitherMode {
    /// Every selectable mode, in the order shown in the UI.
    pub(crate) const ALL: [Self; 9] = [
        Self::None,
        Self::Ordered,
        Self::BlueNoise,
        Self::FloydSteinberg,
        Self::Atkinson,
        Self::JarvisJudiceNinke,
        Self::Stucki,
        Self::Burkes,
        Self::Sierra,
    ];

    /// Human-readable label used by the dither selector.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::None => "Нет",
            Self::Ordered => "Упорядоченный (Bayer)",
            Self::BlueNoise => "Синий шум",
            Self::FloydSteinberg => "Флойд–Стейнберг",
            Self::Atkinson => "Аткинсон",
            Self::JarvisJudiceNinke => "Джарвис–Джадис–Нинке (JJN)",
            Self::Stucki => "Стьюки",
            Self::Burkes => "Бёркс",
            Self::Sierra => "Сьерра (Sierra-3)",
        }
    }

    /// Whether this mode diffuses quantization error to neighboring pixels.
    pub(crate) fn is_error_diffusion(self) -> bool {
        matches!(
            self,
            Self::FloydSteinberg
                | Self::Atkinson
                | Self::JarvisJudiceNinke
                | Self::Stucki
                | Self::Burkes
                | Self::Sierra
        )
    }
}

/// One error-diffusion tap: horizontal/vertical offset and normalized weight.
type DiffusionTap = (i8, i8, f32);

/// Floyd–Steinberg kernel.
const FS_TAPS: &[DiffusionTap] = &[
    (1, 0, 7.0 / 16.0),
    (-1, 1, 3.0 / 16.0),
    (0, 1, 5.0 / 16.0),
    (1, 1, 1.0 / 16.0),
];

/// Atkinson kernel: only 6/8 of the error is propagated.
const ATKINSON_TAPS: &[DiffusionTap] = &[
    (1, 0, 1.0 / 8.0),
    (2, 0, 1.0 / 8.0),
    (-1, 1, 1.0 / 8.0),
    (0, 1, 1.0 / 8.0),
    (1, 1, 1.0 / 8.0),
    (0, 2, 1.0 / 8.0),
];

/// Jarvis–Judice–Ninke kernel.
const JJN_TAPS: &[DiffusionTap] = &[
    (1, 0, 7.0 / 48.0),
    (2, 0, 5.0 / 48.0),
    (-2, 1, 3.0 / 48.0),
    (-1, 1, 5.0 / 48.0),
    (0, 1, 7.0 / 48.0),
    (1, 1, 5.0 / 48.0),
    (2, 1, 3.0 / 48.0),
    (-2, 2, 1.0 / 48.0),
    (-1, 2, 3.0 / 48.0),
    (0, 2, 5.0 / 48.0),
    (1, 2, 3.0 / 48.0),
    (2, 2, 1.0 / 48.0),
];

/// Stucki kernel.
const STUCKI_TAPS: &[DiffusionTap] = &[
    (1, 0, 8.0 / 42.0),
    (2, 0, 4.0 / 42.0),
    (-2, 1, 2.0 / 42.0),
    (-1, 1, 4.0 / 42.0),
    (0, 1, 8.0 / 42.0),
    (1, 1, 4.0 / 42.0),
    (2, 1, 2.0 / 42.0),
    (-2, 2, 1.0 / 42.0),
    (-1, 2, 2.0 / 42.0),
    (0, 2, 4.0 / 42.0),
    (1, 2, 2.0 / 42.0),
    (2, 2, 1.0 / 42.0),
];

/// Burkes kernel.
const BURKES_TAPS: &[DiffusionTap] = &[
    (1, 0, 8.0 / 32.0),
    (2, 0, 4.0 / 32.0),
    (-2, 1, 2.0 / 32.0),
    (-1, 1, 4.0 / 32.0),
    (0, 1, 8.0 / 32.0),
    (1, 1, 4.0 / 32.0),
    (2, 1, 2.0 / 32.0),
];

/// Sierra-3 kernel.
const SIERRA_TAPS: &[DiffusionTap] = &[
    (1, 0, 5.0 / 32.0),
    (2, 0, 3.0 / 32.0),
    (-2, 1, 2.0 / 32.0),
    (-1, 1, 4.0 / 32.0),
    (0, 1, 5.0 / 32.0),
    (1, 1, 4.0 / 32.0),
    (2, 1, 2.0 / 32.0),
    (-1, 2, 2.0 / 32.0),
    (0, 2, 3.0 / 32.0),
    (1, 2, 2.0 / 32.0),
];

/// Diffusion taps belonging to the given error-diffusion mode.
pub(crate) fn diffusion_taps(mode: DitherMode) -> Option<&'static [DiffusionTap]> {
    match mode {
        DitherMode::FloydSteinberg => Some(FS_TAPS),
        DitherMode::Atkinson => Some(ATKINSON_TAPS),
        DitherMode::JarvisJudiceNinke => Some(JJN_TAPS),
        DitherMode::Stucki => Some(STUCKI_TAPS),
        DitherMode::Burkes => Some(BURKES_TAPS),
        DitherMode::Sierra => Some(SIERRA_TAPS),
        _ => None,
    }
}

/// Threshold amplitude for ordered and blue-noise dithering.
///
/// Bigger palettes need less dithering; fewer colors need more. `strength` is a percentage in
/// `0..=100`.
pub(crate) fn threshold_spread(palette: &quantette::PaletteBuf<Srgb<u8>>, strength: u8) -> f32 {
    let levels = (palette.len().max(1)) as f32;
    let strength = (strength.min(100) as f32) / 100.0;
    strength * 255.0 / levels.sqrt()
}

/// Map every pixel to its nearest palette color without dithering.
pub(crate) fn nearest_remap(
    input: &[Srgb<u8>],
    palette: &quantette::PaletteBuf<Srgb<u8>>,
) -> Vec<Srgb<u8>> {
    let map: NearestNeighborParallelColorMap<Srgb<u8>, u8, 3> =
        NearestNeighborParallelColorMap::new(palette.clone());
    input
        .par_iter()
        .map(|color| map.palette_color(color))
        .collect()
}

/// Remap a slice of sRGB pixels to the given palette using ordered (Bayer) dithering.
///
/// A position-dependent threshold is added to each channel before nearest-neighbour lookup,
/// producing a regular dither pattern. `strength` is a percentage in `0..=100`.
pub(crate) fn ordered_remap(
    input: &[Srgb<u8>],
    width: u32,
    palette: &quantette::PaletteBuf<Srgb<u8>>,
    strength: u8,
) -> Vec<Srgb<u8>> {
    let map: NearestNeighborParallelColorMap<Srgb<u8>, u8, 3> =
        NearestNeighborParallelColorMap::new(palette.clone());
    let spread = threshold_spread(palette, strength);
    let width = width.max(1) as usize;

    input
        .par_iter()
        .enumerate()
        .map(|(i, c)| {
            let x = i % width;
            let y = i / width;
            let threshold = (BAYER8[y & 7][x & 7] as f32 + 0.5) / 64.0 - 0.5;
            let offset = threshold * spread;
            let r = (c.red as f32 + offset).round().clamp(0.0, 255.0) as u8;
            let g = (c.green as f32 + offset).round().clamp(0.0, 255.0) as u8;
            let b = (c.blue as f32 + offset).round().clamp(0.0, 255.0) as u8;
            map.palette_color(&Srgb::new(r, g, b))
        })
        .collect()
}

/// Remap a slice of sRGB pixels using a precomputed blue-noise threshold matrix.
pub(crate) fn blue_noise_remap(
    input: &[Srgb<u8>],
    width: u32,
    palette: &quantette::PaletteBuf<Srgb<u8>>,
    strength: u8,
) -> Vec<Srgb<u8>> {
    let map: NearestNeighborParallelColorMap<Srgb<u8>, u8, 3> =
        NearestNeighborParallelColorMap::new(palette.clone());
    let spread = threshold_spread(palette, strength);
    let width = width.max(1) as usize;
    let mask_size = blue_noise::SIZE;
    let mask_area = (mask_size * mask_size) as f32;

    input
        .par_iter()
        .enumerate()
        .map(|(i, c)| {
            let x = i % width;
            let y = i / width;
            let rank = blue_noise::RANKS[y & (mask_size - 1)][x & (mask_size - 1)] as f32;
            let threshold = (rank + 0.5) / mask_area - 0.5;
            let offset = threshold * spread;
            let r = (c.red as f32 + offset).round().clamp(0.0, 255.0) as u8;
            let g = (c.green as f32 + offset).round().clamp(0.0, 255.0) as u8;
            let b = (c.blue as f32 + offset).round().clamp(0.0, 255.0) as u8;
            map.palette_color(&Srgb::new(r, g, b))
        })
        .collect()
}

/// Remap pixels using error diffusion with an arbitrary kernel.
///
/// The adjusted pixel is clamped to `0..=255` before the nearest-color lookup, and the remaining
/// error is scaled by `strength` and distributed according to `taps`. When `serpentine` is set,
/// odd rows are traversed right-to-left and the kernel is mirrored horizontally.
#[allow(clippy::too_many_arguments)]
pub(crate) fn error_diffusion_remap(
    input: &[Srgb<u8>],
    width: u32,
    height: u32,
    palette: &quantette::PaletteBuf<Srgb<u8>>,
    taps: &[DiffusionTap],
    strength: u8,
    serpentine: bool,
    cancel: Option<&AtomicBool>,
) -> Option<Vec<Srgb<u8>>> {
    const PAD: usize = 2;

    let w = width.max(1) as usize;
    let h = height.max(1) as usize;
    let map: NearestNeighborColorMap<Srgb<u8>, u8, 3> =
        NearestNeighborColorMap::new(palette.clone());
    let strength = (strength.clamp(1, 100) as f32) / 100.0;
    let stride = w + 2 * PAD;
    let mut errors = [
        vec![[0.0f32; 3]; stride],
        vec![[0.0f32; 3]; stride],
        vec![[0.0f32; 3]; stride],
    ];
    let mut output = vec![Srgb::new(0u8, 0, 0); w * h];

    for y in 0..h {
        if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            return None;
        }
        let reverse = serpentine && (y & 1 == 1);
        let row = &input[y * w..(y + 1) * w];

        for step in 0..w {
            let x = if reverse { w - 1 - step } else { step };
            let current = errors[0][x + PAD];
            let r = (row[x].red as f32 + current[0]).round().clamp(0.0, 255.0);
            let g = (row[x].green as f32 + current[1]).round().clamp(0.0, 255.0);
            let b = (row[x].blue as f32 + current[2]).round().clamp(0.0, 255.0);
            let color = map.palette_color(&Srgb::new(r as u8, g as u8, b as u8));
            output[y * w + x] = color;

            let error = [
                (r - color.red as f32) * strength,
                (g - color.green as f32) * strength,
                (b - color.blue as f32) * strength,
            ];

            for &(dx, dy, weight) in taps {
                let direction = if reverse { -1 } else { 1 };
                let target_x = x as isize + direction * dx as isize;
                let target_y = y + dy as usize;
                if target_x < 0 || target_x >= w as isize || target_y >= h {
                    continue;
                }
                let target = &mut errors[dy as usize][target_x as usize + PAD];
                target[0] += error[0] * weight;
                target[1] += error[1] * weight;
                target[2] += error[2] * weight;
            }
        }

        errors.rotate_left(1);
        errors[2].fill([0.0; 3]);
    }

    Some(output)
}

/// Dithering algorithm and its tuning, bundled so callers configure it once per render.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DitherEngine {
    pub(crate) mode: DitherMode,
    pub(crate) strength: u8,
    pub(crate) serpentine: bool,
}

impl DitherEngine {
    pub(crate) fn new(mode: DitherMode, strength: u8, serpentine: bool) -> Self {
        Self {
            mode,
            strength,
            serpentine,
        }
    }

    /// Remap a raster to `palette` according to the configured algorithm.
    pub(crate) fn remap(
        &self,
        input: &[Srgb<u8>],
        width: u32,
        height: u32,
        palette: &quantette::PaletteBuf<Srgb<u8>>,
        cancel: Option<&AtomicBool>,
    ) -> Option<Vec<Srgb<u8>>> {
        dither_remap(
            input,
            width,
            height,
            palette,
            self.mode,
            self.strength,
            self.serpentine,
            cancel,
        )
    }
}

/// Select and run the requested dithering algorithm.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dither_remap(
    input: &[Srgb<u8>],
    width: u32,
    height: u32,
    palette: &quantette::PaletteBuf<Srgb<u8>>,
    mode: DitherMode,
    strength: u8,
    serpentine: bool,
    cancel: Option<&AtomicBool>,
) -> Option<Vec<Srgb<u8>>> {
    match mode {
        DitherMode::None => Some(nearest_remap(input, palette)),
        DitherMode::Ordered => Some(ordered_remap(input, width, palette, strength)),
        DitherMode::BlueNoise => Some(blue_noise_remap(input, width, palette, strength)),
        mode => {
            let taps = diffusion_taps(mode)?;
            error_diffusion_remap(
                input, width, height, palette, taps, strength, serpentine, cancel,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use quantette::deps::palette::Srgb;

    #[test]
    fn bayer_matrix_is_permutation_of_0_to_63() {
        let mut seen = [false; 64];
        for row in BAYER8 {
            for value in row {
                assert!(!seen[value as usize], "duplicate value {value}");
                seen[value as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn ordered_remap_only_emits_palette_colors() {
        let palette: quantette::PaletteBuf<Srgb<u8>> =
            quantette::PaletteBuf::try_from(vec![Srgb::new(0u8, 0, 0), Srgb::new(255u8, 255, 255)])
                .unwrap();
        let input = vec![Srgb::new(120u8, 130, 140); 64];
        let output = ordered_remap(&input, 8, &palette, 100);
        assert_eq!(output.len(), 64);
        for color in output {
            assert!(color == palette[0usize] || color == palette[1usize]);
        }
    }

    #[test]
    fn blue_noise_matrix_is_permutation_of_0_to_4095() {
        let mut seen = vec![false; blue_noise::SIZE * blue_noise::SIZE];
        for row in blue_noise::RANKS {
            for rank in row {
                assert!(!seen[rank as usize], "duplicate rank {rank}");
                seen[rank as usize] = true;
            }
        }
        assert!(seen.iter().all(|&seen| seen));
    }

    #[test]
    fn every_dither_mode_only_emits_palette_colors() {
        let palette: quantette::PaletteBuf<Srgb<u8>> = quantette::PaletteBuf::try_from(vec![
            Srgb::new(0u8, 0, 0),
            Srgb::new(255u8, 255, 255),
            Srgb::new(220u8, 30, 40),
            Srgb::new(30u8, 180, 60),
            Srgb::new(40u8, 70, 210),
        ])
        .unwrap();
        let mut input = Vec::with_capacity(16 * 16);
        for y in 0..16u32 {
            for x in 0..16u32 {
                input.push(Srgb::new(
                    (x * 16) as u8,
                    (y * 16) as u8,
                    ((x + y) * 8) as u8,
                ));
            }
        }

        for mode in DitherMode::ALL {
            for serpentine in [false, true] {
                let output = dither_remap(&input, 16, 16, &palette, mode, 80, serpentine, None)
                    .expect("dither produced output");
                assert_eq!(output.len(), input.len());
                for color in output {
                    assert!(
                        (0..palette.len()).any(|i| color == palette[i]),
                        "{mode:?} emitted a color outside the palette"
                    );
                }
            }
        }
    }

    #[test]
    fn serpentine_changes_floyd_steinberg_output() {
        let palette: quantette::PaletteBuf<Srgb<u8>> =
            quantette::PaletteBuf::try_from(vec![Srgb::new(0u8, 0, 0), Srgb::new(255u8, 255, 255)])
                .unwrap();
        let mut input = Vec::with_capacity(16 * 16);
        for y in 0..16u8 {
            for x in 0..16u8 {
                let value = x.wrapping_mul(13).wrapping_add(y.wrapping_mul(7));
                input.push(Srgb::new(value, value, value));
            }
        }

        let raster = dither_remap(
            &input,
            16,
            16,
            &palette,
            DitherMode::FloydSteinberg,
            100,
            false,
            None,
        )
        .unwrap();
        let serpentine = dither_remap(
            &input,
            16,
            16,
            &palette,
            DitherMode::FloydSteinberg,
            100,
            true,
            None,
        )
        .unwrap();
        assert_ne!(raster, serpentine);
    }
}
