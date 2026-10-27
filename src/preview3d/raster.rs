use super::mesh::Mesh3d;
use image::RgbaImage;
use std::sync::atomic::{AtomicBool, Ordering};

/// Texture filtering used by the software rasterizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SampleFilter {
    Nearest,
    Linear,
}

/// How texture coordinates outside `0..=1` are resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WrapMode {
    Repeat,
    Clamp,
}

/// Repetition and placement of the texture on the model's UV coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct UvTransform {
    pub(crate) tiling: [f32; 2],
    pub(crate) offset: [f32; 2],
    pub(crate) rotation: f32,
}

impl Default for UvTransform {
    fn default() -> Self {
        Self {
            tiling: [1.0, 1.0],
            offset: [0.0, 0.0],
            rotation: 0.0,
        }
    }
}

impl UvTransform {
    /// Apply scale, rotation and offset in the same order as the glTF texture transform.
    pub(crate) fn apply(self, uv: [f32; 2]) -> [f32; 2] {
        let (sin, cos) = self.rotation.sin_cos();
        let x = uv[0] * self.tiling[0];
        let y = uv[1] * self.tiling[1];
        [
            cos * x + sin * y + self.offset[0],
            -sin * x + cos * y + self.offset[1],
        ]
    }
}

/// Camera and material options for one 3D preview frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RenderOptions {
    pub(crate) yaw: f32,
    pub(crate) pitch: f32,
    /// Viewport-relative camera offset, where `[0.5, 0.5]` is the center.
    pub(crate) pan: [f32; 2],
    pub(crate) zoom: f32,
    pub(crate) uvs: UvTransform,
    pub(crate) wrap: WrapMode,
    pub(crate) lighting: bool,
    /// Light direction in view space; it is normalized before use.
    pub(crate) light: [f32; 3],
    pub(crate) filter: SampleFilter,
    /// When set, sample the secondary texture on the right side of this screen fraction.
    pub(crate) split: Option<f32>,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            yaw: 0.55,
            pitch: -0.3,
            pan: [0.0, 0.0],
            zoom: 1.0,
            uvs: UvTransform::default(),
            wrap: WrapMode::Repeat,
            lighting: true,
            light: light_direction(-0.6, 0.9),
            filter: SampleFilter::Nearest,
            split: None,
        }
    }
}

/// Convert azimuth/elevation angles (radians) into a normalized light direction.
pub(crate) fn light_direction(azimuth: f32, elevation: f32) -> [f32; 3] {
    let (sin_az, cos_az) = azimuth.sin_cos();
    let (sin_el, cos_el) = elevation.sin_cos();
    normalize([cos_el * sin_az, sin_el, cos_el * cos_az])
}

/// Alpha values below this are treated as cut-out transparent pixels.
const ALPHA_CUTOFF: u8 = 8;

/// Reusable depth buffer for the software rasterizer.
#[derive(Default)]
pub(crate) struct RasterScratch {
    depth: Vec<f32>,
}

impl RasterScratch {
    /// Render `mesh` using the cached depth buffer. Returns `None` when cancelled.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn render(
        &mut self,
        mesh: &Mesh3d,
        texture: &RgbaImage,
        secondary: Option<&RgbaImage>,
        width: usize,
        height: usize,
        options: RenderOptions,
        cancel: &AtomicBool,
    ) -> Option<RgbaImage> {
        render_internal(
            mesh,
            texture,
            secondary,
            width,
            height,
            options,
            &mut self.depth,
            cancel,
        )
    }
}

/// Render `mesh` with `texture` into an RGBA image with a transparent background.
///
/// The rasterizer uses an orthographic camera looking down -Z with a Z-buffer, which is enough
/// for a texture preview and keeps the implementation portable and testable. Orthographic
/// projection also makes texture interpolation affine, so no perspective correction is needed.
#[cfg(test)]
pub(crate) fn render(
    mesh: &Mesh3d,
    texture: &RgbaImage,
    width: usize,
    height: usize,
    options: RenderOptions,
) -> RgbaImage {
    let mut scratch = RasterScratch::default();
    let cancel = AtomicBool::new(false);
    scratch
        .render(mesh, texture, None, width, height, options, &cancel)
        .expect("an uncancelled render always produces an image")
}

#[allow(clippy::too_many_arguments)]
fn render_internal(
    mesh: &Mesh3d,
    texture: &RgbaImage,
    secondary: Option<&RgbaImage>,
    width: usize,
    height: usize,
    options: RenderOptions,
    depth: &mut Vec<f32>,
    cancel: &AtomicBool,
) -> Option<RgbaImage> {
    let width = width.max(1);
    let height = height.max(1);
    let mut pixels = vec![0u8; width * height * 4];
    depth.clear();
    depth.resize(width * height, f32::NEG_INFINITY);

    let (sin_yaw, cos_yaw) = options.yaw.sin_cos();
    let (sin_pitch, cos_pitch) = options.pitch.sin_cos();
    let scale = width.min(height) as f32 * 0.46 * options.zoom.clamp(0.1, 8.0);
    let cx = width as f32 * (0.5 + options.pan[0]);
    let cy = height as f32 * (0.5 + options.pan[1]);
    let light = normalize(options.light);
    let split_x = options
        .split
        .map(|split| split.clamp(0.0, 1.0) * width as f32);

    for (triangle_index, triangle) in mesh.indices.iter().enumerate() {
        if triangle_index.is_multiple_of(64) && cancel.load(Ordering::Relaxed) {
            return None;
        }
        let (a, b, c) = (
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        );
        let (Some(&pa), Some(&pb), Some(&pc)) = (
            mesh.positions.get(a),
            mesh.positions.get(b),
            mesh.positions.get(c),
        ) else {
            continue;
        };
        let pa = rotate(pa, sin_yaw, cos_yaw, sin_pitch, cos_pitch);
        let pb = rotate(pb, sin_yaw, cos_yaw, sin_pitch, cos_pitch);
        let pc = rotate(pc, sin_yaw, cos_yaw, sin_pitch, cos_pitch);

        let face_normal = cross(sub(pb, pa), sub(pc, pa));
        let length = dot(face_normal, face_normal).sqrt();
        if length <= 1e-9 {
            continue;
        }
        let mut normal = [
            face_normal[0] / length,
            face_normal[1] / length,
            face_normal[2] / length,
        ];
        if normal[2] < 0.0 {
            normal = [-normal[0], -normal[1], -normal[2]];
        }
        let shade = if options.lighting {
            let lambert = dot(normal, light).max(0.0);
            0.28 + 0.72 * lambert
        } else {
            1.0
        };

        let sa = [cx + pa[0] * scale, cy - pa[1] * scale];
        let sb = [cx + pb[0] * scale, cy - pb[1] * scale];
        let sc = [cx + pc[0] * scale, cy - pc[1] * scale];
        let area = edge(sa, sb, sc);
        if area.abs() <= 1e-6 {
            continue;
        }

        let (uv_a, uv_b, uv_c) = (
            mesh.uvs.get(a).copied().unwrap_or([0.0, 0.0]),
            mesh.uvs.get(b).copied().unwrap_or([0.0, 0.0]),
            mesh.uvs.get(c).copied().unwrap_or([0.0, 0.0]),
        );

        let last_x = width as f32 - 1.0;
        let last_y = height as f32 - 1.0;
        let min_x = sa[0].min(sb[0]).min(sc[0]).floor().max(0.0).min(last_x) as usize;
        let max_x = sa[0].max(sb[0]).max(sc[0]).ceil().max(0.0).min(last_x) as usize;
        let min_y = sa[1].min(sb[1]).min(sc[1]).floor().max(0.0).min(last_y) as usize;
        let max_y = sa[1].max(sb[1]).max(sc[1]).ceil().max(0.0).min(last_y) as usize;
        if min_x > max_x || min_y > max_y {
            continue;
        }

        for y in min_y..=max_y {
            if y.is_multiple_of(64) && cancel.load(Ordering::Relaxed) {
                return None;
            }
            for x in min_x..=max_x {
                let p = [x as f32 + 0.5, y as f32 + 0.5];
                let w0 = edge(sb, sc, p) / area;
                let w1 = edge(sc, sa, p) / area;
                let w2 = edge(sa, sb, p) / area;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }

                let depth_value = w0 * pa[2] + w1 * pb[2] + w2 * pc[2];
                let index = y * width + x;
                if depth_value <= depth[index] {
                    continue;
                }

                let u = w0 * uv_a[0] + w1 * uv_b[0] + w2 * uv_c[0];
                let v = w0 * uv_a[1] + w1 * uv_b[1] + w2 * uv_c[1];
                let uv = options.uvs.apply([u, v]);
                let source = match (split_x, secondary) {
                    (Some(split), Some(other)) if p[0] >= split => other,
                    _ => texture,
                };
                let color = sample_texture(source, uv[0], uv[1], options.wrap, options.filter);
                if color[3] < ALPHA_CUTOFF {
                    continue;
                }

                depth[index] = depth_value;
                let offset = index * 4;
                pixels[offset] = mul_channel(color[0], shade);
                pixels[offset + 1] = mul_channel(color[1], shade);
                pixels[offset + 2] = mul_channel(color[2], shade);
                pixels[offset + 3] = 255;
            }
        }
    }

    RgbaImage::from_raw(width as u32, height as u32, pixels)
        .expect("raster buffer dimensions are consistent")
        .into()
}

fn rotate(p: [f32; 3], sin_yaw: f32, cos_yaw: f32, sin_pitch: f32, cos_pitch: f32) -> [f32; 3] {
    let x = p[0] * cos_yaw + p[2] * sin_yaw;
    let z = -p[0] * sin_yaw + p[2] * cos_yaw;
    let y = p[1] * cos_pitch - z * sin_pitch;
    let z = p[1] * sin_pitch + z * cos_pitch;
    [x, y, z]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let length = dot(v, v).sqrt();
    if length <= 1e-9 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / length, v[1] / length, v[2] / length]
    }
}

fn edge(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])
}

fn mul_channel(value: u8, shade: f32) -> u8 {
    (value as f32 * shade).clamp(0.0, 255.0) as u8
}

fn sample_texture(
    texture: &RgbaImage,
    u: f32,
    v: f32,
    wrap: WrapMode,
    filter: SampleFilter,
) -> [u8; 4] {
    match filter {
        SampleFilter::Nearest => sample_nearest(texture, u, v, wrap),
        SampleFilter::Linear => sample_linear(texture, u, v, wrap),
    }
}

fn wrap_coord(value: f32, wrap: WrapMode) -> f32 {
    match wrap {
        WrapMode::Repeat => value - value.floor(),
        WrapMode::Clamp => value.clamp(0.0, 1.0),
    }
}

fn wrap_index(value: i64, size: u32, wrap: WrapMode) -> u32 {
    match wrap {
        WrapMode::Repeat => value.rem_euclid(size as i64) as u32,
        WrapMode::Clamp => value.clamp(0, size as i64 - 1) as u32,
    }
}

fn sample_nearest(texture: &RgbaImage, u: f32, v: f32, wrap: WrapMode) -> [u8; 4] {
    let (width, height) = (texture.width(), texture.height());
    let x = wrap_index(
        (wrap_coord(u, wrap) * width as f32).floor() as i64,
        width,
        wrap,
    );
    let y = wrap_index(
        (wrap_coord(v, wrap) * height as f32).floor() as i64,
        height,
        wrap,
    );
    texture.get_pixel(x, y).0
}

fn sample_linear(texture: &RgbaImage, u: f32, v: f32, wrap: WrapMode) -> [u8; 4] {
    let (width, height) = (texture.width(), texture.height());
    let x = wrap_coord(u, wrap) * width as f32 - 0.5;
    let y = wrap_coord(v, wrap) * height as f32 - 0.5;
    let x0 = x.floor();
    let y0 = y.floor();
    let tx = x - x0;
    let ty = y - y0;
    let x0 = wrap_index(x0 as i64, width, wrap);
    let y0 = wrap_index(y0 as i64, height, wrap);
    let x1 = wrap_index(x0 as i64 + 1, width, wrap);
    let y1 = wrap_index(y0 as i64 + 1, height, wrap);

    let p00 = texture.get_pixel(x0, y0).0;
    let p10 = texture.get_pixel(x1, y0).0;
    let p01 = texture.get_pixel(x0, y1).0;
    let p11 = texture.get_pixel(x1, y1).0;
    let mut out = [0u8; 4];
    for i in 0..4 {
        let top = p00[i] as f32 * (1.0 - tx) + p10[i] as f32 * tx;
        let bottom = p01[i] as f32 * (1.0 - tx) + p11[i] as f32 * tx;
        out[i] = (top * (1.0 - ty) + bottom * ty).round() as u8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    fn solid_texture(color: Rgba<u8>) -> RgbaImage {
        RgbaImage::from_pixel(2, 2, color)
    }

    #[test]
    fn nearest_sampling_uses_top_left_image_origin() {
        let mut texture = RgbaImage::new(2, 2);
        texture.put_pixel(0, 0, Rgba([255, 0, 0, 255]));
        texture.put_pixel(1, 0, Rgba([0, 255, 0, 255]));
        texture.put_pixel(0, 1, Rgba([0, 0, 255, 255]));
        texture.put_pixel(1, 1, Rgba([255, 255, 255, 255]));

        assert_eq!(
            sample_nearest(&texture, 0.25, 0.25, WrapMode::Repeat),
            [255, 0, 0, 255]
        );
        assert_eq!(
            sample_nearest(&texture, 0.75, 0.75, WrapMode::Repeat),
            [255, 255, 255, 255]
        );
    }

    #[test]
    fn clamp_wrap_samples_edge_pixels() {
        let mut texture = RgbaImage::new(2, 1);
        texture.put_pixel(0, 0, Rgba([255, 0, 0, 255]));
        texture.put_pixel(1, 0, Rgba([0, 255, 0, 255]));

        assert_eq!(
            sample_nearest(&texture, -1.0, 0.5, WrapMode::Clamp),
            [255, 0, 0, 255]
        );
        assert_eq!(
            sample_nearest(&texture, 2.0, 0.5, WrapMode::Clamp),
            [0, 255, 0, 255]
        );
    }

    #[test]
    fn uv_transform_applies_scale_rotation_and_offset() {
        let transform = UvTransform {
            tiling: [2.0, 3.0],
            offset: [0.1, 0.2],
            rotation: std::f32::consts::FRAC_PI_2,
        };
        let uv = transform.apply([1.0, 0.0]);
        assert!((uv[0] - 0.1).abs() < 1e-5);
        assert!((uv[1] + 1.8).abs() < 1e-5);
    }

    #[test]
    fn renders_plane_and_keeps_background_transparent() {
        let texture = solid_texture(Rgba([220, 40, 30, 255]));
        let image = render(
            &Mesh3d::plane(),
            &texture,
            64,
            64,
            RenderOptions {
                yaw: 0.0,
                pitch: 0.0,
                lighting: false,
                ..Default::default()
            },
        );

        let center = image.get_pixel(32, 32).0;
        assert_eq!(&center[..3], &[220, 40, 30]);
        assert_eq!(center[3], 255);
        assert_eq!(image.get_pixel(0, 0).0[3], 0);
    }

    #[test]
    fn pan_moves_the_rendered_geometry() {
        let texture = solid_texture(Rgba([220, 40, 30, 255]));
        let image = render(
            &Mesh3d::plane(),
            &texture,
            64,
            64,
            RenderOptions {
                yaw: 0.0,
                pitch: 0.0,
                lighting: false,
                pan: [0.5, 0.0],
                ..Default::default()
            },
        );

        assert_eq!(image.get_pixel(63, 32).0[3], 255);
        assert_eq!(image.get_pixel(0, 32).0[3], 0);
    }

    #[test]
    fn split_renders_secondary_texture_on_the_right() {
        let left = solid_texture(Rgba([255, 0, 0, 255]));
        let right = solid_texture(Rgba([0, 255, 0, 255]));
        let mut scratch = RasterScratch::default();
        let cancel = AtomicBool::new(false);
        let image = scratch
            .render(
                &Mesh3d::plane(),
                &left,
                Some(&right),
                64,
                64,
                RenderOptions {
                    yaw: 0.0,
                    pitch: 0.0,
                    lighting: false,
                    split: Some(0.5),
                    ..Default::default()
                },
                &cancel,
            )
            .unwrap();

        assert_eq!(&image.get_pixel(24, 32).0[..3], &[255, 0, 0]);
        assert_eq!(&image.get_pixel(40, 32).0[..3], &[0, 255, 0]);
    }

    #[test]
    fn cancelled_render_is_aborted() {
        let cancel = AtomicBool::new(true);
        let mut scratch = RasterScratch::default();
        assert!(scratch
            .render(
                &Mesh3d::plane(),
                &solid_texture(Rgba([255, 255, 255, 255])),
                None,
                64,
                64,
                RenderOptions::default(),
                &cancel,
            )
            .is_none());
    }

    #[test]
    fn z_buffer_keeps_nearer_triangle() {
        let mut texture = RgbaImage::new(2, 1);
        texture.put_pixel(0, 0, Rgba([255, 0, 0, 255]));
        texture.put_pixel(1, 0, Rgba([0, 255, 0, 255]));

        let mesh = Mesh3d {
            positions: vec![
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
            ],
            uvs: vec![
                [0.0, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
                [0.0, 0.5],
                [0.5, 0.5],
                [1.0, 0.5],
                [1.0, 0.5],
                [0.5, 0.5],
            ],
            indices: vec![[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
            uvs_generated: false,
        };

        let image = render(
            &mesh,
            &texture,
            64,
            64,
            RenderOptions {
                yaw: 0.0,
                pitch: 0.0,
                lighting: false,
                ..Default::default()
            },
        );
        let center = image.get_pixel(32, 32).0;
        assert_eq!(&center[..3], &[255, 0, 0]);
    }

    #[test]
    fn fully_offscreen_triangle_is_ignored() {
        let mut mesh = Mesh3d::plane();
        for p in &mut mesh.positions {
            p[0] += 100.0;
        }
        let image = render(
            &mesh,
            &solid_texture(Rgba([255, 255, 255, 255])),
            64,
            64,
            RenderOptions::default(),
        );
        assert!(image.pixels().all(|p| p.0[3] == 0));
    }

    #[test]
    fn alpha_cutout_leaves_pixels_transparent() {
        let texture = solid_texture(Rgba([10, 20, 30, 0]));
        let image = render(
            &Mesh3d::plane(),
            &texture,
            32,
            32,
            RenderOptions {
                yaw: 0.0,
                pitch: 0.0,
                ..Default::default()
            },
        );
        assert_eq!(image.get_pixel(16, 16).0[3], 0);
    }
}
