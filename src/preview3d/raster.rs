use super::mesh::Mesh3d;
use image::RgbaImage;

/// Texture filtering used by the software rasterizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SampleFilter {
    Nearest,
    Linear,
}

/// Camera and material options for one 3D preview frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RenderOptions {
    pub(crate) yaw: f32,
    pub(crate) pitch: f32,
    pub(crate) zoom: f32,
    pub(crate) tiling: f32,
    pub(crate) lighting: bool,
    pub(crate) filter: SampleFilter,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            yaw: 0.55,
            pitch: -0.3,
            zoom: 1.0,
            tiling: 1.0,
            lighting: true,
            filter: SampleFilter::Nearest,
        }
    }
}

/// Alpha values below this are treated as cut-out transparent pixels.
const ALPHA_CUTOFF: u8 = 8;

/// Render `mesh` with `texture` into an RGBA image with a transparent background.
///
/// The rasterizer uses an orthographic camera looking down -Z with a Z-buffer, which is enough
/// for a texture preview and keeps the implementation portable and testable. Orthographic
/// projection also makes texture interpolation affine, so no perspective correction is needed.
pub(crate) fn render(
    mesh: &Mesh3d,
    texture: &RgbaImage,
    width: usize,
    height: usize,
    options: RenderOptions,
) -> RgbaImage {
    let width = width.max(1);
    let height = height.max(1);
    let mut pixels = vec![0u8; width * height * 4];
    let mut depth = vec![f32::NEG_INFINITY; width * height];

    let (sin_yaw, cos_yaw) = options.yaw.sin_cos();
    let (sin_pitch, cos_pitch) = options.pitch.sin_cos();
    let scale = width.min(height) as f32 * 0.46 * options.zoom.clamp(0.1, 8.0);
    let cx = width as f32 * 0.5;
    let cy = height as f32 * 0.5;
    let light = normalize([-0.35, -0.55, 0.76]);
    let tiling = options.tiling.max(1.0);

    for triangle in &mesh.indices {
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
                let color = sample_texture(texture, u * tiling, v * tiling, options.filter);
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

fn sample_texture(texture: &RgbaImage, u: f32, v: f32, filter: SampleFilter) -> [u8; 4] {
    match filter {
        SampleFilter::Nearest => sample_nearest(texture, u, v),
        SampleFilter::Linear => sample_linear(texture, u, v),
    }
}

fn wrap(value: f32) -> f32 {
    value - value.floor()
}

fn wrap_index(value: i64, size: u32) -> u32 {
    value.rem_euclid(size as i64) as u32
}

fn sample_nearest(texture: &RgbaImage, u: f32, v: f32) -> [u8; 4] {
    let (width, height) = (texture.width(), texture.height());
    let x = wrap_index((wrap(u) * width as f32).floor() as i64, width);
    let y = wrap_index((wrap(v) * height as f32).floor() as i64, height);
    texture.get_pixel(x, y).0
}

fn sample_linear(texture: &RgbaImage, u: f32, v: f32) -> [u8; 4] {
    let (width, height) = (texture.width(), texture.height());
    let x = wrap(u) * width as f32 - 0.5;
    let y = wrap(v) * height as f32 - 0.5;
    let x0 = x.floor();
    let y0 = y.floor();
    let tx = x - x0;
    let ty = y - y0;
    let x0 = wrap_index(x0 as i64, width);
    let y0 = wrap_index(y0 as i64, height);
    let x1 = wrap_index(x0 as i64 + 1, width);
    let y1 = wrap_index(y0 as i64 + 1, height);

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
