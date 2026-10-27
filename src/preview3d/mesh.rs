use std::f32::consts::PI;

/// Built-in shapes offered by the 3D preview.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BuiltinShape {
    Cube,
    Sphere,
    Plane,
}

impl BuiltinShape {
    pub(crate) const ALL: [Self; 3] = [Self::Cube, Self::Sphere, Self::Plane];

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Cube => "Куб",
            Self::Sphere => "Сфера",
            Self::Plane => "Плоскость",
        }
    }
}

/// A triangle mesh with per-vertex texture coordinates.
///
/// UVs use the same top-left origin as `image::RgbaImage` (V grows downwards). Importers must
/// convert from the source format's convention before storing coordinates here.
///
/// Normals are intentionally not stored: the rasterizer computes flat face normals, which is
/// robust for imported models that may have missing or inconsistent normal data.
#[derive(Clone, Debug, Default)]
pub(crate) struct Mesh3d {
    pub(crate) positions: Vec<[f32; 3]>,
    pub(crate) uvs: Vec<[f32; 2]>,
    pub(crate) indices: Vec<[u32; 3]>,
    /// True when at least part of the mesh had no UV data and a projection was generated.
    pub(crate) uvs_generated: bool,
}

impl Mesh3d {
    pub(crate) fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    pub(crate) fn is_valid(&self) -> bool {
        self.positions.len() == self.uvs.len()
            && self
                .indices
                .iter()
                .flatten()
                .all(|&i| (i as usize) < self.positions.len())
    }

    pub(crate) fn builtin(shape: BuiltinShape) -> Self {
        match shape {
            BuiltinShape::Cube => Self::cube(),
            BuiltinShape::Sphere => Self::sphere(64, 32),
            BuiltinShape::Plane => Self::plane(),
        }
    }

    pub(crate) fn cube() -> Self {
        let mut mesh = Self::default();
        // Each face is a quad: origin + u + v, with `u × v` pointing outwards.
        let faces = [
            // +X
            ([0.5, -0.5, -0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            // -X
            ([-0.5, -0.5, 0.5], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]),
            // +Y
            ([-0.5, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
            // -Y
            ([-0.5, -0.5, -0.5], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            // +Z
            ([-0.5, -0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            // -Z
            ([0.5, -0.5, -0.5], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ];
        for (origin, u, v) in faces {
            let base = mesh.positions.len() as u32;
            for (su, sv) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
                mesh.positions.push([
                    origin[0] + u[0] * su + v[0] * sv,
                    origin[1] + u[1] * su + v[1] * sv,
                    origin[2] + u[2] * su + v[2] * sv,
                ]);
                mesh.uvs.push([su, 1.0 - sv]);
            }
            mesh.indices.push([base, base + 1, base + 2]);
            mesh.indices.push([base, base + 2, base + 3]);
        }
        mesh
    }

    pub(crate) fn sphere(segments: usize, rings: usize) -> Self {
        let segments = segments.max(3);
        let rings = rings.max(2);
        let mut mesh = Self {
            positions: Vec::with_capacity((segments + 1) * (rings + 1)),
            uvs: Vec::with_capacity((segments + 1) * (rings + 1)),
            indices: Vec::with_capacity(segments * rings * 2),
            uvs_generated: false,
        };

        for i in 0..=rings {
            let v = i as f32 / rings as f32;
            let theta = -PI * 0.5 + v * PI;
            let y = theta.sin();
            let radius = theta.cos();
            for j in 0..=segments {
                let u = j as f32 / segments as f32;
                let phi = u * PI * 2.0;
                mesh.positions
                    .push([radius * phi.cos(), y, radius * phi.sin()]);
                mesh.uvs.push([u, 1.0 - v]);
            }
        }

        let row = segments + 1;
        for i in 0..rings {
            for j in 0..segments {
                let a = (i * row + j) as u32;
                let b = ((i + 1) * row + j) as u32;
                let c = ((i + 1) * row + j + 1) as u32;
                let d = (i * row + j + 1) as u32;
                mesh.indices.push([a, b, c]);
                mesh.indices.push([a, c, d]);
            }
        }
        mesh
    }

    pub(crate) fn plane() -> Self {
        Self {
            positions: vec![
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ],
            uvs: vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            indices: vec![[0, 1, 2], [0, 2, 3]],
            uvs_generated: false,
        }
    }

    /// Center the mesh at the origin and uniformly scale it to a unit-radius sphere.
    pub(crate) fn normalize(&mut self) {
        if self.positions.is_empty() {
            return;
        }
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for p in &self.positions {
            for axis in 0..3 {
                min[axis] = min[axis].min(p[axis]);
                max[axis] = max[axis].max(p[axis]);
            }
        }
        let center = [
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        ];
        let mut max_radius: f32 = 0.0;
        for p in &self.positions {
            let dx = p[0] - center[0];
            let dy = p[1] - center[1];
            let dz = p[2] - center[2];
            max_radius = max_radius.max((dx * dx + dy * dy + dz * dz).sqrt());
        }
        if !max_radius.is_finite() || max_radius <= 1e-6 {
            return;
        }
        let scale = 1.0 / max_radius;
        for p in &mut self.positions {
            p[0] = (p[0] - center[0]) * scale;
            p[1] = (p[1] - center[1]) * scale;
            p[2] = (p[2] - center[2]) * scale;
        }
    }

    /// Generate a fallback spherical projection for the given vertices.
    pub(crate) fn spherical_uvs(positions: &[[f32; 3]]) -> Vec<[f32; 2]> {
        let center = Self::bounds_center(positions);
        positions
            .iter()
            .map(|&position| Self::spherical_uv(position, center))
            .collect()
    }

    /// Center of the axis-aligned bounds, used as the origin for the fallback projection.
    pub(crate) fn bounds_center(positions: &[[f32; 3]]) -> [f32; 3] {
        if positions.is_empty() {
            return [0.0; 3];
        }
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for p in positions {
            for axis in 0..3 {
                min[axis] = min[axis].min(p[axis]);
                max[axis] = max[axis].max(p[axis]);
            }
        }
        [
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        ]
    }

    /// Spherical UV for one point, with `center` as the projection origin.
    pub(crate) fn spherical_uv(position: [f32; 3], center: [f32; 3]) -> [f32; 2] {
        let x = position[0] - center[0];
        let y = position[1] - center[1];
        let z = position[2] - center[2];
        let radius = (x * x + y * y + z * z).sqrt();
        if radius <= 1e-6 {
            return [0.5, 0.5];
        }
        let u = z.atan2(x) / (2.0 * PI) + 0.5;
        let v = 0.5 - (y / radius).asin() / PI;
        [u, v]
    }

    /// Fill in spherical UVs for models that do not ship texture coordinates.
    pub(crate) fn ensure_uvs(&mut self) {
        if self.uvs.len() == self.positions.len() {
            return;
        }
        self.uvs = Self::spherical_uvs(&self.positions);
        self.uvs_generated = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_meshes_are_valid() {
        for shape in BuiltinShape::ALL {
            let mesh = Mesh3d::builtin(shape);
            assert!(mesh.is_valid(), "{shape:?} produced invalid indices");
            assert!(!mesh.indices.is_empty());
            assert!(mesh.uvs.iter().flatten().all(|&v| (0.0..=1.0).contains(&v)));
        }
        assert_eq!(Mesh3d::cube().triangle_count(), 12);
    }

    #[test]
    fn normalize_fits_mesh_into_unit_radius() {
        let mut mesh = Mesh3d::cube();
        for p in &mut mesh.positions {
            p[0] = p[0] * 10.0 + 4.0;
            p[1] = p[1] * 10.0 - 2.0;
            p[2] = p[2] * 10.0 + 7.0;
        }
        mesh.normalize();
        let max_radius = mesh
            .positions
            .iter()
            .map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt())
            .fold(0.0_f32, f32::max);
        assert!((max_radius - 1.0).abs() < 1e-5);
        assert!(mesh.is_valid());
    }

    #[test]
    fn ensure_uvs_fills_missing_coordinates() {
        let mut mesh = Mesh3d::plane();
        mesh.uvs.clear();
        mesh.ensure_uvs();
        assert_eq!(mesh.uvs.len(), mesh.positions.len());
        assert!(mesh.uvs.iter().flatten().all(|&v| (0.0..=1.0).contains(&v)));
    }
}
