use super::mesh::Mesh3d;
use std::path::Path;

/// Load a triangle mesh from an OBJ, GLB or glTF file.
pub(crate) fn load_model(path: &Path) -> Result<Mesh3d, String> {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let mut mesh = match extension.as_str() {
        "obj" => load_obj(path)?,
        "glb" | "gltf" => load_gltf(path)?,
        "fbx" => load_fbx(path)?,
        _ => {
            return Err(format!(
                "Неподдерживаемый формат модели «{extension}». Используйте OBJ, GLB, glTF или FBX."
            ))
        }
    };
    if mesh.indices.is_empty() {
        return Err("В модели не найдено треугольников.".into());
    }
    mesh.normalize();
    if mesh.uvs.len() != mesh.positions.len() {
        mesh.uvs.clear();
        mesh.ensure_uvs();
    }
    if !mesh.is_valid() {
        return Err("Модель содержит некорректные индексы вершин.".into());
    }
    Ok(mesh)
}

fn load_obj(path: &Path) -> Result<Mesh3d, String> {
    let options = tobj::LoadOptions {
        triangulate: true,
        single_index: true,
        ignore_points: true,
        ignore_lines: true,
    };
    let (models, _materials) =
        tobj::load_obj(path, &options).map_err(|e| format!("Не удалось прочитать OBJ: {e}"))?;

    let mut mesh = Mesh3d::default();
    for model in models {
        let source = model.mesh;
        let base = mesh.positions.len() as u32;
        let vertex_count = source.positions.len() / 3;
        let positions: Vec<[f32; 3]> = source
            .positions
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| [p[0], p[1], p[2]])
            .collect();
        mesh.positions.extend_from_slice(&positions);

        if source.texcoords.len() == vertex_count * 2 {
            // OBJ texture coordinates have a bottom-left origin, while the preview samples
            // images from the top-left, so flip V on import.
            mesh.uvs.extend(
                source
                    .texcoords
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|uv| [uv[0], 1.0 - uv[1]]),
            );
        } else {
            mesh.uvs.extend(Mesh3d::spherical_uvs(&positions));
            mesh.uvs_generated = true;
        }

        mesh.indices.extend(
            source
                .indices
                .as_chunks::<3>()
                .0
                .iter()
                .map(|i| [base + i[0], base + i[1], base + i[2]]),
        );
    }

    Ok(mesh)
}

fn load_gltf(path: &Path) -> Result<Mesh3d, String> {
    let gltf = if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("glb"))
    {
        let file = std::fs::File::open(path).map_err(|e| format!("Не удалось открыть GLB: {e}"))?;
        let glb =
            gltf::Glb::from_reader(file).map_err(|e| format!("Не удалось разобрать GLB: {e}"))?;
        let mut gltf = gltf::Gltf::from_slice(&glb.json)
            .map_err(|e| format!("Не удалось разобрать glTF внутри GLB: {e}"))?;
        gltf.blob = glb.bin.map(|bin| bin.into_owned());
        gltf
    } else {
        gltf::Gltf::open(path).map_err(|e| format!("Не удалось прочитать glTF: {e}"))?
    };

    let base_dir = path.parent();
    let buffers = gltf::import_buffers(&gltf.document, base_dir, gltf.blob)
        .map_err(|e| format!("Не удалось прочитать буферы glTF: {e}"))?;

    let mut mesh = Mesh3d::default();
    let identity = identity_matrix();

    if let Some(scene) = gltf
        .document
        .default_scene()
        .or_else(|| gltf.document.scenes().next())
    {
        for node in scene.nodes() {
            walk_node(&node, identity, &buffers, &mut mesh);
        }
    } else {
        for node in gltf.document.nodes() {
            walk_node(&node, identity, &buffers, &mut mesh);
        }
    }

    Ok(mesh)
}

fn walk_node(
    node: &gltf::Node<'_>,
    parent: [[f32; 4]; 4],
    buffers: &[gltf::buffer::Data],
    mesh: &mut Mesh3d,
) {
    let world = multiply(parent, node.transform().matrix());
    if let Some(source) = node.mesh() {
        for primitive in source.primitives() {
            append_primitive(&primitive, world, buffers, mesh);
        }
    }
    for child in node.children() {
        walk_node(&child, world, buffers, mesh);
    }
}

/// Material UV settings that affect how the previewed texture should be sampled.
#[derive(Clone, Copy, Debug)]
struct UvLayout {
    tex_coord: u32,
    offset: [f32; 2],
    rotation: f32,
    scale: [f32; 2],
}

impl Default for UvLayout {
    fn default() -> Self {
        Self {
            tex_coord: 0,
            offset: [0.0, 0.0],
            rotation: 0.0,
            scale: [1.0, 1.0],
        }
    }
}

impl UvLayout {
    fn apply(self, uv: [f32; 2]) -> [f32; 2] {
        let (sin, cos) = self.rotation.sin_cos();
        let x = uv[0] * self.scale[0];
        let y = uv[1] * self.scale[1];
        [
            cos * x + sin * y + self.offset[0],
            -sin * x + cos * y + self.offset[1],
        ]
    }
}

fn primitive_uv_layout(primitive: &gltf::Primitive<'_>) -> UvLayout {
    let info = primitive
        .material()
        .pbr_metallic_roughness()
        .base_color_texture();
    let Some(info) = info else {
        return UvLayout::default();
    };

    let transform = info.texture_transform();
    let tex_coord = transform
        .as_ref()
        .and_then(|transform| transform.tex_coord())
        .unwrap_or_else(|| info.tex_coord());
    let (offset, rotation, scale) = match transform {
        Some(transform) => (transform.offset(), transform.rotation(), transform.scale()),
        None => ([0.0, 0.0], 0.0, [1.0, 1.0]),
    };
    UvLayout {
        tex_coord,
        offset,
        rotation,
        scale,
    }
}

fn append_primitive(
    primitive: &gltf::Primitive<'_>,
    transform: [[f32; 4]; 4],
    buffers: &[gltf::buffer::Data],
    mesh: &mut Mesh3d,
) {
    let uv_layout = primitive_uv_layout(primitive);
    let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(|data| &data.0[..]));
    let Some(positions) = reader.read_positions() else {
        return;
    };
    let positions: Vec<[f32; 3]> = positions.map(|p| transform_point(transform, p)).collect();
    if positions.is_empty() {
        return;
    }

    let uvs: Option<Vec<[f32; 2]>> = reader
        .read_tex_coords(uv_layout.tex_coord)
        .or_else(|| {
            (uv_layout.tex_coord != 0)
                .then(|| reader.read_tex_coords(0))
                .flatten()
        })
        .map(|tex_coords| {
            tex_coords
                .into_f32()
                .map(|uv| uv_layout.apply(uv))
                .collect()
        });

    let base = mesh.positions.len() as u32;
    let source_indices: Vec<u32> = match reader.read_indices() {
        Some(indices) => indices.into_u32().collect(),
        None => (0..positions.len() as u32).collect(),
    };

    let triangles = primitive_triangles(primitive.mode(), &source_indices);
    if triangles.is_empty() {
        return;
    }

    let vertex_count = positions.len();
    mesh.positions.extend(positions);
    match uvs {
        Some(uvs) if uvs.len() == vertex_count => mesh.uvs.extend(uvs),
        _ => {
            mesh.uvs
                .extend(Mesh3d::spherical_uvs(&mesh.positions[base as usize..]));
            mesh.uvs_generated = true;
        }
    }

    mesh.indices.extend(
        triangles
            .into_iter()
            .map(|[a, b, c]| [base + a, base + b, base + c]),
    );
}

/// Convert glTF primitive indices into a triangle list.
fn primitive_triangles(mode: gltf::mesh::Mode, indices: &[u32]) -> Vec<[u32; 3]> {
    match mode {
        gltf::mesh::Mode::Triangles => indices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|tri| [tri[0], tri[1], tri[2]])
            .collect(),
        gltf::mesh::Mode::TriangleStrip => (0..indices.len().saturating_sub(2))
            .map(|i| {
                if i % 2 == 0 {
                    [indices[i], indices[i + 1], indices[i + 2]]
                } else {
                    [indices[i + 1], indices[i], indices[i + 2]]
                }
            })
            .collect(),
        gltf::mesh::Mode::TriangleFan => (0..indices.len().saturating_sub(2))
            .map(|i| [indices[0], indices[i + 1], indices[i + 2]])
            .collect(),
        _ => Vec::new(),
    }
}

fn load_fbx(path: &Path) -> Result<Mesh3d, String> {
    let data = std::fs::read(path).map_err(|e| format!("Не удалось открыть FBX: {e}"))?;
    let options = ufbx::LoadOpts {
        ignore_animation: true,
        ignore_embedded: true,
        target_axes: ufbx::CoordinateAxes::right_handed_y_up(),
        ..Default::default()
    };
    let scene = ufbx::load_memory(&data, options)
        .map_err(|e| format!("Не удалось разобрать FBX: {e:?}"))?;

    let mut mesh = Mesh3d::default();
    for node in &scene.nodes {
        if !node.visible || node.is_geometry_transform_helper || node.is_scale_helper {
            continue;
        }
        let Some(source) = node.mesh.as_ref() else {
            continue;
        };
        append_fbx_mesh(node, source, &mut mesh);
    }

    Ok(mesh)
}

fn append_fbx_mesh(node: &ufbx::Node, source: &ufbx::Mesh, mesh: &mut Mesh3d) {
    let first_vertex = mesh.positions.len();
    let mut corner_map = vec![None; source.num_indices];
    let mut triangle_indices = Vec::new();
    let mut missing_uv = Vec::new();

    for face in source.faces.iter().copied() {
        if face.num_indices < 3 {
            continue;
        }
        triangle_indices.resize((face.num_indices as usize - 2) * 3, 0);
        let triangle_count = source.triangulate_face(&mut triangle_indices, face);
        triangle_indices.truncate(triangle_count as usize * 3);

        for &triangle in triangle_indices.as_chunks::<3>().0 {
            let [Some(a), Some(b), Some(c)] = triangle.map(|corner| {
                fbx_corner_vertex(
                    &node.geometry_to_world,
                    source,
                    corner,
                    mesh,
                    &mut missing_uv,
                    &mut corner_map,
                )
            }) else {
                continue;
            };
            mesh.indices.push([a, b, c]);
        }
    }

    if !missing_uv.is_empty() {
        let center = Mesh3d::bounds_center(&mesh.positions[first_vertex..]);
        for index in missing_uv {
            let position = mesh.positions[index as usize];
            mesh.uvs[index as usize] = Mesh3d::spherical_uv(position, center);
        }
        mesh.uvs_generated = true;
    }
}

fn fbx_corner_vertex(
    transform: &ufbx::Matrix,
    source: &ufbx::Mesh,
    corner: u32,
    mesh: &mut Mesh3d,
    missing_uv: &mut Vec<u32>,
    corner_map: &mut [Option<u32>],
) -> Option<u32> {
    let corner = corner as usize;
    if let Some(Some(index)) = corner_map.get(corner) {
        return Some(*index);
    }

    let position = fbx_position(source, corner)?;
    let uv = fbx_uv(source, corner);
    let position = ufbx::transform_position(transform, position);
    let index = mesh.positions.len() as u32;
    mesh.positions
        .push([position.x as f32, position.y as f32, position.z as f32]);
    match uv {
        // FBX UVs have a bottom-left origin; convert them to the preview's top-left space.
        Some(uv) => mesh.uvs.push([uv.x as f32, 1.0 - uv.y as f32]),
        None => {
            missing_uv.push(index);
            mesh.uvs.push([0.0, 0.0]);
        }
    }
    if let Some(slot) = corner_map.get_mut(corner) {
        *slot = Some(index);
    }
    Some(index)
}

fn fbx_position(source: &ufbx::Mesh, corner: usize) -> Option<ufbx::Vec3> {
    if source.vertex_position.exists {
        let index = *source.vertex_position.indices.get(corner)? as usize;
        source.vertex_position.values.get(index).copied()
    } else {
        let vertex = *source.vertex_indices.get(corner)? as usize;
        source.vertices.get(vertex).copied()
    }
}

fn fbx_uv(source: &ufbx::Mesh, corner: usize) -> Option<ufbx::Vec2> {
    if !source.vertex_uv.exists {
        return None;
    }
    let index = *source.vertex_uv.indices.get(corner)? as usize;
    source.vertex_uv.values.get(index).copied()
}

fn identity_matrix() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn multiply(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for (col, out_col) in out.iter_mut().enumerate() {
        for (row, value) in out_col.iter_mut().enumerate() {
            *value = (0..4).map(|k| a[k][row] * b[col][k]).sum();
        }
    }
    out
}

fn transform_point(m: [[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_dir;

    #[test]
    fn loads_obj_quad() {
        let dir = temp_dir("model_obj");
        let path = dir.join("quad.obj");
        std::fs::write(
            &path,
            "\
v -1 -1 0
v 1 -1 0
v 1 1 0
v -1 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
f 1/1 2/2 3/3
f 1/1 3/3 4/4
",
        )
        .unwrap();

        let mesh = load_model(&path).expect("obj should load");
        assert_eq!(mesh.triangle_count(), 2);
        assert!(mesh.is_valid());
        assert_eq!(mesh.uvs.len(), mesh.positions.len());
        assert!(!mesh.uvs_generated);
        assert_eq!(
            mesh.uvs,
            vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
        );
        let radius = mesh
            .positions
            .iter()
            .map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt())
            .fold(0.0_f32, f32::max);
        assert!((radius - 1.0).abs() < 1e-4);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn obj_keeps_uvs_from_textured_parts_and_fills_missing_ones() {
        let dir = temp_dir("model_obj_mixed_uv");
        let path = dir.join("mixed.obj");
        std::fs::write(
            &path,
            "\
v -1 -1 0
v 1 -1 0
v 1 1 0
vt 0 0
vt 1 0
vt 1 1
o Textured
f 1/1 2/2 3/3
o Untextured
v 3 -1 0
v 5 -1 0
v 5 1 0
f 4 5 6
",
        )
        .unwrap();

        let mesh = load_model(&path).expect("mixed obj should load");
        assert_eq!(mesh.positions.len(), 6);
        assert_eq!(mesh.uvs.len(), mesh.positions.len());
        assert!(mesh.uvs_generated);
        assert_eq!(&mesh.uvs[..3], &[[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]);
        assert!(mesh.uvs[3..].iter().any(|uv| *uv != [0.0, 0.0]));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_unknown_extension() {
        let error = load_model(Path::new("model.stl")).unwrap_err();
        assert!(error.contains("OBJ"));
    }

    #[test]
    fn loads_glb_triangle_and_generates_uvs() {
        let dir = temp_dir("model_glb");
        let path = dir.join("triangle.glb");
        let positions: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mut bin = Vec::new();
        for position in positions {
            for component in position {
                bin.extend_from_slice(&component.to_le_bytes());
            }
        }
        let json = r#"{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],"buffers":[{"byteLength":36}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}]}"#;
        let mut json_bytes = json.as_bytes().to_vec();
        while !json_bytes.len().is_multiple_of(4) {
            json_bytes.push(b' ');
        }
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }

        let total_len = 12 + 8 + json_bytes.len() + 8 + bin.len();
        let mut glb = Vec::with_capacity(total_len);
        glb.extend_from_slice(&0x4654_6C67_u32.to_le_bytes());
        glb.extend_from_slice(&2_u32.to_le_bytes());
        glb.extend_from_slice(&(total_len as u32).to_le_bytes());
        glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        glb.extend_from_slice(&0x4E4F_534A_u32.to_le_bytes());
        glb.extend_from_slice(&json_bytes);
        glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        glb.extend_from_slice(&0x004E_4942_u32.to_le_bytes());
        glb.extend_from_slice(&bin);
        std::fs::write(&path, glb).unwrap();

        let mesh = load_model(&path).expect("glb should load");
        assert_eq!(mesh.triangle_count(), 1);
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.uvs.len(), 3);
        assert_ne!(mesh.uvs[0], mesh.uvs[1]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn gltf_uses_material_texture_coordinate_set() {
        let dir = temp_dir("model_glb_uv_set");
        let path = dir.join("uv_set.glb");
        let mut bin = Vec::new();
        for position in [[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for component in position {
                bin.extend_from_slice(&component.to_le_bytes());
            }
        }
        let uv0 = [[0.1_f32, 0.1], [0.2, 0.2], [0.3, 0.3]];
        let uv1 = [[0.6_f32, 0.7], [0.8, 0.9], [0.4, 0.5]];
        for uv in uv0.into_iter().chain(uv1) {
            for component in uv {
                bin.extend_from_slice(&component.to_le_bytes());
            }
        }
        let json = r#"{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"TEXCOORD_0":1,"TEXCOORD_1":2},"material":0}]}],"materials":[{"pbrMetallicRoughness":{"baseColorTexture":{"index":0,"texCoord":1}}}],"textures":[{"source":0}],"images":[{"uri":"dummy.png"}],"buffers":[{"byteLength":84}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":24},{"buffer":0,"byteOffset":60,"byteLength":24}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]},{"bufferView":1,"componentType":5126,"count":3,"type":"VEC2"},{"bufferView":2,"componentType":5126,"count":3,"type":"VEC2"}]}"#;
        let mut json_bytes = json.as_bytes().to_vec();
        while !json_bytes.len().is_multiple_of(4) {
            json_bytes.push(b' ');
        }
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }

        let total_len = 12 + 8 + json_bytes.len() + 8 + bin.len();
        let mut glb = Vec::with_capacity(total_len);
        glb.extend_from_slice(&0x4654_6C67_u32.to_le_bytes());
        glb.extend_from_slice(&2_u32.to_le_bytes());
        glb.extend_from_slice(&(total_len as u32).to_le_bytes());
        glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        glb.extend_from_slice(&0x4E4F_534A_u32.to_le_bytes());
        glb.extend_from_slice(&json_bytes);
        glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        glb.extend_from_slice(&0x004E_4942_u32.to_le_bytes());
        glb.extend_from_slice(&bin);
        std::fs::write(&path, glb).unwrap();

        let mesh = load_model(&path).expect("glb should load");
        assert_eq!(mesh.uvs, uv1.to_vec());
        assert!(!mesh.uvs_generated);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn loads_ascii_fbx_quad() {
        let dir = temp_dir("model_fbx");
        let path = dir.join("quad.fbx");
        std::fs::write(
            &path,
            r#"; FBX 7.4.0 project file
FBXHeaderExtension:  {
    FBXHeaderVersion: 1003
    FBXVersion: 7400
    Creator: "texture-forge test"
}
GlobalSettings:  {
    Version: 1000
    Properties70:  {
        P: "UpAxis", "int", "Integer", "",1
        P: "UpAxisSign", "int", "Integer", "",1
        P: "FrontAxis", "int", "Integer", "",2
        P: "FrontAxisSign", "int", "Integer", "",1
        P: "CoordAxis", "int", "Integer", "",0
        P: "CoordAxisSign", "int", "Integer", "",1
        P: "UnitScaleFactor", "double", "Number", "",1
    }
}
Objects:  {
    Geometry: 1, "Geometry::Quad", "Mesh" {
        Vertices: *12 {
            a: -1,-1,0, 1,-1,0, 1,1,0, -1,1,0
        }
        PolygonVertexIndex: *4 {
            a: 0,1,2,-4
        }
        LayerElementUV: 0 {
            Version: 101
            Name: "UVMap"
            MappingInformationType: "ByPolygonVertex"
            ReferenceInformationType: "IndexToDirect"
            UV: *8 {
                a: 0,0.1, 1,0.2, 1,0.3, 0,0.4
            }
            UVIndex: *4 {
                a: 0,1,2,3
            }
        }
        Layer: 0 {
            Version: 100
            LayerElement:  {
                Type: "LayerElementUV"
                TypedIndex: 0
            }
        }
    }
    Model: 2, "Model::Quad", "Mesh" {
        Version: 232
        Properties70:  {
            P: "DefaultAttributeIndex", "int", "Integer", "",0
        }
    }
}
Connections:  {
    C: "OO",1,2
    C: "OO",2,0
}
"#,
        )
        .unwrap();

        let mesh = load_model(&path).expect("ascii fbx should load");
        assert_eq!(mesh.triangle_count(), 2);
        assert_eq!(mesh.positions.len(), 4);
        assert_eq!(mesh.uvs.len(), 4);
        assert!(mesh.is_valid());
        assert!(mesh
            .uvs
            .iter()
            .flatten()
            .all(|&uv| (0.0..=1.0).contains(&uv)));
        assert!(mesh.uvs.iter().all(|uv| uv[1] >= 0.6));
        assert!(!mesh.uvs_generated);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn uv_layout_applies_scale_and_rotation() {
        let layout = UvLayout {
            tex_coord: 0,
            offset: [0.1, 0.2],
            rotation: std::f32::consts::FRAC_PI_2,
            scale: [2.0, 3.0],
        };
        let uv = layout.apply([1.0, 0.0]);
        assert!((uv[0] - 0.1).abs() < 1e-5);
        assert!((uv[1] + 1.8).abs() < 1e-5);
    }

    #[test]
    fn triangle_strip_is_converted() {
        let triangles = primitive_triangles(gltf::mesh::Mode::TriangleStrip, &[0, 1, 2, 3]);
        assert_eq!(triangles, vec![[0, 1, 2], [2, 1, 3]]);
    }

    #[test]
    fn node_transform_moves_points() {
        let matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [2.0, 3.0, 4.0, 1.0],
        ];
        assert_eq!(transform_point(matrix, [1.0, 1.0, 1.0]), [3.0, 4.0, 5.0]);
    }
}
