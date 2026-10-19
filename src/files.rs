use crate::model::export::OutputFormat;
use std::path::{Path, PathBuf};

/// Compare two paths for equality, tolerating differences in relative vs. absolute form.
pub(crate) fn paths_equal(a: &Path, b: &Path) -> bool {
    let canon_a = a.canonicalize();
    let canon_b = b.canonicalize();
    match (canon_a, canon_b) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

/// Image extensions accepted for opening and batch input.
pub(crate) const IMAGE_EXTENSIONS: [&str; 5] = ["png", "jpg", "jpeg", "bmp", "webp"];

/// Whether `path` looks like a supported image, by extension.
pub(crate) fn is_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| IMAGE_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Non-recursively collect supported image files from `dir`, sorted by name for stable ordering.
pub(crate) fn collect_images_in_dir(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && is_image_path(path))
        .collect();
    files.sort();
    files
}

/// Build an exported file name: `{stem}{suffix}.{ext}`.
pub(crate) fn export_file_name(stem: &str, suffix: &str, format: OutputFormat) -> String {
    format!("{stem}{suffix}.{}", format.extension())
}

/// Derive the destination path for a batch output: `output_dir/{stem}{suffix}.{ext}`.
pub(crate) fn batch_output_path(
    input: &Path,
    output_dir: &Path,
    suffix: &str,
    format: OutputFormat,
) -> PathBuf {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("texture");
    output_dir.join(export_file_name(stem, suffix, format))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{temp_dir, write_test_png};
    use std::path::PathBuf;

    #[test]
    fn export_file_name_inserts_suffix_before_extension() {
        assert_eq!(
            export_file_name("texture", "", OutputFormat::Png),
            "texture.png"
        );
        assert_eq!(
            export_file_name("texture", "_stylized", OutputFormat::Jpeg),
            "texture_stylized.jpg"
        );
        assert_eq!(
            export_file_name("hero", "-retro", OutputFormat::WebP),
            "hero-retro.webp"
        );
    }

    #[test]
    fn batch_output_path_uses_stem_suffix_and_extension() {
        let out = PathBuf::from("/tmp/out");
        assert_eq!(
            batch_output_path(Path::new("/tmp/in/hero.png"), &out, "", OutputFormat::Png),
            out.join("hero.png")
        );
        assert_eq!(
            batch_output_path(
                Path::new("/tmp/in/hero.jpg"),
                &out,
                "_stylized",
                OutputFormat::Jpeg
            ),
            out.join("hero_stylized.jpg")
        );
        assert_eq!(
            batch_output_path(
                Path::new("/tmp/in/tile.bmp"),
                &out,
                "-q",
                OutputFormat::WebP
            ),
            out.join("tile-q.webp")
        );
    }

    #[test]
    fn is_image_path_matches_extensions_case_insensitively() {
        assert!(is_image_path(Path::new("a.PNG")));
        assert!(is_image_path(Path::new("a.WebP")));
        assert!(!is_image_path(Path::new("a.txt")));
        assert!(!is_image_path(Path::new("no_extension")));
    }

    #[test]
    fn collect_images_in_dir_keeps_only_supported_files_sorted() {
        let dir = temp_dir("collect");
        write_test_png(&dir.join("b.png"));
        write_test_png(&dir.join("a.png"));
        std::fs::write(dir.join("notes.txt"), b"nope").unwrap();
        std::fs::create_dir_all(dir.join("sub.png")).unwrap();

        let files = collect_images_in_dir(&dir);
        let names: Vec<String> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["a.png", "b.png"]);

        std::fs::remove_dir_all(&dir).ok();
    }
}
