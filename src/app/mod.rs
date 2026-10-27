mod batch_state;
mod model_preview;
mod preview;
mod ui;

use self::batch_state::BatchState;
use self::model_preview::ModelPreview;
use self::preview::{CropDrag, PreviewView};
use crate::files::{collect_images_in_dir, export_file_name, IMAGE_EXTENSIONS};
use crate::model::export::OutputFormat;
use crate::model::history::History;
use crate::model::presets::{Preset, PresetStore};
use crate::model::settings::Settings;
use crate::render::batch::BatchJob;
use crate::render::{RenderEvent, Renderer};
use eframe::egui;
use image::DynamicImage;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub(crate) struct TextureForge {
    source: Option<Arc<DynamicImage>>,
    source_path: Option<PathBuf>,
    /// Size of the source file on disk, in bytes.
    source_size: Option<u64>,
    /// Size of the result encoded to the selected output format, in bytes.
    result_size: Option<u64>,
    /// Cached stylized image; used for size estimation and saving without re-running `stylize`.
    result_image: Option<Arc<DynamicImage>>,
    preview_original: Option<egui::TextureHandle>,
    preview_result: Option<egui::TextureHandle>,
    settings: Settings,
    history: History<Settings>,
    /// Baseline of the in-progress interaction, committed once the pointer is released.
    pending: Option<Settings>,
    jpeg_quality: u8,
    output_format: OutputFormat,
    /// Suffix inserted before the extension of every exported file (e.g. `_stylized`).
    export_suffix: String,
    /// User-defined presets loaded from disk, plus the name currently selected in the manager.
    presets: PresetStore,
    /// Built-in + user presets, kept in sync with [`PresetStore`] so the UI does not rebuild it
    /// every frame.
    presets_cache: Vec<Preset>,
    selected_preset: Option<String>,
    /// Text buffer for the "save preset as" field.
    preset_name_input: String,
    /// Index of the tone-curve control point currently being dragged, if any.
    curve_drag: Option<usize>,
    /// Whether the original preview is in interactive crop-editing mode.
    crop_edit: bool,
    /// Active crop drag over the original preview, if any.
    crop_drag: Option<CropDrag>,
    /// Batch-processing window and its state.
    show_batch: bool,
    batch: BatchState,
    /// Floating 3D texture preview window and its state.
    show_model_preview: bool,
    model_preview: ModelPreview,
    status: String,
    /// Fraction of the preview width occupied by the left (original) pane.
    split_fraction: f32,
    view_original: PreviewView,
    view_result: PreviewView,
    /// Background worker that owns the heavy image processing.
    renderer: Renderer,
}

impl TextureForge {
    pub(crate) fn new(ctx: &egui::Context) -> Self {
        let presets = PresetStore::load();
        let presets_cache = presets.all();
        Self {
            source: None,
            source_path: None,
            source_size: None,
            result_size: None,
            result_image: None,
            preview_original: None,
            preview_result: None,
            settings: Settings::default(),
            history: History::new(),
            pending: None,
            jpeg_quality: 88,
            output_format: OutputFormat::Png,
            export_suffix: String::new(),
            presets,
            presets_cache,
            selected_preset: None,
            preset_name_input: String::new(),
            curve_drag: None,
            crop_edit: false,
            crop_drag: None,
            show_batch: false,
            batch: BatchState::default(),
            show_model_preview: false,
            model_preview: ModelPreview::new(ctx.clone()),
            status: "Откройте изображение, чтобы начать.".into(),
            split_fraction: 0.5,
            view_original: PreviewView::default(),
            view_result: PreviewView::default(),
            renderer: Renderer::spawn(ctx.clone()),
        }
    }

    fn open_image(&mut self, ctx: &egui::Context) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "webp"])
            .pick_file()
        {
            self.load_image(&path, ctx);
        }
    }

    /// Load an image from a path on disk, resetting all per-image state on success.
    fn load_image(&mut self, path: &std::path::Path, ctx: &egui::Context) {
        match image::open(path) {
            Ok(img) => {
                let size = std::fs::metadata(path).ok().map(|m| m.len());
                self.apply_loaded_image(img, Some(path.to_path_buf()), size, ctx);
            }
            Err(e) => self.status = format!("Не удалось открыть файл: {e}"),
        }
    }

    /// Load an image from an in-memory encoded buffer (e.g. a dropped web file).
    fn load_image_bytes(&mut self, name: &str, bytes: &[u8], ctx: &egui::Context) {
        match image::load_from_memory(bytes) {
            Ok(img) => self.apply_loaded_image(img, None, Some(bytes.len() as u64), ctx),
            Err(e) => self.status = format!("Не удалось прочитать «{name}»: {e}"),
        }
    }

    /// Install a freshly decoded image and reset everything tied to the previous one.
    fn apply_loaded_image(
        &mut self,
        img: DynamicImage,
        path: Option<PathBuf>,
        size: Option<u64>,
        ctx: &egui::Context,
    ) {
        self.source_size = size;
        self.source_path = path;
        self.source = Some(Arc::new(img));
        self.history = History::new();
        self.pending = None;
        self.result_size = None;
        self.result_image = None;
        self.view_original.reset();
        self.view_result.reset();
        self.preview_original = self.source.as_ref().map(|src| {
            ctx.load_texture(
                "original-preview",
                Self::color_image(src),
                egui::TextureOptions::LINEAR,
            )
        });
        self.model_preview.mark_dirty();
        self.model_preview.reset_history();
        self.status = "Изображение загружено.".into();
        self.refresh_previews(ctx);
    }

    /// Open the first image dropped onto the window, from bytes or a filesystem path.
    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        let Some(file) = dropped.into_iter().next() else {
            return;
        };
        if let Some(bytes) = file.bytes.as_ref() {
            if is_model_name(&file.name) {
                self.status =
                    "Модель из веб-перетаскивания не поддерживается. Перетащите файл из файловой системы.".into();
            } else {
                self.load_image_bytes(&file.name, bytes, ctx);
            }
        } else if let Some(path) = file.path.as_ref() {
            if is_model_path(path) {
                self.show_model_preview = true;
                self.start_model_load_from(path.clone());
            } else {
                self.load_image(path, ctx);
            }
        }
    }

    /// Queue a stylize job for the current source and settings on the worker thread.
    fn request_stylize(&mut self) {
        let Some(source) = self.source.clone() else {
            self.result_image = None;
            self.result_size = None;
            return;
        };
        self.renderer.submit_stylize(source, self.settings.clone());
    }

    /// Queue a size-only re-encode of the cached result in the current output format.
    fn request_reencode_size(&mut self) {
        let Some(image) = self.result_image.clone() else {
            return;
        };
        self.renderer.submit_reencode(
            image,
            self.output_format,
            self.jpeg_quality,
            self.settings.background,
        );
    }

    fn color_image(img: &DynamicImage) -> egui::ColorImage {
        let rgba = img.to_rgba8();
        let size = [rgba.width() as usize, rgba.height() as usize];
        egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw())
    }

    /// Queue a background stylize of the current source.
    ///
    /// The original preview texture is uploaded once per loaded image (see
    /// [`TextureForge::apply_loaded_image`]) and never changes with the settings.
    fn refresh_previews(&mut self, _ctx: &egui::Context) {
        self.request_stylize();
    }

    /// Apply results coming back from the render thread. Stale renders (superseded by a newer
    /// request) are ignored so a slow background job can never clobber a fresh one.
    fn poll_renderer(&mut self, ctx: &egui::Context) {
        self.renderer.tick(Instant::now());
        if let Some(delay) = self.renderer.time_until_dispatch() {
            // Wake up when the debounce window ends, even if the user stops interacting.
            ctx.request_repaint_after(delay);
        }
        while let Some(event) = self.renderer.try_recv() {
            match event {
                RenderEvent::Rendered { id, image, size } => {
                    if id == self.renderer.latest_stylize {
                        self.renderer.in_flight = false;
                        self.renderer.active_cancel = None;
                        if let Some(img) = image {
                            self.preview_result = Some(ctx.load_texture(
                                "result-preview",
                                Self::color_image(&img),
                                egui::TextureOptions::NEAREST,
                            ));
                            self.result_image = Some(img);
                            self.model_preview.mark_dirty();
                            self.request_reencode_size();
                        } else {
                            self.result_image = None;
                            self.result_size = None;
                            self.model_preview.mark_dirty();
                        }
                    }
                    if let Some(size) = size {
                        if id >= self.renderer.applied_size_id {
                            self.renderer.applied_size_id = id;
                            self.result_size = Some(size);
                        }
                    }
                }
                RenderEvent::Saved { result } => {
                    self.status = match result {
                        Ok(path) => format!("Сохранено: {}", path.display()),
                        Err(e) => format!("Ошибка сохранения: {e}"),
                    };
                }
                RenderEvent::BatchProgress {
                    done,
                    total,
                    current,
                    saved,
                    errors,
                } => {
                    self.batch
                        .apply_progress(done, total, current, saved, errors);
                }
                RenderEvent::BatchFinished {
                    saved,
                    errors,
                    cancelled,
                } => {
                    self.batch.finish(saved, errors, cancelled);
                    self.status = if cancelled {
                        format!("Пакетная обработка отменена: сохранено {saved}.")
                    } else {
                        format!("Пакетная обработка завершена: сохранено {saved}.")
                    };
                }
            }
        }
    }

    /// Apply a preset's settings, format and quality as a single undoable change.
    fn apply_preset(&mut self, preset: &Preset, ctx: &egui::Context) {
        let before = self.settings.clone();
        self.settings = preset.settings.clone();
        self.output_format = preset.format;
        self.jpeg_quality = preset.quality;
        let settings_changed = before != self.settings;
        if settings_changed {
            self.pending = None;
            self.history.record(before);
        }
        self.selected_preset = Some(preset.name.clone());
        self.status = format!("Применён пресет «{}».", preset.name);
        if settings_changed {
            // Re-stylize; the fresh result triggers a size re-encode when it arrives.
            self.refresh_previews(ctx);
        } else {
            // Only the export format/quality changed: reuse the cached result image.
            self.request_reencode_size();
        }
    }

    /// Save the current settings as a named user preset.
    fn save_current_preset(&mut self) {
        let name = self.preset_name_input.trim();
        if name.is_empty() {
            self.status = "Введите имя пресета.".into();
            return;
        }
        let preset = Preset::new(
            name,
            self.settings.clone(),
            self.output_format,
            self.jpeg_quality,
        );
        self.presets.insert(preset);
        self.presets_cache = self.presets.all();
        self.status = match self.presets.save() {
            Ok(()) => format!("Пресет «{name}» сохранён."),
            Err(e) => format!("Пресет сохранён в сессии, но не записан на диск: {e}"),
        };
        self.selected_preset = Some(name.to_owned());
        self.preset_name_input.clear();
    }

    /// Delete the currently selected user preset.
    fn delete_selected_preset(&mut self) {
        let Some(name) = self.selected_preset.clone() else {
            return;
        };
        if PresetStore::is_builtin(&name) {
            self.status = "Встроенные пресеты удалить нельзя.".into();
            return;
        }
        self.presets.remove(&name);
        self.presets_cache = self.presets.all();
        self.status = match self.presets.save() {
            Ok(()) => format!("Пресет «{name}» удалён."),
            Err(e) => format!("Пресет удалён из сессии, но не обновлён на диске: {e}"),
        };
        self.selected_preset = None;
    }

    /// Add files to the batch queue, skipping duplicates already present.
    fn add_batch_files(&mut self) {
        if let Some(files) = rfd::FileDialog::new()
            .add_filter("Images", &IMAGE_EXTENSIONS)
            .pick_files()
        {
            self.extend_batch_inputs(files);
        }
    }

    /// Add all top-level images from a chosen folder to the batch queue.
    fn add_batch_folder(&mut self) {
        if let Some(dir) = rfd::FileDialog::new().pick_folder() {
            let files = collect_images_in_dir(&dir);
            if files.is_empty() {
                self.status = "В папке нет поддерживаемых изображений.".into();
            }
            self.extend_batch_inputs(files);
        }
    }

    fn extend_batch_inputs(&mut self, files: Vec<PathBuf>) {
        self.batch.enqueue(files);
    }

    /// Start processing the batch queue with the current settings and output format.
    fn start_batch(&mut self) {
        if self.batch.running || self.batch.inputs.is_empty() {
            return;
        }
        let Some(output_dir) = self.batch.output_dir.clone() else {
            self.status = "Выберите папку вывода для пакетной обработки.".into();
            return;
        };
        if let Err(e) = std::fs::create_dir_all(&output_dir) {
            self.status = format!("Не удалось создать папку вывода: {e}");
            return;
        }
        self.batch.begin(Arc::new(AtomicBool::new(false)));

        let job = BatchJob {
            inputs: self.batch.inputs.clone(),
            output_dir,
            settings: self.settings.clone(),
            format: self.output_format,
            quality: self.jpeg_quality,
            suffix: self.export_suffix.clone(),
            cancel: self.batch.cancel.clone(),
        };
        self.renderer.submit_batch(job);
        self.status = "Пакетная обработка запущена…".into();
    }

    fn cancel_batch(&mut self) {
        if self.batch.running {
            self.batch.cancel.store(true, Ordering::Relaxed);
            self.status = "Отмена пакетной обработки…".into();
        }
    }

    fn undo_action(&mut self, ctx: &egui::Context) {
        if self.history.undo(&mut self.settings) {
            self.pending = None;
            self.status = "Действие отменено.".into();
            self.refresh_previews(ctx);
        }
    }

    fn redo_action(&mut self, ctx: &egui::Context) {
        if self.history.redo(&mut self.settings) {
            self.pending = None;
            self.status = "Действие повторено.".into();
            self.refresh_previews(ctx);
        }
    }

    /// Record the baseline captured at the start of an interaction once the pointer is released,
    /// so an entire slider drag collapses into a single undo step.
    fn commit_pending(&mut self) {
        if let Some(baseline) = self.pending.take() {
            if baseline != self.settings {
                self.history.record(baseline);
            }
        }
    }

    fn save_result(&mut self) {
        let Some(result) = self.result_image.clone() else {
            self.status = "Сначала откройте изображение.".into();
            return;
        };
        let (filter, extension) = match self.output_format {
            OutputFormat::Png => ("PNG image", "png"),
            OutputFormat::Jpeg => ("JPEG image", "jpg"),
            OutputFormat::WebP => ("WebP image", "webp"),
        };
        let stem = self
            .source_path
            .as_deref()
            .and_then(|path| path.file_stem())
            .and_then(|stem| stem.to_str())
            .unwrap_or("texture");
        let file_name = export_file_name(stem, &self.export_suffix, self.output_format);
        let path = rfd::FileDialog::new()
            .add_filter(filter, &[extension])
            .set_file_name(file_name)
            .save_file();
        if let Some(path) = path {
            self.renderer.submit_save(
                result,
                self.output_format,
                self.jpeg_quality,
                self.settings.background,
                path,
            );
            self.status = "Сохранение…".into();
        }
    }
}

fn is_model_path(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(is_model_extension)
}

fn is_model_name(name: &str) -> bool {
    std::path::Path::new(name)
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(is_model_extension)
}

fn is_model_extension(extension: &str) -> bool {
    ["obj", "glb", "gltf", "fbx"]
        .iter()
        .any(|candidate| extension.eq_ignore_ascii_case(candidate))
}

impl Default for TextureForge {
    fn default() -> Self {
        Self::new(&egui::Context::default())
    }
}

/// Start the native application.
pub(crate) fn run() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Texture Forge")
            .with_inner_size([1100.0, 720.0])
            .with_min_inner_size([760.0, 480.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Texture Forge",
        options,
        Box::new(|cc| Ok(Box::new(TextureForge::new(&cc.egui_ctx)))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imaging::pipeline::stylize_image;
    use eframe::egui;
    use image::RgbaImage;
    use image::{DynamicImage, ImageFormat};

    #[test]
    fn load_image_bytes_decodes_and_resets_state() {
        let mut img = RgbaImage::new(4, 4);
        for (x, y, p) in img.enumerate_pixels_mut() {
            p.0 = [(x * 40) as u8, (y * 40) as u8, 10, 255];
        }
        let mut png = std::io::Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(img)
            .write_to(&mut png, ImageFormat::Png)
            .expect("encode png");
        let bytes = png.into_inner();

        let ctx = egui::Context::default();
        let mut app = TextureForge::default();
        app.load_image_bytes("test.png", &bytes, &ctx);

        let source = app.source.clone().expect("image decoded from bytes");
        assert_eq!((source.width(), source.height()), (4, 4));
        assert_eq!(app.source_path, None);
        assert_eq!(app.source_size, Some(bytes.len() as u64));
        assert!(app.preview_original.is_some());
        // Stylization happens on the worker, so wait for the result to come back.
        assert!(
            wait_for_result(&mut app, &ctx),
            "no stylized preview produced"
        );
    }

    #[test]
    fn renderer_applies_only_the_latest_request() {
        let mut img = RgbaImage::new(32, 32);
        for (x, y, p) in img.enumerate_pixels_mut() {
            p.0 = [(x * 8) as u8, (y * 8) as u8, ((x + y) * 4) as u8, 255];
        }

        let ctx = egui::Context::default();
        let mut app = TextureForge {
            source: Some(Arc::new(DynamicImage::ImageRgba8(img))),
            ..Default::default()
        };

        // Two requests in quick succession: the second must win.
        app.settings.palette_size = 4;
        app.refresh_previews(&ctx);
        app.settings.palette_size = 64;
        app.refresh_previews(&ctx);

        assert!(
            wait_for_result(&mut app, &ctx),
            "no stylized preview produced"
        );

        let expected = stylize_image(app.source.as_ref().unwrap(), &app.settings)
            .expect("expected render")
            .to_rgba8();
        let actual = app.result_image.as_ref().unwrap().to_rgba8();
        assert_eq!(actual.dimensions(), expected.dimensions());
        assert_eq!(actual.as_raw(), expected.as_raw());
    }

    #[test]
    fn model_path_detection_is_case_insensitive() {
        assert!(is_model_path(std::path::Path::new("model.OBJ")));
        assert!(is_model_path(std::path::Path::new("model.glb")));
        assert!(is_model_name("model.fbx"));
        assert!(!is_model_path(std::path::Path::new("texture.png")));
        assert!(!is_model_name("README"));
    }

    fn wait_for_result(app: &mut TextureForge, ctx: &egui::Context) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while app.preview_result.is_none() && std::time::Instant::now() < deadline {
            app.poll_renderer(ctx);
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        app.preview_result.is_some()
    }
}
