use crate::app::preview::paint_checkerboard;
use crate::app::TextureForge;
use crate::model::history::History;
use crate::preview3d::loader::load_model;
use crate::preview3d::mesh::{BuiltinShape, Mesh3d};
use crate::preview3d::raster::{
    light_direction, RenderOptions, SampleFilter, UvTransform, WrapMode,
};
use crate::preview3d::worker::{
    PreviewEvent, PreviewRender, PreviewSave, PreviewTextures, PreviewWorker,
};
use eframe::egui;
use image::{DynamicImage, RgbaImage};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TextureChoice {
    Result,
    Original,
}

impl TextureChoice {
    fn label(self) -> &'static str {
        match self {
            Self::Result => "Результат",
            Self::Original => "Исходник",
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Result => 0,
            Self::Original => 1,
        }
    }

    fn opposite(self) -> Self {
        match self {
            Self::Result => Self::Original,
            Self::Original => Self::Result,
        }
    }
}

#[derive(Default)]
struct TextureCache {
    entries: [Option<CachedTexture>; 2],
}

struct CachedTexture {
    image: Arc<DynamicImage>,
    rgba: Arc<RgbaImage>,
}

/// Camera/state that identifies a rendered frame. A change means the worker needs a new frame.
#[derive(Clone, Copy, Debug, PartialEq)]
struct RenderKey {
    mesh: usize,
    primary: usize,
    secondary: usize,
    options: RenderOptions,
    size: [usize; 2],
}

/// Everything that can be undone in the 3D preview. Loaded model data and the texture itself are
/// documents, not actions: replacing either one resets the history.
#[derive(Clone, Debug, PartialEq)]
struct PreviewState {
    builtin: BuiltinShape,
    use_custom: bool,
    texture: TextureChoice,
    uv_tiling: [f32; 2],
    uv_offset: [f32; 2],
    uv_rotation: f32,
    wrap: WrapMode,
    filter: SampleFilter,
    lighting: bool,
    light_azimuth: f32,
    light_elevation: f32,
    yaw: f32,
    pitch: f32,
    pan: [f32; 2],
    zoom: f32,
    auto_rotate: bool,
    auto_rotate_speed: f32,
    compare: bool,
    compare_split: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DragMode {
    Rotate,
    Pan,
    Split,
}

#[derive(Clone, Copy, Debug)]
struct DragOrigin {
    pos: egui::Pos2,
    mode: DragMode,
    yaw: f32,
    pitch: f32,
    pan: [f32; 2],
    split: f32,
}

pub(crate) struct ModelPreview {
    builtin: BuiltinShape,
    use_custom: bool,
    custom_mesh: Option<Arc<Mesh3d>>,
    custom_name: Option<String>,
    builtin_cache: Option<(BuiltinShape, Arc<Mesh3d>)>,
    texture: TextureChoice,
    texture_cache: TextureCache,
    uv_tiling: [f32; 2],
    uv_offset: [f32; 2],
    uv_rotation: f32,
    wrap: WrapMode,
    filter: SampleFilter,
    lighting: bool,
    light_azimuth: f32,
    light_elevation: f32,
    yaw: f32,
    pitch: f32,
    pan: [f32; 2],
    zoom: f32,
    auto_rotate: bool,
    auto_rotate_speed: f32,
    compare: bool,
    compare_split: f32,
    save_resolution: usize,
    drag_origin: Option<DragOrigin>,
    last_interaction: Option<Instant>,
    history: History<PreviewState>,
    /// State captured before the current interaction; committed when the pointer is released.
    pending_change: Option<PreviewState>,
    /// Last painted window rectangle, used to route undo/redo shortcuts to the preview.
    window_rect: Option<egui::Rect>,
    texture_handle: Option<egui::TextureHandle>,
    rendered_size: [usize; 2],
    render_key: Option<RenderKey>,
    dirty: bool,
    worker: PreviewWorker,
    next_render_id: u64,
    latest_render_id: u64,
    applied_render_id: u64,
    pending_render: Option<u64>,
    /// Cancellation token shared by all frames of the current render epoch.
    render_cancel: Arc<AtomicBool>,
    last_submitted_size: [usize; 2],
    loading: bool,
    load_rx: Option<Receiver<Result<(Mesh3d, String), String>>>,
    status: String,
    save_status: String,
}

impl Default for ModelPreview {
    fn default() -> Self {
        Self::new(egui::Context::default())
    }
}

impl ModelPreview {
    pub(crate) fn new(ctx: egui::Context) -> Self {
        Self {
            builtin: BuiltinShape::Cube,
            use_custom: false,
            custom_mesh: None,
            custom_name: None,
            builtin_cache: None,
            texture: TextureChoice::Result,
            texture_cache: TextureCache::default(),
            uv_tiling: [1.0, 1.0],
            uv_offset: [0.0, 0.0],
            uv_rotation: 0.0,
            wrap: WrapMode::Repeat,
            filter: SampleFilter::Nearest,
            lighting: true,
            light_azimuth: -0.6,
            light_elevation: 0.9,
            yaw: 0.55,
            pitch: -0.3,
            pan: [0.0, 0.0],
            zoom: 1.0,
            auto_rotate: false,
            auto_rotate_speed: 1.2,
            compare: false,
            compare_split: 0.5,
            save_resolution: 1024,
            drag_origin: None,
            last_interaction: None,
            history: History::new(),
            pending_change: None,
            window_rect: None,
            texture_handle: None,
            rendered_size: [0, 0],
            render_key: None,
            dirty: true,
            worker: PreviewWorker::spawn(ctx),
            next_render_id: 1,
            latest_render_id: 0,
            applied_render_id: 0,
            pending_render: None,
            render_cancel: Arc::new(AtomicBool::new(false)),
            last_submitted_size: [0, 0],
            loading: false,
            load_rx: None,
            status: String::new(),
            save_status: String::new(),
        }
    }

    pub(crate) fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    fn touch(&mut self) {
        self.last_interaction = Some(Instant::now());
    }

    fn reset_view(&mut self) {
        self.yaw = 0.55;
        self.pitch = -0.3;
        self.pan = [0.0, 0.0];
        self.zoom = 1.0;
        self.auto_rotate = false;
        self.drag_origin = None;
        self.dirty = true;
    }

    fn state(&self) -> PreviewState {
        PreviewState {
            builtin: self.builtin,
            use_custom: self.use_custom,
            texture: self.texture,
            uv_tiling: self.uv_tiling,
            uv_offset: self.uv_offset,
            uv_rotation: self.uv_rotation,
            wrap: self.wrap,
            filter: self.filter,
            lighting: self.lighting,
            light_azimuth: self.light_azimuth,
            light_elevation: self.light_elevation,
            yaw: self.yaw,
            pitch: self.pitch,
            pan: self.pan,
            zoom: self.zoom,
            auto_rotate: self.auto_rotate,
            auto_rotate_speed: self.auto_rotate_speed,
            compare: self.compare,
            compare_split: self.compare_split,
        }
    }

    fn apply_state(&mut self, state: PreviewState) {
        self.builtin = state.builtin;
        self.use_custom = state.use_custom;
        self.texture = state.texture;
        self.uv_tiling = state.uv_tiling;
        self.uv_offset = state.uv_offset;
        self.uv_rotation = state.uv_rotation;
        self.wrap = state.wrap;
        self.filter = state.filter;
        self.lighting = state.lighting;
        self.light_azimuth = state.light_azimuth;
        self.light_elevation = state.light_elevation;
        self.yaw = state.yaw;
        self.pitch = state.pitch;
        self.pan = state.pan;
        self.zoom = state.zoom;
        self.auto_rotate = state.auto_rotate;
        self.auto_rotate_speed = state.auto_rotate_speed;
        self.compare = state.compare;
        self.compare_split = state.compare_split;
        self.drag_origin = None;
        self.dirty = true;
    }

    /// Intermediate frames are useful during a drag: drop only results older than the last frame
    /// already shown, not everything that is not the newest request.
    fn should_apply_render(&mut self, id: u64) -> bool {
        if id <= self.applied_render_id {
            false
        } else {
            self.applied_render_id = id;
            true
        }
    }

    /// Abort every frame that belongs to the current epoch and start a fresh one.
    fn cancel_renders(&mut self) {
        self.render_cancel.store(true, Ordering::Relaxed);
        self.render_cancel = Arc::new(AtomicBool::new(false));
    }

    /// Cancel the current epoch and ignore results that were already on their way.
    fn reset_renders(&mut self) {
        self.cancel_renders();
        self.applied_render_id = self.next_render_id.saturating_sub(1);
        self.pending_render = None;
    }

    fn can_undo(&self) -> bool {
        self.history.can_undo()
    }

    fn can_redo(&self) -> bool {
        self.history.can_redo()
    }

    fn undo(&mut self) -> bool {
        let mut current = self.state();
        if self.history.undo(&mut current) {
            self.pending_change = None;
            self.apply_state(current);
            true
        } else {
            false
        }
    }

    fn redo(&mut self) -> bool {
        let mut current = self.state();
        if self.history.redo(&mut current) {
            self.pending_change = None;
            self.apply_state(current);
            true
        } else {
            false
        }
    }

    fn commit_pending_change(&mut self) {
        if let Some(baseline) = self.pending_change.take() {
            if baseline != self.state() {
                self.history.record(baseline);
            }
        }
    }

    pub(crate) fn reset_history(&mut self) {
        self.history = History::new();
        self.pending_change = None;
        self.reset_renders();
    }

    /// True when the pointer is over the preview window, so undo/redo shortcuts belong to it.
    pub(crate) fn shortcuts_active(&self, ctx: &egui::Context) -> bool {
        self.window_rect.is_some_and(|rect| {
            ctx.input(|i| {
                i.pointer
                    .hover_pos()
                    .is_some_and(|position| rect.contains(position))
            })
        })
    }

    fn active_mesh(&mut self) -> Option<Arc<Mesh3d>> {
        if self.use_custom {
            return self.custom_mesh.clone();
        }
        let cached = self
            .builtin_cache
            .as_ref()
            .is_some_and(|(shape, _)| *shape == self.builtin);
        if !cached {
            let mesh = Arc::new(Mesh3d::builtin(self.builtin));
            self.builtin_cache = Some((self.builtin, mesh));
        }
        self.builtin_cache.as_ref().map(|(_, mesh)| mesh.clone())
    }

    fn triangle_count(&self) -> usize {
        if self.use_custom {
            self.custom_mesh
                .as_ref()
                .map_or(0, |mesh| mesh.triangle_count())
        } else {
            self.builtin_cache
                .as_ref()
                .filter(|(shape, _)| *shape == self.builtin)
                .map_or(0, |(_, mesh)| mesh.triangle_count())
        }
    }

    fn rgba_for(&mut self, choice: TextureChoice, image: &Arc<DynamicImage>) -> Arc<RgbaImage> {
        let entry = self.texture_cache.entries[choice.index()].as_ref();
        if let Some(cached) = entry {
            if Arc::ptr_eq(&cached.image, image) {
                return cached.rgba.clone();
            }
        }
        let rgba = Arc::new(image.to_rgba8());
        self.texture_cache.entries[choice.index()] = Some(CachedTexture {
            image: image.clone(),
            rgba: rgba.clone(),
        });
        rgba
    }

    fn render_options(&self) -> RenderOptions {
        RenderOptions {
            yaw: self.yaw,
            pitch: self.pitch,
            pan: self.pan,
            zoom: self.zoom,
            uvs: UvTransform {
                tiling: self.uv_tiling,
                offset: self.uv_offset,
                rotation: self.uv_rotation,
            },
            wrap: self.wrap,
            lighting: self.lighting,
            light: light_direction(self.light_azimuth, self.light_elevation),
            filter: self.filter,
            split: self.compare.then_some(self.compare_split),
        }
    }

    fn request_render_if_needed(
        &mut self,
        primary_image: &Arc<DynamicImage>,
        primary_choice: TextureChoice,
        secondary_image: Option<(&Arc<DynamicImage>, TextureChoice)>,
        target: [usize; 2],
    ) {
        let Some(mesh) = self.active_mesh() else {
            return;
        };
        let primary = self.rgba_for(primary_choice, primary_image);
        let secondary = secondary_image.map(|(image, choice)| self.rgba_for(choice, image));
        let options = self.render_options();
        let key = RenderKey {
            mesh: Arc::as_ptr(&mesh) as usize,
            primary: Arc::as_ptr(&primary) as usize,
            secondary: secondary
                .as_ref()
                .map_or(0, |texture| Arc::as_ptr(texture) as usize),
            options,
            size: target,
        };
        if !self.dirty && self.render_key == Some(key) {
            return;
        }
        self.dirty = false;
        self.render_key = Some(key);

        // Do not cancel the in-flight frame on every submit: that would abort every render
        // during a drag before it can reach the screen. Only cancel a slow full-quality render
        // when the requested resolution drops for interaction.
        let target_area = target[0].saturating_mul(target[1]);
        let previous_area = self.last_submitted_size[0].saturating_mul(self.last_submitted_size[1]);
        if previous_area > target_area.saturating_mul(2) {
            self.cancel_renders();
        }
        self.last_submitted_size = target;

        let id = self.next_render_id;
        self.next_render_id = self.next_render_id.wrapping_add(1);
        self.latest_render_id = id;
        self.pending_render = Some(id);
        self.worker.submit_render(PreviewRender {
            id,
            mesh,
            textures: PreviewTextures { primary, secondary },
            width: target[0],
            height: target[1],
            options,
            cancel: self.render_cancel.clone(),
        });
    }

    fn request_save(
        &mut self,
        primary_image: &Arc<DynamicImage>,
        primary_choice: TextureChoice,
        secondary_image: Option<(&Arc<DynamicImage>, TextureChoice)>,
        size: [usize; 2],
        path: PathBuf,
    ) {
        let Some(mesh) = self.active_mesh() else {
            return;
        };
        let primary = self.rgba_for(primary_choice, primary_image);
        let secondary = secondary_image.map(|(image, choice)| self.rgba_for(choice, image));
        self.worker.submit_save(PreviewSave {
            mesh,
            textures: PreviewTextures { primary, secondary },
            width: size[0],
            height: size[1],
            options: self.render_options(),
            path,
        });
        self.save_status = "Сохранение кадра…".into();
    }
}

impl TextureForge {
    pub(crate) fn poll_model_load(&mut self, ctx: &egui::Context) {
        let message = self.model_preview.load_rx.as_ref().map(|rx| rx.try_recv());
        match message {
            Some(Ok(Ok((mesh, name)))) => {
                let triangles = mesh.triangle_count();
                let uv_note = if mesh.uvs_generated {
                    " UV отсутствуют у части модели — использована сферическая проекция."
                } else {
                    ""
                };
                self.model_preview.status =
                    format!("Модель «{name}» загружена: {triangles} треугольников.{uv_note}");
                self.model_preview.custom_mesh = Some(Arc::new(mesh));
                self.model_preview.custom_name = Some(name);
                self.model_preview.use_custom = true;
                self.model_preview.loading = false;
                self.model_preview.load_rx = None;
                self.model_preview.reset_history();
                self.model_preview.reset_view();
            }
            Some(Ok(Err(error))) => {
                self.model_preview.status = error;
                self.model_preview.loading = false;
                self.model_preview.load_rx = None;
            }
            Some(Err(TryRecvError::Disconnected)) => {
                self.model_preview.status = "Загрузка модели была прервана.".into();
                self.model_preview.loading = false;
                self.model_preview.load_rx = None;
            }
            Some(Err(TryRecvError::Empty)) => {
                ctx.request_repaint_after(Duration::from_millis(80));
            }
            None => {}
        }
    }

    pub(crate) fn poll_model_preview(&mut self, ctx: &egui::Context) {
        while let Some(event) = self.model_preview.worker.try_recv() {
            match event {
                PreviewEvent::Rendered { id, image } => {
                    if !self.show_model_preview {
                        continue;
                    }
                    // Accept intermediate frames while a newer one is still queued: otherwise
                    // the preview would look frozen during a drag and only update on release.
                    if !self.model_preview.should_apply_render(id) {
                        continue;
                    }
                    let size = image.size;
                    if let Some(handle) = &mut self.model_preview.texture_handle {
                        handle.set(image, egui::TextureOptions::NEAREST);
                    } else {
                        self.model_preview.texture_handle = Some(ctx.load_texture(
                            "model-preview",
                            image,
                            egui::TextureOptions::NEAREST,
                        ));
                    }
                    self.model_preview.rendered_size = size;
                    if id == self.model_preview.latest_render_id {
                        self.model_preview.pending_render = None;
                    }
                }
                PreviewEvent::Saved { result } => {
                    self.model_preview.save_status = match result {
                        Ok(path) => format!("Кадр сохранён: {}", path.display()),
                        Err(error) => format!("Не удалось сохранить кадр: {error}"),
                    };
                }
            }
        }
        if self.model_preview.pending_render.is_some() {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
    }

    fn start_model_load(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("3D models", &["obj", "glb", "gltf", "fbx"])
            .pick_file()
        else {
            return;
        };
        self.start_model_load_from(path);
    }

    pub(super) fn start_model_load_from(&mut self, path: PathBuf) {
        if self.model_preview.loading {
            return;
        }
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model")
            .to_owned();
        let (tx, rx) = mpsc::channel();
        self.model_preview.loading = true;
        self.model_preview.load_rx = Some(rx);
        self.model_preview.status = format!("Загрузка модели «{name}»…");
        std::thread::spawn(move || {
            let result = load_model(&path).map(|mesh| (mesh, name));
            let _ = tx.send(result);
        });
    }

    pub(super) fn show_model_window(&mut self, ctx: &egui::Context) {
        if !self.show_model_preview {
            return;
        }
        let mut open = true;
        let window = egui::Window::new("Предпросмотр на модели")
            .open(&mut open)
            .default_size([640.0, 700.0])
            .min_size([360.0, 360.0])
            .resizable(true)
            .show(ctx, |ui| self.model_preview_ui(ctx, ui));
        self.model_preview.window_rect = window.map(|window| window.response.rect);
        if !open {
            self.show_model_preview = false;
            self.model_preview.window_rect = None;
            self.model_preview.commit_pending_change();
            self.model_preview.texture_cache = TextureCache::default();
            self.model_preview.texture_handle = None;
            self.model_preview.rendered_size = [0, 0];
            self.model_preview.render_key = None;
            self.model_preview.dirty = true;
            self.model_preview.reset_renders();
        }
    }

    fn model_preview_ui(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let mut load_requested = false;
        let mut reset_requested = false;
        let mut save_requested = false;
        let mut undo_requested = false;
        let mut redo_requested = false;
        let mut history_navigated = false;

        if self.model_preview.shortcuts_active(ctx)
            && !ctx.is_using_pointer()
            && !ctx.wants_keyboard_input()
        {
            let undo_shortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Z);
            let redo_shift_shortcut = egui::KeyboardShortcut::new(
                egui::Modifiers::COMMAND.plus(egui::Modifiers::SHIFT),
                egui::Key::Z,
            );
            let redo_alt_shortcut =
                egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Y);
            // Ctrl+Shift+Z must be checked before Ctrl+Z, otherwise the extra Shift is ignored.
            if ctx.input_mut(|i| i.consume_shortcut(&redo_shift_shortcut)) {
                self.model_preview.redo();
                history_navigated = true;
            } else if ctx.input_mut(|i| i.consume_shortcut(&undo_shortcut)) {
                self.model_preview.undo();
                history_navigated = true;
            } else if ctx.input_mut(|i| i.consume_shortcut(&redo_alt_shortcut)) {
                self.model_preview.redo();
                history_navigated = true;
            }
        }

        // Advance auto-rotation before capturing the history baseline: animation must not
        // generate undo steps on every frame.
        if self.model_preview.auto_rotate {
            let delta = ctx.input(|i| i.stable_dt).min(0.1);
            self.model_preview.yaw = (self.model_preview.yaw
                + delta * self.model_preview.auto_rotate_speed)
                % std::f32::consts::TAU;
            self.model_preview.dirty = true;
            ctx.request_repaint_after(Duration::from_millis(16));
        }
        let frame_start = self.model_preview.state();

        let has_mesh = if self.model_preview.use_custom {
            self.model_preview.custom_mesh.is_some()
        } else {
            true
        };
        let can_compare = self.source.is_some() && self.result_image.is_some();
        let primary_present = match self.model_preview.texture {
            TextureChoice::Result => self.result_image.is_some(),
            TextureChoice::Original => self.source.is_some(),
        };

        ui.horizontal(|ui| {
            ui.label("Фигура");
            let custom_name = self.model_preview.custom_name.clone();
            let has_custom = self.model_preview.custom_mesh.is_some();
            let mut use_custom = self.model_preview.use_custom;
            let mut builtin = self.model_preview.builtin;
            let selected_text = if use_custom {
                custom_name
                    .clone()
                    .unwrap_or_else(|| "Своя модель".to_owned())
            } else {
                builtin.label().to_owned()
            };
            egui::ComboBox::from_id_salt("model-shape")
                .selected_text(selected_text)
                .show_ui(ui, |ui| {
                    for shape in BuiltinShape::ALL {
                        if ui
                            .selectable_label(!use_custom && builtin == shape, shape.label())
                            .clicked()
                        {
                            use_custom = false;
                            builtin = shape;
                        }
                    }
                    if has_custom
                        && ui
                            .selectable_label(
                                use_custom,
                                custom_name.as_deref().unwrap_or("Своя модель"),
                            )
                            .clicked()
                    {
                        use_custom = true;
                    }
                });
            if use_custom != self.model_preview.use_custom || builtin != self.model_preview.builtin
            {
                self.model_preview.use_custom = use_custom;
                self.model_preview.builtin = builtin;
                self.model_preview.dirty = true;
            }
            if ui
                .add_enabled(
                    !self.model_preview.loading,
                    egui::Button::new("Загрузить модель…"),
                )
                .on_hover_text("OBJ, GLB, glTF или FBX. Модель также можно перетащить в окно.")
                .clicked()
            {
                load_requested = true;
            }
            if self.model_preview.loading {
                ui.spinner();
            }
            ui.separator();
            if ui
                .add_enabled(self.model_preview.can_undo(), egui::Button::new("Отменить"))
                .on_hover_text("Ctrl+Z")
                .clicked()
            {
                undo_requested = true;
            }
            if ui
                .add_enabled(
                    self.model_preview.can_redo(),
                    egui::Button::new("Повторить"),
                )
                .on_hover_text("Ctrl+Shift+Z / Ctrl+Y")
                .clicked()
            {
                redo_requested = true;
            }
        });

        if undo_requested {
            self.model_preview.undo();
            history_navigated = true;
        } else if redo_requested {
            self.model_preview.redo();
            history_navigated = true;
        }

        ui.horizontal(|ui| {
            ui.label("Текстура");
            let mut texture = self.model_preview.texture;
            egui::ComboBox::from_id_salt("model-texture")
                .selected_text(texture.label())
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut texture, TextureChoice::Result, "Результат");
                    ui.selectable_value(&mut texture, TextureChoice::Original, "Исходник");
                });
            if texture != self.model_preview.texture {
                self.model_preview.texture = texture;
                self.model_preview.dirty = true;
            }
            if ui
                .add_enabled(
                    can_compare,
                    egui::Checkbox::new(&mut self.model_preview.compare, "Сравнить"),
                )
                .on_hover_text("Показать результат и исходник по разные стороны разделителя.")
                .changed()
            {
                self.model_preview.dirty = true;
            }
            if !can_compare && self.model_preview.compare {
                self.model_preview.compare = false;
                self.model_preview.dirty = true;
            }
            if self.model_preview.compare {
                ui.separator();
                ui.label("Разделитель");
                if ui
                    .add(
                        egui::Slider::new(&mut self.model_preview.compare_split, 0.05..=0.95)
                            .show_value(false),
                    )
                    .changed()
                {
                    self.model_preview.dirty = true;
                    self.model_preview.touch();
                }
            }
        });

        ui.horizontal(|ui| {
            ui.label("Фильтр");
            let mut filter = self.model_preview.filter;
            egui::ComboBox::from_id_salt("model-filter")
                .selected_text(match filter {
                    SampleFilter::Nearest => "Nearest",
                    SampleFilter::Linear => "Linear",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut filter, SampleFilter::Nearest, "Nearest");
                    ui.selectable_value(&mut filter, SampleFilter::Linear, "Linear");
                });
            if filter != self.model_preview.filter {
                self.model_preview.filter = filter;
                self.model_preview.dirty = true;
            }
            if ui
                .checkbox(&mut self.model_preview.lighting, "Освещение")
                .changed()
            {
                self.model_preview.dirty = true;
            }
            if ui
                .checkbox(&mut self.model_preview.auto_rotate, "Авто")
                .on_hover_text("Автоматическое вращение. Пробел — пауза.")
                .changed()
            {
                self.model_preview.dirty = true;
            }
            if self.model_preview.auto_rotate
                && ui
                    .add(
                        egui::DragValue::new(&mut self.model_preview.auto_rotate_speed)
                            .range(0.1..=8.0)
                            .speed(0.05)
                            .suffix(" рад/с"),
                    )
                    .changed()
            {
                self.model_preview.touch();
            }
            if ui.button("Сбросить вид").on_hover_text("R").clicked() {
                reset_requested = true;
            }
            if ui
                .add_enabled(
                    has_mesh && primary_present,
                    egui::Button::new("Сохранить кадр…"),
                )
                .clicked()
            {
                save_requested = true;
            }
            egui::ComboBox::from_id_salt("model-save-size")
                .selected_text(format!("{} px", self.model_preview.save_resolution))
                .show_ui(ui, |ui| {
                    for size in [512, 1024, 2048] {
                        ui.selectable_value(
                            &mut self.model_preview.save_resolution,
                            size,
                            format!("{size} px"),
                        );
                    }
                });
        });

        let mut uv_changed = false;
        egui::CollapsingHeader::new("UV-развёртка и свет")
            .default_open(false)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Повтор U");
                    uv_changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.model_preview.uv_tiling[0])
                                .range(0.1..=16.0)
                                .speed(0.1),
                        )
                        .changed();
                    ui.label("V");
                    uv_changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.model_preview.uv_tiling[1])
                                .range(0.1..=16.0)
                                .speed(0.1),
                        )
                        .changed();
                    ui.separator();
                    ui.label("Смещение U");
                    uv_changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.model_preview.uv_offset[0])
                                .range(-1.0..=1.0)
                                .speed(0.01),
                        )
                        .changed();
                    ui.label("V");
                    uv_changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.model_preview.uv_offset[1])
                                .range(-1.0..=1.0)
                                .speed(0.01),
                        )
                        .changed();
                });
                ui.horizontal(|ui| {
                    ui.label("Поворот UV");
                    let mut degrees = self.model_preview.uv_rotation.to_degrees();
                    if ui
                        .add(egui::Slider::new(&mut degrees, -180.0..=180.0).suffix("°"))
                        .changed()
                    {
                        self.model_preview.uv_rotation = degrees.to_radians();
                        uv_changed = true;
                    }
                    ui.separator();
                    ui.label("Wrap");
                    let mut wrap = self.model_preview.wrap;
                    egui::ComboBox::from_id_salt("model-wrap")
                        .selected_text(match wrap {
                            WrapMode::Repeat => "Repeat",
                            WrapMode::Clamp => "Clamp",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut wrap, WrapMode::Repeat, "Repeat");
                            ui.selectable_value(&mut wrap, WrapMode::Clamp, "Clamp");
                        });
                    if wrap != self.model_preview.wrap {
                        self.model_preview.wrap = wrap;
                        uv_changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Свет: азимут");
                    let mut azimuth = self.model_preview.light_azimuth.to_degrees();
                    if ui
                        .add(egui::Slider::new(&mut azimuth, -180.0..=180.0).suffix("°"))
                        .changed()
                    {
                        self.model_preview.light_azimuth = azimuth.to_radians();
                        uv_changed = true;
                    }
                    ui.label("высота");
                    let mut elevation = self.model_preview.light_elevation.to_degrees();
                    if ui
                        .add(egui::Slider::new(&mut elevation, -90.0..=90.0).suffix("°"))
                        .changed()
                    {
                        self.model_preview.light_elevation = elevation.to_radians();
                        uv_changed = true;
                    }
                });
            });
        if uv_changed {
            self.model_preview.dirty = true;
            self.model_preview.touch();
        }

        if !self.model_preview.status.is_empty() {
            ui.label(
                egui::RichText::new(&self.model_preview.status)
                    .small()
                    .weak(),
            );
        }
        if !self.model_preview.save_status.is_empty() {
            ui.label(
                egui::RichText::new(&self.model_preview.save_status)
                    .small()
                    .weak(),
            );
        }

        let hint_height = 26.0;
        let available = ui.available_size();
        let size = egui::vec2(
            available.x.max(160.0),
            (available.y - hint_height).max(160.0),
        );
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());

        if response.hovered() {
            let (scroll, zoom_delta) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let factor = if (zoom_delta - 1.0).abs() > 1e-4 {
                zoom_delta
            } else if scroll.abs() > 0.0 {
                (scroll * 0.0015).exp()
            } else {
                1.0
            };
            if factor != 1.0 {
                self.model_preview.zoom = (self.model_preview.zoom * factor).clamp(0.25, 5.0);
                self.model_preview.dirty = true;
                self.model_preview.touch();
            }
        }

        if response.drag_started() {
            if let Some(pos) = response.interact_pointer_pos() {
                let split_x = rect.left() + rect.width() * self.model_preview.compare_split;
                let mode = if self.model_preview.compare && (pos.x - split_x).abs() <= 8.0 {
                    DragMode::Split
                } else if ui.input(|i| {
                    i.modifiers.shift || i.pointer.button_down(egui::PointerButton::Secondary)
                }) {
                    DragMode::Pan
                } else {
                    DragMode::Rotate
                };
                self.model_preview.drag_origin = Some(DragOrigin {
                    pos,
                    mode,
                    yaw: self.model_preview.yaw,
                    pitch: self.model_preview.pitch,
                    pan: self.model_preview.pan,
                    split: self.model_preview.compare_split,
                });
            }
        }
        if response.dragged() {
            if let (Some(origin), Some(pos)) = (
                self.model_preview.drag_origin,
                response.interact_pointer_pos(),
            ) {
                let dx = pos.x - origin.pos.x;
                let dy = pos.y - origin.pos.y;
                match origin.mode {
                    DragMode::Rotate => {
                        self.model_preview.yaw = origin.yaw + dx * 0.012;
                        self.model_preview.pitch = (origin.pitch + dy * 0.012).clamp(
                            -std::f32::consts::FRAC_PI_2 + 0.05,
                            std::f32::consts::FRAC_PI_2 - 0.05,
                        );
                    }
                    DragMode::Pan => {
                        self.model_preview.pan = [
                            (origin.pan[0] + dx / rect.width().max(1.0)).clamp(-2.0, 2.0),
                            (origin.pan[1] + dy / rect.height().max(1.0)).clamp(-2.0, 2.0),
                        ];
                    }
                    DragMode::Split => {
                        self.model_preview.compare_split =
                            (origin.split + dx / rect.width().max(1.0)).clamp(0.05, 0.95);
                    }
                }
                self.model_preview.dirty = true;
                self.model_preview.touch();
            }
        }
        if response.drag_stopped() {
            self.model_preview.drag_origin = None;
            self.model_preview.touch();
        }
        if response.double_clicked() {
            reset_requested = true;
        }
        if response.hovered() && !ctx.wants_keyboard_input() {
            if ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::R)) {
                reset_requested = true;
            }
            if ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Space)) {
                self.model_preview.auto_rotate = !self.model_preview.auto_rotate;
                self.model_preview.dirty = true;
            }
        }
        if response.hovered() {
            let over_split = self.model_preview.compare
                && response.hover_pos().is_some_and(|pos| {
                    let split_x = rect.left() + rect.width() * self.model_preview.compare_split;
                    (pos.x - split_x).abs() <= 8.0
                });
            let cursor = if over_split {
                egui::CursorIcon::ResizeHorizontal
            } else if response.dragged() {
                egui::CursorIcon::Grabbing
            } else {
                egui::CursorIcon::Grab
            };
            ui.ctx().set_cursor_icon(cursor);
        }

        if load_requested {
            self.start_model_load();
        }
        if reset_requested {
            self.model_preview.reset_view();
            self.model_preview.touch();
        }

        let primary_choice = self.model_preview.texture;
        let primary_image = match primary_choice {
            TextureChoice::Result => self.result_image.clone(),
            TextureChoice::Original => self.source.clone(),
        };
        let secondary_choice = primary_choice.opposite();
        let secondary_image = if self.model_preview.compare {
            match secondary_choice {
                TextureChoice::Result => self.result_image.clone(),
                TextureChoice::Original => self.source.clone(),
            }
        } else {
            None
        };

        if save_requested {
            if let Some(primary) = &primary_image {
                let file_name = self
                    .source_path
                    .as_deref()
                    .and_then(|path| path.file_stem())
                    .and_then(|stem| stem.to_str())
                    .map_or_else(
                        || "model-preview.png".to_owned(),
                        |stem| format!("{stem}_3d.png"),
                    );
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("PNG image", &["png"])
                    .set_file_name(file_name)
                    .save_file()
                {
                    self.model_preview.request_save(
                        primary,
                        primary_choice,
                        secondary_image
                            .as_ref()
                            .map(|image| (image, secondary_choice)),
                        save_target(rect, self.model_preview.save_resolution),
                        path,
                    );
                }
            }
        }

        if let Some(primary) = &primary_image {
            if has_mesh {
                let ppp = ctx.pixels_per_point();
                let full = fit_within(
                    [
                        ((rect.width() * ppp).round() as usize).max(32),
                        ((rect.height() * ppp).round() as usize).max(32),
                    ],
                    2048,
                );
                let recently = self
                    .model_preview
                    .last_interaction
                    .is_some_and(|time| time.elapsed() < Duration::from_millis(160));
                let interactive = self.model_preview.auto_rotate || recently;
                let target = if interactive {
                    fit_within(
                        [
                            (full[0] as f32 * 0.5).round() as usize,
                            (full[1] as f32 * 0.5).round() as usize,
                        ],
                        512,
                    )
                } else {
                    full
                };
                self.model_preview.request_render_if_needed(
                    primary,
                    primary_choice,
                    secondary_image
                        .as_ref()
                        .map(|image| (image, secondary_choice)),
                    target,
                );
                if interactive || self.model_preview.pending_render.is_some() {
                    ctx.request_repaint_after(Duration::from_millis(16));
                }
            }
        }

        let painter = ui.painter().with_clip_rect(rect);
        paint_checkerboard(&painter, rect, ui.visuals().dark_mode);
        if primary_image.is_some() && has_mesh {
            if let Some(handle) = &self.model_preview.texture_handle {
                painter.image(
                    handle.id(),
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            }
        }
        if self.model_preview.compare && secondary_image.is_some() {
            let x = rect.left() + rect.width() * self.model_preview.compare_split;
            let stroke = egui::Stroke::new(2.0_f32, egui::Color32::WHITE);
            painter.line_segment(
                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                stroke,
            );
            painter.circle_filled(
                egui::pos2(x, rect.center().y),
                6.0,
                ui.visuals().widgets.active.bg_fill,
            );
        }
        if primary_image.is_some() && has_mesh {
            painter.text(
                rect.right_top() + egui::vec2(-8.0, 8.0),
                egui::Align2::RIGHT_TOP,
                format!("{:.0}%", self.model_preview.zoom * 100.0),
                egui::FontId::monospace(12.0),
                egui::Color32::from_white_alpha(200),
            );
        }
        let placeholder = if primary_image.is_none() {
            if self.source.is_some() {
                Some("Текстура ещё не готова.")
            } else {
                Some("Откройте текстуру, чтобы увидеть её на модели.")
            }
        } else if !has_mesh {
            Some("Загрузите модель или выберите встроенную фигуру.")
        } else {
            None
        };
        if let Some(text) = placeholder {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                text,
                egui::FontId::proportional(15.0),
                ui.visuals().text_color(),
            );
        }
        painter.rect_stroke(
            rect,
            0.0,
            egui::Stroke::new(1.0_f32, ui.visuals().widgets.noninteractive.bg_stroke.color),
            egui::StrokeKind::Inside,
        );

        ui.horizontal(|ui| {
            let triangles = self.model_preview.triangle_count();
            if triangles > 0 {
                ui.label(format!("Треугольников: {triangles}"));
                ui.separator();
            }
            ui.label(
                "ЛКМ — вращать, ПКМ/Shift — панорама, колесо — зум, R — сброс, пробел — авто, Ctrl+Z — отмена.",
            );
        });

        // Collapse a whole drag or slider movement into one undo step, exactly like the editor.
        if !history_navigated
            && self.model_preview.pending_change.is_none()
            && frame_start != self.model_preview.state()
        {
            self.model_preview.pending_change = Some(frame_start);
        }
        if !ctx.is_using_pointer() {
            self.model_preview.commit_pending_change();
        }
    }
}

fn fit_within([width, height]: [usize; 2], max_side: usize) -> [usize; 2] {
    let width = width.max(1);
    let height = height.max(1);
    let longest = width.max(height);
    if longest <= max_side {
        [width, height]
    } else {
        let scale = max_side as f32 / longest as f32;
        [
            ((width as f32 * scale).round() as usize).max(1),
            ((height as f32 * scale).round() as usize).max(1),
        ]
    }
}

fn save_target(rect: egui::Rect, resolution: usize) -> [usize; 2] {
    let aspect = rect.width().max(1.0) / rect.height().max(1.0);
    let resolution = resolution.max(32);
    if aspect >= 1.0 {
        [
            resolution,
            ((resolution as f32 / aspect).round() as usize).max(32),
        ]
    } else {
        [
            ((resolution as f32 * aspect).round() as usize).max(32),
            resolution,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_history_undoes_and_redoes_view_changes() {
        let mut preview = ModelPreview::new(egui::Context::default());
        let original = preview.state();

        preview.yaw = 1.5;
        preview.filter = SampleFilter::Linear;
        preview.pending_change = Some(original.clone());
        preview.commit_pending_change();

        assert!(preview.can_undo());
        assert!(preview.undo());
        assert_eq!(preview.state(), original);
        assert!(preview.can_redo());

        assert!(preview.redo());
        assert_eq!(preview.yaw, 1.5);
        assert_eq!(preview.filter, SampleFilter::Linear);
    }

    #[test]
    fn preview_applies_intermediate_render_results() {
        let mut preview = ModelPreview::new(egui::Context::default());
        preview.latest_render_id = 5;

        assert!(preview.should_apply_render(3));
        assert!(!preview.should_apply_render(2));
        assert!(preview.should_apply_render(5));
        assert!(!preview.should_apply_render(4));
    }

    #[test]
    fn preview_history_clears_redo_after_a_new_change() {
        let mut preview = ModelPreview::new(egui::Context::default());
        let original = preview.state();
        preview.zoom = 2.0;
        preview.pending_change = Some(original);
        preview.commit_pending_change();
        assert!(preview.undo());
        assert!(preview.can_redo());

        let baseline = preview.state();
        preview.zoom = 3.0;
        preview.pending_change = Some(baseline);
        preview.commit_pending_change();

        assert!(!preview.can_redo());
        assert!(preview.undo());
        assert_eq!(preview.zoom, 1.0);
    }
}
