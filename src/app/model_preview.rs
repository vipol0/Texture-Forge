use crate::app::preview::paint_checkerboard;
use crate::app::TextureForge;
use crate::preview3d::loader::load_model;
use crate::preview3d::mesh::{BuiltinShape, Mesh3d};
use crate::preview3d::raster::{render, RenderOptions, SampleFilter};
use eframe::egui;
use image::{DynamicImage, RgbaImage};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::Duration;

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
}

struct TextureCache {
    image: Arc<DynamicImage>,
    choice: TextureChoice,
    rgba: Arc<RgbaImage>,
}

pub(crate) struct ModelPreview {
    builtin: BuiltinShape,
    use_custom: bool,
    custom_mesh: Option<Arc<Mesh3d>>,
    custom_name: Option<String>,
    builtin_cache: Option<(BuiltinShape, Arc<Mesh3d>)>,
    texture: TextureChoice,
    texture_cache: Option<TextureCache>,
    tiling: u32,
    filter: SampleFilter,
    lighting: bool,
    yaw: f32,
    pitch: f32,
    zoom: f32,
    drag_origin: Option<(egui::Pos2, f32, f32)>,
    texture_handle: Option<egui::TextureHandle>,
    rendered_size: [usize; 2],
    dirty: bool,
    loading: bool,
    load_rx: Option<Receiver<Result<(Mesh3d, String), String>>>,
    status: String,
}

impl Default for ModelPreview {
    fn default() -> Self {
        Self {
            builtin: BuiltinShape::Cube,
            use_custom: false,
            custom_mesh: None,
            custom_name: None,
            builtin_cache: None,
            texture: TextureChoice::Result,
            texture_cache: None,
            tiling: 1,
            filter: SampleFilter::Nearest,
            lighting: true,
            yaw: 0.55,
            pitch: -0.3,
            zoom: 1.0,
            drag_origin: None,
            texture_handle: None,
            rendered_size: [0, 0],
            dirty: true,
            loading: false,
            load_rx: None,
            status: String::new(),
        }
    }
}

impl ModelPreview {
    pub(crate) fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    fn reset_view(&mut self) {
        self.yaw = 0.55;
        self.pitch = -0.3;
        self.zoom = 1.0;
        self.drag_origin = None;
        self.dirty = true;
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

    fn render_if_needed(
        &mut self,
        ctx: &egui::Context,
        image: &Arc<DynamicImage>,
        target: [usize; 2],
    ) {
        let cache_matches = self
            .texture_cache
            .as_ref()
            .is_some_and(|cache| Arc::ptr_eq(&cache.image, image) && cache.choice == self.texture);
        if !cache_matches {
            self.texture_cache = Some(TextureCache {
                image: image.clone(),
                choice: self.texture,
                rgba: Arc::new(image.to_rgba8()),
            });
            self.dirty = true;
        }
        if !self.dirty && self.rendered_size == target {
            return;
        }
        let Some(mesh) = self.active_mesh() else {
            return;
        };
        let Some(cache) = self.texture_cache.as_ref() else {
            return;
        };

        let options = RenderOptions {
            yaw: self.yaw,
            pitch: self.pitch,
            zoom: self.zoom,
            tiling: self.tiling as f32,
            lighting: self.lighting,
            filter: self.filter,
        };
        let rgba = render(&mesh, &cache.rgba, target[0], target[1], options);
        let color = egui::ColorImage::from_rgba_unmultiplied(target, rgba.as_raw());
        if let Some(handle) = &mut self.texture_handle {
            handle.set(color, egui::TextureOptions::NEAREST);
        } else {
            self.texture_handle =
                Some(ctx.load_texture("model-preview", color, egui::TextureOptions::NEAREST));
        }
        self.rendered_size = target;
        self.dirty = false;
    }
}

impl TextureForge {
    pub(crate) fn poll_model_load(&mut self, ctx: &egui::Context) {
        let message = self.model_preview.load_rx.as_ref().map(|rx| rx.try_recv());
        match message {
            Some(Ok(Ok((mesh, name)))) => {
                self.model_preview.status = format!(
                    "Модель «{name}» загружена: {} треугольников.",
                    mesh.triangle_count()
                );
                self.model_preview.custom_mesh = Some(Arc::new(mesh));
                self.model_preview.custom_name = Some(name);
                self.model_preview.use_custom = true;
                self.model_preview.loading = false;
                self.model_preview.load_rx = None;
                self.model_preview.dirty = true;
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

    fn start_model_load(&mut self) {
        if self.model_preview.loading {
            return;
        }
        let Some(path) = rfd::FileDialog::new()
            .add_filter("3D models", &["obj", "glb", "gltf", "fbx"])
            .pick_file()
        else {
            return;
        };
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
        egui::Window::new("Предпросмотр на модели")
            .open(&mut open)
            .default_size([560.0, 640.0])
            .min_size([360.0, 360.0])
            .resizable(true)
            .show(ctx, |ui| self.model_preview_ui(ctx, ui));
        if !open {
            self.show_model_preview = false;
            self.model_preview.texture_cache = None;
            self.model_preview.texture_handle = None;
            self.model_preview.rendered_size = [0, 0];
            self.model_preview.dirty = true;
        }
    }

    fn model_preview_ui(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let mut load_requested = false;
        let mut reset_requested = false;

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
                .on_hover_text("OBJ, GLB, glTF или FBX")
                .clicked()
            {
                load_requested = true;
            }
            if self.model_preview.loading {
                ui.spinner();
            }
        });

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
            ui.separator();
            ui.label("Повтор");
            if ui
                .add(egui::Slider::new(&mut self.model_preview.tiling, 1..=8).suffix("×"))
                .changed()
            {
                self.model_preview.dirty = true;
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
            if ui.button("Сбросить вид").clicked() {
                reset_requested = true;
            }
        });

        if !self.model_preview.status.is_empty() {
            ui.label(
                egui::RichText::new(&self.model_preview.status)
                    .small()
                    .weak(),
            );
        }

        let side = ui
            .available_width()
            .min((ui.available_height() - 34.0).max(160.0))
            .max(160.0);
        let (rect, response) =
            ui.allocate_exact_size(egui::vec2(side, side), egui::Sense::click_and_drag());

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
            }
        }
        if response.drag_started() {
            if let Some(pos) = response.interact_pointer_pos() {
                self.model_preview.drag_origin =
                    Some((pos, self.model_preview.yaw, self.model_preview.pitch));
            }
        }
        if response.dragged() {
            if let (Some((origin, yaw, pitch)), Some(pos)) = (
                self.model_preview.drag_origin,
                response.interact_pointer_pos(),
            ) {
                self.model_preview.yaw = yaw + (pos.x - origin.x) * 0.012;
                self.model_preview.pitch = (pitch + (pos.y - origin.y) * 0.012).clamp(
                    -std::f32::consts::FRAC_PI_2 + 0.05,
                    std::f32::consts::FRAC_PI_2 - 0.05,
                );
                self.model_preview.dirty = true;
            }
        }
        if response.drag_stopped() {
            self.model_preview.drag_origin = None;
        }
        if response.double_clicked() {
            reset_requested = true;
        }
        if response.hovered() {
            let cursor = if response.dragged() {
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
        }

        let image = match self.model_preview.texture {
            TextureChoice::Result => self.result_image.clone(),
            TextureChoice::Original => self.source.clone(),
        };
        let has_mesh = if self.model_preview.use_custom {
            self.model_preview.custom_mesh.is_some()
        } else {
            true
        };
        if let Some(image) = &image {
            if has_mesh {
                let ppp = ctx.pixels_per_point();
                let target = ((side * ppp).round() as usize).clamp(32, 1024);
                self.model_preview
                    .render_if_needed(ctx, image, [target, target]);
            }
        }

        let painter = ui.painter().with_clip_rect(rect);
        paint_checkerboard(&painter, rect, ui.visuals().dark_mode);
        if image.is_some() && has_mesh {
            if let Some(handle) = &self.model_preview.texture_handle {
                painter.image(
                    handle.id(),
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            }
        }
        let placeholder = if image.is_none() {
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
            ui.label("ЛКМ — вращать, колесо — зум, двойной клик — сброс.");
        });
    }
}
