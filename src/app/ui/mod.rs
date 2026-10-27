mod batch_window;
mod curve_editor;
mod settings_panel;

use crate::app::preview::{draw_preview_pane, CropEdit};
use crate::app::TextureForge;
use eframe::egui;

impl eframe::App for TextureForge {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_renderer(ctx);
        self.poll_model_load(ctx);
        self.poll_model_preview(ctx);
        self.handle_dropped_files(ctx);

        // Keep the spinner in the status bar animating while a render is in progress, and the
        // batch progress bar updating while files are being processed.
        if self.renderer.in_flight || self.batch.running {
            ctx.request_repaint();
        }

        let frame_start = self.settings.clone();

        let undo_shortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Z);
        let redo_shift_shortcut = egui::KeyboardShortcut::new(
            egui::Modifiers::COMMAND.plus(egui::Modifiers::SHIFT),
            egui::Key::Z,
        );
        let redo_alt_shortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Y);
        let mut navigated = false;
        // The preview window owns undo/redo while the pointer is over it; it handles the
        // shortcuts itself so the editor history is not modified at the same time.
        if !self.model_preview.shortcuts_active(ctx) {
            // `consume_shortcut` matches logically and ignores extra Shift, so Ctrl+Shift+Z must
            // be checked before Ctrl+Z or the redo shortcut would be swallowed by undo.
            if ctx.input_mut(|i| i.consume_shortcut(&redo_shift_shortcut)) {
                self.redo_action(ctx);
                navigated = true;
            } else if ctx.input_mut(|i| i.consume_shortcut(&undo_shortcut)) {
                self.undo_action(ctx);
                navigated = true;
            } else if ctx.input_mut(|i| i.consume_shortcut(&redo_alt_shortcut)) {
                self.redo_action(ctx);
                navigated = true;
            }
        }

        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Texture Forge");
                ui.separator();
                if ui.button("Открыть текстуру").clicked() {
                    self.open_image(ctx);
                }
                if ui.button("Сохранить результат").clicked() {
                    self.save_result();
                }
                if ui.button("Пакетная обработка…").clicked() {
                    self.show_batch = true;
                }
                if ui.button("3D-предпросмотр…").clicked() {
                    self.show_model_preview = true;
                    self.model_preview.mark_dirty();
                }
                ui.separator();
                if ui
                    .add_enabled(self.history.can_undo(), egui::Button::new("Отменить"))
                    .on_hover_text("Ctrl+Z")
                    .clicked()
                {
                    self.undo_action(ctx);
                    navigated = true;
                }
                if ui
                    .add_enabled(self.history.can_redo(), egui::Button::new("Повторить"))
                    .on_hover_text("Ctrl+Shift+Z / Ctrl+Y")
                    .clicked()
                {
                    self.redo_action(ctx);
                    navigated = true;
                }
            });
        });

        egui::SidePanel::left("settings")
            .resizable(false)
            .default_width(240.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("settings-scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        self.show_settings_panel(ctx, ui);
                    });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Предпросмотр: до / после");
            ui.label("Колесо — зум, перетаскивание — панорама, двойной клик — сброс вида, тяните разделитель.");
            ui.label("Файл изображения можно просто перетащить в окно.");
            ui.add_space(8.0);
            if self.source.is_none() {
                ui.centered_and_justified(|ui| {
                    ui.label("Откройте PNG, JPEG, BMP или WebP или перетащите файл в окно.");
                });
            } else {
                let full = ui.available_rect_before_wrap();
                let handle_w = 10.0;
                let split_x = full.left() + full.width() * self.split_fraction;
                let left_rect = egui::Rect::from_min_max(
                    full.min,
                    egui::pos2((split_x - handle_w * 0.5).max(full.left()), full.max.y),
                );
                let right_rect = egui::Rect::from_min_max(
                    egui::pos2((split_x + handle_w * 0.5).min(full.right()), full.min.y),
                    full.max,
                );

                let mut crop = self.settings.crop;
                let crop_changed = draw_preview_pane(
                    ui,
                    left_rect,
                    egui::Id::new("pane-original"),
                    self.preview_original.as_ref(),
                    if self.crop_edit { "Исходник — кроп" } else { "Исходник" },
                    &mut self.view_original,
                    self.crop_edit.then_some(CropEdit {
                        crop: &mut crop,
                        drag: &mut self.crop_drag,
                    }),
                );
                if crop_changed {
                    self.settings.crop = crop.sanitized();
                    self.refresh_previews(ctx);
                }
                draw_preview_pane(
                    ui,
                    right_rect,
                    egui::Id::new("pane-result"),
                    self.preview_result.as_ref(),
                    "Результат",
                    &mut self.view_result,
                    None,
                );

                let handle_rect = egui::Rect::from_center_size(
                    egui::pos2(split_x, full.center().y),
                    egui::vec2(handle_w, full.height()),
                );
                let handle = ui.interact(handle_rect, egui::Id::new("split-handle"), egui::Sense::drag());
                if handle.hovered() || handle.dragged() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                }
                if handle.dragged() {
                    if let Some(pos) = handle.interact_pointer_pos() {
                        self.split_fraction = ((pos.x - full.left()) / full.width()).clamp(0.02, 0.98);
                    }
                }
                let handle_color = if handle.hovered() || handle.dragged() {
                    ui.visuals().widgets.hovered.bg_fill
                } else {
                    ui.visuals().widgets.inactive.bg_fill
                };
                let painter = ui.painter();
                painter.rect_filled(handle_rect, 0.0, handle_color);
                painter.line_segment(
                    [egui::pos2(split_x, full.top()), egui::pos2(split_x, full.bottom())],
                    egui::Stroke::new(1.0_f32, egui::Color32::from_black_alpha(160)),
                );
            }
        });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if self.renderer.in_flight || self.batch.running {
                    ui.spinner();
                    ui.label("Обработка…");
                } else {
                    ui.label(&self.status);
                }
            });
        });

        self.show_batch_window(ctx);
        self.show_model_window(ctx);

        // Show an overlay while files are dragged over the window.
        if ctx.input(|i| !i.raw.hovered_files.is_empty()) {
            ctx.request_repaint();
            let rect = ctx.screen_rect();
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("drop-overlay"),
            ));
            painter.rect_filled(rect, 0.0, egui::Color32::from_black_alpha(160));
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Отпустите файл, чтобы открыть текстуру",
                egui::FontId::proportional(22.0),
                egui::Color32::WHITE,
            );
        }

        // Capture the state that preceded an in-progress interaction, then commit it once the
        // pointer is released. This collapses a whole slider drag into one undo step.
        if !navigated && self.pending.is_none() && frame_start != self.settings {
            self.pending = Some(frame_start);
        }
        if !ctx.is_using_pointer() {
            self.commit_pending();
        }
    }
}
