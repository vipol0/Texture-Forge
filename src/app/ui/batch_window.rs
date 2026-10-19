use crate::app::TextureForge;
use eframe::egui;

impl TextureForge {
    pub(super) fn show_batch_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_batch;
        egui::Window::new("Пакетная обработка")
            .open(&mut open)
            .default_width(480.0)
            .collapsible(false)
            .resizable(true)
            .show(ctx, |ui| {
                let idle = !self.batch.running;
                ui.horizontal(|ui| {
                    ui.add_enabled_ui(idle, |ui| {
                        if ui.button("Добавить файлы").clicked() {
                            self.add_batch_files();
                        }
                        if ui.button("Добавить папку").clicked() {
                            self.add_batch_folder();
                        }
                        if ui.button("Очистить").clicked() {
                            self.batch.clear();
                        }
                    });
                });
                ui.label(format!("Файлов в очереди: {}", self.batch.inputs.len()));
                egui::ScrollArea::vertical()
                    .id_salt("batch-inputs")
                    .max_height(120.0)
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                        for path in &self.batch.inputs {
                            ui.label(
                                path.file_name()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or_else(|| path.to_str().unwrap_or("?")),
                            );
                        }
                    });
                ui.separator();
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(idle, egui::Button::new("Папка вывода…"))
                        .clicked()
                    {
                        if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                            self.batch.output_dir = Some(dir);
                        }
                    }
                    match &self.batch.output_dir {
                        Some(dir) => ui.label(dir.display().to_string()),
                        None => ui.label("не выбрана"),
                    };
                });
                ui.horizontal(|ui| {
                    ui.label("Приписка:");
                    ui.add_enabled(
                        idle,
                        egui::TextEdit::singleline(&mut self.export_suffix)
                            .hint_text("_stylized")
                            .desired_width(140.0),
                    );
                });
                ui.label("Приписка общая для одиночного и пакетного экспорта.");
                ui.separator();
                let can_start = self.batch.can_start();
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(can_start, egui::Button::new("Обработать"))
                        .clicked()
                    {
                        self.start_batch();
                    }
                    if ui
                        .add_enabled(self.batch.running, egui::Button::new("Отменить"))
                        .clicked()
                    {
                        self.cancel_batch();
                    }
                });
                if self.batch.running || self.batch.finished {
                    let total = self.batch.total.max(1);
                    let fraction = self.batch.done as f32 / total as f32;
                    ui.add(egui::ProgressBar::new(fraction).show_percentage());
                    if self.batch.running {
                        ui.label(format!(
                            "{} / {}: {}",
                            self.batch.done, self.batch.total, self.batch.current
                        ));
                    } else if self.batch.cancelled {
                        ui.label(format!("Отменено. Сохранено: {}", self.batch.saved));
                    } else {
                        ui.label(format!("Готово. Сохранено: {}", self.batch.saved));
                    }
                    if !self.batch.errors.is_empty() {
                        ui.collapsing(format!("Ошибки ({})", self.batch.errors.len()), |ui| {
                            for error in &self.batch.errors {
                                ui.colored_label(ui.visuals().error_fg_color, error);
                            }
                        });
                    }
                }
                ui.add_space(4.0);
                ui.label(
                    "Используются текущие настройки и формат экспорта. \
                     Исходные файлы не перезаписываются.",
                );
            });
        self.show_batch = open;
    }
}
