use crate::app::ui::curve_editor::curve_editor;
use crate::imaging::dither::DitherMode;
use crate::model::export::OutputFormat;
use crate::model::geometry::{crop_pixel_rect, CropRect, ResizeFilter, ResizeMode};
use crate::model::settings::{default_curves, AlphaMode, PaletteMethod, RESOLUTION_PRESETS};
use eframe::egui;

impl crate::app::TextureForge {
    pub(super) fn show_settings_panel(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        ui.heading("Обработка");
        ui.add_space(8.0);
        ui.label("Пресет");
        let presets = self.presets_cache.clone();
        let mut selection = self.selected_preset.clone();
        let selected_text = selection
            .clone()
            .unwrap_or_else(|| "— выберите пресет —".to_owned());
        let mut apply_requested = false;
        egui::ComboBox::from_id_salt("preset")
            .selected_text(selected_text)
            .show_ui(ui, |ui| {
                for preset in &presets {
                    if ui
                        .selectable_label(
                            selection.as_deref() == Some(preset.name.as_str()),
                            &preset.name,
                        )
                        .clicked()
                    {
                        selection = Some(preset.name.clone());
                        apply_requested = true;
                    }
                }
            });
        if apply_requested {
            if let Some(preset) = selection
                .as_ref()
                .and_then(|name| presets.iter().find(|p| &p.name == name))
                .cloned()
            {
                self.apply_preset(&preset, ctx);
            }
        }
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.preset_name_input)
                    .hint_text("Имя пресета")
                    .desired_width(120.0),
            );
            if ui.button("Сохранить").clicked() {
                self.save_current_preset();
            }
        });
        if ui
            .add_enabled(
                selection.is_some(),
                egui::Button::new("Удалить выбранный пресет"),
            )
            .clicked()
        {
            self.delete_selected_preset();
        }
        ui.separator();
        ui.label("Геометрия");
        let (src_w, src_h) = self
            .source
            .as_ref()
            .map_or((1, 1), |img| (img.width().max(1), img.height().max(1)));
        let has_source = self.source.is_some();
        let mut crop_edit = self.crop_edit;
        if ui
            .add_enabled(
                has_source,
                egui::Checkbox::new(&mut crop_edit, "Ручной кроп"),
            )
            .on_hover_text("Перетаскивайте рамку и маркеры на панели «Исходник»")
            .changed()
        {
            self.crop_edit = crop_edit;
            if !crop_edit {
                self.crop_drag = None;
            }
        }
        let (crop_x, crop_y, crop_w, crop_h) = crop_pixel_rect(self.settings.crop, src_w, src_h);
        let mut px = [crop_x, crop_y, crop_w, crop_h];
        let mut crop_px_changed = false;
        ui.add_enabled_ui(has_source, |ui| {
            ui.horizontal(|ui| {
                ui.label("X");
                crop_px_changed |= ui
                    .add(
                        egui::DragValue::new(&mut px[0])
                            .range(0..=src_w.saturating_sub(1))
                            .speed(0.5),
                    )
                    .changed();
                ui.label("Y");
                crop_px_changed |= ui
                    .add(
                        egui::DragValue::new(&mut px[1])
                            .range(0..=src_h.saturating_sub(1))
                            .speed(0.5),
                    )
                    .changed();
            });
            ui.horizontal(|ui| {
                ui.label("Ш");
                crop_px_changed |= ui
                    .add(egui::DragValue::new(&mut px[2]).range(1..=src_w).speed(0.5))
                    .changed();
                ui.label("В");
                crop_px_changed |= ui
                    .add(egui::DragValue::new(&mut px[3]).range(1..=src_h).speed(0.5))
                    .changed();
            });
        });
        if crop_px_changed {
            px[0] = px[0].min(src_w - 1);
            px[1] = px[1].min(src_h - 1);
            px[2] = px[2].clamp(1, src_w - px[0]);
            px[3] = px[3].clamp(1, src_h - px[1]);
            self.settings.crop = CropRect {
                x: px[0] as f32 / src_w as f32,
                y: px[1] as f32 / src_h as f32,
                w: px[2] as f32 / src_w as f32,
                h: px[3] as f32 / src_h as f32,
            }
            .sanitized();
            self.refresh_previews(ctx);
        }
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    has_source && !self.settings.crop.is_full(),
                    egui::Button::new("Весь кадр"),
                )
                .clicked()
            {
                self.settings.crop = CropRect::FULL;
                self.crop_drag = None;
                self.refresh_previews(ctx);
            }
            ui.label(format!("Кроп: {crop_w} × {crop_h} px"));
        });

        let mut output_enabled = self.settings.output_size.is_some();
        if ui
            .checkbox(&mut output_enabled, "Задать размер результата")
            .changed()
        {
            self.settings.output_size = if output_enabled {
                if has_source {
                    Some([crop_w.max(1), crop_h.max(1)])
                } else {
                    Some([256, 256])
                }
            } else {
                None
            };
            self.refresh_previews(ctx);
        }
        if let Some([out_w, out_h]) = self.settings.output_size {
            let mut ow = out_w.max(1);
            let mut oh = out_h.max(1);
            let mut size_changed = false;
            ui.horizontal(|ui| {
                ui.label("W");
                size_changed |= ui
                    .add(egui::DragValue::new(&mut ow).range(1..=16384).speed(1.0))
                    .changed();
                ui.label("H");
                size_changed |= ui
                    .add(egui::DragValue::new(&mut oh).range(1..=16384).speed(1.0))
                    .changed();
            });
            if size_changed {
                self.settings.output_size = Some([ow.max(1), oh.max(1)]);
                self.refresh_previews(ctx);
            }
        }
        ui.add_enabled_ui(self.settings.output_size.is_some(), |ui| {
            ui.label("Режим ресайза");
            let mut mode_changed = false;
            egui::ComboBox::from_id_salt("resize_mode")
                .selected_text(match self.settings.resize_mode {
                    ResizeMode::Fit => "Fit (вписать)",
                    ResizeMode::Fill => "Fill (заполнить)",
                    ResizeMode::Stretch => "Stretch (растянуть)",
                })
                .show_ui(ui, |ui| {
                    mode_changed |= ui
                        .selectable_value(
                            &mut self.settings.resize_mode,
                            ResizeMode::Fit,
                            "Fit (вписать)",
                        )
                        .changed();
                    mode_changed |= ui
                        .selectable_value(
                            &mut self.settings.resize_mode,
                            ResizeMode::Fill,
                            "Fill (заполнить)",
                        )
                        .changed();
                    mode_changed |= ui
                        .selectable_value(
                            &mut self.settings.resize_mode,
                            ResizeMode::Stretch,
                            "Stretch (растянуть)",
                        )
                        .changed();
                });
            if mode_changed {
                self.refresh_previews(ctx);
            }
        });
        ui.label("Фильтр ресайза");
        let mut filter_changed = false;
        egui::ComboBox::from_id_salt("resize_filter")
            .selected_text(match self.settings.resize_filter {
                ResizeFilter::Nearest => "Nearest (пиксели)",
                ResizeFilter::Lanczos => "Lanczos",
            })
            .show_ui(ui, |ui| {
                filter_changed |= ui
                    .selectable_value(
                        &mut self.settings.resize_filter,
                        ResizeFilter::Nearest,
                        "Nearest (пиксели)",
                    )
                    .changed();
                filter_changed |= ui
                    .selectable_value(
                        &mut self.settings.resize_filter,
                        ResizeFilter::Lanczos,
                        "Lanczos",
                    )
                    .changed();
            });
        if filter_changed {
            self.refresh_previews(ctx);
        }
        if ui
            .add_enabled(
                !self.settings.geometry_is_neutral(),
                egui::Button::new("Сбросить геометрию"),
            )
            .clicked()
        {
            self.settings.reset_geometry();
            self.crop_edit = false;
            self.crop_drag = None;
            self.refresh_previews(ctx);
        }
        ui.separator();
        ui.label("Пикселизация (длинная сторона)")
            .on_hover_text("Рабочее разрешение эффектов; не задаёт размер итогового файла");
        let mut resolution_changed = false;
        let selected_text = match self.settings.resolution {
            None => "Без пикселизации (исходное)".to_owned(),
            Some(value) => format!("{value} px"),
        };
        egui::ComboBox::from_id_salt("resolution")
            .selected_text(selected_text)
            .show_ui(ui, |ui| {
                resolution_changed |= ui
                    .selectable_value(
                        &mut self.settings.resolution,
                        None,
                        "Без пикселизации (исходное)",
                    )
                    .changed();
                for preset in RESOLUTION_PRESETS {
                    resolution_changed |= ui
                        .selectable_value(
                            &mut self.settings.resolution,
                            Some(preset),
                            format!("{preset} px"),
                        )
                        .changed();
                }
            });
        if resolution_changed {
            self.refresh_previews(ctx);
        }
        ui.add_space(8.0);
        ui.label("Цветов в палитре");
        if ui
            .add(egui::Slider::new(&mut self.settings.palette_size, 2..=256))
            .changed()
        {
            self.refresh_previews(ctx);
        }
        ui.add_space(8.0);
        ui.label("Алгоритм палитры");
        let mut method_changed = false;
        egui::ComboBox::from_id_salt("palette_method")
            .selected_text(match self.settings.palette_method {
                PaletteMethod::Wu => "Wu (быстрый)",
                PaletteMethod::KMeans => "K-means (точный)",
            })
            .show_ui(ui, |ui| {
                method_changed |= ui
                    .selectable_value(
                        &mut self.settings.palette_method,
                        PaletteMethod::Wu,
                        "Wu (быстрый)",
                    )
                    .changed();
                method_changed |= ui
                    .selectable_value(
                        &mut self.settings.palette_method,
                        PaletteMethod::KMeans,
                        "K-means (точный)",
                    )
                    .changed();
            });
        if method_changed {
            self.refresh_previews(ctx);
        }
        ui.add_space(8.0);
        ui.label("Дизеринг");
        let mut dither_changed = false;
        egui::ComboBox::from_id_salt("dither")
            .selected_text(self.settings.dither.label())
            .show_ui(ui, |ui| {
                for mode in DitherMode::ALL {
                    dither_changed |= ui
                        .selectable_value(&mut self.settings.dither, mode, mode.label())
                        .changed();
                }
            });
        if dither_changed {
            self.refresh_previews(ctx);
        }
        if self.settings.dither != DitherMode::None {
            ui.add_space(8.0);
            ui.label("Сила дизеринга");
            if ui
                .add(egui::Slider::new(&mut self.settings.dither_strength, 1..=100).suffix("%"))
                .changed()
            {
                self.refresh_previews(ctx);
            }
        }
        if self.settings.dither.is_error_diffusion()
            && ui
                .checkbox(&mut self.settings.dither_serpentine, "Змейкой (serpentine)")
                .changed()
        {
            self.refresh_previews(ctx);
        }
        ui.add_space(8.0);
        ui.separator();
        ui.label("Цвет");
        if ui
            .add(egui::Slider::new(&mut self.settings.brightness, -100..=100).text("Яркость"))
            .changed()
        {
            self.refresh_previews(ctx);
        }
        if ui
            .add(
                egui::Slider::new(&mut self.settings.saturation, 0..=200)
                    .suffix("%")
                    .text("Насыщенность"),
            )
            .changed()
        {
            self.refresh_previews(ctx);
        }
        if ui
            .add(
                egui::Slider::new(&mut self.settings.contrast, 0..=200)
                    .suffix("%")
                    .text("Контрастность"),
            )
            .changed()
        {
            self.refresh_previews(ctx);
        }
        if ui
            .add(
                egui::Slider::new(&mut self.settings.gamma, 10..=300)
                    .suffix("%")
                    .text("Гамма"),
            )
            .changed()
        {
            self.refresh_previews(ctx);
        }
        if ui
            .add(egui::Slider::new(&mut self.settings.temperature, -100..=100).text("Температура"))
            .changed()
        {
            self.refresh_previews(ctx);
        }
        if ui
            .add(egui::Slider::new(&mut self.settings.tint, -100..=100).text("Оттенок"))
            .changed()
        {
            self.refresh_previews(ctx);
        }
        ui.add_space(4.0);
        if ui
            .checkbox(&mut self.settings.auto_contrast, "Авто-контраст")
            .on_hover_text("Растянуть уровни по гистограмме каждого изображения")
            .changed()
        {
            self.refresh_previews(ctx);
        }
        ui.label("Уровни");
        ui.add_enabled_ui(!self.settings.auto_contrast, |ui| {
            if ui
                .add(
                    egui::Slider::new(&mut self.settings.black_point, 0..=254).text("Чёрная точка"),
                )
                .changed()
            {
                if self.settings.black_point >= self.settings.white_point {
                    self.settings.white_point =
                        (self.settings.black_point as u16 + 1).min(255) as u8;
                }
                self.refresh_previews(ctx);
            }
            if ui
                .add(egui::Slider::new(&mut self.settings.white_point, 1..=255).text("Белая точка"))
                .changed()
            {
                if self.settings.white_point <= self.settings.black_point {
                    self.settings.black_point = self.settings.white_point.saturating_sub(1);
                }
                self.refresh_previews(ctx);
            }
        });
        ui.add_space(4.0);
        ui.label("Кривые").on_hover_text(
            "Клик — добавить точку, перетаскивание — изменить, правый клик — удалить",
        );
        if curve_editor(ui, &mut self.settings.curves, &mut self.curve_drag) {
            self.refresh_previews(ctx);
        }
        if ui.button("Сбросить кривую").clicked() {
            self.settings.curves = default_curves();
            self.curve_drag = None;
            self.refresh_previews(ctx);
        }
        ui.add_space(4.0);
        let mut toggles_changed = false;
        toggles_changed |= ui.checkbox(&mut self.settings.invert, "Инверсия").changed();
        toggles_changed |= ui.checkbox(&mut self.settings.grayscale, "Ч/Б").changed();
        toggles_changed |= ui.checkbox(&mut self.settings.sepia, "Сепия").changed();
        if toggles_changed {
            self.refresh_previews(ctx);
        }
        let color_neutral = self.settings.color_is_neutral();
        if ui
            .add_enabled(!color_neutral, egui::Button::new("Сбросить цвет"))
            .on_hover_text("Вернуть все параметры цветокоррекции к значениям по умолчанию")
            .clicked()
        {
            self.settings.reset_color();
            self.curve_drag = None;
            self.refresh_previews(ctx);
        }
        ui.add_space(8.0);
        ui.label("Формат экспорта");
        let format_before = self.output_format;
        egui::ComboBox::from_id_salt("format")
            .selected_text(match self.output_format {
                OutputFormat::Png => "PNG (без потерь)",
                OutputFormat::Jpeg => "JPEG (с потерями)",
                OutputFormat::WebP => "WebP",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut self.output_format,
                    OutputFormat::Png,
                    "PNG (без потерь)",
                );
                ui.selectable_value(
                    &mut self.output_format,
                    OutputFormat::Jpeg,
                    "JPEG (с потерями)",
                );
                ui.selectable_value(&mut self.output_format, OutputFormat::WebP, "WebP");
            });
        let format_changed = self.output_format != format_before;
        if self.output_format == OutputFormat::Jpeg {
            ui.label("Качество JPEG");
            if ui
                .add(egui::Slider::new(&mut self.jpeg_quality, 10..=100))
                .changed()
            {
                self.request_reencode_size();
            }
        }
        if format_changed {
            self.request_reencode_size();
        }
        ui.add_space(8.0);
        ui.separator();
        ui.label("Альфа и JPEG");
        let alpha_before = self.settings.alpha_mode;
        let threshold_before = self.settings.alpha_threshold;
        let background_before = self.settings.background;
        egui::ComboBox::from_id_salt("alpha_mode")
            .selected_text(match self.settings.alpha_mode {
                AlphaMode::Save => "Сохранить (прозрачность)",
                AlphaMode::Threshold => "Порог",
                AlphaMode::Fill => "Залить фоном",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut self.settings.alpha_mode,
                    AlphaMode::Save,
                    "Сохранить (прозрачность)",
                );
                ui.selectable_value(&mut self.settings.alpha_mode, AlphaMode::Threshold, "Порог");
                ui.selectable_value(
                    &mut self.settings.alpha_mode,
                    AlphaMode::Fill,
                    "Залить фоном",
                );
            });
        if self.settings.alpha_mode == AlphaMode::Threshold {
            ui.label("Порог альфы");
            ui.add(egui::Slider::new(
                &mut self.settings.alpha_threshold,
                0..=255,
            ));
        }
        let uses_background =
            self.settings.alpha_mode == AlphaMode::Fill || self.output_format == OutputFormat::Jpeg;
        if uses_background {
            ui.horizontal(|ui| {
                ui.label("Цвет фона");
                ui.color_edit_button_srgb(&mut self.settings.background);
            });
        }
        ui.label("JPEG не хранит альфу: прозрачные пиксели заливаются цветом фона.");
        let alpha_changed = self.settings.alpha_mode != alpha_before;
        let threshold_changed = self.settings.alpha_threshold != threshold_before;
        let background_changed = self.settings.background != background_before;
        let mut re_stylize = alpha_changed
            || (self.settings.alpha_mode == AlphaMode::Threshold && threshold_changed);
        let mut reencode_size = false;
        if background_changed {
            if self.settings.alpha_mode == AlphaMode::Fill {
                re_stylize = true;
            } else if self.output_format == OutputFormat::Jpeg {
                reencode_size = true;
            }
        }
        if re_stylize {
            self.refresh_previews(ctx);
        } else if reencode_size {
            self.request_reencode_size();
        }
        ui.label("Приписка к имени файла");
        ui.add(
            egui::TextEdit::singleline(&mut self.export_suffix)
                .hint_text("_stylized")
                .desired_width(f32::INFINITY),
        )
        .on_hover_text("Добавляется перед расширением: texture_stylized.png");
        ui.separator();
        if let Some(img) = &self.source {
            ui.label(format!("Размер: {} × {}", img.width(), img.height()));
            ui.label(format!("Формат источника: {:?}", img.color()));
            if let Some(size) = self.source_size {
                ui.label(format!("Размер исходника: {}", format_size(size)));
            }
            match (self.source_size, self.result_size) {
                (before, Some(after)) => {
                    let delta = before
                        .and_then(|before| percent_change(before, after))
                        .map(|p| format!(" ({p:+.0}%)"))
                        .unwrap_or_default();
                    ui.label(format!("Размер результата: {}{delta}", format_size(after)));
                }
                _ => {
                    ui.label("Размер результата: —");
                }
            }
        }
        ui.add_space(12.0);
        ui.label("Подсказка: «Пикселизация» задаёт рабочее разрешение эффектов, а «Геометрия» — кроп и итоговый размер файла. Nearest сохраняет жёсткие пиксели, Lanczos сглаживает.");
    }
}

/// Render a byte count as a human-readable string (B / КБ / МБ).
fn format_size(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    let b = bytes as f64;
    if b < KB {
        format!("{bytes} Б")
    } else if b < MB {
        format!("{:.1} КБ", b / KB)
    } else {
        format!("{:.2} МБ", b / MB)
    }
}

/// Signed percentage change from `before` to `after`, or `None` when undefined.
fn percent_change(before: u64, after: u64) -> Option<f64> {
    if before == 0 {
        None
    } else {
        Some((after as f64 - before as f64) / before as f64 * 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_size_uses_expected_units() {
        assert_eq!(format_size(0), "0 Б");
        assert_eq!(format_size(1023), "1023 Б");
        assert_eq!(format_size(1024), "1.0 КБ");
        assert_eq!(format_size(1024 * 1024), "1.00 МБ");
    }

    #[test]
    fn percent_change_handles_zero_baseline() {
        assert_eq!(percent_change(0, 100), None);
        assert_eq!(percent_change(100, 50), Some(-50.0));
        assert_eq!(percent_change(100, 150), Some(50.0));
    }
}
