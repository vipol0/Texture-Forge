use crate::model::geometry::CropRect;
use eframe::egui;

/// Independent view state (zoom and pan) for one preview pane.
#[derive(Clone, Copy)]
pub(super) struct PreviewView {
    zoom: f32,
    pan: egui::Vec2,
    /// Pointer position when the current pan drag began, if any.
    drag_origin: Option<egui::Pos2>,
    /// Pan value captured at the start of the current drag.
    drag_pan: egui::Vec2,
}

impl Default for PreviewView {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            drag_origin: None,
            drag_pan: egui::Vec2::ZERO,
        }
    }
}

impl PreviewView {
    /// Smallest and largest allowed zoom factor.
    const MIN_ZOOM: f32 = 0.1;
    const MAX_ZOOM: f32 = 16.0;
    /// Distance the pointer must travel before a pan drag starts moving the image.
    const PAN_DEAD_ZONE: f32 = 14.0;
    /// Minimum fraction of each pane axis that must stay covered by the texture while panning.
    const PAN_KEEP_COVERED: f32 = 0.5;

    pub(super) fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Which part of the crop rectangle a drag is manipulating.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum CropHandle {
    Move,
    New,
    N,
    NE,
    E,
    SE,
    S,
    SW,
    W,
    NW,
}

impl CropHandle {
    fn moves_left(self) -> bool {
        matches!(self, CropHandle::W | CropHandle::NW | CropHandle::SW)
    }

    fn moves_right(self) -> bool {
        matches!(self, CropHandle::E | CropHandle::NE | CropHandle::SE)
    }

    fn moves_top(self) -> bool {
        matches!(self, CropHandle::N | CropHandle::NE | CropHandle::NW)
    }

    fn moves_bottom(self) -> bool {
        matches!(self, CropHandle::S | CropHandle::SE | CropHandle::SW)
    }
}

/// Active crop drag, captured when the pointer goes down over the original pane.
pub(super) struct CropDrag {
    handle: CropHandle,
    start_pointer: egui::Pos2,
    start_crop: CropRect,
    image_rect: egui::Rect,
}

/// Mutable crop-editing state borrowed by [`draw_preview_pane`].
pub(super) struct CropEdit<'a> {
    pub(super) crop: &'a mut CropRect,
    pub(super) drag: &'a mut Option<CropDrag>,
}

impl CropDrag {
    /// Return the crop updated for the current pointer position.
    fn updated_crop(&self, current: egui::Pos2) -> CropRect {
        let size = self.image_rect.size();
        let to_norm = |p: egui::Pos2| {
            egui::vec2(
                ((p.x - self.image_rect.left()) / size.x.max(1e-6)).clamp(0.0, 1.0),
                ((p.y - self.image_rect.top()) / size.y.max(1e-6)).clamp(0.0, 1.0),
            )
        };
        let start = to_norm(self.start_pointer);
        let current = to_norm(current);
        let min_x = (1.0 / size.x.max(1.0)).max(CropRect::MIN_SIDE);
        let min_y = (1.0 / size.y.max(1.0)).max(CropRect::MIN_SIDE);

        if self.handle == CropHandle::New {
            let left = start.x.min(current.x);
            let right = start.x.max(current.x).max(left + min_x).min(1.0);
            let top = start.y.min(current.y);
            let bottom = start.y.max(current.y).max(top + min_y).min(1.0);
            return CropRect {
                x: left,
                y: top,
                w: right - left,
                h: bottom - top,
            }
            .sanitized();
        }

        let crop = self.start_crop.sanitized();
        if self.handle == CropHandle::Move {
            return CropRect {
                x: (crop.x + current.x - start.x).clamp(0.0, 1.0 - crop.w),
                y: (crop.y + current.y - start.y).clamp(0.0, 1.0 - crop.h),
                w: crop.w,
                h: crop.h,
            };
        }

        let mut left = crop.x;
        let mut right = crop.x + crop.w;
        let mut top = crop.y;
        let mut bottom = crop.y + crop.h;
        let dx = current.x - start.x;
        let dy = current.y - start.y;
        if self.handle.moves_left() {
            left = (left + dx).clamp(0.0, right - min_x);
        }
        if self.handle.moves_right() {
            right = (right + dx).clamp(left + min_x, 1.0);
        }
        if self.handle.moves_top() {
            top = (top + dy).clamp(0.0, bottom - min_y);
        }
        if self.handle.moves_bottom() {
            bottom = (bottom + dy).clamp(top + min_y, 1.0);
        }
        CropRect {
            x: left,
            y: top,
            w: right - left,
            h: bottom - top,
        }
        .sanitized()
    }
}

fn crop_screen_rect(image_rect: egui::Rect, crop: CropRect) -> egui::Rect {
    let crop = crop.sanitized();
    egui::Rect::from_min_max(
        egui::pos2(
            image_rect.left() + crop.x * image_rect.width(),
            image_rect.top() + crop.y * image_rect.height(),
        ),
        egui::pos2(
            image_rect.left() + (crop.x + crop.w) * image_rect.width(),
            image_rect.top() + (crop.y + crop.h) * image_rect.height(),
        ),
    )
}

/// Handle positions in screen space, in the same order as [`CropHandle`] corner/edge variants.
fn crop_handle_positions(rect: egui::Rect) -> [(CropHandle, egui::Pos2); 8] {
    let (l, r, t, b) = (rect.left(), rect.right(), rect.top(), rect.bottom());
    let (cx, cy) = (rect.center().x, rect.center().y);
    [
        (CropHandle::NW, egui::pos2(l, t)),
        (CropHandle::N, egui::pos2(cx, t)),
        (CropHandle::NE, egui::pos2(r, t)),
        (CropHandle::E, egui::pos2(r, cy)),
        (CropHandle::SE, egui::pos2(r, b)),
        (CropHandle::S, egui::pos2(cx, b)),
        (CropHandle::SW, egui::pos2(l, b)),
        (CropHandle::W, egui::pos2(l, cy)),
    ]
}

fn crop_handle_at(
    crop_rect: egui::Rect,
    image_rect: egui::Rect,
    pos: egui::Pos2,
) -> Option<CropHandle> {
    const HIT_RADIUS: f32 = 11.0;
    let mut best: Option<(f32, CropHandle)> = None;
    for (handle, center) in crop_handle_positions(crop_rect) {
        let distance = center.distance_sq(pos);
        if distance <= HIT_RADIUS * HIT_RADIUS && best.is_none_or(|(closest, _)| distance < closest)
        {
            best = Some((distance, handle));
        }
    }
    if let Some((_, handle)) = best {
        return Some(handle);
    }
    if crop_rect.contains(pos) {
        Some(CropHandle::Move)
    } else if image_rect.contains(pos) {
        Some(CropHandle::New)
    } else {
        None
    }
}

fn crop_cursor(handle: CropHandle) -> egui::CursorIcon {
    match handle {
        CropHandle::Move | CropHandle::New => egui::CursorIcon::Crosshair,
        CropHandle::N | CropHandle::S => egui::CursorIcon::ResizeVertical,
        CropHandle::E | CropHandle::W => egui::CursorIcon::ResizeHorizontal,
        CropHandle::NE | CropHandle::SW => egui::CursorIcon::ResizeNeSw,
        CropHandle::NW | CropHandle::SE => egui::CursorIcon::ResizeNwSe,
    }
}

/// Paint a checkerboard behind the preview image so transparency is visible.
fn paint_checkerboard(painter: &egui::Painter, rect: egui::Rect, dark_mode: bool) {
    const CELL: f32 = 16.0;
    let visible = rect.intersect(painter.clip_rect());
    if visible.width() <= 0.0 || visible.height() <= 0.0 {
        return;
    }
    let (light, dark) = if dark_mode {
        (egui::Color32::from_gray(55), egui::Color32::from_gray(80))
    } else {
        (egui::Color32::from_gray(200), egui::Color32::from_gray(230))
    };
    painter.rect_filled(visible, 0.0, light);

    let first_col = ((visible.left() - rect.left()) / CELL).floor() as i64;
    let last_col = ((visible.right() - rect.left()) / CELL).ceil() as i64;
    let first_row = ((visible.top() - rect.top()) / CELL).floor() as i64;
    let last_row = ((visible.bottom() - rect.top()) / CELL).ceil() as i64;
    for row in first_row..last_row {
        for col in first_col..last_col {
            if (row + col) % 2 == 0 {
                continue;
            }
            let cell = egui::Rect::from_min_size(
                egui::pos2(
                    rect.left() + col as f32 * CELL,
                    rect.top() + row as f32 * CELL,
                ),
                egui::vec2(CELL, CELL),
            )
            .intersect(visible);
            if cell.width() > 0.0 && cell.height() > 0.0 {
                painter.rect_filled(cell, 0.0, dark);
            }
        }
    }
}

/// Draw a single zoomable/pannable preview pane, updating `view` in place.
pub(super) fn draw_preview_pane(
    ui: &mut egui::Ui,

    pane_rect: egui::Rect,
    id: egui::Id,
    texture: Option<&egui::TextureHandle>,
    label: &str,
    view: &mut PreviewView,
    mut crop_edit: Option<CropEdit<'_>>,
) -> bool {
    let crop_editing = crop_edit.is_some();
    let response = ui.interact(pane_rect, id, egui::Sense::click_and_drag());
    let painter = ui.painter().with_clip_rect(pane_rect);
    painter.rect_filled(pane_rect, 0.0, ui.visuals().extreme_bg_color);

    let Some(tex) = texture else {
        return false;
    };

    let tex_size = tex.size_vec2();
    let fit_scale = (pane_rect.width() / tex_size.x).min(pane_rect.height() / tex_size.y);
    let fit_size = tex_size * fit_scale.max(0.0001);

    // Zoom around the cursor, keeping the point under it fixed.
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
            let new_zoom = (view.zoom * factor).clamp(PreviewView::MIN_ZOOM, PreviewView::MAX_ZOOM);
            if let Some(cursor) = response.hover_pos() {
                let old_size = fit_size * view.zoom;
                let old_min = pane_rect.center() + view.pan - old_size / 2.0;
                let u = egui::vec2(
                    (cursor.x - old_min.x) / old_size.x.max(1e-6),
                    (cursor.y - old_min.y) / old_size.y.max(1e-6),
                );
                let new_size = fit_size * new_zoom;
                let new_min = cursor - egui::vec2(u.x * new_size.x, u.y * new_size.y);
                view.pan = new_min + new_size / 2.0 - pane_rect.center();
            }
            view.zoom = new_zoom;
        }
    }

    if !crop_editing {
        if response.drag_started() {
            view.drag_origin = response.interact_pointer_pos();
            view.drag_pan = view.pan;
        }
        if response.dragged() {
            if let (Some(origin), Some(current)) =
                (view.drag_origin, response.interact_pointer_pos())
            {
                let delta = current - origin;
                // Ignore small movements so the image does not jitter before the pan really starts.
                view.pan = if delta.length() >= PreviewView::PAN_DEAD_ZONE {
                    view.drag_pan + delta
                } else {
                    view.drag_pan
                };
            }
        }
        if response.drag_stopped() {
            view.drag_origin = None;
        }
        if response.double_clicked() {
            view.reset();
        }
    }

    // Allow free panning, but never let the texture drift too far: keep at least half of the
    // pane covered, or the whole image visible when it is smaller than the pane.
    let size = fit_size * view.zoom;
    let keep = egui::vec2(
        (pane_rect.width() * PreviewView::PAN_KEEP_COVERED).min(size.x),
        (pane_rect.height() * PreviewView::PAN_KEEP_COVERED).min(size.y),
    );
    let limit = egui::vec2(
        (pane_rect.width() / 2.0 - keep.x + size.x / 2.0).max(0.0),
        (pane_rect.height() / 2.0 - keep.y + size.y / 2.0).max(0.0),
    );
    view.pan.x = view.pan.x.clamp(-limit.x, limit.x);
    view.pan.y = view.pan.y.clamp(-limit.y, limit.y);

    let image_rect = egui::Rect::from_center_size(pane_rect.center() + view.pan, size);
    paint_checkerboard(&painter, image_rect, ui.visuals().dark_mode);
    painter.image(
        tex.id(),
        image_rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );

    let mut crop_changed = false;
    let mut hovered_handle = None;
    if let Some(edit) = crop_edit.as_mut() {
        let crop_rect = &mut *edit.crop;
        let crop_drag = &mut *edit.drag;
        let screen_crop = crop_screen_rect(image_rect, *crop_rect);
        if response.drag_started() {
            if let Some(pos) = response.interact_pointer_pos() {
                if let Some(mut handle) = crop_handle_at(screen_crop, image_rect, pos) {
                    // A full-frame crop cannot be moved, so dragging inside it starts a new box.
                    if handle == CropHandle::Move && crop_rect.is_full() {
                        handle = CropHandle::New;
                    }
                    *crop_drag = Some(CropDrag {
                        handle,
                        start_pointer: pos,
                        start_crop: *crop_rect,
                        image_rect,
                    });
                }
            }
        }
        if response.dragged() {
            if let (Some(drag), Some(pos)) = (crop_drag.as_ref(), response.interact_pointer_pos()) {
                let updated = drag.updated_crop(pos);
                if updated != *crop_rect {
                    *crop_rect = updated;
                    crop_changed = true;
                }
            }
        }
        if response.drag_stopped() {
            *crop_drag = None;
        }
        if let Some(pos) = response.hover_pos() {
            hovered_handle = crop_handle_at(screen_crop, image_rect, pos);
        }
    }

    // Draw the crop overlay after the image so the dimmed outside area and handles are visible.
    if let Some(crop_rect) = crop_edit.as_ref().map(|edit| &*edit.crop) {
        let screen = crop_screen_rect(image_rect, *crop_rect);
        let dim = egui::Color32::from_black_alpha(150);
        painter.rect_filled(
            egui::Rect::from_min_max(pane_rect.min, egui::pos2(pane_rect.right(), screen.top())),
            0.0,
            dim,
        );
        painter.rect_filled(
            egui::Rect::from_min_max(egui::pos2(pane_rect.left(), screen.bottom()), pane_rect.max),
            0.0,
            dim,
        );
        painter.rect_filled(
            egui::Rect::from_min_max(
                egui::pos2(pane_rect.left(), screen.top()),
                egui::pos2(screen.left(), screen.bottom()),
            ),
            0.0,
            dim,
        );
        painter.rect_filled(
            egui::Rect::from_min_max(
                egui::pos2(screen.right(), screen.top()),
                egui::pos2(pane_rect.right(), screen.bottom()),
            ),
            0.0,
            dim,
        );

        let accent = ui.visuals().selection.stroke.color;
        painter.rect_stroke(
            screen,
            0.0,
            egui::Stroke::new(2.0_f32, accent),
            egui::StrokeKind::Inside,
        );
        for frac in [1.0 / 3.0, 2.0 / 3.0] {
            painter.line_segment(
                [
                    egui::pos2(screen.left() + screen.width() * frac, screen.top()),
                    egui::pos2(screen.left() + screen.width() * frac, screen.bottom()),
                ],
                egui::Stroke::new(1.0_f32, accent.gamma_multiply(0.45)),
            );
            painter.line_segment(
                [
                    egui::pos2(screen.left(), screen.top() + screen.height() * frac),
                    egui::pos2(screen.right(), screen.top() + screen.height() * frac),
                ],
                egui::Stroke::new(1.0_f32, accent.gamma_multiply(0.45)),
            );
        }
        for (_, pos) in crop_handle_positions(screen) {
            painter.circle_filled(pos, 5.0, egui::Color32::WHITE);
            painter.circle_stroke(pos, 5.0, egui::Stroke::new(1.0_f32, egui::Color32::BLACK));
        }
    }

    // Label over a translucent backing for readability.
    let text_pos = pane_rect.left_top() + egui::vec2(8.0, 6.0);
    let galley = painter.layout_no_wrap(
        label.to_owned(),
        egui::FontId::proportional(14.0),
        egui::Color32::WHITE,
    );
    let bg = egui::Rect::from_min_size(
        text_pos - egui::vec2(6.0, 3.0),
        galley.size() + egui::vec2(12.0, 6.0),
    );
    painter.rect_filled(bg, 3.0, egui::Color32::from_black_alpha(140));
    painter.galley(text_pos, galley, egui::Color32::WHITE);

    if response.hovered() {
        let cursor = if crop_editing {
            hovered_handle
                .map(crop_cursor)
                .unwrap_or(egui::CursorIcon::Crosshair)
        } else if response.dragged() {
            egui::CursorIcon::Grabbing
        } else {
            egui::CursorIcon::Grab
        };
        ui.ctx().set_cursor_icon(cursor);
    }

    crop_changed
}
