use crate::imaging::color::build_curve_lut;
use crate::model::settings::sanitize_curves;
use eframe::egui;

/// Index of the curve control point nearest to `pos`, if it is within the hit radius.
fn nearest_curve_point(points: &[[f32; 2]], pos: egui::Pos2, rect: egui::Rect) -> Option<usize> {
    let to_screen = |p: [f32; 2]| {
        egui::pos2(
            rect.left() + p[0] * rect.width(),
            rect.bottom() - p[1] * rect.height(),
        )
    };
    points
        .iter()
        .enumerate()
        .map(|(index, &point)| (index, to_screen(point).distance(pos)))
        .filter(|(_, distance)| *distance <= 12.0)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(index, _)| index)
}

/// Interactive tone-curve editor: click to add a point, drag to move, right-click to remove.
///
/// Returns `true` when the curve changed. Endpoints stay pinned to x = 0 and x = 1; interior
/// points are kept between their neighbours so the curve remains a function of x.
pub(super) fn curve_editor(
    ui: &mut egui::Ui,
    points: &mut Vec<[f32; 2]>,
    drag: &mut Option<usize>,
) -> bool {
    let width = ui.available_width().clamp(120.0, 220.0);
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(width, 170.0), egui::Sense::click_and_drag());
    let painter = ui.painter_at(rect);
    let visuals = ui.visuals().clone();

    painter.rect_filled(rect, 3.0, visuals.extreme_bg_color);
    let grid_color = visuals
        .widgets
        .noninteractive
        .bg_stroke
        .color
        .gamma_multiply(0.6);
    let grid_stroke = egui::Stroke::new(1.0_f32, grid_color);
    for step in 1..4 {
        let t = step as f32 / 4.0;
        let x = rect.left() + rect.width() * t;
        let y = rect.bottom() - rect.height() * t;
        painter.line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
            grid_stroke,
        );
        painter.line_segment(
            [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
            grid_stroke,
        );
    }
    painter.line_segment(
        [rect.left_bottom(), rect.right_top()],
        egui::Stroke::new(1.0_f32, grid_color),
    );

    *points = sanitize_curves(points);

    let to_screen = |p: [f32; 2]| {
        egui::pos2(
            rect.left() + p[0] * rect.width(),
            rect.bottom() - p[1] * rect.height(),
        )
    };
    let from_screen = |pos: egui::Pos2| {
        [
            ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0),
            ((rect.bottom() - pos.y) / rect.height()).clamp(0.0, 1.0),
        ]
    };

    let mut changed = false;

    if response.clicked_by(egui::PointerButton::Secondary) {
        if let Some(pos) = response.interact_pointer_pos() {
            if let Some(index) = nearest_curve_point(points, pos, rect) {
                if index != 0 && index != points.len() - 1 {
                    points.remove(index);
                    *drag = None;
                    changed = true;
                }
            }
        }
    }

    if response.clicked_by(egui::PointerButton::Primary) {
        if let Some(pos) = response.interact_pointer_pos() {
            if nearest_curve_point(points, pos, rect).is_none() {
                let point = from_screen(pos);
                if point[0] > 1e-3 && point[0] < 1.0 - 1e-3 {
                    let index = points
                        .iter()
                        .position(|p| p[0] > point[0])
                        .unwrap_or(points.len());
                    points.insert(index, point);
                    changed = true;
                }
            }
        }
    }

    if response.drag_started_by(egui::PointerButton::Primary) {
        if let Some(pos) = response.interact_pointer_pos() {
            if let Some(index) = nearest_curve_point(points, pos, rect) {
                *drag = Some(index);
            } else {
                let point = from_screen(pos);
                if point[0] > 1e-3 && point[0] < 1.0 - 1e-3 {
                    let index = points
                        .iter()
                        .position(|p| p[0] > point[0])
                        .unwrap_or(points.len());
                    points.insert(index, point);
                    *drag = Some(index);
                    changed = true;
                }
            }
        }
    }

    if response.dragged_by(egui::PointerButton::Primary) {
        if let (Some(index), Some(pos)) = (*drag, response.interact_pointer_pos()) {
            let len = points.len();
            if index < len {
                let mut updated = from_screen(pos);
                if index == 0 {
                    updated[0] = 0.0;
                } else if index == len - 1 {
                    updated[0] = 1.0;
                } else {
                    let min = points[index - 1][0] + 1e-3;
                    let max = points[index + 1][0] - 1e-3;
                    updated[0] = updated[0].clamp(min, max);
                }
                if updated != points[index] {
                    points[index] = updated;
                    changed = true;
                }
            }
        }
    }
    if response.drag_stopped() {
        *drag = None;
    }

    let lut = build_curve_lut(points);
    const SAMPLES: usize = 96;
    let line: Vec<egui::Pos2> = (0..=SAMPLES)
        .map(|i| {
            let x = i as f32 / SAMPLES as f32;
            let y = lut[(x * 255.0).round().clamp(0.0, 255.0) as usize] as f32 / 255.0;
            to_screen([x, y])
        })
        .collect();
    painter.add(egui::Shape::line(
        line,
        egui::Stroke::new(2.0_f32, visuals.selection.stroke.color),
    ));

    for (index, &point) in points.iter().enumerate() {
        let center = to_screen(point);
        let radius = if *drag == Some(index) { 5.5 } else { 4.5 };
        painter.circle_filled(center, radius, visuals.selection.stroke.color);
        painter.circle_stroke(
            center,
            radius,
            egui::Stroke::new(1.0_f32, visuals.strong_text_color()),
        );
    }

    changed
}
