use crate::imaging::encode::ImageEncoder;
use crate::imaging::pipeline::StylizeCache;
use crate::render::batch::run_batch;
use crate::render::{RenderEvent, RenderJob};
use eframe::egui;
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc};

/// Worker loop: pull jobs, optionally coalescing queued renders down to the newest one, then
/// process them and wake the UI. `Save` jobs are never dropped by coalescing.
pub(super) fn renderer_loop(
    ctx: egui::Context,
    jobs: mpsc::Receiver<RenderJob>,
    events: mpsc::Sender<RenderEvent>,
) {
    // Sits on the worker thread so decoded/pixelized/palette layers survive between jobs.
    let mut cache = StylizeCache::default();
    while let Ok(mut job) = jobs.recv() {
        let mut deferred: Vec<RenderJob> = Vec::new();
        if job.is_render() {
            while let Ok(next) = jobs.try_recv() {
                if next.is_render() {
                    job = next;
                } else {
                    deferred.push(next);
                }
            }
        }
        process_job(&job, &mut cache, &events, &ctx);
        for extra in deferred {
            process_job(&extra, &mut cache, &events, &ctx);
        }
        ctx.request_repaint();
    }
}

pub(crate) fn process_job(
    job: &RenderJob,
    cache: &mut StylizeCache,
    events: &mpsc::Sender<RenderEvent>,
    ctx: &egui::Context,
) {
    match job {
        RenderJob::Stylize {
            id,
            source,
            settings,
            cancel,
        } => {
            // A superseded job may already have been cancelled before it even started.
            if cancel.load(Ordering::Relaxed) {
                return;
            }
            let image = cache
                .stylize_cancellable(source.clone(), settings, Some(cancel))
                .map(Arc::new);
            // Do not report a superseded result; the newer request owns the preview now.
            if cancel.load(Ordering::Relaxed) {
                return;
            }
            let _ = events.send(RenderEvent::Rendered {
                id: *id,
                image,
                size: None,
            });
        }
        RenderJob::Reencode {
            id,
            image,
            format,
            quality,
            background,
        } => {
            let size = ImageEncoder::for_format(*format, *quality, *background)
                .encode(image)
                .ok()
                .map(|b| b.len() as u64);
            let _ = events.send(RenderEvent::Rendered {
                id: *id,
                image: None,
                size,
            });
        }
        RenderJob::Save {
            image,
            format,
            quality,
            background,
            path,
        } => {
            let result = ImageEncoder::for_format(*format, *quality, *background)
                .encode(image)
                .map_err(|e| e.to_string())
                .and_then(|bytes| std::fs::write(path, bytes).map_err(|e| e.to_string()))
                .map(|_| path.clone());
            let _ = events.send(RenderEvent::Saved { result });
        }
        RenderJob::Batch(job) => run_batch(job, events, ctx),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::settings::Settings;
    use crate::render::{RenderEvent, RenderJob};
    use image::DynamicImage;
    use image::RgbaImage;
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc;

    #[test]
    fn cancelled_stylize_job_reports_no_result() {
        let source = Arc::new(DynamicImage::ImageRgba8(RgbaImage::new(8, 8)));
        let (tx, rx) = mpsc::channel::<RenderEvent>();
        let cancel = Arc::new(AtomicBool::new(true));
        let job = RenderJob::Stylize {
            id: 7,
            source,
            settings: Settings::default(),
            cancel,
        };

        process_job(
            &job,
            &mut StylizeCache::default(),
            &tx,
            &egui::Context::default(),
        );
        assert!(
            rx.try_recv().is_err(),
            "cancelled job must not report a result"
        );
    }
}
