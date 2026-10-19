pub(crate) mod batch;
pub(crate) mod worker;

use crate::model::export::OutputFormat;
use crate::model::settings::Settings;
use crate::render::batch::BatchJob;
use crate::render::worker::renderer_loop;
use eframe::egui;
use image::DynamicImage;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

/// How long to wait after the last settings change before starting a render.
///
/// Dragging a slider or typing a value can produce dozens of changes per second; waiting a short
/// moment collapses them into a single job instead of starting one render per frame.
pub(crate) const RENDER_DEBOUNCE: Duration = Duration::from_millis(90);

/// A unit of work handed to the background render thread.
pub(crate) enum RenderJob {
    /// Re-run the stylization pipeline over `source` with `settings`.
    Stylize {
        id: u64,
        source: Arc<DynamicImage>,
        settings: Settings,
        /// Set when a newer render supersedes this one; the worker aborts at the next checkpoint.
        cancel: Arc<AtomicBool>,
    },
    /// Re-encode an already stylized image only to refresh the estimated output size.
    Reencode {
        id: u64,
        image: Arc<DynamicImage>,
        format: OutputFormat,
        quality: u8,
        background: [u8; 3],
    },
    /// Encode and write the result to `path` on disk.
    Save {
        image: Arc<DynamicImage>,
        format: OutputFormat,
        quality: u8,
        background: [u8; 3],
        path: PathBuf,
    },
    /// Stylize and save an entire list of files sequentially, reporting progress as it goes.
    Batch(BatchJob),
}

impl RenderJob {
    /// Whether this job produces a stylized image (and therefore participates in coalescing).
    pub(crate) fn is_render(&self) -> bool {
        matches!(self, RenderJob::Stylize { .. } | RenderJob::Reencode { .. })
    }
}

/// A completion notification sent back from the render thread to the UI.
pub(crate) enum RenderEvent {
    /// A stylize or re-encode job finished. `image` is `Some` for stylize jobs.
    Rendered {
        id: u64,
        image: Option<Arc<DynamicImage>>,
        size: Option<u64>,
    },
    /// A save job finished.
    Saved { result: Result<PathBuf, String> },
    /// A batch job made progress: `done` of `total` files processed, `current` is the active name.
    BatchProgress {
        done: usize,
        total: usize,
        current: String,
        saved: usize,
        errors: Vec<String>,
    },
    /// A batch job finished, either because it ran out of files or was cancelled.
    BatchFinished {
        saved: usize,
        errors: Vec<String>,
        cancelled: bool,
    },
}

/// A stylize request that is waiting for the debounce window to elapse.
pub(crate) struct PendingStylize {
    id: u64,
    source: Arc<DynamicImage>,
    settings: Settings,
    due: Instant,
}

/// Handle to the dedicated background render thread.
///
/// The UI thread submits jobs and polls [`RenderEvent`]s; the heavy quantization and encoding
/// work happens on the worker so the interface never blocks on large images.
pub(crate) struct Renderer {
    pub(crate) jobs: mpsc::Sender<RenderJob>,
    pub(crate) events: mpsc::Receiver<RenderEvent>,
    /// Monotonic id generator shared by all job kinds.
    pub(crate) next_id: u64,
    /// Id of the newest stylize job; results with an older id are stale and ignored.
    pub(crate) latest_stylize: u64,
    /// Id of the newest size estimate applied, so out-of-order results cannot regress it.
    pub(crate) applied_size_id: u64,
    /// Whether a stylize job is still running or waiting to be dispatched.
    pub(crate) in_flight: bool,
    /// Newest request, coalesced until [`RENDER_DEBOUNCE`] passes without further changes.
    pub(crate) pending_stylize: Option<PendingStylize>,
    /// Cancellation flag of the stylize job currently running on the worker.
    pub(crate) active_cancel: Option<Arc<AtomicBool>>,
}

impl Renderer {
    pub(crate) fn spawn(ctx: egui::Context) -> Self {
        let (job_tx, job_rx) = mpsc::channel::<RenderJob>();
        let (event_tx, event_rx) = mpsc::channel::<RenderEvent>();
        thread::spawn(move || renderer_loop(ctx, job_rx, event_tx));
        Self {
            jobs: job_tx,
            events: event_rx,
            next_id: 0,
            latest_stylize: 0,
            applied_size_id: 0,
            in_flight: false,
            pending_stylize: None,
            active_cancel: None,
        }
    }

    /// Record a stylize request and return its id.
    ///
    /// Any older render is invalidated immediately: a queued job is replaced and a running one is
    /// asked to stop at its next checkpoint. The request is dispatched from [`Renderer::tick`]
    /// once the debounce window has elapsed, so rapid slider changes collapse into one render.
    pub(crate) fn submit_stylize(&mut self, source: Arc<DynamicImage>, settings: Settings) -> u64 {
        self.next_id += 1;
        self.latest_stylize = self.next_id;
        self.applied_size_id = self.next_id;
        self.in_flight = true;
        self.cancel_active();
        self.pending_stylize = Some(PendingStylize {
            id: self.next_id,
            source,
            settings,
            due: Instant::now() + RENDER_DEBOUNCE,
        });
        self.next_id
    }

    /// Dispatch the pending stylize request once its debounce window has elapsed.
    pub(crate) fn tick(&mut self, now: Instant) {
        let Some(pending) = self.pending_stylize.as_ref() else {
            return;
        };
        if now < pending.due {
            return;
        }
        let pending = self
            .pending_stylize
            .take()
            .expect("pending stylize present");
        let cancel = Arc::new(AtomicBool::new(false));
        let _ = self.jobs.send(RenderJob::Stylize {
            id: pending.id,
            source: pending.source,
            settings: pending.settings,
            cancel: cancel.clone(),
        });
        self.active_cancel = Some(cancel);
    }

    /// Ask the stylize job currently running on the worker to stop.
    pub(crate) fn cancel_active(&mut self) {
        if let Some(cancel) = self.active_cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
    }

    /// Time until the pending stylize request should be dispatched, if one is waiting.
    pub(crate) fn time_until_dispatch(&self) -> Option<Duration> {
        self.pending_stylize
            .as_ref()
            .map(|pending| pending.due.saturating_duration_since(Instant::now()))
    }

    /// Queue a size-only re-encode of an already stylized image.
    pub(crate) fn submit_reencode(
        &mut self,
        image: Arc<DynamicImage>,
        format: OutputFormat,
        quality: u8,
        background: [u8; 3],
    ) -> u64 {
        self.next_id += 1;
        let _ = self.jobs.send(RenderJob::Reencode {
            id: self.next_id,
            image,
            format,
            quality,
            background,
        });
        self.next_id
    }

    pub(crate) fn submit_save(
        &self,
        image: Arc<DynamicImage>,
        format: OutputFormat,
        quality: u8,
        background: [u8; 3],
        path: PathBuf,
    ) {
        let _ = self.jobs.send(RenderJob::Save {
            image,
            format,
            quality,
            background,
            path,
        });
    }

    /// Queue a batch pass over `job` and return the cancellation flag it will honor.
    pub(crate) fn submit_batch(&self, job: BatchJob) -> Arc<AtomicBool> {
        let cancel = job.cancel.clone();
        let _ = self.jobs.send(RenderJob::Batch(job));
        cancel
    }

    pub(crate) fn try_recv(&self) -> Option<RenderEvent> {
        self.events.try_recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::settings::Settings;
    use image::DynamicImage;
    use image::RgbaImage;

    use std::time::Duration;

    #[test]
    fn renderer_debounces_rapid_stylize_requests() {
        let mut img = RgbaImage::new(16, 16);
        for (x, y, p) in img.enumerate_pixels_mut() {
            p.0 = [(x * 16) as u8, (y * 16) as u8, 0, 255];
        }
        let source = Arc::new(DynamicImage::ImageRgba8(img));
        let ctx = egui::Context::default();
        let mut renderer = Renderer::spawn(ctx);

        for size in [4u32, 8, 16, 32] {
            renderer.submit_stylize(
                source.clone(),
                Settings {
                    resolution: None,
                    palette_size: size,
                    ..Default::default()
                },
            );
        }
        assert_eq!(renderer.latest_stylize, 4);

        // Nothing is dispatched during the debounce window, so no stale result can arrive first.
        renderer.tick(Instant::now());
        assert!(renderer.active_cancel.is_none());

        // Once the window passes, only the newest request is sent to the worker.
        renderer.tick(Instant::now() + RENDER_DEBOUNCE);
        assert!(renderer.active_cancel.is_some());
        assert!(renderer.pending_stylize.is_none());

        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            if let Some(RenderEvent::Rendered { id, image, .. }) = renderer.try_recv() {
                assert_eq!(id, 4);
                assert!(image.is_some());
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("debounced render produced no result");
    }

    #[test]
    fn new_request_cancels_the_running_stylize() {
        let source = Arc::new(DynamicImage::ImageRgba8(RgbaImage::new(8, 8)));
        let mut renderer = Renderer::spawn(egui::Context::default());

        renderer.submit_stylize(source.clone(), Settings::default());
        renderer.tick(Instant::now() + RENDER_DEBOUNCE);
        let cancel = renderer.active_cancel.clone().expect("render dispatched");

        renderer.submit_stylize(source, Settings::default());
        assert!(
            cancel.load(Ordering::Relaxed),
            "superseded render was not cancelled"
        );
    }
}
