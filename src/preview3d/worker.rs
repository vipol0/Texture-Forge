use super::mesh::Mesh3d;
use super::raster::{RasterScratch, RenderOptions};
use eframe::egui;
use image::RgbaImage;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;

/// Textures used by one preview frame. `secondary` is sampled on the right side of the split.
pub(crate) struct PreviewTextures {
    pub(crate) primary: Arc<RgbaImage>,
    pub(crate) secondary: Option<Arc<RgbaImage>>,
}

/// One interactive 3D preview frame.
pub(crate) struct PreviewRender {
    pub(crate) id: u64,
    pub(crate) mesh: Arc<Mesh3d>,
    pub(crate) textures: PreviewTextures,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) options: RenderOptions,
    pub(crate) cancel: Arc<AtomicBool>,
}

/// A full-quality preview frame that is written to disk.
pub(crate) struct PreviewSave {
    pub(crate) mesh: Arc<Mesh3d>,
    pub(crate) textures: PreviewTextures,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) options: RenderOptions,
    pub(crate) path: PathBuf,
}

pub(crate) enum PreviewEvent {
    Rendered { id: u64, image: egui::ColorImage },
    Saved { result: Result<PathBuf, String> },
}

/// Handles to the background 3D preview threads.
///
/// Rasterization and the conversion to `egui::ColorImage` run on a dedicated render thread, so
/// camera interaction never blocks the UI thread. Saving is performed by a second thread: a slow
/// PNG encode cannot stall interactive rendering. The UI thread only uploads the finished texture
/// and paints it.
pub(crate) struct PreviewWorker {
    render_jobs: mpsc::Sender<PreviewRender>,
    save_jobs: mpsc::Sender<PreviewSave>,
    events: mpsc::Receiver<PreviewEvent>,
}

impl PreviewWorker {
    pub(crate) fn spawn(ctx: egui::Context) -> Self {
        let (render_tx, render_rx) = mpsc::channel();
        let (save_tx, save_rx) = mpsc::channel();
        let (event_tx, event_rx) = mpsc::channel();

        let render_ctx = ctx.clone();
        let render_events = event_tx.clone();
        thread::spawn(move || render_loop(render_ctx, render_rx, render_events));

        let save_ctx = ctx;
        thread::spawn(move || save_loop(save_ctx, save_rx, event_tx));

        Self {
            render_jobs: render_tx,
            save_jobs: save_tx,
            events: event_rx,
        }
    }

    pub(crate) fn submit_render(&self, job: PreviewRender) {
        let _ = self.render_jobs.send(job);
    }

    pub(crate) fn submit_save(&self, job: PreviewSave) {
        let _ = self.save_jobs.send(job);
    }

    pub(crate) fn try_recv(&self) -> Option<PreviewEvent> {
        self.events.try_recv().ok()
    }
}

fn render_loop(
    ctx: egui::Context,
    jobs: mpsc::Receiver<PreviewRender>,
    events: mpsc::Sender<PreviewEvent>,
) {
    let mut scratch = RasterScratch::default();

    while let Ok(mut render) = jobs.recv() {
        // Drop every queued render except the newest one, so camera drags cannot accumulate
        // a backlog while a frame is being rasterized. The in-flight frame is intentionally not
        // cancelled: letting it finish is what makes rotation visible during the drag.
        while let Ok(next) = jobs.try_recv() {
            render = next;
        }
        if render.cancel.load(Ordering::Relaxed) {
            continue;
        }

        let image = scratch.render(
            &render.mesh,
            &render.textures.primary,
            render.textures.secondary.as_deref(),
            render.width,
            render.height,
            render.options,
            &render.cancel,
        );
        if let Some(image) = image {
            if !render.cancel.load(Ordering::Relaxed) {
                let image = egui::ColorImage::from_rgba_unmultiplied(
                    [image.width() as usize, image.height() as usize],
                    image.as_raw(),
                );
                let _ = events.send(PreviewEvent::Rendered {
                    id: render.id,
                    image,
                });
            }
        }
        ctx.request_repaint();
    }
}

fn save_loop(
    ctx: egui::Context,
    jobs: mpsc::Receiver<PreviewSave>,
    events: mpsc::Sender<PreviewEvent>,
) {
    let mut scratch = RasterScratch::default();

    while let Ok(save) = jobs.recv() {
        let cancel = AtomicBool::new(false);
        let result = scratch
            .render(
                &save.mesh,
                &save.textures.primary,
                save.textures.secondary.as_deref(),
                save.width,
                save.height,
                save.options,
                &cancel,
            )
            .ok_or_else(|| "Не удалось отрисовать кадр.".to_owned())
            .and_then(|image| {
                image
                    .save_with_format(&save.path, image::ImageFormat::Png)
                    .map_err(|e| e.to_string())
            })
            .map(|()| save.path);
        let _ = events.send(PreviewEvent::Saved { result });
        ctx.request_repaint();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    fn textured(color: [u8; 4]) -> Arc<RgbaImage> {
        Arc::new(RgbaImage::from_pixel(2, 2, Rgba(color)))
    }

    fn plane_options() -> RenderOptions {
        RenderOptions {
            yaw: 0.0,
            pitch: 0.0,
            lighting: false,
            ..Default::default()
        }
    }

    #[test]
    fn render_thread_coalesces_frames_and_keeps_the_latest() {
        let worker = PreviewWorker::spawn(egui::Context::default());
        let mesh = Arc::new(Mesh3d::plane());

        for (id, color) in [(1, [255, 0, 0, 255]), (2, [0, 255, 0, 255])] {
            worker.submit_render(PreviewRender {
                id,
                mesh: mesh.clone(),
                textures: PreviewTextures {
                    primary: textured(color),
                    secondary: None,
                },
                width: 32,
                height: 32,
                options: plane_options(),
                cancel: Arc::new(AtomicBool::new(false)),
            });
        }

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            if let Some(PreviewEvent::Rendered { id, image }) = worker.try_recv() {
                if id == 2 {
                    assert_eq!(image.size, [32, 32]);
                    let center = 16 * 32 + 16;
                    assert_eq!(image.pixels[center], egui::Color32::from_rgb(0, 255, 0));
                    return;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        panic!("render thread did not produce the latest frame");
    }

    #[test]
    fn save_thread_writes_a_png() {
        let worker = PreviewWorker::spawn(egui::Context::default());
        let mesh = Arc::new(Mesh3d::plane());
        let path = std::env::temp_dir().join(format!(
            "texture-forge-preview-save-{}-{}.png",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos())
        ));

        worker.submit_save(PreviewSave {
            mesh,
            textures: PreviewTextures {
                primary: textured([255, 0, 0, 255]),
                secondary: None,
            },
            width: 16,
            height: 16,
            options: plane_options(),
            path: path.clone(),
        });

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            if let Some(PreviewEvent::Saved { result }) = worker.try_recv() {
                let saved_path = result.expect("save thread failed");
                assert_eq!(saved_path, path);
                let saved = image::open(&saved_path).expect("saved PNG should decode");
                assert_eq!((saved.width(), saved.height()), (16, 16));
                let _ = std::fs::remove_file(saved_path);
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let _ = std::fs::remove_file(path);
        panic!("save thread did not finish");
    }
}
