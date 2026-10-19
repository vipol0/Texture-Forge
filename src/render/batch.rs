use crate::files::{batch_output_path, paths_equal};
use crate::imaging::encode::ImageEncoder;
use crate::imaging::pipeline::StylizeCache;
use crate::model::export::OutputFormat;
use crate::model::settings::Settings;
use crate::render::RenderEvent;
use eframe::egui;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

/// Everything the worker needs to run one batch pass.
pub(crate) struct BatchJob {
    pub(crate) inputs: Vec<PathBuf>,
    pub(crate) output_dir: PathBuf,
    pub(crate) settings: Settings,
    pub(crate) format: OutputFormat,
    pub(crate) quality: u8,
    pub(crate) suffix: String,
    pub(crate) cancel: Arc<AtomicBool>,
}

/// Process every file in `job`, emitting progress after each one so the UI can update.
///
/// Individual failures are collected rather than aborting the pass, and cancellation is checked
/// between files so a long batch can be stopped promptly.
pub(crate) fn run_batch(job: &BatchJob, events: &mpsc::Sender<RenderEvent>, ctx: &egui::Context) {
    let total = job.inputs.len();
    let mut saved = 0usize;
    let mut errors: Vec<String> = Vec::new();
    let mut used: HashSet<PathBuf> = HashSet::new();
    let mut cache = StylizeCache::default();

    // Make sure the destination exists; individual writes report any remaining failure.
    if let Err(e) = std::fs::create_dir_all(&job.output_dir) {
        errors.push(format!("Не удалось создать папку вывода: {e}"));
    }

    for (index, input) in job.inputs.iter().enumerate() {
        if job.cancel.load(Ordering::Relaxed) {
            let _ = events.send(RenderEvent::BatchFinished {
                saved,
                errors,
                cancelled: true,
            });
            ctx.request_repaint();
            return;
        }

        let name = input
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(без имени)")
            .to_owned();
        let _ = events.send(RenderEvent::BatchProgress {
            done: index,
            total,
            current: name.clone(),
            saved,
            errors: errors.clone(),
        });
        ctx.request_repaint();

        let mut output = batch_output_path(input, &job.output_dir, &job.suffix, job.format);
        // Never overwrite the source, and disambiguate files that map to the same output name.
        if paths_equal(input, &output) {
            errors.push(format!("{name}: пропущено — совпадает с исходником"));
            continue;
        }
        let mut counter = 1u32;
        while used.contains(&output) {
            output = batch_output_path(
                input,
                &job.output_dir,
                &format!("{}_{counter}", job.suffix),
                job.format,
            );
            counter += 1;
        }
        used.insert(output.clone());

        match image::open(input) {
            Ok(img) => match cache.stylize(Arc::new(img), &job.settings) {
                Some(stylized) => {
                    let encoder =
                        ImageEncoder::for_format(job.format, job.quality, job.settings.background);
                    match encoder.encode(&stylized) {
                        Ok(bytes) => match std::fs::write(&output, bytes) {
                            Ok(()) => saved += 1,
                            Err(e) => errors.push(format!("{name}: {e}")),
                        },
                        Err(e) => errors.push(format!("{name}: {e}")),
                    }
                }
                None => errors.push(format!("{name}: не удалось обработать изображение")),
            },
            Err(e) => errors.push(format!("{name}: {e}")),
        }

        let _ = events.send(RenderEvent::BatchProgress {
            done: index + 1,
            total,
            current: name,
            saved,
            errors: errors.clone(),
        });
        ctx.request_repaint();
    }

    let _ = events.send(RenderEvent::BatchFinished {
        saved,
        errors,
        cancelled: false,
    });
    ctx.request_repaint();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::collect_images_in_dir;
    use crate::model::settings::Settings;
    use crate::test_support::{temp_dir, write_test_png};

    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    #[test]
    fn run_batch_writes_outputs_and_reports_progress() {
        let dir = temp_dir("batch");
        let input_dir = dir.join("in");
        let output_dir = dir.join("out");
        std::fs::create_dir_all(&input_dir).unwrap();
        write_test_png(&input_dir.join("one.png"));
        write_test_png(&input_dir.join("two.png"));

        let inputs = collect_images_in_dir(&input_dir);
        assert_eq!(inputs.len(), 2);

        let (tx, rx) = mpsc::channel::<RenderEvent>();
        let ctx = egui::Context::default();
        let job = BatchJob {
            inputs: inputs.clone(),
            output_dir: output_dir.clone(),
            settings: Settings {
                resolution: Some(16),
                palette_size: 4,
                ..Default::default()
            },
            format: OutputFormat::Png,
            quality: 88,
            suffix: "_stylized".to_owned(),
            cancel: Arc::new(AtomicBool::new(false)),
        };
        run_batch(&job, &tx, &ctx);

        assert!(output_dir.join("one_stylized.png").is_file());
        assert!(output_dir.join("two_stylized.png").is_file());

        let events: Vec<RenderEvent> = rx.try_iter().collect();
        let finished = events.iter().any(|e| {
            matches!(
                e,
                RenderEvent::BatchFinished { saved: 2, errors, cancelled: false } if errors.is_empty()
            )
        });
        assert!(finished, "batch did not report a clean finish");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_batch_skips_overwrite_and_missing_files() {
        let dir = temp_dir("batch_conflict");
        let input_dir = dir.join("in");
        std::fs::create_dir_all(&input_dir).unwrap();
        write_test_png(&input_dir.join("hero.png"));

        // Output directory equals the input directory with the same format and no suffix, so the
        // target path would clobber the source and must be skipped with an error.
        let (tx, rx) = mpsc::channel::<RenderEvent>();
        let ctx = egui::Context::default();
        let job = BatchJob {
            inputs: vec![input_dir.join("hero.png"), input_dir.join("missing.png")],
            output_dir: input_dir.clone(),
            settings: Settings::default(),
            format: OutputFormat::Png,
            quality: 88,
            suffix: String::new(),
            cancel: Arc::new(AtomicBool::new(false)),
        };
        run_batch(&job, &tx, &ctx);

        let events: Vec<RenderEvent> = rx.try_iter().collect();
        let errors: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                RenderEvent::BatchFinished { errors, .. } => Some(errors.clone()),
                _ => None,
            })
            .flatten()
            .collect();
        assert_eq!(errors.len(), 2, "expected both files to fail: {errors:?}");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_batch_honors_cancellation() {
        let dir = temp_dir("batch_cancel");
        let input_dir = dir.join("in");
        let output_dir = dir.join("out");
        std::fs::create_dir_all(&input_dir).unwrap();
        write_test_png(&input_dir.join("one.png"));

        let (tx, rx) = mpsc::channel::<RenderEvent>();
        let ctx = egui::Context::default();
        let cancel = Arc::new(AtomicBool::new(true));
        let job = BatchJob {
            inputs: collect_images_in_dir(&input_dir),
            output_dir,
            settings: Settings::default(),
            format: OutputFormat::Png,
            quality: 88,
            suffix: String::new(),
            cancel,
        };
        run_batch(&job, &tx, &ctx);

        let events: Vec<RenderEvent> = rx.try_iter().collect();
        assert!(events.iter().any(|e| matches!(
            e,
            RenderEvent::BatchFinished {
                cancelled: true,
                ..
            }
        )));

        std::fs::remove_dir_all(&dir).ok();
    }
}
