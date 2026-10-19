use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Progress and configuration of the batch-processing window.
pub(crate) struct BatchState {
    /// Input files to process, in the order they were added.
    pub(crate) inputs: Vec<PathBuf>,
    /// Destination folder for stylized files.
    pub(crate) output_dir: Option<PathBuf>,
    /// Whether a batch job is currently running on the worker thread.
    pub(crate) running: bool,
    pub(crate) done: usize,
    pub(crate) total: usize,
    /// Name of the file currently being processed.
    pub(crate) current: String,
    /// Per-file failures collected during the last run.
    pub(crate) errors: Vec<String>,
    /// Number of files successfully written by the last run.
    pub(crate) saved: usize,
    /// Set to request cancellation of the running batch.
    pub(crate) cancel: Arc<AtomicBool>,
    /// Whether the last run finished (used to show the summary).
    pub(crate) finished: bool,
    pub(crate) cancelled: bool,
}

impl Default for BatchState {
    fn default() -> Self {
        Self {
            inputs: Vec::new(),
            output_dir: None,
            running: false,
            done: 0,
            total: 0,
            current: String::new(),
            errors: Vec::new(),
            saved: 0,
            cancel: Arc::new(AtomicBool::new(false)),
            finished: false,
            cancelled: false,
        }
    }
}

impl BatchState {
    /// Add files to the queue, skipping duplicates.
    pub(crate) fn enqueue(&mut self, files: Vec<PathBuf>) {
        for file in files {
            if !self.inputs.contains(&file) {
                self.inputs.push(file);
            }
        }
        self.finished = false;
        self.done = 0;
        self.total = self.inputs.len();
    }

    /// Drop every queued file and clear the previous run's report.
    pub(crate) fn clear(&mut self) {
        self.inputs.clear();
        self.errors.clear();
        self.finished = false;
        self.done = 0;
        self.total = 0;
    }

    /// Prepare the state for a fresh run.
    pub(crate) fn begin(&mut self, cancel: Arc<AtomicBool>) {
        self.cancel = cancel;
        self.running = true;
        self.finished = false;
        self.cancelled = false;
        self.done = 0;
        self.total = self.inputs.len();
        self.errors.clear();
        self.saved = 0;
    }

    pub(crate) fn apply_progress(
        &mut self,
        done: usize,
        total: usize,
        current: String,
        saved: usize,
        errors: Vec<String>,
    ) {
        self.done = done;
        self.total = total;
        self.current = current;
        self.saved = saved;
        self.errors = errors;
    }

    pub(crate) fn finish(&mut self, saved: usize, errors: Vec<String>, cancelled: bool) {
        self.running = false;
        self.saved = saved;
        self.errors = errors;
        self.cancelled = cancelled;
        self.finished = true;
        self.current.clear();
    }

    pub(crate) fn can_start(&self) -> bool {
        !self.running && !self.inputs.is_empty() && self.output_dir.is_some()
    }
}
