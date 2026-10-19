/// Undo/redo stack over cloned snapshots of `T`.
pub(crate) struct History<T> {
    pub(crate) undo: Vec<T>,
    pub(crate) redo: Vec<T>,
}

impl<T: Clone + PartialEq> History<T> {
    /// Maximum number of retained undo steps.
    pub(crate) const LIMIT: usize = 100;

    pub(crate) fn new() -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }

    /// Push the state that preceded a change, discarding the redo branch.
    pub(crate) fn record(&mut self, previous: T) {
        if self.undo.last() == Some(&previous) {
            return;
        }
        self.undo.push(previous);
        if self.undo.len() > Self::LIMIT {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    /// Restore the previous state into `current`, if any.
    pub(crate) fn undo(&mut self, current: &mut T) -> bool {
        match self.undo.pop() {
            Some(previous) => {
                self.redo.push(current.clone());
                *current = previous;
                true
            }
            None => false,
        }
    }

    /// Re-apply the last undone state into `current`, if any.
    pub(crate) fn redo(&mut self, current: &mut T) -> bool {
        match self.redo.pop() {
            Some(next) => {
                self.undo.push(current.clone());
                *current = next;
                true
            }
            None => false,
        }
    }

    pub(crate) fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub(crate) fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_undo_and_redo_restore_snapshots() {
        let mut history: History<u8> = History::new();
        let mut current = 1u8;

        history.record(current);
        current = 2;
        history.record(current);
        current = 3;

        assert!(history.can_undo());
        assert!(!history.can_redo());

        assert!(history.undo(&mut current));
        assert_eq!(current, 2);
        assert!(history.undo(&mut current));
        assert_eq!(current, 1);
        assert!(!history.undo(&mut current));

        assert!(history.can_redo());
        assert!(history.redo(&mut current));
        assert_eq!(current, 2);
        assert!(history.redo(&mut current));
        assert_eq!(current, 3);
        assert!(!history.redo(&mut current));
    }

    #[test]
    fn history_record_clears_redo_and_skips_duplicates() {
        let mut history: History<u8> = History::new();
        let mut current = 1u8;

        history.record(current);
        current = 2;
        history.record(current);
        current = 1;
        assert!(history.undo(&mut current));
        assert!(history.can_redo());

        // A new change must invalidate the redo branch.
        history.record(current);
        assert!(!history.can_redo());

        // Consecutive identical snapshots are ignored.
        let depth = history.undo.len();
        history.record(current);
        assert_eq!(history.undo.len(), depth);
    }

    #[test]
    fn history_is_bounded_by_limit() {
        let mut history: History<u32> = History::new();
        for value in 0..(History::<u32>::LIMIT as u32 + 50) {
            history.record(value);
        }
        assert_eq!(history.undo.len(), History::<u32>::LIMIT);
    }
}
