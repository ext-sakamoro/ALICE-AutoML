//! Early stopping (`EarlyStopping`).

// Early Stopping
// ---------------------------------------------------------------------------

/// Patience-based early stopping.
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    patience: usize,
    min_delta: f64,
    minimize: bool,
    best: f64,
    counter: usize,
}

impl EarlyStopping {
    pub const fn new(patience: usize, min_delta: f64, minimize: bool) -> Self {
        let best = if minimize {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
        Self {
            patience,
            min_delta,
            minimize,
            best,
            counter: 0,
        }
    }

    /// Report a metric. Returns true if training should stop.
    pub fn should_stop(&mut self, metric: f64) -> bool {
        let improved = if self.minimize {
            metric < self.best - self.min_delta
        } else {
            metric > self.best + self.min_delta
        };
        if improved {
            self.best = metric;
            self.counter = 0;
        } else {
            self.counter += 1;
        }
        self.counter >= self.patience
    }

    pub const fn best_value(&self) -> f64 {
        self.best
    }

    pub const fn counter(&self) -> usize {
        self.counter
    }

    /// Reset state.
    pub const fn reset(&mut self) {
        self.best = if self.minimize {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
        self.counter = 0;
    }
}
