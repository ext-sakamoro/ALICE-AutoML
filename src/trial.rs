//! Trial tracking (`TrialStatus` / `Trial` / `TrialTracker`).

use crate::search_space::ParamValue;
use std::collections::HashMap;

// Trial Tracking
// ---------------------------------------------------------------------------

/// Status of a trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrialStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Pruned,
}

/// A single trial record.
#[derive(Debug, Clone)]
pub struct Trial {
    pub id: usize,
    pub params: HashMap<String, ParamValue>,
    pub metric: Option<f64>,
    pub status: TrialStatus,
    pub epoch_metrics: Vec<f64>,
}

impl Trial {
    pub const fn new(id: usize, params: HashMap<String, ParamValue>) -> Self {
        Self {
            id,
            params,
            metric: None,
            status: TrialStatus::Pending,
            epoch_metrics: Vec::new(),
        }
    }

    /// Record a metric for an epoch (used for early stopping).
    pub fn report_epoch(&mut self, value: f64) {
        self.epoch_metrics.push(value);
    }

    /// Set final metric and mark completed.
    pub const fn complete(&mut self, metric: f64) {
        self.metric = Some(metric);
        self.status = TrialStatus::Completed;
    }

    /// Mark trial as failed.
    pub const fn fail(&mut self) {
        self.status = TrialStatus::Failed;
    }

    /// Mark trial as pruned.
    pub const fn prune(&mut self) {
        self.status = TrialStatus::Pruned;
    }
}

/// Tracks all trials in a study.
#[derive(Debug, Clone)]
pub struct TrialTracker {
    trials: Vec<Trial>,
}

impl TrialTracker {
    pub const fn new() -> Self {
        Self { trials: Vec::new() }
    }

    pub fn add_trial(&mut self, trial: Trial) {
        self.trials.push(trial);
    }

    pub fn trials(&self) -> &[Trial] {
        &self.trials
    }

    pub fn completed_trials(&self) -> Vec<&Trial> {
        self.trials
            .iter()
            .filter(|t| t.status == TrialStatus::Completed)
            .collect()
    }

    /// Return the best trial (minimize=true: lowest metric, else highest).
    pub fn best_trial(&self, minimize: bool) -> Option<&Trial> {
        self.completed_trials().into_iter().min_by(|a, b| {
            let (ma, mb) = (a.metric.unwrap_or(f64::NAN), b.metric.unwrap_or(f64::NAN));
            if minimize {
                ma.partial_cmp(&mb).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                mb.partial_cmp(&ma).unwrap_or(std::cmp::Ordering::Equal)
            }
        })
    }

    pub const fn len(&self) -> usize {
        self.trials.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.trials.is_empty()
    }
}

impl Default for TrialTracker {
    fn default() -> Self {
        Self::new()
    }
}
