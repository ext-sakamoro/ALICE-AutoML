//! Search strategies (`GridSearch` / `RandomSearch` / `BayesianOptimizer`).

use crate::acquisition::expected_improvement;
use crate::gaussian_process::GaussianProcess;
use crate::rng::Rng;
use crate::search_space::{ParamValue, SearchSpace};
use crate::trial::{Trial, TrialStatus, TrialTracker};
use std::collections::HashMap;

// Search Strategies
// ---------------------------------------------------------------------------

/// Grid search: enumerate all parameter combinations.
pub struct GridSearch {
    space: SearchSpace,
    continuous_steps: usize,
}

impl GridSearch {
    pub fn new(space: SearchSpace, continuous_steps: usize) -> Self {
        Self {
            space,
            continuous_steps: continuous_steps.max(2),
        }
    }

    /// Generate all candidate parameter sets.
    pub fn candidates(&self) -> Vec<HashMap<String, ParamValue>> {
        let dims: Vec<usize> = self
            .space
            .params()
            .iter()
            .map(|p| p.grid_size(self.continuous_steps))
            .collect();
        let total: usize = dims.iter().product();
        let mut results = Vec::with_capacity(total);
        for i in 0..total {
            let mut params = HashMap::new();
            let mut idx = i;
            for (d, ps) in dims.iter().zip(self.space.params()) {
                let local = idx % d;
                idx /= d;
                params.insert(
                    ps.name().to_string(),
                    ps.grid_value(local, self.continuous_steps),
                );
            }
            results.push(params);
        }
        results
    }

    /// Run grid search with the given objective function. Returns tracker.
    pub fn run<F: FnMut(&HashMap<String, ParamValue>) -> f64>(
        &self,
        mut objective: F,
    ) -> TrialTracker {
        let mut tracker = TrialTracker::new();
        for (i, params) in self.candidates().into_iter().enumerate() {
            let metric = objective(&params);
            let mut trial = Trial::new(i, params);
            trial.status = TrialStatus::Running;
            trial.complete(metric);
            tracker.add_trial(trial);
        }
        tracker
    }
}

/// Random search: sample parameters randomly.
pub struct RandomSearch {
    space: SearchSpace,
    n_trials: usize,
    seed: u64,
}

impl RandomSearch {
    pub const fn new(space: SearchSpace, n_trials: usize, seed: u64) -> Self {
        Self {
            space,
            n_trials,
            seed,
        }
    }

    /// Sample random parameter sets.
    pub fn candidates(&self) -> Vec<HashMap<String, ParamValue>> {
        let mut rng = Rng::new(self.seed);
        (0..self.n_trials)
            .map(|_| {
                let mut params = HashMap::new();
                for ps in self.space.params() {
                    params.insert(ps.name().to_string(), ps.sample(&mut rng));
                }
                params
            })
            .collect()
    }

    pub fn run<F: FnMut(&HashMap<String, ParamValue>) -> f64>(
        &self,
        mut objective: F,
    ) -> TrialTracker {
        let mut tracker = TrialTracker::new();
        for (i, params) in self.candidates().into_iter().enumerate() {
            let metric = objective(&params);
            let mut trial = Trial::new(i, params);
            trial.status = TrialStatus::Running;
            trial.complete(metric);
            tracker.add_trial(trial);
        }
        tracker
    }
}

/// Bayesian optimization with Gaussian Process and Expected Improvement.
pub struct BayesianOptimizer {
    space: SearchSpace,
    n_trials: usize,
    n_initial: usize,
    minimize: bool,
    seed: u64,
    length_scale: f64,
    noise: f64,
    n_candidates: usize,
}

impl BayesianOptimizer {
    pub const fn new(space: SearchSpace, n_trials: usize) -> Self {
        Self {
            space,
            n_trials,
            n_initial: 5,
            minimize: true,
            seed: 42,
            length_scale: 1.0,
            noise: 1e-5,
            n_candidates: 200,
        }
    }

    pub const fn minimize(mut self, minimize: bool) -> Self {
        self.minimize = minimize;
        self
    }

    pub const fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub const fn n_initial(mut self, n: usize) -> Self {
        self.n_initial = n;
        self
    }

    pub const fn length_scale(mut self, ls: f64) -> Self {
        self.length_scale = ls;
        self
    }

    pub const fn noise(mut self, noise: f64) -> Self {
        self.noise = noise;
        self
    }

    pub const fn n_candidates(mut self, n: usize) -> Self {
        self.n_candidates = n;
        self
    }

    fn normalize_params(&self, params: &HashMap<String, ParamValue>) -> Vec<f64> {
        self.space
            .params()
            .iter()
            .map(|ps| ps.normalize(params.get(ps.name()).unwrap()))
            .collect()
    }

    pub fn run<F: FnMut(&HashMap<String, ParamValue>) -> f64>(
        &self,
        mut objective: F,
    ) -> TrialTracker {
        let mut tracker = TrialTracker::new();
        let mut rng = Rng::new(self.seed);
        let mut gp = GaussianProcess::new(self.length_scale, self.noise);

        let n_init = self.n_initial.min(self.n_trials);

        // Initial random exploration
        for i in 0..n_init {
            let mut params = HashMap::new();
            for ps in self.space.params() {
                params.insert(ps.name().to_string(), ps.sample(&mut rng));
            }
            let metric = objective(&params);
            let mut trial = Trial::new(i, params);
            trial.status = TrialStatus::Running;
            trial.complete(metric);
            tracker.add_trial(trial);
        }

        // Bayesian optimization loop
        for i in n_init..self.n_trials {
            // Fit GP
            let completed = tracker.completed_trials();
            let x_train: Vec<Vec<f64>> = completed
                .iter()
                .map(|t| self.normalize_params(&t.params))
                .collect();
            let y_train: Vec<f64> = completed.iter().map(|t| t.metric.unwrap()).collect();

            let best_y = if self.minimize {
                y_train.iter().copied().fold(f64::INFINITY, f64::min)
            } else {
                y_train.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            };

            gp.fit(x_train, y_train);

            // Optimize acquisition by random sampling
            let mut best_ei = f64::NEG_INFINITY;
            let mut best_params = HashMap::new();
            for _ in 0..self.n_candidates {
                let mut params = HashMap::new();
                for ps in self.space.params() {
                    params.insert(ps.name().to_string(), ps.sample(&mut rng));
                }
                let x_norm = self.normalize_params(&params);
                let (mean, var) = gp.predict(&x_norm);
                let ei = expected_improvement(mean, var, best_y, self.minimize);
                if ei > best_ei {
                    best_ei = ei;
                    best_params = params;
                }
            }

            let metric = objective(&best_params);
            let mut trial = Trial::new(i, best_params);
            trial.status = TrialStatus::Running;
            trial.complete(metric);
            tracker.add_trial(trial);
        }

        tracker
    }
}
