#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::should_implement_trait)]

//! ALICE-AutoML: Pure Rust `AutoML` framework.
//!
//! Provides hyperparameter search (grid, random, Bayesian optimization),
//! neural architecture search (NAS), early stopping, cross-validation,
//! model selection, search space definition, and trial tracking.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Pseudo-random number generator (Xoshiro256**)
// ---------------------------------------------------------------------------

/// A simple PRNG based on xoshiro256**.
#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Create a new RNG from a seed.
    pub fn new(seed: u64) -> Self {
        let mut s = [0u64; 4];
        // SplitMix64 to initialize state
        let mut z = seed;
        for slot in &mut s {
            z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            *slot = x ^ (x >> 31);
        }
        Self { s }
    }

    /// Generate next u64.
    pub const fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Generate a uniform f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Generate a uniform f64 in [lo, hi).
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        (hi - lo).mul_add(self.next_f64(), lo)
    }

    /// Generate a usize in [0, n).
    pub const fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Generate a standard normal via Box-Muller.
    pub fn normal(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        std.mul_add(z, mean)
    }
}

// ---------------------------------------------------------------------------
// Search Space
// ---------------------------------------------------------------------------

/// A single hyperparameter dimension.
#[derive(Debug, Clone)]
pub enum ParamSpace {
    /// Continuous parameter in [low, high].
    Continuous { name: String, low: f64, high: f64 },
    /// Discrete parameter in [low, high] (inclusive integer range).
    Discrete { name: String, low: i64, high: i64 },
    /// Categorical parameter with named choices.
    Categorical { name: String, choices: Vec<String> },
}

impl ParamSpace {
    pub fn continuous(name: &str, low: f64, high: f64) -> Self {
        Self::Continuous {
            name: name.to_string(),
            low,
            high,
        }
    }

    pub fn discrete(name: &str, low: i64, high: i64) -> Self {
        Self::Discrete {
            name: name.to_string(),
            low,
            high,
        }
    }

    pub fn categorical(name: &str, choices: &[&str]) -> Self {
        Self::Categorical {
            name: name.to_string(),
            choices: choices.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Continuous { name, .. }
            | Self::Discrete { name, .. }
            | Self::Categorical { name, .. } => name,
        }
    }

    /// Sample a random value.
    fn sample(&self, rng: &mut Rng) -> ParamValue {
        match self {
            Self::Continuous { low, high, .. } => ParamValue::Continuous(rng.uniform(*low, *high)),
            Self::Discrete { low, high, .. } => {
                let range = (*high - *low + 1) as u64;
                let v = *low + (rng.next_u64() % range) as i64;
                ParamValue::Discrete(v)
            }
            Self::Categorical { choices, .. } => {
                let idx = rng.next_usize(choices.len());
                ParamValue::Categorical(idx)
            }
        }
    }

    /// Return the number of grid points for grid search.
    const fn grid_size(&self, continuous_steps: usize) -> usize {
        match self {
            Self::Continuous { .. } => continuous_steps,
            Self::Discrete { low, high, .. } => (*high - *low + 1) as usize,
            Self::Categorical { choices, .. } => choices.len(),
        }
    }

    /// Return the i-th grid value.
    fn grid_value(&self, idx: usize, continuous_steps: usize) -> ParamValue {
        match self {
            Self::Continuous { low, high, .. } => {
                let steps = continuous_steps.max(2);
                let t = if steps == 1 {
                    0.5
                } else {
                    idx as f64 / (steps - 1) as f64
                };
                ParamValue::Continuous(*low + t * (*high - *low))
            }
            Self::Discrete { low, .. } => ParamValue::Discrete(*low + idx as i64),
            Self::Categorical { .. } => ParamValue::Categorical(idx),
        }
    }

    /// Normalize a value to [0, 1] for GP.
    fn normalize(&self, val: &ParamValue) -> f64 {
        match (self, val) {
            (Self::Continuous { low, high, .. }, ParamValue::Continuous(v)) => {
                if (high - low).abs() < f64::EPSILON {
                    0.5
                } else {
                    (v - low) / (high - low)
                }
            }
            (Self::Discrete { low, high, .. }, ParamValue::Discrete(v)) => {
                let range = high - low;
                if range == 0 {
                    0.5
                } else {
                    (*v - low) as f64 / range as f64
                }
            }
            (Self::Categorical { choices, .. }, ParamValue::Categorical(idx)) => {
                if choices.len() <= 1 {
                    0.5
                } else {
                    *idx as f64 / (choices.len() - 1) as f64
                }
            }
            _ => 0.5,
        }
    }
}

/// A concrete parameter value.
#[derive(Debug, Clone)]
pub enum ParamValue {
    Continuous(f64),
    Discrete(i64),
    Categorical(usize),
}

impl fmt::Display for ParamValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Continuous(v) => write!(f, "{v:.6}"),
            Self::Discrete(v) => write!(f, "{v}"),
            Self::Categorical(v) => write!(f, "choice[{v}]"),
        }
    }
}

impl ParamValue {
    pub const fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Continuous(v) => Some(*v),
            Self::Discrete(v) => Some(*v as f64),
            Self::Categorical(_) => None,
        }
    }

    pub const fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Discrete(v) => Some(*v),
            Self::Continuous(v) => Some(*v as i64),
            Self::Categorical(_) => None,
        }
    }

    pub const fn as_category(&self) -> Option<usize> {
        if let Self::Categorical(v) = self {
            Some(*v)
        } else {
            None
        }
    }
}

/// A complete search space definition.
#[derive(Debug, Clone)]
pub struct SearchSpace {
    params: Vec<ParamSpace>,
}

impl SearchSpace {
    pub const fn new() -> Self {
        Self { params: Vec::new() }
    }

    pub fn add(mut self, param: ParamSpace) -> Self {
        self.params.push(param);
        self
    }

    pub fn params(&self) -> &[ParamSpace] {
        &self.params
    }

    pub const fn dim(&self) -> usize {
        self.params.len()
    }
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Cross-Validation
// ---------------------------------------------------------------------------

/// K-fold cross-validation splitter.
#[derive(Debug, Clone)]
pub struct KFold {
    k: usize,
}

impl KFold {
    pub fn new(k: usize) -> Self {
        assert!(k >= 2, "k must be >= 2");
        Self { k }
    }

    /// Return k folds as (`train_indices`, `val_indices`) pairs.
    pub fn split(&self, n: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let fold_size = n / self.k;
        let remainder = n % self.k;
        let mut folds = Vec::with_capacity(self.k);
        let mut start = 0;
        for i in 0..self.k {
            let size = fold_size + usize::from(i < remainder);
            let end = start + size;
            let val: Vec<usize> = (start..end).collect();
            let train: Vec<usize> = (0..start).chain(end..n).collect();
            folds.push((train, val));
            start = end;
        }
        folds
    }

    pub const fn k(&self) -> usize {
        self.k
    }
}

/// Stratified K-fold: preserves label distribution in each fold.
#[derive(Debug, Clone)]
pub struct StratifiedKFold {
    k: usize,
}

impl StratifiedKFold {
    pub fn new(k: usize) -> Self {
        assert!(k >= 2, "k must be >= 2");
        Self { k }
    }

    /// Split with labels for stratification.
    pub fn split(&self, labels: &[usize]) -> Vec<(Vec<usize>, Vec<usize>)> {
        let n = labels.len();
        // Group indices by label
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            groups.entry(label).or_default().push(i);
        }

        let mut folds: Vec<(Vec<usize>, Vec<usize>)> =
            (0..self.k).map(|_| (Vec::new(), Vec::new())).collect();

        for indices in groups.values() {
            let sub_folds = KFold::new(self.k).split(indices.len());
            for (fold_idx, (sub_train, sub_val)) in sub_folds.iter().enumerate() {
                for &si in sub_train {
                    folds[fold_idx].0.push(indices[si]);
                }
                for &si in sub_val {
                    folds[fold_idx].1.push(indices[si]);
                }
            }
        }

        // Verify all indices are covered
        for (train, val) in &folds {
            debug_assert_eq!(train.len() + val.len(), n);
        }

        folds
    }
}

// ---------------------------------------------------------------------------
// Gaussian Process (for Bayesian Optimization)
// ---------------------------------------------------------------------------

/// Gaussian Process with RBF (squared exponential) kernel.
#[derive(Debug, Clone)]
struct GaussianProcess {
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    length_scale: f64,
    noise: f64,
    alpha: Vec<f64>, // K^{-1} y
    y_mean: f64,
}

impl GaussianProcess {
    const fn new(length_scale: f64, noise: f64) -> Self {
        Self {
            x_train: Vec::new(),
            y_train: Vec::new(),
            length_scale,
            noise,
            alpha: Vec::new(),
            y_mean: 0.0,
        }
    }

    fn rbf_kernel(&self, a: &[f64], b: &[f64]) -> f64 {
        let sq_dist: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum();
        (-0.5 * sq_dist / (self.length_scale * self.length_scale)).exp()
    }

    /// Fit the GP to training data. Uses Cholesky solve.
    fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) {
        let n = x.len();
        self.y_mean = if n > 0 {
            y.iter().sum::<f64>() / n as f64
        } else {
            0.0
        };
        let y_centered: Vec<f64> = y.iter().map(|yi| yi - self.y_mean).collect();
        self.x_train = x;
        self.y_train = y;

        if n == 0 {
            self.alpha = Vec::new();
            return;
        }

        // Build K + noise*I
        let mut k_mat = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                k_mat[i * n + j] = self.rbf_kernel(&self.x_train[i], &self.x_train[j]);
                if i == j {
                    k_mat[i * n + j] += self.noise;
                }
            }
        }

        // Cholesky decomposition (lower triangular)
        let l = Self::cholesky(&k_mat, n);

        // Solve L z = y_centered
        let z = Self::forward_sub(&l, &y_centered, n);
        // Solve L^T alpha = z
        self.alpha = Self::backward_sub(&l, &z, n);
    }

    fn cholesky(a: &[f64], n: usize) -> Vec<f64> {
        let mut l = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut s = 0.0;
                for k in 0..j {
                    s += l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    let diag = a[i * n + i] - s;
                    l[i * n + j] = if diag > 0.0 { diag.sqrt() } else { 1e-10 };
                } else {
                    l[i * n + j] = (a[i * n + j] - s) / l[j * n + j];
                }
            }
        }
        l
    }

    fn forward_sub(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mut x = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..i {
                s += l[i * n + j] * x[j];
            }
            x[i] = (b[i] - s) / l[i * n + i];
        }
        x
    }

    fn backward_sub(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = 0.0;
            for j in (i + 1)..n {
                s += l[j * n + i] * x[j]; // L^T
            }
            x[i] = (b[i] - s) / l[i * n + i];
        }
        x
    }

    /// Predict mean and variance at a point.
    fn predict(&self, x: &[f64]) -> (f64, f64) {
        let n = self.x_train.len();
        if n == 0 {
            return (self.y_mean, 1.0);
        }
        let mut k_star = Vec::with_capacity(n);
        for xi in &self.x_train {
            k_star.push(self.rbf_kernel(x, xi));
        }
        let mean: f64 = k_star
            .iter()
            .zip(self.alpha.iter())
            .map(|(k, a)| k * a)
            .sum::<f64>()
            + self.y_mean;

        let k_ss = self.rbf_kernel(x, x) + self.noise;
        // Approximate variance (without full K^-1 k_star solve, use diagonal approx)
        let k_star_sq: f64 = k_star.iter().map(|k| k * k).sum();
        let var = (k_ss - k_star_sq / (n as f64 + self.noise)).max(1e-10);
        (mean, var)
    }
}

// ---------------------------------------------------------------------------
// Acquisition Functions
// ---------------------------------------------------------------------------

/// Expected Improvement acquisition function.
fn expected_improvement(mean: f64, var: f64, best: f64, minimize: bool) -> f64 {
    let std = var.sqrt();
    if std < 1e-12 {
        return 0.0;
    }
    let (diff, z) = if minimize {
        let d = best - mean;
        (d, d / std)
    } else {
        let d = mean - best;
        (d, d / std)
    };
    diff.mul_add(normal_cdf(z), std * normal_pdf(z))
}

/// Standard normal PDF.
fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Approximate standard normal CDF using Abramowitz & Stegun.
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / 0.231_641_9f64.mul_add(x.abs(), 1.0);
    let d = 1.330_274_429f64.mul_add(
        t.powi(5),
        1.821_255_978f64.mul_add(
            -t.powi(4),
            1.781_477_937f64.mul_add(
                t.powi(3),
                0.319_381_530f64.mul_add(t, -(0.356_563_782 * t * t)),
            ),
        ),
    );
    let approx = normal_pdf(x.abs()).mul_add(-d, 1.0);
    if x >= 0.0 {
        approx
    } else {
        1.0 - approx
    }
}

// ---------------------------------------------------------------------------
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
            let mut trial = Trial::new(i, params.clone());
            trial.status = TrialStatus::Running;
            let metric = objective(&params);
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
            let mut trial = Trial::new(i, params.clone());
            trial.status = TrialStatus::Running;
            let metric = objective(&params);
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
            let mut trial = Trial::new(i, params.clone());
            trial.status = TrialStatus::Running;
            let metric = objective(&params);
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

            let mut trial = Trial::new(i, best_params.clone());
            trial.status = TrialStatus::Running;
            let metric = objective(&best_params);
            trial.complete(metric);
            tracker.add_trial(trial);
        }

        tracker
    }
}

// ---------------------------------------------------------------------------
// Neural Architecture Search (NAS)
// ---------------------------------------------------------------------------

/// Activation function type for NAS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU,
    Swish,
    GELU,
}

impl Activation {
    pub const ALL: [Self; 6] = [
        Self::ReLU,
        Self::Sigmoid,
        Self::Tanh,
        Self::LeakyReLU,
        Self::Swish,
        Self::GELU,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::ReLU => "relu",
            Self::Sigmoid => "sigmoid",
            Self::Tanh => "tanh",
            Self::LeakyReLU => "leaky_relu",
            Self::Swish => "swish",
            Self::GELU => "gelu",
        }
    }

    pub fn apply(self, x: f64) -> f64 {
        match self {
            Self::ReLU => x.max(0.0),
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::Tanh => x.tanh(),
            Self::LeakyReLU => {
                if x >= 0.0 {
                    x
                } else {
                    0.01 * x
                }
            }
            Self::Swish => x * (1.0 / (1.0 + (-x).exp())),
            Self::GELU => {
                0.5 * x * (1.0 + (0.797_884_56 * 0.044_715f64.mul_add(x.powi(3), x)).tanh())
            }
        }
    }
}

/// A layer specification in a neural architecture.
#[derive(Debug, Clone)]
pub struct LayerSpec {
    pub units: usize,
    pub activation: Activation,
    pub dropout: f64,
}

/// A candidate neural architecture.
#[derive(Debug, Clone)]
pub struct Architecture {
    pub layers: Vec<LayerSpec>,
    pub learning_rate: f64,
    pub batch_size: usize,
}

impl Architecture {
    pub fn total_params(&self, input_dim: usize, output_dim: usize) -> usize {
        let mut total = 0;
        let mut prev = input_dim;
        for layer in &self.layers {
            total += prev * layer.units + layer.units; // weights + bias
            prev = layer.units;
        }
        total += prev * output_dim + output_dim;
        total
    }
}

/// NAS search space definition.
#[derive(Debug, Clone)]
pub struct NasSearchSpace {
    pub min_layers: usize,
    pub max_layers: usize,
    pub min_units: usize,
    pub max_units: usize,
    pub unit_step: usize,
    pub activations: Vec<Activation>,
    pub learning_rates: Vec<f64>,
    pub batch_sizes: Vec<usize>,
    pub dropout_range: (f64, f64),
}

impl Default for NasSearchSpace {
    fn default() -> Self {
        Self {
            min_layers: 1,
            max_layers: 5,
            min_units: 16,
            max_units: 512,
            unit_step: 16,
            activations: Activation::ALL.to_vec(),
            learning_rates: vec![0.1, 0.01, 0.001, 0.000_1],
            batch_sizes: vec![16, 32, 64, 128, 256],
            dropout_range: (0.0, 0.5),
        }
    }
}

/// Neural Architecture Search engine.
pub struct NasSearch {
    search_space: NasSearchSpace,
    n_trials: usize,
    seed: u64,
    minimize: bool,
}

impl NasSearch {
    pub const fn new(search_space: NasSearchSpace, n_trials: usize) -> Self {
        Self {
            search_space,
            n_trials,
            seed: 42,
            minimize: true,
        }
    }

    pub const fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub const fn minimize(mut self, minimize: bool) -> Self {
        self.minimize = minimize;
        self
    }

    fn sample_architecture(&self, rng: &mut Rng) -> Architecture {
        let n_layers = self.search_space.min_layers
            + rng.next_usize(self.search_space.max_layers - self.search_space.min_layers + 1);
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let n_steps = (self.search_space.max_units - self.search_space.min_units)
                / self.search_space.unit_step
                + 1;
            let units =
                self.search_space.min_units + rng.next_usize(n_steps) * self.search_space.unit_step;
            let act_idx = rng.next_usize(self.search_space.activations.len());
            let dropout = rng.uniform(
                self.search_space.dropout_range.0,
                self.search_space.dropout_range.1,
            );
            layers.push(LayerSpec {
                units,
                activation: self.search_space.activations[act_idx],
                dropout,
            });
        }
        let lr_idx = rng.next_usize(self.search_space.learning_rates.len());
        let bs_idx = rng.next_usize(self.search_space.batch_sizes.len());
        Architecture {
            layers,
            learning_rate: self.search_space.learning_rates[lr_idx],
            batch_size: self.search_space.batch_sizes[bs_idx],
        }
    }

    /// Run NAS with a user-provided evaluation function.
    pub fn run<F: FnMut(&Architecture) -> f64>(
        &self,
        mut evaluate: F,
    ) -> (Vec<(Architecture, f64)>, Option<Architecture>) {
        let mut rng = Rng::new(self.seed);
        let mut results = Vec::with_capacity(self.n_trials);
        let mut best_score = if self.minimize {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
        let mut best_arch: Option<Architecture> = None;

        for _ in 0..self.n_trials {
            let arch = self.sample_architecture(&mut rng);
            let score = evaluate(&arch);
            let is_better = if self.minimize {
                score < best_score
            } else {
                score > best_score
            };
            if is_better {
                best_score = score;
                best_arch = Some(arch.clone());
            }
            results.push((arch, score));
        }

        (results, best_arch)
    }
}

// ---------------------------------------------------------------------------
// Model Selection
// ---------------------------------------------------------------------------

/// Model selection from a set of named model evaluations.
#[derive(Debug, Clone)]
pub struct ModelSelection {
    results: Vec<(String, f64)>,
    minimize: bool,
}

impl ModelSelection {
    pub const fn new(minimize: bool) -> Self {
        Self {
            results: Vec::new(),
            minimize,
        }
    }

    /// Add a model evaluation result.
    pub fn add(&mut self, name: &str, score: f64) {
        self.results.push((name.to_string(), score));
    }

    /// Return the best model name and score.
    pub fn best(&self) -> Option<(&str, f64)> {
        if self.results.is_empty() {
            return None;
        }
        let best = if self.minimize {
            self.results
                .iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        } else {
            self.results
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        };
        best.map(|(name, score)| (name.as_str(), *score))
    }

    /// Return results sorted by score.
    pub fn ranked(&self) -> Vec<(&str, f64)> {
        let mut sorted: Vec<(&str, f64)> =
            self.results.iter().map(|(n, s)| (n.as_str(), *s)).collect();
        if self.minimize {
            sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        sorted
    }

    pub fn results(&self) -> &[(String, f64)] {
        &self.results
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Rng tests ----

    #[test]
    fn test_rng_deterministic() {
        let mut r1 = Rng::new(123);
        let mut r2 = Rng::new(123);
        for _ in 0..100 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn test_rng_different_seeds() {
        let mut r1 = Rng::new(1);
        let mut r2 = Rng::new(2);
        let v1: Vec<u64> = (0..10).map(|_| r1.next_u64()).collect();
        let v2: Vec<u64> = (0..10).map(|_| r2.next_u64()).collect();
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_rng_f64_range() {
        let mut rng = Rng::new(42);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_rng_uniform_range() {
        let mut rng = Rng::new(7);
        for _ in 0..1000 {
            let v = rng.uniform(2.0, 5.0);
            assert!(v >= 2.0 && v < 5.0);
        }
    }

    #[test]
    fn test_rng_next_usize_range() {
        let mut rng = Rng::new(99);
        for _ in 0..1000 {
            let v = rng.next_usize(10);
            assert!(v < 10);
        }
    }

    #[test]
    fn test_rng_normal_distribution() {
        let mut rng = Rng::new(42);
        let samples: Vec<f64> = (0..10000).map(|_| rng.normal(0.0, 1.0)).collect();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 0.1, "mean={mean}");
    }

    // ---- ParamSpace tests ----

    #[test]
    fn test_param_continuous() {
        let p = ParamSpace::continuous("lr", 0.001, 1.0);
        assert_eq!(p.name(), "lr");
        let mut rng = Rng::new(1);
        for _ in 0..100 {
            if let ParamValue::Continuous(v) = p.sample(&mut rng) {
                assert!(v >= 0.001 && v <= 1.0);
            } else {
                panic!("wrong variant");
            }
        }
    }

    #[test]
    fn test_param_discrete() {
        let p = ParamSpace::discrete("depth", 1, 10);
        assert_eq!(p.name(), "depth");
        let mut rng = Rng::new(2);
        for _ in 0..100 {
            if let ParamValue::Discrete(v) = p.sample(&mut rng) {
                assert!((1..=10).contains(&v));
            } else {
                panic!("wrong variant");
            }
        }
    }

    #[test]
    fn test_param_categorical() {
        let p = ParamSpace::categorical("optimizer", &["sgd", "adam", "rmsprop"]);
        assert_eq!(p.name(), "optimizer");
        let mut rng = Rng::new(3);
        for _ in 0..100 {
            if let ParamValue::Categorical(v) = p.sample(&mut rng) {
                assert!(v < 3);
            } else {
                panic!("wrong variant");
            }
        }
    }

    #[test]
    fn test_param_grid_size_continuous() {
        let p = ParamSpace::continuous("x", 0.0, 1.0);
        assert_eq!(p.grid_size(5), 5);
    }

    #[test]
    fn test_param_grid_size_discrete() {
        let p = ParamSpace::discrete("n", 1, 5);
        assert_eq!(p.grid_size(10), 5);
    }

    #[test]
    fn test_param_grid_size_categorical() {
        let p = ParamSpace::categorical("c", &["a", "b", "c"]);
        assert_eq!(p.grid_size(10), 3);
    }

    #[test]
    fn test_param_grid_value_continuous() {
        let p = ParamSpace::continuous("x", 0.0, 1.0);
        if let ParamValue::Continuous(v) = p.grid_value(0, 3) {
            assert!((v - 0.0).abs() < 1e-10);
        }
        if let ParamValue::Continuous(v) = p.grid_value(2, 3) {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_param_normalize_continuous() {
        let p = ParamSpace::continuous("x", 0.0, 10.0);
        let n = p.normalize(&ParamValue::Continuous(5.0));
        assert!((n - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_param_normalize_discrete() {
        let p = ParamSpace::discrete("n", 0, 4);
        let n = p.normalize(&ParamValue::Discrete(2));
        assert!((n - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_param_normalize_categorical() {
        let p = ParamSpace::categorical("c", &["a", "b", "c"]);
        let n = p.normalize(&ParamValue::Categorical(1));
        assert!((n - 0.5).abs() < 1e-10);
    }

    // ---- ParamValue tests ----

    #[test]
    fn test_param_value_as_f64() {
        assert_eq!(ParamValue::Continuous(3.14).as_f64(), Some(3.14));
        assert_eq!(ParamValue::Discrete(5).as_f64(), Some(5.0));
        assert_eq!(ParamValue::Categorical(0).as_f64(), None);
    }

    #[test]
    fn test_param_value_as_i64() {
        assert_eq!(ParamValue::Discrete(5).as_i64(), Some(5));
        assert_eq!(ParamValue::Continuous(3.9).as_i64(), Some(3));
        assert_eq!(ParamValue::Categorical(0).as_i64(), None);
    }

    #[test]
    fn test_param_value_as_category() {
        assert_eq!(ParamValue::Categorical(2).as_category(), Some(2));
        assert_eq!(ParamValue::Continuous(1.0).as_category(), None);
    }

    #[test]
    fn test_param_value_display() {
        let s = format!("{}", ParamValue::Continuous(1.5));
        assert!(s.contains("1.5"));
        let s = format!("{}", ParamValue::Discrete(42));
        assert_eq!(s, "42");
        let s = format!("{}", ParamValue::Categorical(3));
        assert_eq!(s, "choice[3]");
    }

    // ---- SearchSpace tests ----

    #[test]
    fn test_search_space_new() {
        let ss = SearchSpace::new();
        assert_eq!(ss.dim(), 0);
        assert!(ss.params().is_empty());
    }

    #[test]
    fn test_search_space_add() {
        let ss = SearchSpace::new()
            .add(ParamSpace::continuous("lr", 0.001, 1.0))
            .add(ParamSpace::discrete("depth", 1, 10));
        assert_eq!(ss.dim(), 2);
    }

    #[test]
    fn test_search_space_default() {
        let ss = SearchSpace::default();
        assert_eq!(ss.dim(), 0);
    }

    // ---- Trial tests ----

    #[test]
    fn test_trial_new() {
        let t = Trial::new(0, HashMap::new());
        assert_eq!(t.id, 0);
        assert_eq!(t.status, TrialStatus::Pending);
        assert!(t.metric.is_none());
    }

    #[test]
    fn test_trial_complete() {
        let mut t = Trial::new(1, HashMap::new());
        t.complete(0.95);
        assert_eq!(t.status, TrialStatus::Completed);
        assert_eq!(t.metric, Some(0.95));
    }

    #[test]
    fn test_trial_fail() {
        let mut t = Trial::new(2, HashMap::new());
        t.fail();
        assert_eq!(t.status, TrialStatus::Failed);
    }

    #[test]
    fn test_trial_prune() {
        let mut t = Trial::new(3, HashMap::new());
        t.prune();
        assert_eq!(t.status, TrialStatus::Pruned);
    }

    #[test]
    fn test_trial_report_epoch() {
        let mut t = Trial::new(0, HashMap::new());
        t.report_epoch(0.5);
        t.report_epoch(0.4);
        t.report_epoch(0.3);
        assert_eq!(t.epoch_metrics.len(), 3);
        assert!((t.epoch_metrics[2] - 0.3).abs() < f64::EPSILON);
    }

    // ---- TrialTracker tests ----

    #[test]
    fn test_tracker_new() {
        let tracker = TrialTracker::new();
        assert!(tracker.is_empty());
        assert_eq!(tracker.len(), 0);
    }

    #[test]
    fn test_tracker_add() {
        let mut tracker = TrialTracker::new();
        tracker.add_trial(Trial::new(0, HashMap::new()));
        assert_eq!(tracker.len(), 1);
        assert!(!tracker.is_empty());
    }

    #[test]
    fn test_tracker_completed_trials() {
        let mut tracker = TrialTracker::new();
        let mut t1 = Trial::new(0, HashMap::new());
        t1.complete(1.0);
        let t2 = Trial::new(1, HashMap::new());
        let mut t3 = Trial::new(2, HashMap::new());
        t3.complete(0.5);
        tracker.add_trial(t1);
        tracker.add_trial(t2);
        tracker.add_trial(t3);
        assert_eq!(tracker.completed_trials().len(), 2);
    }

    #[test]
    fn test_tracker_best_trial_minimize() {
        let mut tracker = TrialTracker::new();
        let mut t1 = Trial::new(0, HashMap::new());
        t1.complete(1.0);
        let mut t2 = Trial::new(1, HashMap::new());
        t2.complete(0.3);
        let mut t3 = Trial::new(2, HashMap::new());
        t3.complete(0.7);
        tracker.add_trial(t1);
        tracker.add_trial(t2);
        tracker.add_trial(t3);
        let best = tracker.best_trial(true).unwrap();
        assert_eq!(best.id, 1);
        assert_eq!(best.metric, Some(0.3));
    }

    #[test]
    fn test_tracker_best_trial_maximize() {
        let mut tracker = TrialTracker::new();
        let mut t1 = Trial::new(0, HashMap::new());
        t1.complete(1.0);
        let mut t2 = Trial::new(1, HashMap::new());
        t2.complete(0.3);
        tracker.add_trial(t1);
        tracker.add_trial(t2);
        let best = tracker.best_trial(false).unwrap();
        assert_eq!(best.id, 0);
    }

    #[test]
    fn test_tracker_best_trial_empty() {
        let tracker = TrialTracker::new();
        assert!(tracker.best_trial(true).is_none());
    }

    #[test]
    fn test_tracker_default() {
        let tracker = TrialTracker::default();
        assert!(tracker.is_empty());
    }

    // ---- EarlyStopping tests ----

    #[test]
    fn test_early_stopping_no_stop() {
        let mut es = EarlyStopping::new(3, 0.0, true);
        assert!(!es.should_stop(1.0));
        assert!(!es.should_stop(0.9));
        assert!(!es.should_stop(0.8));
    }

    #[test]
    fn test_early_stopping_triggers() {
        let mut es = EarlyStopping::new(3, 0.0, true);
        assert!(!es.should_stop(1.0));
        assert!(!es.should_stop(1.1)); // no improvement
        assert!(!es.should_stop(1.2)); // counter=2
        assert!(es.should_stop(1.3)); // counter=3 -> stop
    }

    #[test]
    fn test_early_stopping_maximize() {
        let mut es = EarlyStopping::new(2, 0.0, false);
        assert!(!es.should_stop(0.5));
        assert!(!es.should_stop(0.6)); // improved
        assert!(!es.should_stop(0.55)); // no improvement, counter=1
        assert!(es.should_stop(0.4)); // counter=2 -> stop
    }

    #[test]
    fn test_early_stopping_min_delta() {
        let mut es = EarlyStopping::new(2, 0.1, true);
        assert!(!es.should_stop(1.0));
        // 0.95 is not enough improvement (delta < 0.1)
        assert!(!es.should_stop(0.95)); // counter=1
        assert!(es.should_stop(0.92)); // counter=2 -> stop
    }

    #[test]
    fn test_early_stopping_reset() {
        let mut es = EarlyStopping::new(2, 0.0, true);
        es.should_stop(1.0);
        es.should_stop(2.0);
        es.reset();
        assert_eq!(es.counter(), 0);
        assert!(es.best_value().is_infinite());
    }

    #[test]
    fn test_early_stopping_best_value() {
        let mut es = EarlyStopping::new(5, 0.0, true);
        es.should_stop(3.0);
        es.should_stop(2.0);
        es.should_stop(4.0);
        assert!((es.best_value() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_early_stopping_resets_counter_on_improvement() {
        let mut es = EarlyStopping::new(3, 0.0, true);
        es.should_stop(1.0);
        es.should_stop(1.1); // counter=1
        es.should_stop(0.5); // improvement, counter=0
        assert_eq!(es.counter(), 0);
    }

    // ---- KFold tests ----

    #[test]
    fn test_kfold_split() {
        let kf = KFold::new(5);
        let folds = kf.split(100);
        assert_eq!(folds.len(), 5);
        for (train, val) in &folds {
            assert_eq!(train.len() + val.len(), 100);
            assert_eq!(val.len(), 20);
        }
    }

    #[test]
    fn test_kfold_split_uneven() {
        let kf = KFold::new(3);
        let folds = kf.split(10);
        assert_eq!(folds.len(), 3);
        // 10/3 = 3 remainder 1, so first fold has 4 val, others 3
        let val_sizes: Vec<usize> = folds.iter().map(|(_, v)| v.len()).collect();
        assert_eq!(val_sizes.iter().sum::<usize>(), 10);
    }

    #[test]
    fn test_kfold_no_overlap() {
        let kf = KFold::new(3);
        let folds = kf.split(9);
        for (train, val) in &folds {
            for v in val {
                assert!(!train.contains(v));
            }
        }
    }

    #[test]
    fn test_kfold_k() {
        let kf = KFold::new(10);
        assert_eq!(kf.k(), 10);
    }

    #[test]
    #[should_panic]
    fn test_kfold_k_less_than_2() {
        let _ = KFold::new(1);
    }

    // ---- StratifiedKFold tests ----

    #[test]
    fn test_stratified_kfold() {
        let skf = StratifiedKFold::new(3);
        // 6 samples of class 0, 3 samples of class 1
        let labels = vec![0, 0, 0, 0, 0, 0, 1, 1, 1];
        let folds = skf.split(&labels);
        assert_eq!(folds.len(), 3);
        for (train, val) in &folds {
            assert_eq!(train.len() + val.len(), 9);
        }
    }

    #[test]
    fn test_stratified_kfold_preserves_ratio() {
        let skf = StratifiedKFold::new(2);
        let labels = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let folds = skf.split(&labels);
        for (_, val) in &folds {
            let n_class0 = val.iter().filter(|&&i| labels[i] == 0).count();
            let n_class1 = val.iter().filter(|&&i| labels[i] == 1).count();
            assert_eq!(n_class0, n_class1);
        }
    }

    // ---- Normal PDF/CDF tests ----

    #[test]
    fn test_normal_pdf_at_zero() {
        let v = normal_pdf(0.0);
        let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!((v - expected).abs() < 1e-10);
    }

    #[test]
    fn test_normal_cdf_at_zero() {
        let v = normal_cdf(0.0);
        assert!((v - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_normal_cdf_large_positive() {
        let v = normal_cdf(5.0);
        assert!(v > 0.999);
    }

    #[test]
    fn test_normal_cdf_large_negative() {
        let v = normal_cdf(-5.0);
        assert!(v < 0.001);
    }

    // ---- GP tests ----

    #[test]
    fn test_gp_fit_predict() {
        let mut gp = GaussianProcess::new(1.0, 1e-5);
        let x = vec![vec![0.0], vec![0.5], vec![1.0]];
        let y = vec![0.0, 0.25, 1.0];
        gp.fit(x, y);
        let (mean, var) = gp.predict(&[0.5]);
        assert!((mean - 0.25).abs() < 0.2, "mean={mean}");
        assert!(var > 0.0);
    }

    #[test]
    fn test_gp_empty() {
        let mut gp = GaussianProcess::new(1.0, 1e-5);
        gp.fit(vec![], vec![]);
        let (mean, var) = gp.predict(&[0.5]);
        assert!((mean - 0.0).abs() < f64::EPSILON);
        assert!((var - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gp_rbf_kernel_same_point() {
        let gp = GaussianProcess::new(1.0, 1e-5);
        let k = gp.rbf_kernel(&[1.0, 2.0], &[1.0, 2.0]);
        assert!((k - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gp_rbf_kernel_distant() {
        let gp = GaussianProcess::new(0.1, 1e-5);
        let k = gp.rbf_kernel(&[0.0], &[100.0]);
        assert!(k < 1e-10);
    }

    // ---- Expected Improvement tests ----

    #[test]
    fn test_ei_zero_variance() {
        let ei = expected_improvement(0.5, 0.0, 1.0, true);
        assert!((ei - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ei_positive() {
        let ei = expected_improvement(0.5, 1.0, 1.0, true);
        assert!(ei > 0.0);
    }

    #[test]
    fn test_ei_maximize() {
        let ei = expected_improvement(2.0, 1.0, 1.0, false);
        assert!(ei > 0.0);
    }

    // ---- GridSearch tests ----

    #[test]
    fn test_grid_search_candidates_count() {
        let space = SearchSpace::new()
            .add(ParamSpace::continuous("x", 0.0, 1.0))
            .add(ParamSpace::discrete("n", 1, 3));
        let gs = GridSearch::new(space, 3);
        let cands = gs.candidates();
        assert_eq!(cands.len(), 3 * 3); // 3 continuous steps * 3 discrete values
    }

    #[test]
    fn test_grid_search_run() {
        let space = SearchSpace::new().add(ParamSpace::continuous("x", 0.0, 1.0));
        let gs = GridSearch::new(space, 5);
        let tracker = gs.run(|params| {
            let x = params["x"].as_f64().unwrap();
            (x - 0.5).powi(2)
        });
        let best = tracker.best_trial(true).unwrap();
        assert!(best.metric.unwrap() < 0.1);
    }

    #[test]
    fn test_grid_search_categorical() {
        let space = SearchSpace::new().add(ParamSpace::categorical("opt", &["a", "b"]));
        let gs = GridSearch::new(space, 3);
        let cands = gs.candidates();
        assert_eq!(cands.len(), 2);
    }

    #[test]
    fn test_grid_search_all_completed() {
        let space = SearchSpace::new().add(ParamSpace::discrete("n", 1, 3));
        let gs = GridSearch::new(space, 5);
        let tracker = gs.run(|_| 1.0);
        assert_eq!(tracker.completed_trials().len(), 3);
    }

    // ---- RandomSearch tests ----

    #[test]
    fn test_random_search_candidates_count() {
        let space = SearchSpace::new().add(ParamSpace::continuous("x", 0.0, 1.0));
        let rs = RandomSearch::new(space, 20, 42);
        let cands = rs.candidates();
        assert_eq!(cands.len(), 20);
    }

    #[test]
    fn test_random_search_deterministic() {
        let space = SearchSpace::new().add(ParamSpace::continuous("x", 0.0, 1.0));
        let c1 = RandomSearch::new(space.clone(), 10, 42).candidates();
        let c2 = RandomSearch::new(space, 10, 42).candidates();
        for (a, b) in c1.iter().zip(c2.iter()) {
            let va = a["x"].as_f64().unwrap();
            let vb = b["x"].as_f64().unwrap();
            assert!((va - vb).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_random_search_run() {
        let space = SearchSpace::new().add(ParamSpace::continuous("x", -5.0, 5.0));
        let rs = RandomSearch::new(space, 50, 0);
        let tracker = rs.run(|p| {
            let x = p["x"].as_f64().unwrap();
            x * x
        });
        let best = tracker.best_trial(true).unwrap();
        assert!(best.metric.unwrap() < 2.0);
    }

    #[test]
    fn test_random_search_different_seeds() {
        let space = SearchSpace::new().add(ParamSpace::continuous("x", 0.0, 1.0));
        let c1 = RandomSearch::new(space.clone(), 10, 1).candidates();
        let c2 = RandomSearch::new(space, 10, 2).candidates();
        let same = c1.iter().zip(c2.iter()).all(|(a, b)| {
            (a["x"].as_f64().unwrap() - b["x"].as_f64().unwrap()).abs() < f64::EPSILON
        });
        assert!(!same);
    }

    // ---- BayesianOptimizer tests ----

    #[test]
    fn test_bayesian_optimizer_basic() {
        let space = SearchSpace::new().add(ParamSpace::continuous("x", -5.0, 5.0));
        let bo = BayesianOptimizer::new(space, 20).seed(42);
        let tracker = bo.run(|p| {
            let x = p["x"].as_f64().unwrap();
            x * x
        });
        let best = tracker.best_trial(true).unwrap();
        assert!(
            best.metric.unwrap() < 5.0,
            "metric={}",
            best.metric.unwrap()
        );
    }

    #[test]
    fn test_bayesian_optimizer_maximize() {
        let space = SearchSpace::new().add(ParamSpace::continuous("x", 0.0, 1.0));
        let bo = BayesianOptimizer::new(space, 15).minimize(false).seed(7);
        let tracker = bo.run(|p| {
            let x = p["x"].as_f64().unwrap();
            -(x - 0.8).powi(2) + 1.0
        });
        let best = tracker.best_trial(false).unwrap();
        assert!(best.metric.unwrap() > 0.5);
    }

    #[test]
    fn test_bayesian_optimizer_n_trials() {
        let space = SearchSpace::new().add(ParamSpace::continuous("x", 0.0, 1.0));
        let bo = BayesianOptimizer::new(space, 10);
        let tracker = bo.run(|_| 1.0);
        assert_eq!(tracker.len(), 10);
    }

    #[test]
    fn test_bayesian_optimizer_multi_dim() {
        let space = SearchSpace::new()
            .add(ParamSpace::continuous("x", -2.0, 2.0))
            .add(ParamSpace::continuous("y", -2.0, 2.0));
        let bo = BayesianOptimizer::new(space, 25).seed(123);
        let tracker = bo.run(|p| {
            let x = p["x"].as_f64().unwrap();
            let y = p["y"].as_f64().unwrap();
            x * x + y * y
        });
        let best = tracker.best_trial(true).unwrap();
        assert!(best.metric.unwrap() < 3.0);
    }

    #[test]
    fn test_bayesian_builder_methods() {
        let space = SearchSpace::new().add(ParamSpace::continuous("x", 0.0, 1.0));
        let bo = BayesianOptimizer::new(space, 5)
            .seed(99)
            .n_initial(2)
            .length_scale(0.5)
            .noise(1e-3)
            .n_candidates(50)
            .minimize(true);
        let tracker = bo.run(|_| 1.0);
        assert_eq!(tracker.len(), 5);
    }

    #[test]
    fn test_bayesian_discrete_and_categorical() {
        let space = SearchSpace::new()
            .add(ParamSpace::discrete("n", 1, 5))
            .add(ParamSpace::categorical("act", &["relu", "tanh"]));
        let bo = BayesianOptimizer::new(space, 10).seed(42);
        let tracker = bo.run(|p| {
            let n = p["n"].as_i64().unwrap() as f64;
            let act = p["act"].as_category().unwrap();
            n + act as f64
        });
        assert_eq!(tracker.len(), 10);
    }

    // ---- Activation tests ----

    #[test]
    fn test_activation_relu() {
        assert!((Activation::ReLU.apply(1.0) - 1.0).abs() < f64::EPSILON);
        assert!((Activation::ReLU.apply(-1.0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_activation_sigmoid() {
        let v = Activation::Sigmoid.apply(0.0);
        assert!((v - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_activation_tanh() {
        let v = Activation::Tanh.apply(0.0);
        assert!(v.abs() < 1e-10);
    }

    #[test]
    fn test_activation_leaky_relu() {
        assert!((Activation::LeakyReLU.apply(1.0) - 1.0).abs() < f64::EPSILON);
        assert!((Activation::LeakyReLU.apply(-1.0) - (-0.01)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_activation_swish() {
        assert!((Activation::Swish.apply(0.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_activation_gelu() {
        // GELU(0) ~ 0
        assert!(Activation::GELU.apply(0.0).abs() < 1e-5);
    }

    #[test]
    fn test_activation_name() {
        assert_eq!(Activation::ReLU.name(), "relu");
        assert_eq!(Activation::Sigmoid.name(), "sigmoid");
        assert_eq!(Activation::Tanh.name(), "tanh");
        assert_eq!(Activation::LeakyReLU.name(), "leaky_relu");
        assert_eq!(Activation::Swish.name(), "swish");
        assert_eq!(Activation::GELU.name(), "gelu");
    }

    #[test]
    fn test_activation_all_count() {
        assert_eq!(Activation::ALL.len(), 6);
    }

    // ---- Architecture tests ----

    #[test]
    fn test_architecture_total_params() {
        let arch = Architecture {
            layers: vec![
                LayerSpec {
                    units: 64,
                    activation: Activation::ReLU,
                    dropout: 0.0,
                },
                LayerSpec {
                    units: 32,
                    activation: Activation::ReLU,
                    dropout: 0.0,
                },
            ],
            learning_rate: 0.01,
            batch_size: 32,
        };
        // input=10, output=1
        // layer1: 10*64+64 = 704
        // layer2: 64*32+32 = 2080
        // output: 32*1+1 = 33
        // total: 2817
        assert_eq!(arch.total_params(10, 1), 2817);
    }

    #[test]
    fn test_architecture_single_layer() {
        let arch = Architecture {
            layers: vec![LayerSpec {
                units: 128,
                activation: Activation::Sigmoid,
                dropout: 0.1,
            }],
            learning_rate: 0.001,
            batch_size: 64,
        };
        // 5*128+128 + 128*2+2 = 768 + 258 = 1026
        assert_eq!(arch.total_params(5, 2), 1026);
    }

    // ---- NAS tests ----

    #[test]
    fn test_nas_search_basic() {
        let nas = NasSearch::new(NasSearchSpace::default(), 10).seed(42);
        let (results, best) = nas.run(|arch| arch.total_params(10, 1) as f64);
        assert_eq!(results.len(), 10);
        assert!(best.is_some());
    }

    #[test]
    fn test_nas_search_maximize() {
        let nas = NasSearch::new(NasSearchSpace::default(), 10)
            .minimize(false)
            .seed(1);
        let (_, best) = nas.run(|arch| -(arch.total_params(10, 1) as f64));
        assert!(best.is_some());
    }

    #[test]
    fn test_nas_search_custom_space() {
        let space = NasSearchSpace {
            min_layers: 2,
            max_layers: 3,
            min_units: 32,
            max_units: 64,
            unit_step: 32,
            activations: vec![Activation::ReLU, Activation::Tanh],
            learning_rates: vec![0.01],
            batch_sizes: vec![32],
            dropout_range: (0.0, 0.1),
        };
        let nas = NasSearch::new(space, 5).seed(7);
        let (results, _) = nas.run(|arch| {
            assert!(arch.layers.len() >= 2 && arch.layers.len() <= 3);
            for l in &arch.layers {
                assert!(l.units == 32 || l.units == 64);
            }
            0.0
        });
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_nas_deterministic() {
        let space = NasSearchSpace::default();
        let r1 = NasSearch::new(space.clone(), 5)
            .seed(42)
            .run(|a| a.total_params(10, 1) as f64);
        let r2 = NasSearch::new(space, 5)
            .seed(42)
            .run(|a| a.total_params(10, 1) as f64);
        for (a, b) in r1.0.iter().zip(r2.0.iter()) {
            assert!((a.1 - b.1).abs() < f64::EPSILON);
        }
    }

    // ---- ModelSelection tests ----

    #[test]
    fn test_model_selection_minimize() {
        let mut ms = ModelSelection::new(true);
        ms.add("linear", 0.5);
        ms.add("tree", 0.3);
        ms.add("svm", 0.7);
        let (name, score) = ms.best().unwrap();
        assert_eq!(name, "tree");
        assert!((score - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_model_selection_maximize() {
        let mut ms = ModelSelection::new(false);
        ms.add("linear", 0.5);
        ms.add("tree", 0.9);
        ms.add("svm", 0.7);
        let (name, score) = ms.best().unwrap();
        assert_eq!(name, "tree");
        assert!((score - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_model_selection_empty() {
        let ms = ModelSelection::new(true);
        assert!(ms.best().is_none());
    }

    #[test]
    fn test_model_selection_ranked() {
        let mut ms = ModelSelection::new(true);
        ms.add("a", 3.0);
        ms.add("b", 1.0);
        ms.add("c", 2.0);
        let ranked = ms.ranked();
        assert_eq!(ranked[0].0, "b");
        assert_eq!(ranked[1].0, "c");
        assert_eq!(ranked[2].0, "a");
    }

    #[test]
    fn test_model_selection_ranked_maximize() {
        let mut ms = ModelSelection::new(false);
        ms.add("a", 3.0);
        ms.add("b", 1.0);
        ms.add("c", 2.0);
        let ranked = ms.ranked();
        assert_eq!(ranked[0].0, "a");
        assert_eq!(ranked[2].0, "b");
    }

    #[test]
    fn test_model_selection_results() {
        let mut ms = ModelSelection::new(true);
        ms.add("x", 1.0);
        assert_eq!(ms.results().len(), 1);
    }

    // ---- Integration tests ----

    #[test]
    fn test_full_pipeline_grid() {
        let space = SearchSpace::new().add(ParamSpace::continuous("x", -2.0, 2.0));
        let gs = GridSearch::new(space, 21);
        let tracker = gs.run(|p| {
            let x = p["x"].as_f64().unwrap();
            (x - 0.3).powi(2)
        });
        let best = tracker.best_trial(true).unwrap();
        assert!(best.metric.unwrap() < 0.05);
    }

    #[test]
    fn test_full_pipeline_random_with_early_stopping() {
        let space = SearchSpace::new()
            .add(ParamSpace::continuous("lr", 0.001, 1.0))
            .add(ParamSpace::discrete("epochs", 10, 100));
        let rs = RandomSearch::new(space, 30, 42);
        let tracker = rs.run(|p| {
            let lr = p["lr"].as_f64().unwrap();
            let epochs = p["epochs"].as_i64().unwrap() as f64;
            // Simulate: use early stopping per trial
            let mut es = EarlyStopping::new(5, 0.001, true);
            let mut loss = 10.0;
            for _ in 0..epochs as usize {
                loss *= 1.0 - lr * 0.1;
                if es.should_stop(loss) {
                    break;
                }
            }
            loss
        });
        let best = tracker.best_trial(true).unwrap();
        assert!(best.metric.unwrap() < 10.0);
    }

    #[test]
    fn test_full_pipeline_bayesian_with_cv() {
        let space = SearchSpace::new().add(ParamSpace::continuous("alpha", 0.01, 10.0));
        let bo = BayesianOptimizer::new(space, 15).seed(0);
        let tracker = bo.run(|p| {
            let alpha = p["alpha"].as_f64().unwrap();
            // Simulate cross-validation
            let kf = KFold::new(3);
            let data: Vec<f64> = (0..30).map(|i| i as f64).collect();
            let folds = kf.split(data.len());
            let mut total = 0.0;
            for (_, val) in &folds {
                let val_mean: f64 = val.iter().map(|&i| data[i]).sum::<f64>() / val.len() as f64;
                total += (val_mean - alpha).powi(2);
            }
            total / folds.len() as f64
        });
        assert_eq!(tracker.len(), 15);
        let best = tracker.best_trial(true).unwrap();
        assert!(best.metric.is_some());
    }

    #[test]
    fn test_cholesky_identity() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let l = GaussianProcess::cholesky(&a, 2);
        assert!((l[0] - 1.0).abs() < 1e-10);
        assert!((l[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_forward_backward_sub() {
        let l = vec![2.0, 0.0, 1.0, 3.0];
        let b = vec![4.0, 7.0];
        let z = GaussianProcess::forward_sub(&l, &b, 2);
        // z[0] = 4/2 = 2, z[1] = (7-1*2)/3 = 5/3
        assert!((z[0] - 2.0).abs() < 1e-10);
        assert!((z[1] - 5.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_nas_with_model_selection() {
        let nas = NasSearch::new(NasSearchSpace::default(), 10).seed(42);
        let (results, _) = nas.run(|arch| {
            let params = arch.total_params(10, 1);
            // Prefer smaller models
            params as f64
        });
        let mut ms = ModelSelection::new(true);
        for (i, (_, score)) in results.iter().enumerate() {
            ms.add(&format!("arch_{i}"), *score);
        }
        assert!(ms.best().is_some());
    }

    #[test]
    fn test_early_stopping_with_trial() {
        let mut trial = Trial::new(0, HashMap::new());
        let mut es = EarlyStopping::new(3, 0.0, true);
        let metrics = [1.0, 0.9, 0.8, 0.85, 0.86, 0.87];
        let mut stopped_at = metrics.len();
        for (i, &m) in metrics.iter().enumerate() {
            trial.report_epoch(m);
            if es.should_stop(m) {
                stopped_at = i;
                trial.prune();
                break;
            }
        }
        assert!(stopped_at < metrics.len());
        assert_eq!(trial.status, TrialStatus::Pruned);
    }
}
