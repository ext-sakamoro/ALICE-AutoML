//! Neural Architecture Search (NAS).

use crate::rng::Rng;

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
