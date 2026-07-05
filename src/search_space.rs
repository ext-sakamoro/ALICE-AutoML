//! Search space (`ParamSpace` / `ParamValue` / `SearchSpace`).

use crate::rng::Rng;
use std::fmt;

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
    pub fn sample(&self, rng: &mut Rng) -> ParamValue {
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
    pub const fn grid_size(&self, continuous_steps: usize) -> usize {
        match self {
            Self::Continuous { .. } => continuous_steps,
            Self::Discrete { low, high, .. } => (*high - *low + 1) as usize,
            Self::Categorical { choices, .. } => choices.len(),
        }
    }

    /// Return the i-th grid value.
    pub fn grid_value(&self, idx: usize, continuous_steps: usize) -> ParamValue {
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
    pub fn normalize(&self, val: &ParamValue) -> f64 {
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
