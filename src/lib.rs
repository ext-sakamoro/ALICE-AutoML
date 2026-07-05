//! ALICE-AutoML: Pure Rust `AutoML` framework.
//!
//! Provides hyperparameter search (grid, random, Bayesian optimization),
//! neural architecture search (NAS), early stopping, cross-validation,
//! model selection, search space definition, and trial tracking.

#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::module_name_repetitions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::missing_errors_doc,
    clippy::many_single_char_names,
    clippy::cast_possible_wrap,
    clippy::should_implement_trait,
    clippy::wildcard_imports,
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::cast_lossless,
    clippy::float_cmp,
    clippy::suboptimal_flops
)]

pub mod acquisition;
pub mod cross_val;
pub mod early_stopping;
pub mod gaussian_process;
pub mod model_selection;
pub mod nas;
pub mod prelude;
pub mod rng;
pub mod search_space;
pub mod search_strategies;
pub mod trial;

#[cfg(test)]
mod integration_tests;

// Backward-compat re-exports.
pub use crate::cross_val::*;
pub use crate::early_stopping::*;
pub use crate::model_selection::*;
pub use crate::nas::*;
pub use crate::rng::*;
pub use crate::search_space::*;
pub use crate::search_strategies::*;
pub use crate::trial::*;
