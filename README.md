**English** | [日本語](README_JP.md)

# ALICE-AutoML

AutoML framework for [Project A.L.I.C.E.](https://github.com/anthropics/alice)

## Overview

`alice-automl` is a pure Rust AutoML framework providing hyperparameter optimization, neural architecture search, early stopping, and cross-validation — all with zero external dependencies.

## Features

- **Search Space Definition** — continuous, discrete, and categorical parameter spaces
- **Grid Search** — exhaustive search over parameter grid
- **Random Search** — randomized hyperparameter sampling
- **Bayesian Optimization** — surrogate model-based optimization
- **Neural Architecture Search (NAS)** — automated model architecture discovery
- **Early Stopping** — patience-based training termination
- **Cross-Validation** — k-fold evaluation with score aggregation
- **Trial Tracking** — experiment logging with parameter/metric history
- **Built-in PRNG** — xoshiro256** with Box-Muller normal distribution

## Quick Start

```rust
use alice_automl::{ParamSpace, SearchSpace, Rng};

let space = SearchSpace::new(vec![
    ParamSpace::Continuous { name: "lr".into(), low: 1e-5, high: 1e-1 },
    ParamSpace::Discrete { name: "layers".into(), low: 1, high: 8 },
    ParamSpace::Categorical { name: "act".into(), choices: vec!["relu".into(), "gelu".into()] },
]);

let mut rng = Rng::new(42);
let sample = space.sample(&mut rng);
```

## Architecture

```
alice-automl
├── Rng            — xoshiro256** PRNG
├── ParamSpace     — parameter dimension definitions
├── SearchSpace    — multi-dimensional search space
├── GridSearch     — exhaustive parameter search
├── RandomSearch   — randomized sampling
├── BayesianOpt    — surrogate-based optimization
├── NAS            — neural architecture search
├── EarlyStopping  — patience-based termination
└── CrossValidator — k-fold cross-validation
```

## License

MIT OR Apache-2.0
