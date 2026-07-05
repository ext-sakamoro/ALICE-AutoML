//! Integration tests spanning multiple modules.

#![allow(
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::explicit_iter_loop,
    clippy::bool_to_int_with_if,
    clippy::approx_constant,
    clippy::cast_lossless,
    clippy::redundant_clone,
    clippy::format_collect,
    clippy::similar_names,
    clippy::needless_collect,
    clippy::iter_cloned_collect,
    clippy::suboptimal_flops,
    clippy::should_panic_without_expect,
    clippy::manual_range_contains
)]

use crate::acquisition::{expected_improvement, normal_cdf, normal_pdf};
use crate::cross_val::*;
use crate::early_stopping::*;
use crate::gaussian_process::GaussianProcess;
use crate::model_selection::*;
use crate::nas::*;
use crate::rng::*;
use crate::search_space::*;
use crate::search_strategies::*;
use crate::trial::*;
use std::collections::HashMap;

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
    let same = c1
        .iter()
        .zip(c2.iter())
        .all(|(a, b)| (a["x"].as_f64().unwrap() - b["x"].as_f64().unwrap()).abs() < f64::EPSILON);
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
