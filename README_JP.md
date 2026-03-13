[English](README.md) | **日本語**

# ALICE-AutoML

[Project A.L.I.C.E.](https://github.com/anthropics/alice) のAutoMLフレームワーク

## 概要

`alice-automl` は純RustのAutoMLフレームワークです。ハイパーパラメータ最適化、ニューラルアーキテクチャ探索、早期停止、交差検証を外部依存ゼロで提供します。

## 機能

- **探索空間定義** — 連続値・離散値・カテゴリカルのパラメータ空間
- **グリッドサーチ** — パラメータグリッドの網羅的探索
- **ランダムサーチ** — ランダム化ハイパーパラメータサンプリング
- **ベイズ最適化** — 代理モデルベースの最適化
- **ニューラルアーキテクチャ探索 (NAS)** — モデル構造の自動探索
- **早期停止** — Patienceベースの学習終了判定
- **交差検証** — k分割評価とスコア集約
- **試行追跡** — パラメータ・メトリクス履歴付き実験ログ
- **内蔵PRNG** — xoshiro256** + Box-Muller正規分布

## クイックスタート

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

## アーキテクチャ

```
alice-automl
├── Rng            — xoshiro256** 擬似乱数生成器
├── ParamSpace     — パラメータ次元定義
├── SearchSpace    — 多次元探索空間
├── GridSearch     — 網羅的パラメータ探索
├── RandomSearch   — ランダムサンプリング
├── BayesianOpt    — 代理モデルベース最適化
├── NAS            — ニューラルアーキテクチャ探索
├── EarlyStopping  — Patienceベース終了判定
└── CrossValidator — k分割交差検証
```

## ライセンス

MIT OR Apache-2.0
