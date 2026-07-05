//! Cross-validation (`KFold` / `StratifiedKFold`).

use std::collections::HashMap;

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
