//! Cross-validation (`KFold` / `StratifiedKFold`).

use std::collections::BTreeMap;

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
        // Group indices by label, in ascending label order (a HashMap's order differs between runs, and the
        // order of each fold's indices follows it)
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
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

#[cfg(test)]
mod tests {
    use super::*;

    /// oracle (closed form, not the implementation): labels grouped in ascending label order, each group split by
    /// KFold over its own indices, the groups concatenated in that order. labels 0: [1, 3, 6], 1: [0, 2, 7],
    /// 2: [4, 5, 8]; KFold(2) over 3 indices gives validation [0, 1] then [2]
    #[test]
    fn stratified_folds_are_in_label_order_and_the_same_every_call() {
        let labels = [1, 0, 1, 0, 2, 2, 0, 1, 2];
        let want = vec![
            (vec![6, 7, 8], vec![1, 3, 0, 2, 4, 5]),
            (vec![1, 3, 0, 2, 4, 5], vec![6, 7, 8]),
        ];
        // a HashMap's iteration order differs between maps (each gets its own random keys), so repeated calls on
        // fresh maps expose an order taken from one
        for _ in 0..64 {
            assert_eq!(StratifiedKFold::new(2).split(&labels), want);
        }
    }
}
