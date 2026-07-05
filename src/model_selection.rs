//! Model selection (`ModelSelection`).

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
