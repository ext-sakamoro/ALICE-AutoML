//! Gaussian Process (for Bayesian Optimization).

// Gaussian Process (for Bayesian Optimization)
// ---------------------------------------------------------------------------

/// Gaussian Process with RBF (squared exponential) kernel.
#[derive(Debug, Clone)]
pub struct GaussianProcess {
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    length_scale: f64,
    noise: f64,
    alpha: Vec<f64>, // K^{-1} y
    y_mean: f64,
}

impl GaussianProcess {
    pub const fn new(length_scale: f64, noise: f64) -> Self {
        Self {
            x_train: Vec::new(),
            y_train: Vec::new(),
            length_scale,
            noise,
            alpha: Vec::new(),
            y_mean: 0.0,
        }
    }

    pub fn rbf_kernel(&self, a: &[f64], b: &[f64]) -> f64 {
        let sq_dist: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum();
        (-0.5 * sq_dist / (self.length_scale * self.length_scale)).exp()
    }

    /// Fit the GP to training data. Uses Cholesky solve.
    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) {
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

    pub fn cholesky(a: &[f64], n: usize) -> Vec<f64> {
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

    pub fn forward_sub(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
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

    pub fn backward_sub(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
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
    pub fn predict(&self, x: &[f64]) -> (f64, f64) {
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
