//! Acquisition functions (Expected Improvement + normal pdf/cdf).

// Acquisition Functions
// ---------------------------------------------------------------------------

/// Expected Improvement acquisition function.
pub fn expected_improvement(mean: f64, var: f64, best: f64, minimize: bool) -> f64 {
    let std = var.sqrt();
    if std < 1e-12 {
        return 0.0;
    }
    let (diff, z) = if minimize {
        let d = best - mean;
        (d, d / std)
    } else {
        let d = mean - best;
        (d, d / std)
    };
    diff.mul_add(normal_cdf(z), std * normal_pdf(z))
}

/// Standard normal PDF.
pub fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Approximate standard normal CDF using Abramowitz & Stegun.
pub fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / 0.231_641_9f64.mul_add(x.abs(), 1.0);
    let d = 1.330_274_429f64.mul_add(
        t.powi(5),
        1.821_255_978f64.mul_add(
            -t.powi(4),
            1.781_477_937f64.mul_add(
                t.powi(3),
                0.319_381_530f64.mul_add(t, -(0.356_563_782 * t * t)),
            ),
        ),
    );
    let approx = normal_pdf(x.abs()).mul_add(-d, 1.0);
    if x >= 0.0 {
        approx
    } else {
        1.0 - approx
    }
}
