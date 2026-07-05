//! Xoshiro256** PRNG (`Rng`).

// Pseudo-random number generator (Xoshiro256**)
// ---------------------------------------------------------------------------

/// A simple PRNG based on xoshiro256**.
#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Create a new RNG from a seed.
    pub fn new(seed: u64) -> Self {
        let mut s = [0u64; 4];
        // SplitMix64 to initialize state
        let mut z = seed;
        for slot in &mut s {
            z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            *slot = x ^ (x >> 31);
        }
        Self { s }
    }

    /// Generate next u64.
    pub const fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Generate a uniform f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Generate a uniform f64 in [lo, hi).
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        (hi - lo).mul_add(self.next_f64(), lo)
    }

    /// Generate a usize in [0, n).
    pub const fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Generate a standard normal via Box-Muller.
    pub fn normal(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        std.mul_add(z, mean)
    }
}
