//! Shared scalar helpers: decibel conversion and denormal (flush-to-zero)
//! protection.
//!
//! Floating-point denormals are the classic silent killer of real-time audio
//! filters: once a filter's state decays into the denormal range, every
//! multiply on that state can trigger a microcode assist, and the process
//! path stalls by orders of magnitude. Every recursive state update in this
//! crate is flushed through [`flush_denormal`], which zeroes anything below
//! [`DENORMAL_EPSILON`] (1e-20 ≈ −400 dBFS — 200 dB below any audible f64
//! signal, and far above the 1e-308 denormal range).

/// Magnitude below which samples are flushed to exactly zero by
/// [`flush_denormal`]. 1e-20 ≈ −400 dBFS.
pub const DENORMAL_EPSILON: f64 = 1e-20;

/// Flush-to-zero: returns `0.0` for any value with magnitude below
/// [`DENORMAL_EPSILON`], otherwise the value unchanged.
///
/// NaN and ±∞ pass through unchanged — this crate never produces them from
/// finite inputs, and flushing would hide upstream corruption.
#[inline]
#[must_use]
pub fn flush_denormal(value: f64) -> f64 {
    if value.abs() < DENORMAL_EPSILON {
        0.0
    } else {
        value
    }
}

/// Convert decibels to a linear amplitude ratio: `10^(db / 20)`.
#[inline]
#[must_use]
pub fn db_to_linear(db: f64) -> f64 {
    libm::pow(10.0, db / 20.0)
}

/// Convert a linear amplitude ratio to decibels: `20·log10(linear)`.
///
/// A zero (or denormal) input yields −∞ by IEEE semantics; dynamics blocks
/// in this crate clamp before converting so ±∞ never enters state.
#[inline]
#[must_use]
pub fn linear_to_db(linear: f64) -> f64 {
    20.0 * libm::log10(linear)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    // Scalar helper tests; tolerance-based assertions, no slice indexing.

    use super::*;

    #[test]
    fn db_round_trip() {
        assert_eq!(db_to_linear(0.0), 1.0);
        assert!((db_to_linear(20.0) - 10.0).abs() < 1e-12);
        assert!((db_to_linear(-6.0206) - 0.5).abs() < 1e-4);
        assert!((linear_to_db(10.0) - 20.0).abs() < 1e-12);
        assert!((linear_to_db(db_to_linear(-13.7)) + 13.7).abs() < 1e-9);
    }

    #[test]
    fn flush_zeroes_below_epsilon() {
        assert_eq!(flush_denormal(0.0), 0.0);
        assert_eq!(flush_denormal(1e-300), 0.0);
        assert_eq!(flush_denormal(-1e-25), 0.0);
        assert_eq!(flush_denormal(1e-20), 1e-20); // at threshold: kept
        assert_eq!(flush_denormal(1e-19), 1e-19);
        assert_eq!(flush_denormal(-4.2), -4.2);
        // NaN/inf pass through so upstream corruption is not masked.
        assert!(flush_denormal(f64::NAN).is_nan());
        assert_eq!(flush_denormal(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn linear_to_db_edge() {
        assert!(linear_to_db(0.0) == f64::NEG_INFINITY);
    }
}
