//! Windowing — pre-computed cosine-sum and Kaiser windows.
//!
//! [`Window::new`] materializes the window into a fixed buffer once; the
//! process path ([`Window::process`]) is a multiply over that buffer, in
//! place, allocation-free. All cosine-sum shapes (Hann, Hamming, Blackman,
//! Blackman–Harris, rectangular) support both *symmetric* (`θ = 2πn/(N−1)`,
//! the STFT convention) and *periodic* (`θ = 2πn/N`, the spectral-analysis
//! convention) variants; Kaiser supports both via the same α-geometry.
//!
//! # Example
//!
//! ```
//! use dsp_core::window::{Window, WindowKind};
//!
//! let hann = Window::new(WindowKind::Hann, 8, true, 0.0).expect("valid length");
//! let mut frame = [1.0f64; 8];
//! hann.process(&mut frame).expect("length matches");
//! assert_eq!(frame[0], 0.0); // periodic Hann starts at zero
//! assert!(frame.iter().all(|v| *v >= 0.0 && *v <= 1.0));
//! ```

use crate::error::DspError;
use crate::math::flush_denormal;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::f64::consts::TAU;
use libm::cos;

/// Window shapes supported by [`Window::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowKind {
    /// Rectangular (Dirichlet): `w[n] = 1`. `kaiser_beta` ignored.
    Rectangular,
    /// Hann (raised cosine): `0.5 − 0.5·cos θ`. Main lobe ≈ 4 bins
    /// (periodic convention).
    Hann,
    /// Hamming: `0.54 − 0.46·cos θ`.
    Hamming,
    /// Exact Blackman: `0.42 − 0.5·cos θ + 0.08·cos 2θ`.
    Blackman,
    /// 4-term Blackman–Harris: `0.35875 − 0.48829·cos θ + 0.14128·cos 2θ −
    /// 0.01168·cos 3θ`. −92 dB sidelobes.
    BlackmanHarris,
    /// Kaiser (Bessel-derived): `I0(β·α)/I0(β)` with
    /// `α = √(1 − (2n/(M−1) − 1)²)`. The β parameter is
    /// `Window::new`'s `kaiser_beta` argument and selects the
    /// sidelobe/transition trade-off (0 ≈ rectangular, 6 ≈ strong).
    Kaiser,
}

/// Cosine-sum coefficients `(a0, a1, …)` for each non-Kaiser shape:
/// `w[n] = Σ a_k·cos(kθ)`.
fn cosine_coefficients(kind: WindowKind) -> Option<&'static [f64]> {
    match kind {
        WindowKind::Rectangular => Some(&[1.0]),
        WindowKind::Hann => Some(&[0.5, -0.5]),
        WindowKind::Hamming => Some(&[0.54, -0.46]),
        WindowKind::Blackman => Some(&[0.42, -0.5, 0.08]),
        WindowKind::BlackmanHarris => Some(&[0.35875, -0.48829, 0.14128, -0.01168]),
        WindowKind::Kaiser => None,
    }
}

/// Modified Bessel function of the first kind, order zero — power series
/// `I0(x) = Σ ((x/2)^k / k!)²`, iterated to convergence (no libm needed).
fn bessel_i0(x: f64) -> f64 {
    let half = x / 2.0;
    let mut term = 1.0; // k = 0
    let mut sum = 1.0;
    let mut k = 1u32;
    while term > 1e-18 * sum && k < 64 {
        term *= (half / k as f64) * (half / k as f64);
        sum += term;
        k += 1;
    }
    sum
}

/// A pre-computed window: built once, applied by a single multiply.
pub struct Window {
    kind: WindowKind,
    periodic: bool,
    coeffs: Box<[f64]>,
}

impl core::fmt::Debug for Window {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Window")
            .field("kind", &self.kind)
            .field("periodic", &self.periodic)
            .field("len", &self.coeffs.len())
            .finish()
    }
}

impl Window {
    /// Generate `len` samples of `kind`.
    ///
    /// `periodic = true` selects the spectral-analysis convention (ideal for
    /// FFT work: the wrap-around sample is not duplicated); `periodic =
    /// false` the symmetric STFT convention. For non-Kaiser shapes
    /// `kaiser_beta` is ignored; for [`WindowKind::Kaiser`] it is the β
    /// parameter and must be finite and ≥ 0.
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when a parameter is outside its valid range (see the parameter docs).
    pub fn new(
        kind: WindowKind,
        len: usize,
        periodic: bool,
        kaiser_beta: f64,
    ) -> Result<Self, DspError> {
        if len == 0 {
            return Err(DspError::InvalidParameter {
                name: "len",
                reason: "must be > 0",
            });
        }
        if let WindowKind::Kaiser = kind {
            if !kaiser_beta.is_finite() || kaiser_beta < 0.0 {
                return Err(DspError::InvalidParameter {
                    name: "kaiser_beta",
                    reason: "must be finite and >= 0",
                });
            }
        }
        // Single-point windows are [1.0] by convention.
        if len == 1 {
            return Ok(Self {
                kind,
                periodic,
                coeffs: Box::new([1.0]),
            });
        }
        // θ step denominator: symmetric windows share endpoints (2π/(N−1));
        // periodic ones wrap (2π/N). Symmetric single-point windows are
        // defined by convention as [1.0].
        let denom = if periodic || len == 1 {
            len as f64
        } else {
            (len - 1) as f64
        };
        let coeffs: Vec<f64> = (0..len)
            .map(|n| {
                let center = 2.0 * (n as f64) / denom - 1.0; // −1 .. 1
                let value = match cosine_coefficients(kind) {
                    Some(a) => {
                        let theta = TAU * (n as f64) / denom;
                        let mut acc = 0.0;
                        for (term, &coef) in a.iter().enumerate() {
                            acc += coef * cos(term as f64 * theta);
                        }
                        acc
                    }
                    None => {
                        // Kaiser: I0(β·√(1 − center²)) / I0(β).
                        let radicand = 1.0 - center * center;
                        let alpha = libm::sqrt(radicand.max(0.0));
                        bessel_i0(kaiser_beta * alpha) / bessel_i0(kaiser_beta)
                    }
                };
                flush_denormal(value)
            })
            .collect();
        Ok(Self {
            kind,
            periodic,
            coeffs: coeffs.into_boxed_slice(),
        })
    }

    /// The window shape.
    #[must_use]
    pub fn kind(&self) -> WindowKind {
        self.kind
    }

    /// Whether this is the periodic variant.
    #[must_use]
    pub fn is_periodic(&self) -> bool {
        self.periodic
    }

    /// Number of taps.
    #[must_use]
    pub fn len(&self) -> usize {
        self.coeffs.len()
    }

    /// True when the window has no taps (never the case — construction
    /// rejects length 0).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// The pre-computed taps.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.coeffs
    }

    /// Apply the window in place: `frame[i] *= w[i]`. `frame.len()` must
    /// equal [`Window::len`].
    ///
    /// # Errors
    ///
    /// Returns [`DspError::InvalidLength`] when `frame.len()` differs from [`Window::len`].
    pub fn process(&self, frame: &mut [f64]) -> Result<(), DspError> {
        if frame.len() != self.coeffs.len() {
            return Err(DspError::InvalidLength {
                expected: self.coeffs.len(),
                got: frame.len(),
            });
        }
        for (sample, w) in frame.iter_mut().zip(self.coeffs.iter()) {
            *sample = flush_denormal(*sample * w);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    // Window tap checks over fixed-size buffers.

    use super::*;
    use alloc::format;

    #[test]
    fn rectangular_is_ones() {
        let w = Window::new(WindowKind::Rectangular, 16, false, 0.0).unwrap();
        assert_eq!(w.len(), 16);
        assert!(!w.is_empty());
        assert_eq!(w.kind(), WindowKind::Rectangular);
        for v in w.as_slice() {
            assert_eq!(*v, 1.0);
        }
    }

    #[test]
    fn symmetric_windows_are_symmetric() {
        for kind in [
            WindowKind::Hann,
            WindowKind::Hamming,
            WindowKind::Blackman,
            WindowKind::BlackmanHarris,
            WindowKind::Kaiser,
        ] {
            let w = Window::new(kind, 33, false, 4.0).unwrap();
            for n in 0..17 {
                let a = w.as_slice()[n];
                let b = w.as_slice()[32 - n];
                assert!((a - b).abs() < 1e-12, "{kind:?} tap {n}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn known_tap_values() {
        // Symmetric Hann of length 33: endpoints 0, center 1.
        let w = Window::new(WindowKind::Hann, 33, false, 0.0).unwrap();
        assert!(w.as_slice()[0].abs() < 1e-12);
        assert!((w.as_slice()[16] - 1.0).abs() < 1e-12);
        // Hamming endpoints: 0.54 − 0.46 = 0.08.
        let h = Window::new(WindowKind::Hamming, 33, false, 0.0).unwrap();
        assert!((h.as_slice()[0] - 0.08).abs() < 1e-12);
        // Blackman endpoints: 0.42 − 0.5 + 0.08 = 0.
        let b = Window::new(WindowKind::Blackman, 33, false, 0.0).unwrap();
        assert!(b.as_slice()[0].abs() < 1e-12);
        // Blackman–Harris endpoints: 0.35875 − 0.48829 + 0.14128 − 0.01168.
        let bh = Window::new(WindowKind::BlackmanHarris, 33, false, 0.0).unwrap();
        assert!((bh.as_slice()[0] - 0.00006).abs() < 1e-6);
    }

    #[test]
    fn periodic_hann_sums_to_half_len() {
        // Σ periodic Hann = N/2 exactly, independent of N parity.
        for len in [8usize, 9, 64] {
            let w = Window::new(WindowKind::Hann, len, true, 0.0).unwrap();
            let sum: f64 = w.as_slice().iter().sum();
            assert!((sum - len as f64 / 2.0).abs() < 1e-9, "len {len}: {sum}");
            assert!(w.is_periodic());
        }
    }

    #[test]
    fn periodic_hann_mainlobe_is_four_bins() {
        // A periodic Hann has spectral zeros at ±2 bins: the main lobe
        // spans 4 bins total (±1 around DC at −6 dB, first null at 2).
        let n = 64;
        let w = Window::new(WindowKind::Hann, n, true, 0.0).unwrap();
        // DFT of the window directly (bin 2 must vanish). Complex magnitude,
        // not just the real part — the Hann spectrum is imaginary at ±1.
        let magnitude = |k: usize| -> f64 {
            let (mut re, mut im) = (0.0, 0.0);
            for (j, &v) in w.as_slice().iter().enumerate() {
                let a = TAU * (k * j) as f64 / n as f64;
                re += v * libm::cos(a);
                im -= v * libm::sin(a);
            }
            libm::sqrt(re * re + im * im)
        };
        let dc = magnitude(0);
        assert!((dc - n as f64 / 2.0).abs() < 1e-9);
        assert!(magnitude(1) > 0.2 * dc, "energy at ±1 expected");
        assert!(magnitude(2).abs() < 1e-9, "first null at bin 2");
        assert!(magnitude(n - 2).abs() < 1e-9);
    }

    #[test]
    fn kaiser_beta_zero_is_rectangular() {
        let w = Window::new(WindowKind::Kaiser, 32, false, 0.0).unwrap();
        for v in w.as_slice() {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn kaiser_higher_beta_attenuates_edges() {
        // Odd length puts the exact I0(β) point on a sample.
        let w4 = Window::new(WindowKind::Kaiser, 33, false, 4.0).unwrap();
        let w8 = Window::new(WindowKind::Kaiser, 33, false, 8.0).unwrap();
        // Edge attenuation deepens with β; center stays at 1.
        assert!(w4.as_slice()[0] < 0.1);
        assert!(w8.as_slice()[0] < w4.as_slice()[0]);
        assert!((w4.as_slice()[16] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn process_applies_in_place() {
        let w = Window::new(WindowKind::Hann, 4, true, 0.0).unwrap();
        let mut frame = [1.0f64; 4];
        w.process(&mut frame).unwrap();
        for (got, want) in frame.iter().zip(w.as_slice().iter()) {
            assert_eq!(got, want);
        }
        // Wrong length is an error, not a panic.
        let mut short = [1.0f64; 3];
        assert_eq!(
            w.process(&mut short),
            Err(DspError::InvalidLength {
                expected: 4,
                got: 3
            })
        );
    }

    #[test]
    fn invalid_parameters_are_errors() {
        assert!(matches!(
            Window::new(WindowKind::Hann, 0, true, 0.0),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(matches!(
            Window::new(WindowKind::Kaiser, 8, true, -1.0),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(matches!(
            Window::new(WindowKind::Kaiser, 8, true, f64::NAN),
            Err(DspError::InvalidParameter { .. })
        ));
        // Single-point windows degrade to [1].
        let one = Window::new(WindowKind::Hann, 1, false, 0.0).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one.as_slice()[0], 1.0);
    }

    #[test]
    fn debug_format() {
        let s = format!("{:?}", Window::new(WindowKind::Hann, 8, true, 0.0).unwrap());
        assert!(s.contains("Hann") && s.contains("len: 8"));
    }
}
