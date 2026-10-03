//! Biquad filters — RBJ *Audio EQ Cookbook* designs on a Direct Form I core.
//!
//! [`Coefficients`] computes normalized (a0 = 1) biquad coefficients for all
//! nine cookbook filters from type-safe parameter structs: resonant filters
//! take a [`Resonant`] (frequency + Q), EQ peaking takes an [`Equalizer`]
//! (adds gain), shelves take a [`Shelf`]. There is no way to pass a gain to a
//! lowpass or a Q-only shape to a shelf — the parameter type is the contract.
//!
//! [`Biquad`] holds coefficients plus Direct Form I state and processes one
//! sample (or a slice, in place) with per-sample denormal flushing on both
//! the output and the state words.
//!
//! # Example
//!
//! ```
//! use dsp_core::biquad::{Biquad, Coefficients, Resonant};
//!
//! let mut lowpass = Biquad::new(
//!     Coefficients::lowpass(Resonant {
//!         sample_rate: 48_000.0,
//!         frequency: 1_000.0,
//!         q: core::f64::consts::FRAC_1_SQRT_2,
//!     })
//!     .expect("valid parameters"),
//! );
//!
//! let mut block = [0.0f64; 480];
//! lowpass.process_slice(&mut block);
//! ```

use crate::error::DspError;
use crate::math::flush_denormal;
use core::f64::consts::{PI, TAU};
use libm::{atan2, cos, sin};

/// Parameters for the gain-free RBJ shapes: lowpass, highpass, both
/// band-passes, notch, allpass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Resonant {
    /// Sample rate in Hz. Must be finite and > 0.
    pub sample_rate: f64,
    /// Center/corner frequency in Hz. Must satisfy `0 < frequency <
    /// sample_rate / 2`.
    pub frequency: f64,
    /// Resonance quality factor. Must be finite and > 0. 1/√2 is
    /// Butterworth (maximally flat) for the lowpass/highpass shapes.
    pub q: f64,
}

/// Parameters for the peaking EQ: like [`Resonant`] plus a boost/cut gain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Equalizer {
    /// Sample rate in Hz. Must be finite and > 0.
    pub sample_rate: f64,
    /// Center frequency in Hz. Must satisfy `0 < frequency < sample_rate/2`.
    pub frequency: f64,
    /// Resonance quality factor. Must be finite and > 0.
    pub q: f64,
    /// Peak gain at `frequency` in dB (positive boosts, negative cuts).
    /// Must be finite.
    pub gain_db: f64,
}

/// Parameters for the low/high shelf filters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shelf {
    /// Sample rate in Hz. Must be finite and > 0.
    pub sample_rate: f64,
    /// Corner (half-gain) frequency in Hz. Must satisfy
    /// `0 < frequency < sample_rate / 2`.
    pub frequency: f64,
    /// Shelf slope as a Q-like factor around the corner frequency. Must be
    /// finite and > 0; a Q of 1/√2 gives the classic gentle shelf.
    pub q: f64,
    /// Shelf gain in dB (positive lifts the shelf, negative lowers it).
    /// Must be finite.
    pub gain_db: f64,
}

/// Normalized biquad coefficients (`a0 == 1`), as published by the RBJ
/// cookbook designs.
///
/// Transfer function:
/// `H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coefficients {
    /// Feed-forward numerator tap 0.
    pub b0: f64,
    /// Feed-forward numerator tap 1.
    pub b1: f64,
    /// Feed-forward numerator tap 2.
    pub b2: f64,
    /// Feed-back denominator tap 1 (sign per the transfer function above).
    pub a1: f64,
    /// Feed-back denominator tap 2.
    pub a2: f64,
}

impl Coefficients {
    /// Build coefficients directly from already-normalized values.
    ///
    /// This is an escape hatch for designs outside the cookbook; no
    /// stability check is performed.
    #[must_use]
    pub fn from_raw(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Self {
        Self { b0, b1, b2, a1, a2 }
    }

    fn resonant(p: Resonant, name: &'static str) -> Result<(f64, f64, f64), DspError> {
        validate_freq_q(p.sample_rate, p.frequency, p.q, name)?;
        let w0 = TAU * p.frequency / p.sample_rate;
        let c = cos(w0);
        let s = sin(w0);
        let alpha = s / (2.0 * p.q);
        Ok((c, s, alpha))
    }

    fn shelf_common(p: Shelf, name: &'static str) -> Result<(f64, f64, f64, f64), DspError> {
        validate_freq_q(p.sample_rate, p.frequency, p.q, name)?;
        if !p.gain_db.is_finite() {
            return Err(DspError::InvalidParameter {
                name: "gain_db",
                reason: "must be finite",
            });
        }
        let w0 = TAU * p.frequency / p.sample_rate;
        let c = cos(w0);
        let s = sin(w0);
        let alpha = s / (2.0 * p.q);
        let a = libm::pow(10.0, p.gain_db / 40.0);
        Ok((c, s, alpha, a))
    }

    /// Second-order Butterworth-style lowpass (RBJ cookbook).
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when a parameter is outside its valid range (see the parameter docs).
    pub fn lowpass(p: Resonant) -> Result<Self, DspError> {
        let (c, _s, alpha) = Self::resonant(p, "lowpass")?;
        let one_minus_c = 1.0 - c;
        Ok(Self::normalize(
            one_minus_c / 2.0,
            one_minus_c,
            one_minus_c / 2.0,
            -2.0 * c,
            1.0 - alpha,
            1.0 + alpha,
        ))
    }

    /// Second-order highpass (RBJ cookbook).
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when a parameter is outside its valid range (see the parameter docs).
    pub fn highpass(p: Resonant) -> Result<Self, DspError> {
        let (c, _s, alpha) = Self::resonant(p, "highpass")?;
        let one_plus_c = 1.0 + c;
        Ok(Self::normalize(
            one_plus_c / 2.0,
            -one_plus_c,
            one_plus_c / 2.0,
            -2.0 * c,
            1.0 - alpha,
            1.0 + alpha,
        ))
    }

    /// Bandpass with constant skirt gain — the peak gain equals `Q`
    /// (RBJ cookbook first bandpass form).
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when a parameter is outside its valid range (see the parameter docs).
    pub fn bandpass_constant_skirt(p: Resonant) -> Result<Self, DspError> {
        let (c, s, alpha) = Self::resonant(p, "bandpass_constant_skirt")?;
        let half_sine = s / 2.0;
        Ok(Self::normalize(
            half_sine,
            0.0,
            -half_sine,
            -2.0 * c,
            1.0 - alpha,
            1.0 + alpha,
        ))
    }

    /// Bandpass with constant 0 dB peak gain (RBJ cookbook second bandpass
    /// form).
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when a parameter is outside its valid range (see the parameter docs).
    pub fn bandpass_constant_peak(p: Resonant) -> Result<Self, DspError> {
        let (c, _s, alpha) = Self::resonant(p, "bandpass_constant_peak")?;
        Ok(Self::normalize(
            alpha,
            0.0,
            -alpha,
            -2.0 * c,
            1.0 - alpha,
            1.0 + alpha,
        ))
    }

    /// Notch — unity gain everywhere except a zero at the center frequency.
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when a parameter is outside its valid range (see the parameter docs).
    pub fn notch(p: Resonant) -> Result<Self, DspError> {
        let (c, _s, alpha) = Self::resonant(p, "notch")?;
        Ok(Self::normalize(
            1.0,
            -2.0 * c,
            1.0,
            -2.0 * c,
            1.0 - alpha,
            1.0 + alpha,
        ))
    }

    /// Allpass — unity magnitude, frequency-dependent phase.
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when a parameter is outside its valid range (see the parameter docs).
    pub fn allpass(p: Resonant) -> Result<Self, DspError> {
        let (c, _s, alpha) = Self::resonant(p, "allpass")?;
        Ok(Self::normalize(
            1.0 - alpha,
            -2.0 * c,
            1.0 + alpha,
            -2.0 * c,
            1.0 - alpha,
            1.0 + alpha,
        ))
    }

    /// Peaking EQ — boost/cut `gain_db` at `frequency` with width set by
    /// `q`.
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when a parameter is outside its valid range (see the parameter docs).
    pub fn peaking_eq(p: Equalizer) -> Result<Self, DspError> {
        validate_freq_q(p.sample_rate, p.frequency, p.q, "peaking_eq")?;
        if !p.gain_db.is_finite() {
            return Err(DspError::InvalidParameter {
                name: "gain_db",
                reason: "must be finite",
            });
        }
        let w0 = TAU * p.frequency / p.sample_rate;
        let c = cos(w0);
        let alpha = sin(w0) / (2.0 * p.q);
        let a = libm::pow(10.0, p.gain_db / 40.0);
        // Cookbook peakingEQ: the poles scale by α/A (not α), so the
        // center-frequency gain is exactly A² = 10^(gain_db/20).
        Ok(Self::normalize(
            1.0 + alpha * a,
            -2.0 * c,
            1.0 - alpha * a,
            -2.0 * c,
            1.0 - alpha / a,
            1.0 + alpha / a,
        ))
    }

    /// Low shelf — gains below the corner frequency by `gain_db`.
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when a parameter is outside its valid range (see the parameter docs).
    pub fn low_shelf(p: Shelf) -> Result<Self, DspError> {
        let (c, _s, alpha, a) = Self::shelf_common(p, "low_shelf")?;
        let sq = 2.0 * libm::sqrt(a) * alpha;
        let common = a + 1.0 - (a - 1.0) * c;
        Ok(Self::normalize(
            a * (common + sq),
            2.0 * a * ((a - 1.0) - (a + 1.0) * c),
            a * (common - sq),
            -2.0 * ((a - 1.0) + (a + 1.0) * c),
            a + 1.0 + (a - 1.0) * c - sq,
            a + 1.0 + (a - 1.0) * c + sq,
        ))
    }

    /// High shelf — gains above the corner frequency by `gain_db`.
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when a parameter is outside its valid range (see the parameter docs).
    pub fn high_shelf(p: Shelf) -> Result<Self, DspError> {
        let (c, _s, alpha, a) = Self::shelf_common(p, "high_shelf")?;
        let sq = 2.0 * libm::sqrt(a) * alpha;
        let common = a + 1.0 + (a - 1.0) * c;
        Ok(Self::normalize(
            a * (common + sq),
            -2.0 * a * ((a - 1.0) + (a + 1.0) * c),
            a * (common - sq),
            2.0 * ((a - 1.0) - (a + 1.0) * c),
            a + 1.0 - (a - 1.0) * c - sq,
            a + 1.0 - (a - 1.0) * c + sq,
        ))
    }

    /// Divide every tap by `a0` so the denominator head term is 1.
    fn normalize(b0: f64, b1: f64, b2: f64, a1: f64, a2: f64, a0: f64) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }
}

fn validate_freq_q(
    sample_rate: f64,
    frequency: f64,
    q: f64,
    shape: &'static str,
) -> Result<(), DspError> {
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(DspError::InvalidParameter {
            name: "sample_rate",
            reason: "must be finite and > 0",
        });
    }
    if !frequency.is_finite() || frequency <= 0.0 || frequency >= sample_rate / 2.0 {
        return Err(DspError::InvalidParameter {
            name: "frequency",
            reason: "must be finite and in (0, sample_rate/2)",
        });
    }
    if !q.is_finite() || q <= 0.0 {
        return Err(DspError::InvalidParameter {
            name: "q",
            reason: "must be finite and > 0",
        });
    }
    let _ = shape; // kept for call-site readability
    Ok(())
}

/// Measured/analytic transfer-function value at one frequency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrequencyResponse {
    /// Linear magnitude `|H(e^{jω})|`.
    pub magnitude: f64,
    /// Wrapped phase in radians, in `(−π, π]`.
    pub phase_radians: f64,
}

impl FrequencyResponse {
    /// Magnitude in decibels (`20·log10(|H|)`).
    #[must_use]
    pub fn magnitude_db(&self) -> f64 {
        crate::math::linear_to_db(self.magnitude)
    }
}

/// Direct Form I biquad with coefficients public for direct manipulation and
/// private, denormal-flushed state.
///
/// `process` is allocation-free and panic-free; with finite coefficients and
/// finite input it always returns a finite sample (any denormal intermediate
/// is flushed to zero).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Biquad {
    /// Feed-forward numerator tap 0.
    pub b0: f64,
    /// Feed-forward numerator tap 1.
    pub b1: f64,
    /// Feed-forward numerator tap 2.
    pub b2: f64,
    /// Feed-back denominator tap 1.
    pub a1: f64,
    /// Feed-back denominator tap 2.
    pub a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl Biquad {
    /// A biquad with the given coefficients and zeroed state.
    #[must_use]
    pub fn new(coefficients: Coefficients) -> Self {
        Self {
            b0: coefficients.b0,
            b1: coefficients.b1,
            b2: coefficients.b2,
            a1: coefficients.a1,
            a2: coefficients.a2,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Replace the coefficients, keeping the current state — the usual
    /// live-parameter-change path (no click from a state reset).
    pub fn set_coefficients(&mut self, coefficients: Coefficients) {
        self.b0 = coefficients.b0;
        self.b1 = coefficients.b1;
        self.b2 = coefficients.b2;
        self.a1 = coefficients.a1;
        self.a2 = coefficients.a2;
    }

    /// The current coefficients.
    #[must_use]
    pub fn coefficients(&self) -> Coefficients {
        Coefficients {
            b0: self.b0,
            b1: self.b1,
            b2: self.b2,
            a1: self.a1,
            a2: self.a2,
        }
    }

    /// Zero the filter state (both input and output delay lines).
    pub fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }

    /// Process one sample (Direct Form I), flushing the output and state
    /// words against denormals.
    pub fn process(&mut self, input: f64) -> f64 {
        let output = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = input;
        let output = flush_denormal(output);
        self.y2 = self.y1;
        self.y1 = output;
        output
    }

    /// Process a slice of samples in place (same as folding [`Biquad::process`]
    /// over the slice).
    pub fn process_slice(&mut self, samples: &mut [f64]) {
        for sample in samples.iter_mut() {
            *sample = self.process(*sample);
        }
    }

    /// Analytic frequency response of the current coefficients at
    /// `frequency` (Hz) for the given sample rate:
    ///
    /// `H(e^{jω}) = (b0 + b1·e^{−jω} + b2·e^{−2jω}) / (1 + a1·e^{−jω} +
    /// a2·e^{−2jω})`.
    #[must_use]
    pub fn frequency_response(&self, frequency: f64, sample_rate: f64) -> FrequencyResponse {
        let w = TAU * frequency / sample_rate;
        let (c1, s1) = (cos(w), sin(w));
        let (c2, s2) = (cos(2.0 * w), sin(2.0 * w));
        // e^{-j·kw} = (cos(kw), −sin(kw)); evaluate numerator and denominator.
        let num_re = self.b0 + self.b1 * c1 + self.b2 * c2;
        let num_im = -(self.b1 * s1 + self.b2 * s2);
        let den_re = 1.0 + self.a1 * c1 + self.a2 * c2;
        let den_im = -(self.a1 * s1 + self.a2 * s2);
        let magnitude = libm::sqrt(num_re * num_re + num_im * num_im)
            / libm::sqrt(den_re * den_re + den_im * den_im);
        let mut phase = atan2(num_im, num_re) - atan2(den_im, den_re);
        // Wrap into (−π, π].
        while phase > PI {
            phase -= TAU;
        }
        while phase <= -PI {
            phase += TAU;
        }
        FrequencyResponse {
            magnitude,
            phase_radians: phase,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    // Unit tests over known-size buffers; the module-level deny on
    // unwraps/indexing is relaxed here with fixed-length test buffers.

    use super::*;
    use alloc::{vec, vec::Vec};
    use core::f64::consts::FRAC_PI_2;

    const FS: f64 = 48_000.0;

    fn resonant(f: f64, q: f64) -> Resonant {
        Resonant {
            sample_rate: FS,
            frequency: f,
            q,
        }
    }

    #[test]
    fn lowpass_dc_gain_is_unity() {
        let mut b = Biquad::new(
            Coefficients::lowpass(resonant(1_000.0, core::f64::consts::FRAC_1_SQRT_2)).unwrap(),
        );
        // Feed DC; Direct Form I converges to H(1) = 1.
        let mut last = 0.0;
        for _ in 0..20_000 {
            last = b.process(1.0);
        }
        assert!(
            (last - 1.0).abs() < 1e-9,
            "DC gain {last} should converge to 1"
        );
    }

    #[test]
    fn highpass_nyquist_gain_is_unity() {
        let mut b = Biquad::new(
            Coefficients::highpass(resonant(1_000.0, core::f64::consts::FRAC_1_SQRT_2)).unwrap(),
        );
        // Feed Nyquist (+1/−1); converges to |H(−1)| = 1.
        let mut last = 0.0;
        for n in 0..20_000 {
            last = b.process(if n % 2 == 0 { 1.0 } else { -1.0 });
        }
        assert!(
            (last.abs() - 1.0).abs() < 1e-9,
            "Nyquist gain |{last}| should converge to 1"
        );
    }

    #[test]
    fn impulse_response_matches_transfer_function() {
        // Drive each shape to steady state with a sine at an exact bin of
        // the N = 8192 window; coherent correlation then recovers magnitude
        // and phase, which must match the analytic response (phase < 1e-6).
        let shapes: Vec<(&'static str, Coefficients)> = vec![
            ("lp", Coefficients::lowpass(resonant(2_000.0, 1.3)).unwrap()),
            (
                "hp",
                Coefficients::highpass(resonant(2_000.0, 1.3)).unwrap(),
            ),
            (
                "bp_sk",
                Coefficients::bandpass_constant_skirt(resonant(3_000.0, 4.0)).unwrap(),
            ),
            (
                "bp_pk",
                Coefficients::bandpass_constant_peak(resonant(3_000.0, 4.0)).unwrap(),
            ),
            (
                "notch",
                Coefficients::notch(resonant(3_000.0, 6.0)).unwrap(),
            ),
            (
                "apf",
                Coefficients::allpass(resonant(2_000.0, 2.0)).unwrap(),
            ),
            (
                "peak",
                Coefficients::peaking_eq(Equalizer {
                    sample_rate: FS,
                    frequency: 3_000.0,
                    q: 2.5,
                    gain_db: 6.0,
                })
                .unwrap(),
            ),
            (
                "lshelf",
                Coefficients::low_shelf(Shelf {
                    sample_rate: FS,
                    frequency: 500.0,
                    q: core::f64::consts::FRAC_1_SQRT_2,
                    gain_db: -9.0,
                })
                .unwrap(),
            ),
            (
                "hshelf",
                Coefficients::high_shelf(Shelf {
                    sample_rate: FS,
                    frequency: 4_000.0,
                    q: core::f64::consts::FRAC_1_SQRT_2,
                    gain_db: 7.0,
                })
                .unwrap(),
            ),
        ];
        // Exact bins k·FS/8192 so the correlation window holds whole cycles.
        let test_bins = [16usize, 64, 256, 1024];
        for (name, coeffs) in shapes {
            let filter = Biquad::new(coeffs);
            for k in test_bins {
                let f = FS * (k as f64) / 8192.0;
                let measured = steady_state(&filter, f, 8192, 2048);
                let analytic = filter.frequency_response(f, FS);
                assert!(
                    (measured.0 - analytic.magnitude).abs() < 1e-7,
                    "{name} @ {f}: magnitude {} vs analytic {}",
                    measured.0,
                    analytic.magnitude
                );
                let phase_err = (measured.1 - analytic.phase_radians).abs();
                assert!(
                    phase_err < 1e-6,
                    "{name} @ {f}: phase error {phase_err} (measured {}, analytic {})",
                    measured.1,
                    analytic.phase_radians
                );
            }
        }
    }

    /// Feed `total` samples of a sine at `frequency` through a copy of
    /// `filter`, then coherent-correlate the last `measure` samples (an
    /// exact-bin window) to recover (magnitude, phase) of the output. The
    /// correlation is against a cosine reference while the drive is a sine,
    /// so the recovered phase is offset by +π/2 before returning.
    fn steady_state(filter: &Biquad, frequency: f64, total: usize, measure: usize) -> (f64, f64) {
        let w = TAU * frequency / FS;
        let mut b = *filter;
        let (mut re, mut im) = (0.0, 0.0);
        let start = total - measure;
        for k in 0..total {
            let t = w * (k as f64);
            let y = b.process(t.sin());
            if k >= start {
                re += y * t.cos();
                im -= y * t.sin();
            }
        }
        let span = measure as f64 / 2.0;
        let mut phase = atan2(im, re) + FRAC_PI_2;
        while phase > PI {
            phase -= TAU;
        }
        while phase <= -PI {
            phase += TAU;
        }
        ((libm::sqrt(re * re + im * im) / span), phase)
    }

    #[test]
    fn process_slice_matches_sample_loop() {
        let mut a = Biquad::new(Coefficients::lowpass(resonant(900.0, 0.9)).unwrap());
        let mut b = a;
        let original: Vec<f64> = (0..97).map(|i| ((i as f64) * 0.37).sin()).collect();
        let mut buf = original.clone();
        b.process_slice(&mut buf);
        // `a` (a copy) processes sample-by-sample over the same input.
        for (want, got) in original.iter().zip(buf.iter()) {
            let expected = a.process(*want);
            assert!((expected - got).abs() < 1e-15);
        }
    }

    #[test]
    fn reset_zeroes_state() {
        let mut b = Biquad::new(Coefficients::lowpass(resonant(900.0, 0.9)).unwrap());
        b.process(1.0);
        b.process(1.0);
        b.reset();
        let y = b.process(0.0);
        assert_eq!(y, 0.0);
    }

    #[test]
    fn set_coefficients_keeps_state() {
        let mut b = Biquad::new(Coefficients::lowpass(resonant(900.0, 0.9)).unwrap());
        b.process(1.0);
        let coeffs = Coefficients::highpass(resonant(900.0, 0.9)).unwrap();
        b.set_coefficients(coeffs);
        assert_eq!(b.coefficients(), coeffs);
        // State (x1) survived the swap: processing 0 yields the x1 tail.
        assert!(b.process(0.0) != 0.0);
    }

    #[test]
    fn denormals_are_flushed() {
        let mut b = Biquad::new(
            Coefficients::lowpass(resonant(100.0, core::f64::consts::FRAC_1_SQRT_2)).unwrap(),
        );
        // Feed values below the flush threshold: every output is exactly 0.
        for _ in 0..64 {
            let y = b.process(1e-30);
            assert_eq!(y, 0.0, "denormal output must flush to +0.0");
        }
        // State words must also be flushed, never denormal.
        assert_eq!(b.y1, 0.0);
        assert_eq!(b.y2, 0.0);
    }

    #[test]
    fn invalid_parameters_are_errors() {
        let ok = resonant(1_000.0, core::f64::consts::FRAC_1_SQRT_2);
        assert_eq!(
            Coefficients::lowpass(Resonant {
                sample_rate: 0.0,
                ..ok
            }),
            Err(DspError::InvalidParameter {
                name: "sample_rate",
                reason: "must be finite and > 0"
            })
        );
        assert!(Coefficients::lowpass(Resonant {
            frequency: 0.0,
            ..ok
        })
        .is_err());
        assert!(Coefficients::lowpass(Resonant {
            frequency: FS / 2.0,
            ..ok
        })
        .is_err());
        assert!(Coefficients::lowpass(Resonant {
            frequency: f64::NAN,
            ..ok
        })
        .is_err());
        assert!(Coefficients::lowpass(Resonant { q: -1.0, ..ok }).is_err());
        assert!(Coefficients::lowpass(Resonant {
            q: f64::INFINITY,
            ..ok
        })
        .is_err());
        assert!(Coefficients::highpass(ok).is_ok());
        assert!(Coefficients::bandpass_constant_skirt(ok).is_ok());
        assert!(Coefficients::bandpass_constant_peak(ok).is_ok());
        assert!(Coefficients::notch(ok).is_ok());
        assert!(Coefficients::allpass(ok).is_ok());
        assert!(Coefficients::peaking_eq(Equalizer {
            gain_db: f64::NAN,
            ..Equalizer {
                sample_rate: ok.sample_rate,
                frequency: ok.frequency,
                q: ok.q,
                gain_db: 0.0
            }
        })
        .is_err());
        assert!(Coefficients::low_shelf(Shelf {
            gain_db: f64::INFINITY,
            ..Shelf {
                sample_rate: ok.sample_rate,
                frequency: ok.frequency,
                q: ok.q,
                gain_db: 0.0
            }
        })
        .is_err());
        assert!(Coefficients::high_shelf(Shelf {
            q: 0.0,
            ..Shelf {
                sample_rate: ok.sample_rate,
                frequency: ok.frequency,
                q: 1.0,
                gain_db: 0.0
            }
        })
        .is_err());
        assert!(Coefficients::low_shelf(Shelf {
            sample_rate: -1.0,
            ..Shelf {
                sample_rate: FS,
                frequency: ok.frequency,
                q: ok.q,
                gain_db: 0.0
            }
        })
        .is_err());
    }

    #[test]
    fn frequency_response_known_shapes() {
        // Lowpass magnitude at DC is ~1, at Nyquist ~0; the allpass is flat.
        let lp = Biquad::new(
            Coefficients::lowpass(resonant(1_000.0, core::f64::consts::FRAC_1_SQRT_2)).unwrap(),
        );
        let r = lp.frequency_response(0.0, FS);
        assert!((r.magnitude - 1.0).abs() < 1e-12);
        let r = lp.frequency_response(FS / 2.0 - 1.0, FS);
        assert!(r.magnitude < 0.01);

        let ap = Biquad::new(Coefficients::allpass(resonant(2_000.0, 0.9)).unwrap());
        for f in [100.0, 2_000.0, 10_000.0] {
            let r = ap.frequency_response(f, FS);
            assert!((r.magnitude - 1.0).abs() < 1e-12, "allpass flat at {f}");
        }
        // magnitude_db consistent with magnitude.
        let r = lp.frequency_response(1_000.0, FS);
        assert!((r.magnitude_db() - 20.0 * r.magnitude.log10()).abs() < 1e-12);
    }
}
