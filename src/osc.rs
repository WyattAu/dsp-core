//! Oscillators — phase-accumulator sources with a per-sample frequency
//! (modulation) input.
//!
//! Every oscillator owns a phase in `[0, 1)` advanced by
//! `frequency / sample_rate` each call; the `process(frequency)` argument is
//! the *instantaneous* frequency for that sample, which makes FM and
//! parameter automation trivial (constant frequency → just pass the same
//! value). Sine uses a quadrant-reduced 9th-order polynomial (max error
//! ≈ 4·10⁻⁶); saw and square are band-limited via PolyBLEP correction;
//! triangle is the naive shape (gradient-only discontinuities — aliasing is
//! already far down).
//!
//! # Example
//!
//! ```
//! use dsp_core::osc::{Oscillator, SineOsc};
//!
//! let mut osc = SineOsc::new(48_000.0).expect("valid sample rate");
//! // 1 kHz tone for one second.
//! for _ in 0..48_000 {
//!     let sample = osc.process(1_000.0);
//!     assert!(sample.abs() <= 1.0 + 1e-5);
//! }
//! ```

use crate::error::DspError;
use core::f64::consts::PI;

/// Common phase-accumulator behavior: an instantaneous-frequency input,
/// a readable/resettable phase.
pub trait Oscillator {
    /// Produce the next sample at `frequency` Hz (the FM/automation input).
    fn process(&mut self, frequency: f64) -> f64;

    /// Current phase in `[0, 1)`.
    fn phase(&self) -> f64;

    /// Overwrite the phase (wraps into `[0, 1)`).
    fn set_phase(&mut self, phase: f64);

    /// Restart from phase zero.
    fn reset(&mut self);
}

/// Advance a phase by the per-sample increment `frequency / sample_rate`,
/// wrapping into `[0, 1)`.
fn advance(phase: &mut f64, frequency: f64, sample_rate: f64) {
    let next = *phase + frequency / sample_rate;
    // Single floor-wrap: exact for sane frequencies, terminates for any
    // finite input, and handles negative frequencies too.
    *phase = next - libm::floor(next);
}

/// Quadrant-reduced polynomial sine on a `[0, 1)` phase.
///
/// Taylor through the x⁹ term evaluated on `[0, π/2]`: max absolute error
/// ≈ 4·10⁻⁶ (≈ −108 dB), several multiply-adds, no libm call.
fn fast_sin_phase(phase: f64) -> f64 {
    let t = 2.0 * phase; // [0, 2)
    let sign = if t >= 1.0 { -1.0 } else { 1.0 };
    let u = t - libm::floor(t); // [0, 1)
    let mut x = PI * u; // [0, π)
    if x > core::f64::consts::FRAC_PI_2 {
        x = PI - x; // reduce to [0, π/2]
    }
    let x2 = x * x;
    // Horner form of x − x³/6 + x⁵/120 − x⁷/5040 + x⁹/362880.
    sign * x * (1.0 - x2 / 6.0 * (1.0 - x2 / 20.0 * (1.0 - x2 / 42.0 * (1.0 - x2 / 72.0))))
}

/// PolyBLEP correction for a discontinuity at phase `t` (in `[0, 1)`) with
/// per-sample phase increment `dt`: subtract the returned value from the
/// sample *after* the jump and add it at the sample *before* it. Used
/// internally by [`SawOsc`] and [`SquareOsc`]; exposed for custom
/// band-limited shapes.
#[must_use]
pub fn poly_blep(t: f64, dt: f64) -> f64 {
    if t < dt {
        let x = t / dt;
        x + x - x * x - 1.0
    } else if t > 1.0 - dt {
        let x = (t - 1.0) / dt;
        x * x + x + x + 1.0
    } else {
        0.0
    }
}

fn validate_sample_rate(sample_rate: f64) -> Result<(), DspError> {
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        Err(DspError::InvalidParameter {
            name: "sample_rate",
            reason: "must be finite and > 0",
        })
    } else {
        Ok(())
    }
}

/// Sine oscillator (fast polynomial approximation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SineOsc {
    sample_rate: f64,
    phase: f64,
}

impl SineOsc {
    /// Create with a validated sample rate; phase starts at 0.
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when a parameter is outside its valid range (see the parameter docs).
    pub fn new(sample_rate: f64) -> Result<Self, DspError> {
        validate_sample_rate(sample_rate)?;
        Ok(Self {
            sample_rate,
            phase: 0.0,
        })
    }

    /// Change the sample rate (keeps the current phase).
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when a parameter is outside its valid range (see the parameter docs).
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when `sample_rate` is not finite and > 0.
    pub fn set_sample_rate(&mut self, sample_rate: f64) -> Result<(), DspError> {
        validate_sample_rate(sample_rate)?;
        self.sample_rate = sample_rate;
        Ok(())
    }

    /// The sample rate this oscillator was configured with.
    #[must_use]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }
}

impl Oscillator for SineOsc {
    fn process(&mut self, frequency: f64) -> f64 {
        let out = fast_sin_phase(self.phase);
        advance(&mut self.phase, frequency, self.sample_rate);
        out
    }

    fn phase(&self) -> f64 {
        self.phase
    }

    fn set_phase(&mut self, phase: f64) {
        self.phase = phase - libm::floor(phase);
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }
}

/// Sawtooth oscillator, band-limited with PolyBLEP correction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SawOsc {
    sample_rate: f64,
    phase: f64,
}

impl SawOsc {
    /// Create with a validated sample rate; phase starts at 0.
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when a parameter is outside its valid range (see the parameter docs).
    pub fn new(sample_rate: f64) -> Result<Self, DspError> {
        validate_sample_rate(sample_rate)?;
        Ok(Self {
            sample_rate,
            phase: 0.0,
        })
    }

    /// Change the sample rate (keeps the current phase).
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when `sample_rate` is not finite and > 0.
    pub fn set_sample_rate(&mut self, sample_rate: f64) -> Result<(), DspError> {
        validate_sample_rate(sample_rate)?;
        self.sample_rate = sample_rate;
        Ok(())
    }

    /// The sample rate this oscillator was configured with.
    #[must_use]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }
}

impl Oscillator for SawOsc {
    fn process(&mut self, frequency: f64) -> f64 {
        let dt = frequency / self.sample_rate;
        let p = self.phase;
        let out = 2.0 * p - 1.0 - poly_blep(p, dt);
        advance(&mut self.phase, frequency, self.sample_rate);
        out
    }

    fn phase(&self) -> f64 {
        self.phase
    }

    fn set_phase(&mut self, phase: f64) {
        self.phase = phase - libm::floor(phase);
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }
}

/// Square oscillator (50% duty), band-limited with PolyBLEP correction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SquareOsc {
    sample_rate: f64,
    phase: f64,
}

impl SquareOsc {
    /// Create with a validated sample rate; phase starts at 0.
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when a parameter is outside its valid range (see the parameter docs).
    pub fn new(sample_rate: f64) -> Result<Self, DspError> {
        validate_sample_rate(sample_rate)?;
        Ok(Self {
            sample_rate,
            phase: 0.0,
        })
    }

    /// Change the sample rate (keeps the current phase).
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when `sample_rate` is not finite and > 0.
    pub fn set_sample_rate(&mut self, sample_rate: f64) -> Result<(), DspError> {
        validate_sample_rate(sample_rate)?;
        self.sample_rate = sample_rate;
        Ok(())
    }

    /// The sample rate this oscillator was configured with.
    #[must_use]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }
}

impl Oscillator for SquareOsc {
    fn process(&mut self, frequency: f64) -> f64 {
        let dt = frequency / self.sample_rate;
        let p = self.phase;
        let mut out = if p < 0.5 { 1.0 } else { -1.0 };
        // Rising edge at phase 0 (jump −1 → +1): `poly_blep`'s orientation
        // is for a falling jump, so its correction is applied as-is — both
        // of its branches already carry the correct sign for a rising edge.
        out += poly_blep(p, dt);
        // Falling edge at phase 0.5 (jump +1 → −1): measure the distance to
        // the edge *unwrapped* in (−0.5, 0.5) — wrapping `p + 0.5` can land
        // exactly on 0/1 for accumulated phases, misclassifying which side
        // of the edge a sample sits on.
        let d = p - 0.5;
        if (0.0..dt).contains(&d) {
            out -= poly_blep(d, dt);
        } else if (-dt..0.0).contains(&d) {
            out -= poly_blep(1.0 + d, dt);
        }
        advance(&mut self.phase, frequency, self.sample_rate);
        out
    }

    fn phase(&self) -> f64 {
        self.phase
    }

    fn set_phase(&mut self, phase: f64) {
        self.phase = phase - libm::floor(phase);
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }
}

/// Triangle oscillator (naive shape — gradient-only discontinuities, so its
/// aliasing content is inherently weak).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriangleOsc {
    sample_rate: f64,
    phase: f64,
}

impl TriangleOsc {
    /// Create with a validated sample rate; phase starts at 0.
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when a parameter is outside its valid range (see the parameter docs).
    pub fn new(sample_rate: f64) -> Result<Self, DspError> {
        validate_sample_rate(sample_rate)?;
        Ok(Self {
            sample_rate,
            phase: 0.0,
        })
    }

    /// Change the sample rate (keeps the current phase).
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when `sample_rate` is not finite and > 0.
    pub fn set_sample_rate(&mut self, sample_rate: f64) -> Result<(), DspError> {
        validate_sample_rate(sample_rate)?;
        self.sample_rate = sample_rate;
        Ok(())
    }

    /// The sample rate this oscillator was configured with.
    #[must_use]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }
}

impl Oscillator for TriangleOsc {
    fn process(&mut self, frequency: f64) -> f64 {
        let p = self.phase;
        let out = 4.0 * (p - 0.5).abs() - 1.0;
        advance(&mut self.phase, frequency, self.sample_rate);
        out
    }

    fn phase(&self) -> f64 {
        self.phase
    }

    fn set_phase(&mut self, phase: f64) {
        self.phase = phase - libm::floor(phase);
    }

    fn reset(&mut self) {
        self.phase = 0.0;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    // Oscillator verification over generated buffers.

    use super::*;
    use alloc::{boxed::Box, vec, vec::Vec};

    const FS: f64 = 48_000.0;

    /// Count rising zero crossings of a rendered buffer.
    fn zero_crossings(samples: &[f64]) -> usize {
        let mut count = 0;
        for w in samples.windows(2) {
            if w[0] < 0.0 && w[1] >= 0.0 {
                count += 1;
            }
        }
        count
    }

    #[test]
    fn sine_frequency_accuracy() {
        for freq in [220.0f64, 1_000.0, 7_333.0] {
            let mut osc = SineOsc::new(FS).unwrap();
            let n = 48_000;
            let rendered: Vec<f64> = (0..n).map(|_| osc.process(freq)).collect();
            let crossings = zero_crossings(&rendered);
            let expected = (freq * (n as f64) / FS).round();
            assert!(
                (crossings as f64 - expected).abs() <= 2.0,
                "{freq} Hz: {crossings} crossings vs {expected}"
            );
        }
    }

    #[test]
    fn sine_amplitude_and_accuracy() {
        // Compare the fast polynomial against libm::sin across all phases.
        let mut worst = 0.0f64;
        for k in 0..4096 {
            let phase = k as f64 / 4096.0;
            let err = (fast_sin_phase(phase) - libm::sin(core::f64::consts::TAU * phase)).abs();
            worst = worst.max(err);
        }
        assert!(worst < 5e-6, "fast sine error {worst}");
        // Range bound: a rendered second never meaningfully leaves [−1, 1]
        // (the polynomial overshoots by ≈ 4e-6 at the quadrant boundary).
        let mut osc = SineOsc::new(FS).unwrap();
        for _ in 0..48_000 {
            let s = osc.process(1_234.0);
            assert!((-1.0 - 1.0e-5..=1.0 + 1.0e-5).contains(&s));
        }
    }

    #[test]
    fn saw_and_square_amplitude_bounds() {
        for freq in [110.0, 4_000.0] {
            let mut saw = SawOsc::new(FS).unwrap();
            let mut square = SquareOsc::new(FS).unwrap();
            for _ in 0..4_800 {
                let s = saw.process(freq);
                assert!(s > -1.01 && s < 1.01, "saw {s}");
                let s = square.process(freq);
                assert!(s > -1.01 && s < 1.01, "square {s}");
            }
        }
    }

    #[test]
    fn polyblep_suppresses_aliasing() {
        // Render naive vs PolyBLEP saws at a non-bin-aligned frequency,
        // window, and compare spectral energy in bins far from every
        // harmonic — i.e. genuine aliasing, not leakage.
        const N: usize = 8192;
        let freq = 997.0f64;
        let dt = freq / FS;
        let fund_bin = freq * (N as f64) / FS;

        let render = |blep: bool| -> f64 {
            let mut phase = 0.0f64;
            let samples: Vec<f64> = (0..N)
                .map(|i| {
                    let naive = 2.0 * phase - 1.0;
                    let out = if blep {
                        naive - poly_blep(phase, dt)
                    } else {
                        naive
                    };
                    phase += dt;
                    phase -= libm::floor(phase);
                    // Hann window so harmonic leakage stays local.
                    out * (0.5 * (1.0 - libm::cos(core::f64::consts::TAU * i as f64 / N as f64)))
                })
                .collect();
            let fft = crate::fft::Fft::new(N).unwrap();
            let mut buf: Vec<f64> = samples.iter().flat_map(|v| [*v, 0.0]).collect();
            fft.forward(&mut buf).unwrap();
            (100..N / 2)
                .filter(|k| {
                    let d = (*k as f64 / fund_bin - (*k as f64 / fund_bin).round()) * fund_bin;
                    d.abs() > 16.0
                })
                .map(|k| {
                    let re = buf[2 * k];
                    let im = buf[2 * k + 1];
                    re * re + im * im
                })
                .sum()
        };
        let naive_energy = render(false);
        let blep_energy = render(true);
        // PolyBLEP (1-sample correction) reliably buys an order of
        // magnitude of aliasing suppression in the aggregate non-harmonic
        // band and pushes the residual well under the naive floor.
        let improvement_db = 10.0 * (naive_energy / blep_energy).log10();
        assert!(
            improvement_db >= 10.0,
            "PolyBLEP improvement only {improvement_db:.1} dB: naive {naive_energy}, blep {blep_energy}"
        );
    }

    #[test]
    fn square_mean_is_zero_and_symmetric() {
        let mut osc = SquareOsc::new(FS).unwrap();
        let n = 48_000;
        let sum: f64 = (0..n).map(|_| osc.process(500.0)).sum();
        assert!(sum.abs() < 1.0, "DC of 50% square must vanish: {sum}");
    }

    #[test]
    fn triangle_spans_full_range() {
        let mut osc = TriangleOsc::new(FS).unwrap();
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for _ in 0..4_800 {
            let s = osc.process(500.0);
            min = min.min(s);
            max = max.max(s);
        }
        assert!((max - 1.0).abs() < 1e-12, "max {max}");
        assert!((min + 1.0).abs() < 1e-12, "min {min}");
    }

    #[test]
    fn phase_wraps_and_fm_works() {
        let mut osc = SineOsc::new(FS).unwrap();
        // FM sweep 100 Hz → 2 kHz: average crossing rate ≈ mean frequency.
        let mut samples = Vec::new();
        let mut expected_cycles = 0.0f64;
        for k in 0..48_000 {
            let t = k as f64 / 48_000.0;
            let freq = 100.0 + 1_900.0 * t;
            expected_cycles += freq / FS;
            samples.push(osc.process(freq));
        }
        let crossings = zero_crossings(&samples) as f64;
        assert!(
            (crossings - expected_cycles).abs() <= 2.0,
            "sweep: {crossings} vs {expected_cycles}"
        );
        assert!(osc.phase() >= 0.0 && osc.phase() < 1.0);
        // Phase set wraps.
        osc.set_phase(7.25);
        assert!((osc.phase() - 0.25).abs() < 1e-12);
        osc.set_phase(-3.5);
        assert!((osc.phase() - 0.5).abs() < 1e-12);
        // Reset.
        osc.reset();
        assert_eq!(osc.phase(), 0.0);
    }

    #[test]
    fn negative_frequency_wraps_downward() {
        let mut osc = SawOsc::new(FS).unwrap();
        for _ in 0..1_000 {
            osc.process(-220.0);
            assert!(osc.phase() >= 0.0 && osc.phase() < 1.0);
        }
    }

    #[test]
    fn near_nyquist_stays_finite() {
        let mut osc = SquareOsc::new(FS).unwrap();
        for _ in 0..1_000 {
            assert!(osc.process(0.45 * FS).abs() <= 1.01);
        }
        let mut osc = SineOsc::new(FS).unwrap();
        for _ in 0..1_000 {
            assert!(osc.process(0.45 * FS).abs() <= 1.0 + 1.0e-5);
        }
    }

    #[test]
    fn sample_rate_validation_and_change() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                SineOsc::new(bad),
                Err(DspError::InvalidParameter { .. })
            ));
            assert!(matches!(
                SawOsc::new(bad),
                Err(DspError::InvalidParameter { .. })
            ));
            assert!(matches!(
                SquareOsc::new(bad),
                Err(DspError::InvalidParameter { .. })
            ));
            assert!(matches!(
                TriangleOsc::new(bad),
                Err(DspError::InvalidParameter { .. })
            ));
        }
        let mut osc = SineOsc::new(FS).unwrap();
        assert_eq!(osc.sample_rate(), FS);
        osc.set_sample_rate(96_000.0).unwrap();
        assert_eq!(osc.sample_rate(), 96_000.0);
        assert!(osc.set_sample_rate(0.0).is_err());
        assert_eq!(osc.sample_rate(), 96_000.0, "failed set keeps old rate");
    }

    #[test]
    fn accessors_on_all_types() {
        let fs = FS;
        let mut saw = SawOsc::new(fs).unwrap();
        let mut square = SquareOsc::new(fs).unwrap();
        let mut tri = TriangleOsc::new(fs).unwrap();
        assert_eq!(saw.sample_rate(), fs);
        assert_eq!(square.sample_rate(), fs);
        assert_eq!(tri.sample_rate(), fs);
        saw.set_phase(1.75);
        square.set_phase(1.75);
        tri.set_phase(1.75);
        assert!((saw.phase() - 0.75).abs() < 1e-12);
        assert!((square.phase() - 0.75).abs() < 1e-12);
        assert!((tri.phase() - 0.75).abs() < 1e-12);
        assert!(saw.process(100.0).is_finite());
        assert!(square.process(100.0).is_finite());
        assert!(tri.process(100.0).is_finite());
    }

    #[test]
    fn trait_object_usage() {
        // All four oscillators are usable through the trait.
        let mut oscs: Vec<Box<dyn Oscillator>> = vec![
            Box::new(SineOsc::new(FS).unwrap()),
            Box::new(SawOsc::new(FS).unwrap()),
            Box::new(SquareOsc::new(FS).unwrap()),
            Box::new(TriangleOsc::new(FS).unwrap()),
        ];
        for osc in oscs.iter_mut() {
            for _ in 0..480 {
                let s = osc.process(440.0);
                assert!(s.is_finite());
            }
            osc.reset();
            assert_eq!(osc.phase(), 0.0);
        }
    }
}
