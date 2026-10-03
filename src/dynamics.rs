//! Dynamics — lookahead limiter, soft-knee compressor, hysteresis noise
//! gate.
//!
//! All three blocks share the same discipline: parameters are validated at
//! construction, per-sample state (envelopes, gains, delay rings) is
//! pre-allocated, every recursive state word and output sample is flushed
//! against denormals, and `process` never panics.
//!
//! * [`Limiter`] — lookahead peak limiter: a delay ring provides the
//!   look-ahead window, the gain computer takes the window peak against the
//!   ceiling, and a one-pole smooths the gain with separate attack/release
//!   time constants. With attack ≪ lookahead the output never exceeds the
//!   ceiling.
//! * [`Compressor`] — feed-forward soft-knee compressor: an envelope
//!   follower tracks the input level, a quadratic knee maps level → gain
//!   reduction around the threshold, and the gain is smoothed with
//!   attack/release coefficients plus optional makeup gain.
//! * [`Gate`] — envelope-follower gate with hysteresis: it opens when the
//!   envelope rises past the threshold and closes only after the envelope
//!   falls below `threshold − hysteresis`, so signals hovering near the
//!   threshold do not chatter.

use crate::error::DspError;
use crate::math::{db_to_linear, flush_denormal, linear_to_db};
use alloc::boxed::Box;
use alloc::vec;
use libm::exp;

/// One-pole smoothing coefficient for a time constant given in
/// milliseconds: `exp(−1/(t_seconds·sample_rate))`. A non-positive time
/// constant yields 0 (instant).
fn time_coefficient(time_ms: f64, sample_rate: f64) -> f64 {
    if time_ms <= 0.0 {
        return 0.0;
    }
    let samples = time_ms * 0.001 * sample_rate;
    exp(-1.0 / samples)
}

fn validate_sample_rate(sample_rate: f64) -> Result<(), DspError> {
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(DspError::InvalidParameter {
            name: "sample_rate",
            reason: "must be finite and > 0",
        });
    }
    Ok(())
}

fn validate_time(name: &'static str, ms: f64) -> Result<(), DspError> {
    if !ms.is_finite() || ms < 0.0 {
        return Err(DspError::InvalidParameter {
            name,
            reason: "must be finite and >= 0",
        });
    }
    Ok(())
}

fn validate_db(name: &'static str, db: f64) -> Result<(), DspError> {
    if !db.is_finite() {
        return Err(DspError::InvalidParameter {
            name,
            reason: "must be finite",
        });
    }
    Ok(())
}

/// Parameters for [`Limiter`]. All fields are validated in [`Limiter::new`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LimiterParams {
    /// Sample rate in Hz; must be finite and > 0.
    pub sample_rate: f64,
    /// Output ceiling in dBFS (e.g. −1.0).
    pub ceiling_db: f64,
    /// Lookahead window in milliseconds; ≥ 0.
    pub lookahead_ms: f64,
    /// Gain attack time in milliseconds; ≥ 0 (0 = instant duck).
    pub attack_ms: f64,
    /// Gain release time in milliseconds; ≥ 0 (0 = instant recover).
    pub release_ms: f64,
}

/// Lookahead peak limiter.
///
/// Per sample the input enters a delay ring; the gain computer reads the
/// window peak (past *and* future relative to the delayed output sample) and
/// derives `target = min(1, ceiling / peak)`, which the attack/release
/// one-pole then smooths. The output is the delayed sample times the gain.
pub struct Limiter {
    ceiling: f64,
    attack_coef: f64,
    release_coef: f64,
    delay: Box<[f64]>,
    pos: usize,
    gain: f64,
    params: LimiterParams,
}

impl Limiter {
    /// Validate parameters and allocate the delay ring.
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when a parameter is outside its valid range (see the parameter docs).
    pub fn new(params: LimiterParams) -> Result<Self, DspError> {
        validate_sample_rate(params.sample_rate)?;
        validate_db("ceiling_db", params.ceiling_db)?;
        validate_time("lookahead_ms", params.lookahead_ms)?;
        validate_time("attack_ms", params.attack_ms)?;
        validate_time("release_ms", params.release_ms)?;
        let raw = params.lookahead_ms * 0.001 * params.sample_rate;
        let lookahead_samples = (libm::ceil(raw) as usize).clamp(1, 10_000_000);
        Ok(Self {
            ceiling: db_to_linear(params.ceiling_db),
            attack_coef: time_coefficient(params.attack_ms, params.sample_rate),
            release_coef: time_coefficient(params.release_ms, params.sample_rate),
            delay: vec![0.0; lookahead_samples].into_boxed_slice(),
            pos: 0,
            gain: 1.0,
            params,
        })
    }

    /// The validated parameters.
    #[must_use]
    pub fn params(&self) -> &LimiterParams {
        &self.params
    }

    /// Lookahead window in samples (the buffering latency).
    #[must_use]
    pub fn latency(&self) -> usize {
        self.delay.len()
    }

    /// Clear the delay ring and release the gain envelope to unity.
    pub fn reset(&mut self) {
        self.delay.fill(0.0);
        self.pos = 0;
        self.gain = 1.0;
    }

    /// Process one sample.
    pub fn process(&mut self, input: f64) -> f64 {
        // Lookahead window: the ring still holds the sample about to emerge
        // plus everything up to the just-arrived input, so the gain reacts
        // both to the present sample and to the near future.
        let peak = self.delay.iter().fold(input.abs(), |m, v| m.max(v.abs()));
        let target = self.ceiling / peak.max(self.ceiling);
        self.gain = if target < self.gain {
            self.attack_coef * self.gain + (1.0 - self.attack_coef) * target
        } else {
            self.release_coef * self.gain + (1.0 - self.release_coef) * target
        };
        // Slide through the delay ring after the gain decision.
        let delayed = self.delay.get(self.pos).copied().unwrap_or(input);
        if let Some(slot) = self.delay.get_mut(self.pos) {
            *slot = input;
        }
        self.pos += 1;
        if self.pos >= self.delay.len() {
            self.pos = 0;
        }
        flush_denormal(delayed * self.gain)
    }

    /// Process a slice in place.
    pub fn process_slice(&mut self, samples: &mut [f64]) {
        for sample in samples.iter_mut() {
            *sample = self.process(*sample);
        }
    }
}

/// Parameters for [`Compressor`]. All fields are validated in
/// [`Compressor::new`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressorParams {
    /// Sample rate in Hz; must be finite and > 0.
    pub sample_rate: f64,
    /// Threshold in dBFS where gain reduction begins (knee center).
    pub threshold_db: f64,
    /// Compression ratio ≥ 1 (4 = 4:1).
    pub ratio: f64,
    /// Soft-knee width in dB (total, centered on the threshold); ≥ 0.
    pub knee_db: f64,
    /// Attack time in ms; ≥ 0.
    pub attack_ms: f64,
    /// Release time in ms; ≥ 0.
    pub release_ms: f64,
    /// Makeup gain in dB applied after compression.
    pub makeup_db: f64,
}

/// Feed-forward soft-knee compressor.
///
/// The static curve is the classic quadratic interpolation: below
/// `threshold − knee/2` unity, above `threshold + knee/2` the full
/// `(1 − 1/ratio)` slope, and a smooth quadratic in between. An envelope
/// follower (peak) drives the curve; the resulting gain is smoothed with
/// attack/release one-poles and makeup gain is applied.
pub struct Compressor {
    params: CompressorParams,
    attack_coef: f64,
    release_coef: f64,
    /// Peak-envelope state (linear).
    envelope: f64,
    /// Smoothed gain state (linear).
    gain: f64,
}

impl Compressor {
    /// Validate parameters and initialize state.
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when a parameter is outside its valid range (see the parameter docs).
    pub fn new(params: CompressorParams) -> Result<Self, DspError> {
        validate_sample_rate(params.sample_rate)?;
        validate_db("threshold_db", params.threshold_db)?;
        validate_db("makeup_db", params.makeup_db)?;
        if !params.ratio.is_finite() || params.ratio < 1.0 {
            return Err(DspError::InvalidParameter {
                name: "ratio",
                reason: "must be finite and >= 1",
            });
        }
        if !params.knee_db.is_finite() || params.knee_db < 0.0 {
            return Err(DspError::InvalidParameter {
                name: "knee_db",
                reason: "must be finite and >= 0",
            });
        }
        validate_time("attack_ms", params.attack_ms)?;
        validate_time("release_ms", params.release_ms)?;
        Ok(Self {
            attack_coef: time_coefficient(params.attack_ms, params.sample_rate),
            release_coef: time_coefficient(params.release_ms, params.sample_rate),
            envelope: 0.0,
            gain: 1.0,
            params,
        })
    }

    /// The validated parameters.
    #[must_use]
    pub fn params(&self) -> &CompressorParams {
        &self.params
    }

    /// Zero envelope/gain state.
    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.gain = 1.0;
    }

    /// Static curve: gain reduction in dB for a level of `level_db`.
    #[must_use]
    fn gain_reduction_db(&self, level_db: f64) -> f64 {
        let p = &self.params;
        let slope = 1.0 - 1.0 / p.ratio;
        let over = level_db - p.threshold_db;
        if p.knee_db <= 0.0 {
            if over > 0.0 {
                over * slope
            } else {
                0.0
            }
        } else {
            let half = p.knee_db / 2.0;
            if over <= -half {
                0.0
            } else if over >= half {
                over * slope
            } else {
                slope * (over + half) * (over + half) / (2.0 * p.knee_db)
            }
        }
    }

    /// Process one sample.
    pub fn process(&mut self, input: f64) -> f64 {
        // Envelope follower on |input|.
        let level = input.abs();
        let coef = if level > self.envelope {
            self.attack_coef
        } else {
            self.release_coef
        };
        self.envelope = flush_denormal(coef * self.envelope + (1.0 - coef) * level);
        // Static curve in dB, clamped away from −∞.
        let level_db = linear_to_db(self.envelope.max(1e-12));
        let reduction = self.gain_reduction_db(level_db);
        let target = db_to_linear(self.params.makeup_db - reduction);
        // Smooth the gain.
        let gcoef = if target < self.gain {
            self.attack_coef
        } else {
            self.release_coef
        };
        self.gain = flush_denormal(gcoef * self.gain + (1.0 - gcoef) * target);
        flush_denormal(input * self.gain)
    }

    /// Process a slice in place.
    pub fn process_slice(&mut self, samples: &mut [f64]) {
        for sample in samples.iter_mut() {
            *sample = self.process(*sample);
        }
    }
}

/// Parameters for [`Gate`]. All fields are validated in [`Gate::new`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateParams {
    /// Sample rate in Hz; must be finite and > 0.
    pub sample_rate: f64,
    /// Open threshold in dBFS.
    pub threshold_db: f64,
    /// Hysteresis in dB: the gate closes only below
    /// `threshold_db − hysteresis_db`; ≥ 0.
    pub hysteresis_db: f64,
    /// Open/attack time in ms; ≥ 0.
    pub attack_ms: f64,
    /// Close/release time in ms; ≥ 0.
    pub release_ms: f64,
}

/// Envelope noise gate with hysteresis.
pub struct Gate {
    params: GateParams,
    attack_coef: f64,
    release_coef: f64,
    envelope: f64,
    gain: f64,
    open: bool,
}

impl Gate {
    /// Validate parameters and initialize state.
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when a parameter is outside its valid range (see the parameter docs).
    pub fn new(params: GateParams) -> Result<Self, DspError> {
        validate_sample_rate(params.sample_rate)?;
        validate_db("threshold_db", params.threshold_db)?;
        if !params.hysteresis_db.is_finite() || params.hysteresis_db < 0.0 {
            return Err(DspError::InvalidParameter {
                name: "hysteresis_db",
                reason: "must be finite and >= 0",
            });
        }
        validate_time("attack_ms", params.attack_ms)?;
        validate_time("release_ms", params.release_ms)?;
        Ok(Self {
            params,
            attack_coef: time_coefficient(params.attack_ms, params.sample_rate),
            release_coef: time_coefficient(params.release_ms, params.sample_rate),
            envelope: 0.0,
            gain: 0.0,
            open: false,
        })
    }

    /// The validated parameters.
    #[must_use]
    pub fn params(&self) -> &GateParams {
        &self.params
    }

    /// Whether the gate is currently open (envelope above threshold).
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Zero envelope/gain state (gate closed).
    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.gain = 0.0;
        self.open = false;
    }

    /// Process one sample.
    pub fn process(&mut self, input: f64) -> f64 {
        let level = input.abs();
        let coef = if level > self.envelope {
            self.attack_coef
        } else {
            self.release_coef
        };
        self.envelope = flush_denormal(coef * self.envelope + (1.0 - coef) * level);
        let env_db = linear_to_db(self.envelope.max(1e-12));
        // Hysteresis: open above threshold, close below threshold − hyst.
        if !self.open && env_db >= self.params.threshold_db {
            self.open = true;
        } else if self.open && env_db < self.params.threshold_db - self.params.hysteresis_db {
            self.open = false;
        }
        let target = if self.open { 1.0 } else { 0.0 };
        let gcoef = if target > self.gain {
            self.attack_coef
        } else {
            self.release_coef
        };
        self.gain = flush_denormal(gcoef * self.gain + (1.0 - gcoef) * target);
        flush_denormal(input * self.gain)
    }

    /// Process a slice in place.
    pub fn process_slice(&mut self, samples: &mut [f64]) {
        for sample in samples.iter_mut() {
            *sample = self.process(*sample);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    // Dynamics steady-state tests; fixed buffers, tolerance assertions.

    use super::*;
    use crate::math::linear_to_db;
    use core::f64::consts::TAU;
    use libm::sin;

    const FS: f64 = 48_000.0;

    fn sine(amplitude: f64) -> impl Iterator<Item = f64> {
        (0..).map(move |k| amplitude * sin(TAU * 100.0 * (k as f64) / FS))
    }

    #[test]
    fn limiter_ducks_to_ceiling_and_releases() {
        let mut lim = Limiter::new(LimiterParams {
            sample_rate: FS,
            ceiling_db: -6.0,
            lookahead_ms: 5.0,
            attack_ms: 0.1,
            release_ms: 50.0,
        })
        .unwrap();
        assert_eq!(lim.latency(), 240); // 5 ms @ 48 kHz
        let ceiling = db_to_linear(-6.0);
        // Steady full-scale sine: the settled peaks sit at the ceiling,
        // never above it by more than float slop.
        let mut peak = 0.0f64;
        for (k, s) in sine(1.0).take(24_000).enumerate() {
            let out = lim.process(s);
            if k > 12_000 {
                peak = peak.max(out.abs());
                assert!(out.abs() <= ceiling * 1.001, "overshoot {}", out.abs());
            }
        }
        assert!(
            (peak - ceiling).abs() < ceiling * 0.01,
            "settled peak {peak} vs ceiling {ceiling}"
        );
        // Below the ceiling the limiter returns to unity: a quiet signal
        // passes at its original amplitude after release.
        let mut peak = 0.0f64;
        for (k, s) in sine(0.1).take(24_000).enumerate() {
            let out = lim.process(s);
            if k > 12_000 {
                peak = peak.max(out.abs());
            }
        }
        assert!(
            (linear_to_db(peak) - linear_to_db(0.1)).abs() < 0.2,
            "unity pass-through failed: {peak}"
        );
        // Reset restores unity gain and clears the ring.
        lim.reset();
        assert_eq!(lim.gain, 1.0);
    }

    #[test]
    fn limiter_latency_is_exact() {
        // An impulse emerges exactly `latency` samples later, at unity gain.
        let mut lim = Limiter::new(LimiterParams {
            sample_rate: FS,
            ceiling_db: 0.0,
            lookahead_ms: 2.0,
            attack_ms: 0.0,
            release_ms: 100.0,
        })
        .unwrap();
        let n = lim.latency();
        let mut impulse_seen = None;
        for k in 0..(2 * n + 8) {
            let out = lim.process(if k == 0 { 1.0 } else { 0.0 });
            if out > 0.5 {
                impulse_seen = Some(k);
                break;
            }
        }
        assert_eq!(impulse_seen, Some(n));
    }

    #[test]
    fn limiter_rejects_bad_params() {
        let ok = LimiterParams {
            sample_rate: FS,
            ceiling_db: -1.0,
            lookahead_ms: 1.0,
            attack_ms: 1.0,
            release_ms: 10.0,
        };
        assert!(matches!(
            Limiter::new(LimiterParams {
                sample_rate: 0.0,
                ..ok
            }),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(matches!(
            Limiter::new(LimiterParams {
                ceiling_db: f64::NAN,
                ..ok
            }),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(matches!(
            Limiter::new(LimiterParams {
                lookahead_ms: -1.0,
                ..ok
            }),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(matches!(
            Limiter::new(LimiterParams {
                attack_ms: f64::INFINITY,
                ..ok
            }),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(matches!(
            Limiter::new(LimiterParams {
                release_ms: -0.5,
                ..ok
            }),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(Limiter::new(ok).is_ok());
        // Zero lookahead/attack collapse to single-sample/instant.
        let instant = Limiter::new(LimiterParams {
            lookahead_ms: 0.0,
            attack_ms: 0.0,
            release_ms: 0.0,
            ..ok
        })
        .unwrap();
        assert_eq!(instant.latency(), 1);
        assert_eq!(instant.params().ceiling_db, -1.0);
    }

    #[test]
    fn limiter_silence_is_flushed() {
        let mut lim = Limiter::new(LimiterParams {
            sample_rate: FS,
            ceiling_db: -1.0,
            lookahead_ms: 1.0,
            attack_ms: 1.0,
            release_ms: 10.0,
        })
        .unwrap();
        for _ in 0..1000 {
            assert_eq!(lim.process(0.0), 0.0);
        }
    }

    #[test]
    fn compressor_static_curve() {
        let params = CompressorParams {
            sample_rate: FS,
            threshold_db: -20.0,
            ratio: 4.0,
            knee_db: 6.0,
            attack_ms: 5.0,
            release_ms: 100.0,
            makeup_db: 0.0,
        };
        let comp = Compressor::new(params).unwrap();
        // Below the knee: unity.
        assert_eq!(comp.gain_reduction_db(-40.0), 0.0);
        assert_eq!(comp.gain_reduction_db(-25.0), 0.0);
        // Above the knee: full slope.
        assert!((comp.gain_reduction_db(-10.0) - 7.5).abs() < 1e-12);
        // Knee center: quadratic value (1 − 1/r)·knee/8.
        let expected_center = (1.0 - 0.25) * 6.0 / 8.0;
        assert!((comp.gain_reduction_db(-20.0) - expected_center).abs() < 1e-12);
        // Zero-knee variant: hard below/above; exactly at the threshold
        // there is no reduction yet.
        let hard = Compressor::new(CompressorParams {
            knee_db: 0.0,
            ..params
        })
        .unwrap();
        assert_eq!(hard.gain_reduction_db(-21.0), 0.0);
        assert_eq!(hard.gain_reduction_db(-20.0), 0.0);
        // 6 dB over at 4:1 → 6·(1 − 1/4) = 4.5 dB reduction.
        assert!((hard.gain_reduction_db(-14.0) - 4.5).abs() < 1e-12);
    }

    #[test]
    fn compressor_steady_state_input_output() {
        let mut comp = Compressor::new(CompressorParams {
            sample_rate: FS,
            threshold_db: -20.0,
            ratio: 4.0,
            knee_db: 0.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            makeup_db: 3.0,
        })
        .unwrap();
        // Drive with a −10 dBFS sine (above threshold) long enough to
        // settle; output dB ≈ threshold + (in − thr)/ratio + makeup.
        let amplitude = db_to_linear(-10.0);
        let mut peak = 0.0f64;
        for (k, s) in sine(amplitude).take(48_000).enumerate() {
            let out = comp.process(s);
            if k > 24_000 {
                peak = peak.max(out.abs());
            }
        }
        let expected_db = -20.0 + 10.0 / 4.0 + 3.0;
        let got_db = linear_to_db(peak);
        assert!(
            (got_db - expected_db).abs() < 0.5,
            "output {got_db} dB, expected ≈ {expected_db} dB"
        );
    }

    #[test]
    fn compressor_below_threshold_is_transparent() {
        let mut comp = Compressor::new(CompressorParams {
            sample_rate: FS,
            threshold_db: -10.0,
            ratio: 8.0,
            knee_db: 0.0,
            attack_ms: 1.0,
            release_ms: 50.0,
            makeup_db: 0.0,
        })
        .unwrap();
        let amplitude = db_to_linear(-30.0);
        let mut peak = 0.0f64;
        for (k, s) in sine(amplitude).take(48_000).enumerate() {
            let out = comp.process(s);
            if k > 24_000 {
                peak = peak.max(out.abs());
            }
        }
        assert!((linear_to_db(peak) + 30.0).abs() < 0.2);
        assert!((comp.gain - 1.0).abs() < 1e-9);
    }

    #[test]
    fn compressor_rejects_bad_params() {
        let ok = CompressorParams {
            sample_rate: FS,
            threshold_db: -20.0,
            ratio: 4.0,
            knee_db: 6.0,
            attack_ms: 5.0,
            release_ms: 100.0,
            makeup_db: 0.0,
        };
        assert!(matches!(
            Compressor::new(CompressorParams {
                sample_rate: -1.0,
                ..ok
            }),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(matches!(
            Compressor::new(CompressorParams {
                threshold_db: f64::NAN,
                ..ok
            }),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(matches!(
            Compressor::new(CompressorParams {
                makeup_db: f64::INFINITY,
                ..ok
            }),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(matches!(
            Compressor::new(CompressorParams { ratio: 0.5, ..ok }),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(matches!(
            Compressor::new(CompressorParams {
                knee_db: -1.0,
                ..ok
            }),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(matches!(
            Compressor::new(CompressorParams {
                attack_ms: -1.0,
                ..ok
            }),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(matches!(
            Compressor::new(CompressorParams {
                release_ms: f64::NAN,
                ..ok
            }),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(Compressor::new(ok).is_ok());
    }

    #[test]
    fn compressor_reset_and_silence_flush() {
        let mut comp = Compressor::new(CompressorParams {
            sample_rate: FS,
            threshold_db: -20.0,
            ratio: 4.0,
            knee_db: 6.0,
            attack_ms: 5.0,
            release_ms: 100.0,
            makeup_db: 6.0,
        })
        .unwrap();
        for _ in 0..100 {
            comp.process(0.7);
        }
        comp.reset();
        assert_eq!(comp.envelope, 0.0);
        assert_eq!(comp.gain, 1.0);
        for _ in 0..1000 {
            assert_eq!(comp.process(0.0), 0.0);
        }
    }

    #[test]
    fn gate_opens_closes_with_hysteresis() {
        let mut gate = Gate::new(GateParams {
            sample_rate: FS,
            threshold_db: -20.0,
            hysteresis_db: 6.0,
            attack_ms: 1.0,
            release_ms: 20.0,
        })
        .unwrap();
        assert!(!gate.is_open());
        // Loud signal opens the gate and passes.
        let loud = db_to_linear(-10.0);
        let mut peak = 0.0f64;
        for (k, s) in sine(loud).take(12_000).enumerate() {
            let out = gate.process(s);
            if k > 6_000 {
                peak = peak.max(out.abs());
            }
        }
        assert!(gate.is_open());
        assert!((linear_to_db(peak) + 10.0).abs() < 0.2);
        // Signal between (threshold − hyst) and threshold keeps it open.
        let between = db_to_linear(-23.0); // > −26, < −20
        for s in sine(between).take(12_000) {
            gate.process(s);
        }
        assert!(gate.is_open(), "hysteresis should hold the gate open");
        // Quiet signal closes it.
        let quiet = db_to_linear(-40.0);
        for (k, s) in sine(quiet).take(24_000).enumerate() {
            gate.process(s);
            if k > 12_000 {
                assert!(!gate.is_open());
            }
        }
        assert!(gate.gain < 1e-3, "gain must fall toward zero");
    }

    #[test]
    fn gate_rejects_bad_params() {
        let ok = GateParams {
            sample_rate: FS,
            threshold_db: -40.0,
            hysteresis_db: 3.0,
            attack_ms: 1.0,
            release_ms: 10.0,
        };
        assert!(matches!(
            Gate::new(GateParams {
                sample_rate: 0.0,
                ..ok
            }),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(matches!(
            Gate::new(GateParams {
                threshold_db: f64::NAN,
                ..ok
            }),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(matches!(
            Gate::new(GateParams {
                hysteresis_db: -3.0,
                ..ok
            }),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(matches!(
            Gate::new(GateParams {
                attack_ms: -1.0,
                ..ok
            }),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(matches!(
            Gate::new(GateParams {
                release_ms: f64::INFINITY,
                ..ok
            }),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(Gate::new(ok).is_ok());
    }

    #[test]
    fn gate_reset_and_silence_flush() {
        let mut gate = Gate::new(GateParams {
            sample_rate: FS,
            threshold_db: -40.0,
            hysteresis_db: 3.0,
            attack_ms: 1.0,
            release_ms: 10.0,
        })
        .unwrap();
        for _ in 0..100 {
            gate.process(1.0);
        }
        gate.reset();
        assert!(!gate.is_open());
        assert_eq!(gate.gain, 0.0);
        for _ in 0..1000 {
            assert_eq!(gate.process(0.0), 0.0);
        }
    }

    #[test]
    fn process_slice_matches_sample_loop() {
        let mut lim = Limiter::new(LimiterParams {
            sample_rate: FS,
            ceiling_db: -6.0,
            lookahead_ms: 1.0,
            attack_ms: 0.5,
            release_ms: 20.0,
        })
        .unwrap();
        let mut comp = Compressor::new(CompressorParams {
            sample_rate: FS,
            threshold_db: -20.0,
            ratio: 2.0,
            knee_db: 3.0,
            attack_ms: 2.0,
            release_ms: 30.0,
            makeup_db: 1.0,
        })
        .unwrap();
        let mut gate = Gate::new(GateParams {
            sample_rate: FS,
            threshold_db: -30.0,
            hysteresis_db: 3.0,
            attack_ms: 1.0,
            release_ms: 10.0,
        })
        .unwrap();
        let mut a = [0.5f64; 64];
        for (i, s) in a.iter_mut().enumerate() {
            *s = ((i as f64) * 0.2).sin() * 0.9;
        }
        let mut b = a;
        let mut c = a;
        lim.process_slice(&mut a);
        comp.process_slice(&mut b);
        gate.process_slice(&mut c);
        for s in a.iter().chain(b.iter()).chain(c.iter()) {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn time_coefficient_edges() {
        assert_eq!(time_coefficient(0.0, FS), 0.0);
        assert_eq!(time_coefficient(-1.0, FS), 0.0);
        assert!(time_coefficient(1000.0, FS) > 0.0 && time_coefficient(1000.0, FS) < 1.0);
    }
}
