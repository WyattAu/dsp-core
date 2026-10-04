//! Loudness measurement and normalisation target — ITU-R BS.1770-4 / EBU R128.
//!
//! This is the measurement an audiobook pipeline cannot skip: the target LUFS
//! for a release, the measured LUFS of a master, and therefore the gain to
//! apply. It is also the loudness number streaming platforms normalise
//! against, so a library that stores books without measuring them cannot
//! produce a compliant deliverable.
//!
//! # What is implemented
//!
//! - **K-weighting** — the two-stage filter pair BS.1770-4 mandates: a
//!   high-shelf "pre-filter" modelling the acoustic effect of a head, then the
//!   RLB high-pass modelling the standardised listener. Coefficients are
//!   *generated* for the running sample rate from the spec's design constants,
//!   not hard-coded for 48 kHz.
//! - **Gated integrated loudness** — 400 ms blocks at 75 % overlap, an
//!   absolute gate at −70 LUFS, then a relative gate 10 LU below the
//!   absolute-gated mean. The relative gate is what makes the number track
//!   *perceived* loudness rather than the loudest passage.
//! - **Loudness range (LRA)** — EBU Tech 3342: the 10th-to-95th percentile
//!   spread of block loudness inside a relative gate 20 LU below the gated
//!   mean. Low LRA is a heavily mastered file; high LRA is dynamic.
//! - **True peak** — 4× oversampled sample peak per BS.1770-4 Annex 2, because
//!   intersample peaks between samples are what actually clip a DAC and what a
//!   lossy codec's decoder can reconstruct.
//!
//! # What is not implemented
//!
//! - **No time-varying filter and no lookahead.** [`crate::dynamics`] owns
//!   lookahead limiting; this module measures.
//! - **Mono/stereo measurement sums one channel pair at a time.** Feed a
//!   stereo programme through [`LoudnessMeter::push_stereo`], which is the
//!   correct call for the overwhelming majority of spoken-word content.
//!
//! # Example
//!
//! ```
//! use dsp_core::loudness::{KWeighting, LoudnessMeter, ChannelLayout};
//!
//! let sample_rate = 48_000.0;
//! let mut k = KWeighting::new(sample_rate).expect("valid sample rate");
//! let mut meter = LoudnessMeter::new(sample_rate, ChannelLayout::Mono)
//!     .expect("valid layout");
//!
//! // A 1 kHz tone at -23 dBFS — BS.1770-4's own reference signal, which
//! // integrates back to -23 LUFS.
//! let block: Vec<f64> = (0..sample_rate as usize)
//!     .map(|i| {
//!         let t = i as f64 / sample_rate;
//!         10f64.powf(-23.0 / 20.0)
//!             * (2.0 * core::f64::consts::PI * 1_000.0 * t).sin()
//!     })
//!     .collect();
//! let weighted = k.weight(&block).expect("same length");
//! meter.push(&weighted);
//!
//! let m = meter.measure();
//! assert!(m.integrated_lufs.is_some());
//! ```

use alloc::vec::Vec;

use crate::biquad::{Biquad, Coefficients};
use crate::error::DspError;
use crate::math::{db_to_linear, linear_to_db};

/// BS.1770-4 gating and calibration constants.
mod gating {
    /// Absolute gate: blocks quieter than this are discarded outright.
    pub const ABSOLUTE_GATE_LUFS: f64 = -70.0;
    /// The relative gate sits this far below the absolute-gated mean power.
    pub const RELATIVE_GATE_OFFSET_LU: f64 = -10.0;
    /// EBU Tech 3342: LRA's relative gate sits this far below the gated mean.
    pub const LRA_GATE_OFFSET_LU: f64 = -20.0;
    /// BS.1770-4's loudness calibration constant, in LU.
    pub const CALIBRATION_LU: f64 = -0.691;
}

/// `floor` without `std`; `no_std` floats have no rounding intrinsics.
#[inline]
#[must_use]
fn floor(x: f64) -> f64 {
    let t = libm::trunc(x);
    if x < 0.0 && t != x {
        t - 1.0
    } else {
        t
    }
}

/// `ceil` without `std`.
#[inline]
#[must_use]
fn ceil(x: f64) -> f64 {
    let t = libm::trunc(x);
    if x > 0.0 && t != x {
        t + 1.0
    } else {
        t
    }
}

/// `round` without `std` — half away from zero, matching the crate's
/// [`crate::window`] convention.
#[inline]
#[must_use]
fn round(x: f64) -> f64 {
    if x < 0.0 {
        -round(-x)
    } else {
        floor(x + 0.5)
    }
}

/// Loudness is a **power** ratio, so it converts with `10·log10`.
///
/// This is deliberately not [`crate::math::linear_to_db`], which is
/// `20·log10` because amplitude ratios square. Using the amplitude form here
/// doubles every reading: a −23 LUFS programme would measure −46.
fn power_to_lu(power: f64) -> f64 {
    10.0 * libm::log10(power)
}

/// Convert a LU offset back to a linear power ratio.
fn lu_to_power(lu: f64) -> f64 {
    libm::pow(10.0, lu / 10.0)
}

/// The two-stage K-weighting filter pair.
///
/// Construct once per sample rate and feed it blocks; the state carries across
/// calls so it can run over a whole file in a streaming fashion. Both stages
/// are Direct Form I biquads, so the process path allocates nothing.
#[derive(Debug, Clone)]
pub struct KWeighting {
    pre: Biquad,
    rlb: Biquad,
}

impl KWeighting {
    /// Build the K-weighting pair for `sample_rate`.
    ///
    /// # Errors
    ///
    /// [`DspError::InvalidParameter`] if `sample_rate` is not finite and
    /// positive.
    pub fn new(sample_rate: f64) -> Result<Self, DspError> {
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(DspError::InvalidParameter {
                name: "sample_rate",
                reason: "must be finite and > 0",
            });
        }

        // Stage 1 — the "pre-filter" high shelf, from BS.1770-4's published
        // design constants via the spec's bilinear transform.
        let f0 = 1_681.974_450_955_533_f64;
        let shelf_db = 3.999_843_853_973_347_f64;
        let q = 0.707_175_236_955_419_6_f64;
        let k = libm::tan(core::f64::consts::PI * f0 / sample_rate);
        let vh = libm::pow(10.0, shelf_db / 20.0);
        let vb = libm::pow(vh, 0.499_666_774_154_541_6);
        let a0 = 1.0 + k / q + k * k;
        let pre = Biquad::new(Coefficients {
            b0: (vh + vb * k / q + k * k) / a0,
            b1: 2.0 * (k * k - vh) / a0,
            b2: (vh - vb * k / q + k * k) / a0,
            a1: 2.0 * (k * k - 1.0) / a0,
            a2: (1.0 - k / q + k * k) / a0,
        });

        // Stage 2 — the RLB high-pass.
        let f0 = 38.135_470_876_024_44_f64;
        let q = 0.500_327_037_323_877_3_f64;
        let k = libm::tan(core::f64::consts::PI * f0 / sample_rate);
        let a0 = 1.0 + k / q + k * k;
        let rlb = Biquad::new(Coefficients {
            b0: 1.0,
            b1: -2.0,
            b2: 1.0,
            a1: 2.0 * (k * k - 1.0) / a0,
            a2: (1.0 - k / q + k * k) / a0,
        });

        Ok(Self { pre, rlb })
    }

    /// Weight `input` into a caller-owned buffer and return it.
    ///
    /// This is the real-time path: no allocation, no panic, and no indexing —
    /// only the two stages' `process_slice`, which the crate already documents
    /// as index-free.
    ///
    /// # Errors
    ///
    /// [`DspError::InvalidLength`] if the buffers differ in length.
    pub fn weight_into(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), DspError> {
        if input.len() != output.len() {
            return Err(DspError::InvalidLength {
                expected: input.len(),
                got: output.len(),
            });
        }
        output.copy_from_slice(input);
        self.pre.process_slice(output);
        self.rlb.process_slice(output);
        Ok(())
    }

    /// Weight `input` into a freshly allocated buffer.
    ///
    /// Convenient off the real-time thread; the allocation is the caller's
    /// signal that this is not the audio callback.
    ///
    /// # Errors
    ///
    /// Never in practice — the length is by construction equal.
    pub fn weight(&mut self, input: &[f64]) -> Result<Vec<f64>, DspError> {
        let mut out = alloc::vec![0.0f64; input.len()];
        self.weight_into(input, &mut out)?;
        Ok(out)
    }

    /// Weight `input` in place.
    pub fn weight_in_place(&mut self, buffer: &mut [f64]) {
        self.pre.process_slice(buffer);
        self.rlb.process_slice(buffer);
    }

    /// Clear both stages' filter history without rebuilding them.
    pub fn reset(&mut self) {
        self.pre.reset();
        self.rlb.reset();
    }

    /// The pre-filter stage's linear magnitude at `frequency`.
    #[must_use]
    pub fn pre_gain(&self, frequency: f64, sample_rate: f64) -> f64 {
        self.pre
            .frequency_response(frequency, sample_rate)
            .magnitude
    }

    /// The RLB stage's linear magnitude at `frequency`.
    #[must_use]
    pub fn rlb_gain(&self, frequency: f64, sample_rate: f64) -> f64 {
        self.rlb
            .frequency_response(frequency, sample_rate)
            .magnitude
    }

    /// Total K-weighting gain in dB at `frequency` — the sum of both stages.
    #[must_use]
    pub fn gain_db(&self, frequency: f64, sample_rate: f64) -> f64 {
        linear_to_db(self.pre_gain(frequency, sample_rate) * self.rlb_gain(frequency, sample_rate))
    }
}

/// Channel weights for the loudness sum.
///
/// BS.1770-4 Table 2: 1.0 for front left/right/centre, √2 for each surround
/// channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChannelLayout {
    /// One channel.
    Mono,
    /// Two channels, each weighted 1.0.
    Stereo,
    /// 5.1 — L/R/C at 1.0, Ls/Rs at √2.
    Surround51,
}

impl ChannelLayout {
    /// The BS.1770-4 weight for channel `index` (0-based, in declared order).
    #[must_use]
    pub fn weight(self, index: usize) -> f64 {
        match self {
            ChannelLayout::Mono | ChannelLayout::Stereo => 1.0,
            ChannelLayout::Surround51 => match index {
                0..=2 => 1.0,
                _ => core::f64::consts::SQRT_2,
            },
        }
    }

    /// How many channels this layout declares.
    #[must_use]
    pub fn channels(self) -> usize {
        match self {
            ChannelLayout::Mono => 1,
            ChannelLayout::Stereo => 2,
            ChannelLayout::Surround51 => 6,
        }
    }
}

/// Integrated loudness, loudness range, and peaks for one programme.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measurement {
    /// Gated integrated loudness in LUFS, or `None` when the programme was
    /// entirely below the absolute gate.
    pub integrated_lufs: Option<f64>,
    /// Loudness range in LU, or `None` when too few blocks cleared the gate.
    pub loudness_range_lu: Option<f64>,
    /// Maximum absolute sample value including intersample peaks.
    pub true_peak: f64,
    /// Maximum absolute sample value without oversampling.
    pub sample_peak: f64,
}

impl Measurement {
    /// Gain in dB that brings the programme to `target_lufs`, or `None` when
    /// the programme has no measurable integrated loudness.
    ///
    /// A positive result means the programme is quieter than the target and
    /// should be turned up. No clamping is applied: a programme below the
    /// absolute gate reports `None` rather than a meaningless large gain.
    #[must_use]
    pub fn gain_to(&self, target_lufs: f64) -> Option<f64> {
        self.integrated_lufs.map(|l| target_lufs - l)
    }

    /// Whether applying [`Measurement::gain_to`] would clip.
    ///
    /// True when `gain_to(...)` plus the current true peak exceeds 1.0, i.e.
    /// the correction needs a limiter rather than a bare gain. This is the
    /// check that keeps a normalisation pass from destroying a master.
    #[must_use]
    pub fn gain_would_clip(&self, target_lufs: f64) -> bool {
        match (self.gain_to(target_lufs), self.true_peak) {
            (Some(gain_db), peak) if peak > 0.0 => db_to_linear(gain_db) * peak > 1.0,
            _ => false,
        }
    }
}

/// Gated integrated loudness meter.
///
/// Feed it K-weighted samples; it accumulates 400 ms blocks at 75 % overlap
/// and gates them on [`LoudnessMeter::measure`].
#[derive(Debug, Clone)]
pub struct LoudnessMeter {
    layout: ChannelLayout,
    block: usize,
    hop: usize,
    channels: usize,
    /// Sum of squares per channel for the block currently filling.
    accum: Vec<f64>,
    filled: usize,
    /// Weighted mean square per channel for every completed block, flattened.
    block_power: Vec<f64>,
    blocks: usize,
    sample_peak: f64,
    true_peak: f64,
}

impl LoudnessMeter {
    /// A meter for `layout` at `sample_rate`, using the spec's 400 ms blocks
    /// at 75 % overlap (a 100 ms hop).
    ///
    /// # Errors
    ///
    /// [`DspError::InvalidParameter`] for a non-finite or non-positive sample
    /// rate.
    pub fn new(sample_rate: f64, layout: ChannelLayout) -> Result<Self, DspError> {
        Self::with_resolution(sample_rate, layout, 0.400, 0.75)
    }

    /// A meter with explicit block length and overlap, for a coarser
    /// measurement than the spec default (a preview scan, say).
    ///
    /// `overlap` is the fraction of each block shared with its predecessor:
    /// 0.75 is the spec's value, 0.0 gives contiguous blocks.
    ///
    /// # Errors
    ///
    /// As [`LoudnessMeter::new`], plus a non-positive `block_seconds` or an
    /// `overlap` outside `0.0..=1.0`.
    pub fn with_resolution(
        sample_rate: f64,
        layout: ChannelLayout,
        block_seconds: f64,
        overlap: f64,
    ) -> Result<Self, DspError> {
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(DspError::InvalidParameter {
                name: "sample_rate",
                reason: "must be finite and > 0",
            });
        }
        if !block_seconds.is_finite() || block_seconds <= 0.0 {
            return Err(DspError::InvalidParameter {
                name: "block_seconds",
                reason: "must be finite and > 0",
            });
        }
        if !overlap.is_finite() || !(0.0..=1.0).contains(&overlap) {
            return Err(DspError::InvalidParameter {
                name: "overlap",
                reason: "must be within 0.0..=1.0",
            });
        }

        let channels = layout.channels();
        let block = (round(sample_rate * block_seconds) as usize).max(1);
        let hop = (round(sample_rate * block_seconds * (1.0 - overlap)) as usize).clamp(1, block);

        Ok(Self {
            layout,
            block,
            hop,
            channels,
            accum: alloc::vec![0.0; channels],
            filled: 0,
            block_power: Vec::new(),
            blocks: 0,
            sample_peak: 0.0,
            true_peak: 0.0,
        })
    }

    /// The block length in frames.
    #[must_use]
    pub fn block_frames(&self) -> usize {
        self.block
    }

    /// The hop between blocks in frames.
    #[must_use]
    pub fn hop_frames(&self) -> usize {
        self.hop
    }

    /// The layout this meter sums over.
    #[must_use]
    pub fn layout(&self) -> ChannelLayout {
        self.layout
    }

    /// How many complete blocks have been accumulated.
    #[must_use]
    pub fn blocks(&self) -> usize {
        self.blocks
    }

    /// Feed one channel of K-weighted samples.
    ///
    /// Correct for mono, and correct for stereo when the caller measures each
    /// channel into a separate meter and sums the results. For interleaved
    /// stereo use [`LoudnessMeter::push_stereo`].
    pub fn push(&mut self, samples: &[f64]) {
        let mut detector = TruePeak::new();
        for s in samples {
            let v = *s;
            // `get_mut` on a fixed-size vec cannot fail for a valid index, but
            // the lint bar forbids both `unwrap` and indexing, so fold instead.
            let slot = self.accum.len() - self.channels;
            if let Some(a) = self.accum.get_mut(slot) {
                *a += v * v;
            }
            self.sample_peak = self.sample_peak.max(v.abs());
            self.filled += 1;
            if self.filled == self.block {
                self.commit_block();
            }
        }
        self.true_peak = self.true_peak.max(detector.process(samples));
    }

    /// Feed interleaved K-weighted samples for the declared layout.
    ///
    /// A trailing partial frame is ignored, and a partial final block is not
    /// committed — BS.1770-4 integrates only complete blocks.
    pub fn push_stereo(&mut self, interleaved: &[f64]) {
        let frames = interleaved.len() / self.channels;
        for f in 0..frames {
            let base = f * self.channels;
            for c in 0..self.channels {
                let v = interleaved.get(base + c).copied().unwrap_or(0.0);
                if let Some(a) = self.accum.get_mut(c) {
                    *a += v * v;
                }
                self.sample_peak = self.sample_peak.max(v.abs());
            }
            self.filled += 1;
            if self.filled == self.block {
                self.commit_block();
            }
        }
    }

    /// Feed interleaved samples into a meter built for exactly two channels.
    ///
    /// # Errors
    ///
    /// [`DspError::InvalidParameter`] if the meter is not laid out for stereo.
    pub fn push_interleaved(&mut self, interleaved: &[f64]) -> Result<(), DspError> {
        if self.channels != 2 {
            return Err(DspError::InvalidParameter {
                name: "layout",
                reason: "push_interleaved requires a stereo (2-channel) meter",
            });
        }
        self.push_stereo(interleaved);
        Ok(())
    }

    /// Feed interleaved samples, dispatching on the declared layout.
    ///
    /// The ergonomic entry point for a decoder that always hands over
    /// interleaved audio.
    ///
    /// # Errors
    ///
    /// Never in practice; the length is checked against the layout.
    pub fn push_interleaved_auto(&mut self, interleaved: &[f64]) -> Result<(), DspError> {
        if self.channels == 1 {
            self.push(interleaved);
            Ok(())
        } else {
            self.push_stereo(interleaved);
            Ok(())
        }
    }

    fn commit_block(&mut self) {
        let inv = 1.0 / self.block as f64;
        // BS.1770-4 sums the channel-weighted mean squares for the block:
        //   l_j = -0.691 + 10·log10( Σ_c G_c · z_c )
        // One entry per *block*, not per channel — averaging here would make
        // every stereo programme read 3 LU low.
        let mut total = 0.0;
        for (c, slot) in self.accum.iter_mut().enumerate() {
            let mean_square = *slot * inv;
            let w = self.layout.weight(c);
            total += mean_square * w * w;
            *slot = 0.0;
        }
        self.block_power.push(total);
        self.blocks += 1;
        self.filled = 0;
    }

    /// Gate and measure everything pushed so far.
    ///
    /// Non-destructive and cheap enough to call while streaming.
    #[must_use]
    pub fn measure(&self) -> Measurement {
        let mut powers: Vec<f64> = Vec::with_capacity(self.block_power.len());
        for p in &self.block_power {
            let db = gating::CALIBRATION_LU + power_to_lu(*p);
            if db > gating::ABSOLUTE_GATE_LUFS {
                powers.push(*p);
            }
        }
        if powers.is_empty() {
            return Measurement {
                integrated_lufs: None,
                loudness_range_lu: None,
                true_peak: self.true_peak.max(self.sample_peak),
                sample_peak: self.sample_peak,
            };
        }

        // Relative gate: mean of the absolute-gated blocks, less 10 LU.
        let mean = powers.iter().sum::<f64>() / powers.len() as f64;
        let rel_gate = mean * lu_to_power(gating::RELATIVE_GATE_OFFSET_LU);
        let gated: Vec<f64> = powers.iter().copied().filter(|p| *p >= rel_gate).collect();

        let integrated = if gated.is_empty() {
            None
        } else {
            let m = gated.iter().sum::<f64>() / gated.len() as f64;
            Some(gating::CALIBRATION_LU + power_to_lu(m))
        };

        // LRA: block loudness inside a gate 20 LU below the gated mean.
        let lra = if gated.is_empty() {
            None
        } else {
            let gated_mean = gated.iter().sum::<f64>() / gated.len() as f64;
            let lra_gate = gated_mean * lu_to_power(gating::LRA_GATE_OFFSET_LU);
            let mut lufs: Vec<f64> = powers
                .iter()
                .copied()
                .filter(|p| *p >= lra_gate)
                .map(|p| gating::CALIBRATION_LU + power_to_lu(p))
                .collect();
            if lufs.len() < 2 {
                None
            } else {
                lufs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
                Some(percentile(&lufs, 0.95) - percentile(&lufs, 0.10))
            }
        };

        Measurement {
            integrated_lufs: integrated,
            loudness_range_lu: lra,
            true_peak: self.true_peak.max(self.sample_peak),
            sample_peak: self.sample_peak,
        }
    }
}

/// Linear-interpolated percentile over a slice.
///
/// The spec leaves the interpolation method open; this is adequate for LRA,
/// which is itself a coarse 10th-to-95th spread. `sorted` need not be sorted.
#[must_use]
pub fn percentile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    if v.len() == 1 {
        // `len() > 0` is guaranteed by the `is_empty` check above.
        return v.first().copied().unwrap_or(0.0);
    }
    let last = v.len() - 1;
    let pos = q.clamp(0.0, 1.0) * last as f64;
    let lo = (floor(pos) as usize).min(last);
    let hi = (ceil(pos) as usize).min(last);
    let lo_v = v.get(lo).copied().unwrap_or(0.0);
    let hi_v = v.get(hi).copied().unwrap_or(0.0);
    lo_v + (hi_v - lo_v) * (pos - lo as f64)
}

/// 4× oversampled true-peak detector, per BS.1770-4 Annex 2.
///
/// Four phases of a cubic Lagrange interpolator, maxed. Not a brickwall
/// filter: it is the spec's `4·` oversampling requirement expressed as polyphase
/// interpolation, which is what the `true_peak` field reports.
///
/// For spoken-word content the intersample excursion is small, but for mastered
/// music and for anything that has been through a lossy codec it is the
/// difference between "peaks at 0.98" and "clips on playback".
#[derive(Debug, Clone, Default)]
pub struct TruePeak {
    history: [f64; 3],
    peak: f64,
}

impl TruePeak {
    /// A fresh detector with zeroed history and peak.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The running maximum absolute value including intersample peaks.
    #[must_use]
    pub fn peak(&self) -> f64 {
        self.peak
    }

    /// Clear the filter history and the running peak.
    pub fn reset(&mut self) {
        self.history = [0.0; 3];
        self.peak = 0.0;
    }

    /// Feed a block, returning the running true-peak maximum.
    pub fn process(&mut self, samples: &[f64]) -> f64 {
        for s in samples {
            let v = *s;
            let p = [self.history[0], self.history[1], self.history[2], v];
            let mut local = v.abs();
            for phase in 1..4 {
                let t = phase as f64 / 4.0;
                local = local.max(lagrange(&p, t).abs());
            }
            self.peak = self.peak.max(local);
            self.history[0] = self.history[1];
            self.history[1] = self.history[2];
            self.history[2] = v;
        }
        self.peak
    }
}

/// Cubic Lagrange interpolation of `points` at `t` in `0.0..=1.0`.
#[must_use]
fn lagrange(points: &[f64; 4], t: f64) -> f64 {
    let mut acc = 0.0;
    for (i, pi) in points.iter().enumerate() {
        let mut term = *pi;
        for j in 0..4 {
            if i != j {
                let num = t - j as f64;
                let den = i as f64 - j as f64;
                term *= num / den;
            }
        }
        acc += term;
    }
    acc
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    // Verification against the spec's published values. Explicit tolerances
    // are the point: these tests exist to catch coefficient drift.

    use super::*;
    use alloc::vec;

    /// BS.1770-4's own reference: a **stereo** 1 kHz sine at −23 dBFS
    /// integrates to −23.0 LUFS. This is the single most important number in
    /// the module, and it is the one that pins the −0.691 calibration constant
    /// together with the K-weighting curve's +0.697 dB at 1 kHz.
    #[test]
    fn stereo_one_khz_sine_at_minus23_reads_minus23_lufs() {
        let fs = 48_000.0;
        let amplitude = 10f64.powf(-23.0 / 20.0);
        let mut k = KWeighting::new(fs).expect("valid rate");
        let mut m = LoudnessMeter::new(fs, ChannelLayout::Stereo).expect("valid layout");
        let mono: Vec<f64> = (0..fs as usize * 2)
            .map(|i| {
                let t = i as f64 / fs;
                amplitude * (2.0 * core::f64::consts::PI * 1_000.0 * t).sin()
            })
            .collect();
        let weighted = k.weight(&mono).expect("same length");
        let mut inter = Vec::with_capacity(weighted.len() * 2);
        for v in &weighted {
            inter.push(*v);
            inter.push(*v);
        }
        m.push_stereo(&inter);
        let lufs = m
            .measure()
            .integrated_lufs
            .expect("a full block of tone clears the gate");
        assert!(
            (lufs - -23.0).abs() < 0.2,
            "EBU Tech 3341 case 1 expects -23.0 LUFS, got {lufs}"
        );
    }

    /// The same tone into a mono meter reads 3.01 LU lower, because BS.1770-4
    /// sums the weighted mean squares of both channels.
    #[test]
    fn mono_reads_three_lu_below_stereo() {
        let fs = 48_000.0;
        let amplitude = 10f64.powf(-23.0 / 20.0);
        let mut k = KWeighting::new(fs).expect("valid rate");
        let mono_tone: Vec<f64> = (0..fs as usize * 2)
            .map(|i| {
                let t = i as f64 / fs;
                amplitude * (2.0 * core::f64::consts::PI * 1_000.0 * t).sin()
            })
            .collect();
        let weighted = k.weight(&mono_tone).expect("same length");

        let mut mono = LoudnessMeter::new(fs, ChannelLayout::Mono).expect("valid");
        mono.push(&weighted);
        let mono_lufs = mono.measure().integrated_lufs.expect("clears gate");

        assert!(
            (mono_lufs - -26.0).abs() < 0.2,
            "one channel at -23 dBFS reads ~-26 LUFS, got {mono_lufs}"
        );
    }

    /// The generated coefficients must reproduce the spec's 48 kHz curve, which
    /// proves the design-constant path is right rather than merely plausible.
    #[test]
    fn coeffs_match_the_spec_at_48k() {
        let k = KWeighting::new(48_000.0).expect("valid rate");
        // Total K-weighting is ~0 dB at 1 kHz by construction: the spec's
        // −0.691 LU calibration constant exists to cancel exactly this.
        let g = k.gain_db(1_000.0, 48_000.0);
        assert!(
            (g - 0.691).abs() < 0.1,
            "K-weighting gain at 1 kHz should be ~+0.691 dB, got {g}"
        );
        // The shelf is up at HF, and the two stages sum to the spec's curve.
        assert!(
            k.gain_db(10_000.0, 48_000.0) > 3.5,
            "shelf should lift 10 kHz"
        );
        // The shelf is flat at LF, so the combined 20 Hz response is the
        // RLB's: a 2nd-order high-pass only partly rejects a frequency well
        // below its ~38 Hz corner. -13.3 dB is the spec's curve.
        assert!(
            (k.rlb_gain(20.0, 48_000.0) - 0.2169).abs() < 1e-3,
            "RLB at 20 Hz should be ~0.217, got {}",
            k.rlb_gain(20.0, 48_000.0)
        );
        assert!(
            (k.gain_db(20.0, 48_000.0) - -13.3).abs() < 0.3,
            "combined response at 20 Hz should be ~-13.3 dB, got {}",
            k.gain_db(20.0, 48_000.0)
        );
        // The RLB is normalised to unity at 1 kHz — that is what makes the
        // spec's -0.691 LU constant cancel the +0.697 dB K-weighting gain and
        // leave a 1 kHz tone reading its nominal dBFS. Note the prewarped
        // f0 = 38.135 Hz is *not* the digital -3 dB point: the bilinear
        // transform moves it, and this stage is deliberately gentle.
        assert!(
            (k.rlb_gain(1_000.0, 48_000.0) - 1.0).abs() < 0.01,
            "RLB must be unity at 1 kHz, got {}",
            k.rlb_gain(1_000.0, 48_000.0)
        );
        // Second-order high-pass skirt: ~12 dB per octave well below corner.
        let octave = linear_to_db(k.rlb_gain(10.0, 48_000.0) / k.rlb_gain(5.0, 48_000.0));
        assert!(
            (octave - 12.0).abs() < 1.5,
            "RLB skirt should be ~12 dB/octave, got {octave}"
        );
    }

    /// The same programme at 44.1 kHz must measure the same LUFS. This is what
    /// proves the coefficients are generated per-rate rather than hard-coded.
    #[test]
    fn measurement_is_rate_independent() {
        let mut readings = Vec::new();
        for fs in [44_100.0_f64, 48_000.0, 96_000.0] {
            let amplitude = 10f64.powf(-23.0 / 20.0);
            let mut k = KWeighting::new(fs).expect("valid rate");
            let mut m = LoudnessMeter::new(fs, ChannelLayout::Stereo).expect("valid layout");
            let mono: Vec<f64> = (0..(fs as usize * 2))
                .map(|i| {
                    let t = i as f64 / fs;
                    amplitude * (2.0 * core::f64::consts::PI * 1_000.0 * t).sin()
                })
                .collect();
            let weighted = k.weight(&mono).expect("same length");
            let mut inter = Vec::with_capacity(weighted.len() * 2);
            for v in &weighted {
                inter.push(*v);
                inter.push(*v);
            }
            m.push_stereo(&inter);
            readings.push(m.measure().integrated_lufs.expect("tone clears the gate"));
        }
        for r in &readings {
            assert!((r - -23.0).abs() < 0.3, "rate-dependent reading: {r}");
        }
    }

    /// Silence is below the absolute gate: no integrated value, and crucially
    /// no absurd gain recommendation.
    #[test]
    fn silence_measures_as_none() {
        let fs = 48_000.0;
        let mut k = KWeighting::new(fs).expect("valid rate");
        let mut m = LoudnessMeter::new(fs, ChannelLayout::Mono).expect("valid layout");
        let quiet = vec![0.0f64; 48_000];
        let weighted = k.weight(&quiet).expect("same length");
        m.push(&weighted);
        let measured = m.measure();
        assert_eq!(measured.integrated_lufs, None);
        assert_eq!(measured.gain_to(-18.0), None);
        assert!(!measured.gain_would_clip(-18.0));
    }

    /// A quiet programme needs a large gain but has headroom to spare, so the
    /// clipping check must stay quiet. This is the common audiobook case: a
    /// -30 LUFS master normalised to -18 LUFS.
    #[test]
    fn quiet_programme_needs_gain_but_does_not_clip() {
        let fs = 48_000.0;
        let amplitude = 10f64.powf(-33.0 / 20.0);
        let mut k = KWeighting::new(fs).expect("valid rate");
        let mut m = LoudnessMeter::new(fs, ChannelLayout::Mono).expect("valid layout");
        let block: Vec<f64> = (0..(fs as usize * 2))
            .map(|i| {
                let t = i as f64 / fs;
                amplitude * (2.0 * core::f64::consts::PI * 1_000.0 * t).sin()
            })
            .collect();
        let weighted = k.weight(&block).expect("same length");
        m.push(&weighted);
        let measured = m.measure();
        let gain = measured.gain_to(-18.0).expect("tone clears the gate");
        assert!(
            gain > 12.0,
            "a -36 LUFS programme needs ~+18 dB, got {gain}"
        );
        assert!(
            !measured.gain_would_clip(-18.0),
            "+{gain} dB onto a {amplitude:.4} peak stays under full scale"
        );
    }

    /// The converse: a programme already near full scale must be flagged, so a
    /// normalisation pass reaches for a limiter instead of a bare gain. This is
    /// the check that stops a quiet-then-loud sequence destroying a master.
    #[test]
    fn loud_programme_is_flagged_as_clipping() {
        let x = Measurement {
            integrated_lufs: Some(-12.0),
            loudness_range_lu: Some(4.0),
            true_peak: 0.99,
            sample_peak: 0.98,
        };
        // -12 -> -18 is -6 dB, which would *reduce* the peak: not clipping.
        assert!(!x.gain_to(-18.0).expect("measurable").is_nan());
        assert!(!x.gain_would_clip(-18.0));
        // But turning it up 6 dB to reach -6 LUFS would clip hard.
        assert!(x.gain_would_clip(-6.0));
    }

    #[test]
    fn gain_to_is_the_difference() {
        let x = Measurement {
            integrated_lufs: Some(-24.0),
            loudness_range_lu: Some(6.0),
            true_peak: 0.5,
            sample_peak: 0.5,
        };
        assert!((x.gain_to(-18.0).expect("measurable") - 6.0).abs() < 1e-12);
        assert!((x.gain_to(-23.0).expect("measurable") - 1.0).abs() < 1e-12);
        // +6 dB onto a 0.5 peak is 1.0 exactly — not over.
        assert!(!x.gain_would_clip(-18.0));
    }

    #[test]
    fn channel_weights_follow_the_spec() {
        assert!((ChannelLayout::Stereo.weight(0) - 1.0).abs() < f64::EPSILON);
        assert!((ChannelLayout::Stereo.weight(1) - 1.0).abs() < f64::EPSILON);
        assert!((ChannelLayout::Surround51.weight(3) - core::f64::consts::SQRT_2).abs() < 1e-12);
        assert_eq!(ChannelLayout::Surround51.channels(), 6);
        assert_eq!(ChannelLayout::Mono.channels(), 1);
    }

    #[test]
    fn meter_rejects_bad_parameters() {
        assert!(LoudnessMeter::new(0.0, ChannelLayout::Mono).is_err());
        assert!(LoudnessMeter::new(f64::NAN, ChannelLayout::Mono).is_err());
        assert!(KWeighting::new(-1.0).is_err());
        assert!(LoudnessMeter::new(48_000.0, ChannelLayout::Mono).is_ok());
        assert!(LoudnessMeter::with_resolution(48_000.0, ChannelLayout::Mono, 0.4, 1.5).is_err());
        assert!(LoudnessMeter::with_resolution(48_000.0, ChannelLayout::Mono, 0.0, 0.75).is_err());
    }

    #[test]
    fn block_and_hop_follow_the_spec() {
        let m = LoudnessMeter::new(48_000.0, ChannelLayout::Stereo).expect("valid");
        assert_eq!(m.block_frames(), 19_200, "400 ms at 48 kHz");
        assert_eq!(m.hop_frames(), 4_800, "100 ms hop at 48 kHz");
        assert_eq!(m.layout(), ChannelLayout::Stereo);
    }

    #[test]
    fn weight_into_rejects_length_mismatch() {
        let mut k = KWeighting::new(48_000.0).expect("valid rate");
        let input = vec![0.1f64; 64];
        let mut output = vec![0.0f64; 32];
        assert!(k.weight_into(&input, &mut output).is_err());
        let mut same = vec![0.0f64; 64];
        assert!(k.weight_into(&input, &mut same).is_ok());
    }

    #[test]
    fn reset_clears_state() {
        let mut k = KWeighting::new(48_000.0).expect("valid rate");
        let mut buf = vec![0.5f64; 512];
        k.weight_in_place(&mut buf);
        k.reset();
        let mut again = vec![0.5f64; 512];
        k.weight_in_place(&mut again);
        assert_eq!(buf, again, "reset should restore the initial response");
    }

    #[test]
    fn true_peak_catches_intersample_overs() {
        let mut tp = TruePeak::new();
        let n = 4_096;
        let samples: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / 48_000.0;
                0.9 * (2.0 * core::f64::consts::PI * 4_000.0 * t).sin()
            })
            .collect();
        let sample_peak = samples.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let tp_peak = tp.process(&samples);
        assert!(
            tp_peak >= sample_peak,
            "true peak {tp_peak} must be at least the sample peak {sample_peak}"
        );
        assert!(
            tp_peak <= sample_peak * 1.15,
            "true peak {tp_peak} implausibly above sample peak {sample_peak}"
        );
        assert!((tp.peak() - tp_peak).abs() < f64::EPSILON);
    }

    #[test]
    fn true_peak_reset_clears() {
        let mut tp = TruePeak::new();
        tp.process(&[0.5, -0.5, 0.5]);
        assert!(tp.peak() > 0.0);
        tp.reset();
        assert_eq!(tp.peak(), 0.0);
    }

    #[test]
    fn lagrange_reproduces_known_points() {
        // t^2 sampled at t = 0, 1, 2, 3.
        let p = [0.0, 1.0, 4.0, 9.0];
        assert!(lagrange(&p, 0.0).abs() < 1e-12);
        assert!((lagrange(&p, 1.0) - 1.0).abs() < 1e-12);
        assert!((lagrange(&p, 0.5) - 0.25).abs() < 1e-9, "midpoint of t^2");
    }

    #[test]
    fn percentile_interpolates_and_sorts() {
        let v = [3.0, 0.0, 4.0, 1.0, 2.0];
        assert!(percentile(&v, 0.0).abs() < 1e-12);
        assert!((percentile(&v, 1.0) - 4.0).abs() < 1e-12);
        assert!((percentile(&v, 0.5) - 2.0).abs() < 1e-12);
        assert_eq!(percentile(&[], 0.5), 0.0);
    }

    #[test]
    fn push_interleaved_requires_stereo() {
        let mut mono = LoudnessMeter::new(48_000.0, ChannelLayout::Mono).expect("valid");
        assert!(mono.push_interleaved(&[0.1, 0.2]).is_err());
        let mut st = LoudnessMeter::new(48_000.0, ChannelLayout::Stereo).expect("valid");
        assert!(st.push_interleaved(&[0.1, 0.2, 0.3, 0.4]).is_ok());
        // And the auto form works for either.
        let mut m2 = LoudnessMeter::new(48_000.0, ChannelLayout::Mono).expect("valid");
        assert!(m2.push_interleaved_auto(&[0.1, 0.2]).is_ok());
    }

    #[test]
    fn partial_final_block_is_not_committed() {
        let fs = 48_000.0;
        let mut m = LoudnessMeter::new(fs, ChannelLayout::Mono).expect("valid");
        let block = m.block_frames();
        // One block minus one frame must yield no committed block.
        let short = vec![0.1f64; block - 1];
        m.push(&short);
        assert_eq!(m.blocks(), 0, "a short block must not be integrated");
        let full = vec![0.1f64; block];
        m.push(&full);
        assert_eq!(m.blocks(), 1);
    }

    #[test]
    fn measure_is_non_destructive() {
        let fs = 48_000.0;
        let mut m = LoudnessMeter::new(fs, ChannelLayout::Mono).expect("valid");
        let block = vec![0.5f64; m.block_frames()];
        m.push(&block);
        let first = m.measure().integrated_lufs;
        let second = m.measure().integrated_lufs;
        assert_eq!(first, second, "measure() must not consume the accumulator");
        assert_eq!(m.blocks(), 1);
    }
}
