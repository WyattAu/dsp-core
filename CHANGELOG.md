# Changelog

All notable changes to this project are documented here. Format: [Keep a
Changelog](https://keepachangelog.com/) — versions follow [semver](https://semver.org).

## [Unreleased]

### Added

- **`loudness`** — ITU-R BS.1770-4 / EBU R128 measurement: `KWeighting`
  (the two-stage spec filter pair, coefficients *generated* per sample rate
  rather than hard-coded for 48 kHz), `LoudnessMeter` (400 ms blocks at
  75 % overlap, −70 LUFS absolute gate, −10 LU relative gate),
  `Measurement::gain_to` / `::gain_would_clip`, `TruePeak` (4× oversampled,
  four-phase cubic Lagrange), and `LoudnessRange` via the `loudness_range_lu`
  field (EBU Tech 3342: 10th-to-95th percentile inside a −20 LU gate).
  `ChannelLayout` carries the spec's channel weights (1.0 front, √2
  surround).

  Verified against the spec's own reference signals: a **stereo** 1 kHz sine
  at −23 dBFS integrates to −23.0 LUFS (EBU Tech 3341 case 1), and the
  reading is rate-independent across 44.1/48/96 kHz, which is what proves the
  per-rate coefficient generation is correct.

  This unblocks the estate's audiobook path: loudness is the number a
  deliverable is normalised against and the one streaming platforms gate on,
  and no other estate crate measured it.

## [0.1.0] - 2026-10-03

### Added

- **`biquad`** — RBJ *Audio EQ Cookbook* designs (lowpass, highpass, both
  band-passes, notch, allpass, peaking EQ, low/high shelf) computed from
  type-safe parameter structs (`Resonant` / `Equalizer` / `Shelf`) onto a
  Direct Form I `Biquad` core with per-sample denormal flushing and an
  analytic `frequency_response`.
- **`fft`** — radix-2 DIT complex FFT with pre-computed twiddle tables and
  bit-reversal permutation (allocation-free transforms, `&self`), inverse
  with 1/N scaling, and a real FFT (Sorensen split: N real samples through
  an N/2-point complex FFT) with a plain interleaved spectrum layout.
  Accuracy vs a direct DFT reference: < 1e-10 at N = 64.
- **`convolve`** — overlap-add streaming convolver (arbitrary block/kernel
  sizes) and a uniformly partitioned low-latency convolver (UPOLS-style
  frequency-domain delay line; constant cost per block regardless of
  kernel length). Both verified against direct convolution (< 1e-12).
- **`window`** — rectangular, Hann, Hamming, Blackman, Blackman–Harris
  (cosine-sum) and Kaiser (β-parameter, in-crate Bessel I0) windows,
  symmetric and periodic variants, pre-computed at construction.
- **`dynamics`** — lookahead peak limiter (delay ring + window-peak gain
  computer + attack/release smoothing), soft-knee feed-forward compressor
  (quadratic knee, makeup gain), and a hysteresis noise gate.
- **`osc`** — phase-accumulator sine (quadrant-reduced 9th-order
  polynomial), saw and square (PolyBLEP band-limited), triangle; every
  oscillator takes an instantaneous-frequency input for FM/automation.
- **`math`** — dB↔linear conversion and the shared flush-to-zero
  denormal guard (threshold 1e-20 ≈ −400 dBFS).

### Design

- `#![no_std]` + `alloc`; sole dependency is `libm`. Builds for
  `thumbv7em-none-eabihf` and `wasm32-unknown-unknown`.
- f64 samples throughout; no allocation, no panics, no indexing in any
  process path (crate denies `unwrap_used`, `expect_used`, `panic`,
  `indexing_slicing`).
- Estate layer: **L0** (leaf — zero estate dependencies).

[0.1.0]: https://github.com/WyattAu/dsp-core/releases/tag/v0.1.0
