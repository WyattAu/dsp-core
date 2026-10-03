# Changelog

All notable changes to this project are documented here. Format: [Keep a
Changelog](https://keepachangelog.com/) — versions follow [semver](https://semver.org).

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
