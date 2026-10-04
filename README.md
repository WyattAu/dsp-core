# dsp-core

Foundational audio DSP primitives for the WyattAu estate — biquad filters,
FFT, convolution, windowing, dynamics, and oscillators — with
**allocation-free process paths**.

Estate layer: **L0** (leaf). Zero estate dependencies; the sole runtime
dependency is [`libm`](https://crates.io/crates/libm) (pure-Rust `no_std`
transcendental math). The FFT is implemented in-crate — this crate owns its
numerics rather than deferring to `rustfft`.

## Contents

| Module | What's inside |
|---|---|
| [`biquad`] | RBJ *Audio EQ Cookbook* filters (lowpass, highpass, band-pass ×2, notch, allpass, peaking EQ, low/high shelf) on a Direct Form I core, type-safe parameter structs, analytic `frequency_response` |
| [`fft`] | radix-2 DIT complex FFT (pre-computed twiddles + bit-reversal, unit forward / 1·N⁻¹ inverse) and a real FFT (N real → N/2 complex via the Sorensen split) |
| [`convolve`] | overlap-add streaming convolution and uniformly partitioned low-latency FFT convolution (frequency-domain delay line, constant cost per block) |
| [`window`] | rectangular, Hann, Hamming, Blackman, Blackman–Harris, Kaiser (β) — periodic and symmetric |
| [`dynamics`] | lookahead peak limiter, soft-knee compressor, hysteresis noise gate |
| [`osc`] | sine / saw / square / triangle phase accumulators with an instantaneous-frequency (FM) input; saw + square band-limited via PolyBLEP |
| [`loudness`] | ITU-R BS.1770-4 / EBU R128: K-weighting (coefficients generated per sample rate), gated integrated LUFS, loudness range, 4× oversampled true peak, normalisation gain with a clip check |
| [`math`] | dB↔linear conversion, flush-to-zero denormal guard |

## Design principles

- **f64 samples throughout.** f32 is a consumer concern — convert once at
  the edges.
- **No allocation in process paths.** Twiddle tables, delay rings, kernel
  spectra, and window buffers are built in constructors; every `process*`
  call is pure arithmetic over caller-owned memory.
- **No panics in process paths.** Constructors validate and return
  [`DspError`]; processing is infallible. The crate denies
  `clippy::unwrap_used`, `expect_used`, `panic`, and `indexing_slicing`,
  and forbids `unsafe`.
- **Denormal-safe.** Every recursive state word and output sample is
  flushed to zero below 1e-20 (≈ −400 dBFS) so silent filters cannot
  trigger denormal microcode stalls on a real-time thread.
- **`#![no_std]` + `alloc`.** Builds for `thumbv7em-none-eabihf` and
  `wasm32-unknown-unknown`.
- **Verified numerics.** FFT vs direct-DFT reference (< 1e-10 at N = 64),
  biquads vs analytic transfer functions (< 1e-6 phase error), convolvers
  vs direct convolution (< 1e-12), Parseval's theorem, impulse/sine bin
  placement, PolyBLEP aliasing suppression, limiter ceiling/unity
  behavior — all under `cargo test`.

## Example

```rust
use dsp_core::biquad::{Biquad, Coefficients, Resonant};
use dsp_core::fft::Fft;
use dsp_core::window::{Window, WindowKind};

let fs = 48_000.0;
let mut lowpass = Biquad::new(
    Coefficients::lowpass(Resonant {
        sample_rate: fs,
        frequency: 1_000.0,
        q: core::f64::consts::FRAC_1_SQRT_2,
    })
    .expect("valid parameters"),
);

let n = 64;
let mut frame: Vec<f64> = (0..n).map(|k| (0.05 * k as f64).sin()).collect();
lowpass.process_slice(&mut frame);

let window = Window::new(WindowKind::Hann, n, true, 0.0).expect("valid length");
window.process(&mut frame).expect("length matches");

let fft = Fft::new(n).expect("power of two");
let mut spectrum: Vec<f64> = frame.iter().flat_map(|v| [*v, 0.0]).collect();
fft.forward(&mut spectrum).expect("length matches");
```

## MSRV

Rust 1.85 (edition 2021).

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.

[`biquad`]: https://docs.rs/dsp-core/latest/dsp_core/biquad/
[`fft`]: https://docs.rs/dsp-core/latest/dsp_core/fft/
[`convolve`]: https://docs.rs/dsp-core/latest/dsp_core/convolve/
[`window`]: https://docs.rs/dsp-core/latest/dsp_core/window/
[`dynamics`]: https://docs.rs/dsp-core/latest/dsp_core/dynamics/
[`osc`]: https://docs.rs/dsp-core/latest/dsp_core/osc/
[`loudness`]: https://docs.rs/dsp-core/latest/dsp_core/loudness/
[`math`]: https://docs.rs/dsp-core/latest/dsp_core/math/
[`DspError`]: https://docs.rs/dsp-core/latest/dsp_core/error/enum.DspError.html
