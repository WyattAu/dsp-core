//! Foundational audio DSP primitives — biquad filters, FFT, convolution,
//! windowing, dynamics, oscillators — with allocation-free process paths.
//!
//! `dsp-core` is the estate's L0 audio layer: the numerical kernels every
//! higher audio kit builds on, implemented in-crate (the FFT is a radix-2
//! DIT design owned here, not `rustfft`) with one curated external
//! dependency (`libm`, for `no_std` transcendental math). The crate is
//! `#![no_std]` + `alloc` and builds for embedded (`thumbv7em-none-eabihf`)
//! and wasm targets.
//!
//! # Design
//!
//! - **f64 samples throughout.** f32 is a consumer concern, handled at the
//!   edges (convert once on the way in and out).
//! - **No allocation in process paths.** Twiddle tables, delay rings,
//!   kernels, and window buffers are all built in constructors; every
//!   `process*` call is pure arithmetic on caller-owned memory.
//! - **No panics in process paths.** Fallible construction returns
//!   [`DspError`]; processing is infallible and index-free (the crate
//!   denies `clippy::unwrap_used`, `expect_used`, `panic`, and
//!   `indexing_slicing`).
//! - **Denormal-safe.** Every recursive state word and every output sample
//!   passes a flush-to-zero at 1e-20 (≈ −400 dBFS), so silent filters
//!   cannot stall the real-time thread in denormal microcode.
//! - **Verified numerics.** The FFT is tested against a direct DFT
//!   reference (error < 1e-10 at N = 64), biquads against the analytic
//!   transfer function (phase error < 1e-6 at test frequencies), and
//!   convolvers against direct convolution (error < 1e-12).
//!
//! # Example
//!
//! Filter a block, then look at its spectrum:
//!
//! ```
//! use dsp_core::biquad::{Biquad, Coefficients, Resonant};
//! use dsp_core::fft::Fft;
//! use dsp_core::window::{Window, WindowKind};
//!
//! let fs = 48_000.0;
//! let mut lowpass = Biquad::new(
//!     Coefficients::lowpass(Resonant { sample_rate: fs, frequency: 1_000.0, q: core::f64::consts::FRAC_1_SQRT_2 })
//!         .expect("valid parameters"),
//! );
//!
//! let n = 64;
//! let mut frame: Vec<f64> = (0..n)
//!     .map(|k| (0.05 * k as f64).sin())
//!     .collect();
//! lowpass.process_slice(&mut frame);
//!
//! let window = Window::new(WindowKind::Hann, n, true, 0.0).expect("valid length");
//! window.process(&mut frame).expect("length matches");
//!
//! let fft = Fft::new(n).expect("power of two");
//! let mut spectrum: Vec<f64> = frame.iter().flat_map(|v| [*v, 0.0]).collect();
//! fft.forward(&mut spectrum).expect("length matches");
//! ```
//!
//! # Modules
//!
//! | Module | Contents |
//! |---|---|
//! | [`biquad`] | RBJ cookbook filters on a Direct Form I core |
//! | [`fft`] | radix-2 complex FFT and real FFT |
//! | [`convolve`] | overlap-add and uniformly partitioned convolution |
//! | [`window`] | Hann/Hamming/Blackman/Blackman–Harris/Kaiser/rectangular |
//! | [`dynamics`] | lookahead limiter, soft-knee compressor, hysteresis gate |
//! | [`osc`] | sine/saw/square/triangle phase accumulators |
//! | [`math`] | dB conversion and denormal flushing |

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

extern crate alloc;

pub mod biquad;
pub mod convolve;
pub mod dynamics;
pub mod error;
pub mod fft;
pub mod math;
pub mod osc;
pub mod window;

pub use error::DspError;
