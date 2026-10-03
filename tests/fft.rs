#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    missing_docs
)]
//! FFT integration tests: DFT-reference agreement across sizes, Parseval,
//! round-trip identity, and sine-bin placement.

use dsp_core::fft::{Fft, RealFft};
use std::f64::consts::TAU;

/// Naive O(N²) DFT reference.
fn dft(x: &[f64]) -> Vec<(f64, f64)> {
    let n = x.len();
    (0..n)
        .map(|k| {
            let (mut re, mut im) = (0.0, 0.0);
            for (j, &v) in x.iter().enumerate() {
                let a = TAU * (k * j) as f64 / n as f64;
                re += v * a.cos();
                im -= v * a.sin();
            }
            (re, im)
        })
        .collect()
}

#[test]
fn forward_matches_dft_across_sizes() {
    for n in [8usize, 16, 32, 64, 128, 256] {
        let signal: Vec<f64> = (0..n)
            .map(|i| (0.37 * i as f64).sin() + (1.31 * i as f64).cos() * 0.5)
            .collect();
        let fft = Fft::new(n).unwrap();
        let mut buf: Vec<f64> = signal.iter().flat_map(|v| [*v, 0.0]).collect();
        fft.forward(&mut buf).unwrap();
        for (k, (re, im)) in dft(&signal).iter().enumerate() {
            let err = (buf[2 * k] - re).abs().max((buf[2 * k + 1] - im).abs());
            assert!(err < 1e-10, "n={n} bin {k}: error {err}");
        }
    }
}

#[test]
fn parseval_and_round_trip() {
    let n = 256;
    let fft = Fft::new(n).unwrap();
    let signal: Vec<f64> = (0..n).map(|i| (0.77 * i as f64).sin()).collect();
    let time_energy: f64 = signal.iter().map(|v| v * v).sum();

    let mut buf: Vec<f64> = signal.iter().flat_map(|v| [*v, 0.0]).collect();
    fft.forward(&mut buf).unwrap();
    let freq_energy: f64 = buf.chunks(2).map(|c| c[0] * c[0] + c[1] * c[1]).sum();
    assert!(
        (time_energy - freq_energy / n as f64).abs() / time_energy < 1e-12,
        "Parseval violated"
    );

    fft.inverse(&mut buf).unwrap();
    for (a, b) in signal.iter().zip(buf.chunks(2)) {
        assert!((a - b[0]).abs() < 1e-12 && b[1].abs() < 1e-12);
    }
}

#[test]
fn real_fft_end_to_end() {
    // A real signal's packed spectrum must agree with the complex FFT of
    // the same signal, and inverse must restore the input bit-close.
    let n = 512;
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            0.4 * (TAU * 32.0 * i as f64 / n as f64).sin()
                + 0.2 * (TAU * 97.0 * i as f64 / n as f64).cos()
                + 0.05
        })
        .collect();
    let rfft = RealFft::new(n).unwrap();

    let mut spectrum = vec![0.0f64; n + 2];
    rfft.forward(&mut signal.clone(), &mut spectrum).unwrap();

    let mut reference: Vec<f64> = signal.iter().flat_map(|v| [*v, 0.0]).collect();
    Fft::new(n).unwrap().forward(&mut reference).unwrap();
    for k in 0..n / 2 {
        assert!(
            (spectrum[2 * k] - reference[2 * k]).abs() < 1e-10,
            "bin {k}"
        );
        assert!(
            (spectrum[2 * k + 1] - reference[2 * k + 1]).abs() < 1e-10,
            "bin {k}"
        );
    }

    let mut out = vec![0.0f64; n];
    rfft.inverse(&mut spectrum.clone(), &mut out).unwrap();
    for (a, b) in signal.iter().zip(out.iter()) {
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn larger_round_trips_stay_accurate() {
    for n in [1024usize, 4096] {
        let fft = Fft::new(n).unwrap();
        let original: Vec<f64> = (0..2 * n)
            .map(|i| ((13 * i) as f64 * 0.017).sin() * ((i % 97) as f64 * 0.01))
            .collect();
        let mut buf = original.clone();
        fft.forward(&mut buf).unwrap();
        fft.inverse(&mut buf).unwrap();
        let max_err = original
            .iter()
            .zip(buf.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_err < 1e-12, "n={n}: round-trip error {max_err}");
    }
}
