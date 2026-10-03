#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    missing_docs
)]
//! Window integration: symmetry, normalization sums, and spectral
//! properties.

use dsp_core::fft::Fft;
use dsp_core::window::{Window, WindowKind};

#[test]
fn symmetric_and_periodic_hann_differ_correctly() {
    let n = 64;
    let sym = Window::new(WindowKind::Hann, n, false, 0.0).unwrap();
    let per = Window::new(WindowKind::Hann, n, true, 0.0).unwrap();
    // Symmetric: both endpoints are zero (2π/(N−1) sweep).
    assert!(sym.as_slice()[0].abs() < 1e-12);
    assert!(sym.as_slice()[n - 1].abs() < 1e-12);
    // Periodic: first sample zero, but the wrap-around sample is distinct
    // (2π/N sweep) — no duplicated endpoint.
    assert!(per.as_slice()[0].abs() < 1e-12);
    assert!(per.as_slice()[n - 1] > 0.001);
    // Tap 1 matches each formula.
    let s1 = 0.5 * (1.0 - (std::f64::consts::TAU / (n as f64 - 1.0)).cos());
    let p1 = 0.5 * (1.0 - (std::f64::consts::TAU / n as f64).cos());
    assert!((sym.as_slice()[1] - s1).abs() < 1e-12);
    assert!((per.as_slice()[1] - p1).abs() < 1e-12);
    assert!(per.is_periodic() && !sym.is_periodic());
}

#[test]
fn window_sums() {
    // Periodic Hann sums to N/2; rectangular sums to N.
    for n in [32usize, 33, 256] {
        let per = Window::new(WindowKind::Hann, n, true, 0.0).unwrap();
        let sum: f64 = per.as_slice().iter().sum();
        assert!((sum - n as f64 / 2.0).abs() < 1e-9);
        let rect = Window::new(WindowKind::Rectangular, n, true, 0.0).unwrap();
        let sum: f64 = rect.as_slice().iter().sum();
        assert!((sum - n as f64).abs() < 1e-12);
    }
    // COLA-ish sanity: Hann×Hann periodic sums to 3N/8.
    let n = 64;
    let per = Window::new(WindowKind::Hann, n, true, 0.0).unwrap();
    let sq: f64 = per.as_slice().iter().map(|v| v * v).sum();
    assert!((sq - 3.0 * n as f64 / 8.0).abs() < 1e-9);
}

#[test]
fn hann_mainlobe_width_is_four_bins() {
    // Periodic Hann, N = 64: spectrum zeros exactly at bins ±2 → the main
    // lobe spans 4 bins.
    let n = 64;
    let w = Window::new(WindowKind::Hann, n, true, 0.0).unwrap();
    let fft = Fft::new(n).unwrap();
    let mut buf: Vec<f64> = w.as_slice().iter().flat_map(|v| [*v, 0.0]).collect();
    fft.forward(&mut buf).unwrap();
    let mag = |k: usize| (buf[2 * k] * buf[2 * k] + buf[2 * k + 1] * buf[2 * k + 1]).sqrt();
    let dc = mag(0);
    assert!((dc - n as f64 / 2.0).abs() < 1e-9);
    // −6 dB points at ±1 (Hann main lobe convention).
    let m1 = mag(1);
    assert!((m1 / dc - 0.5).abs() < 1e-9, "|W[1]|/|W[0]| = {}", m1 / dc);
    // First nulls at ±2.
    assert!(mag(2) < 1e-9);
    assert!(mag(n - 2) < 1e-9);
}

#[test]
fn cosine_sum_windows_are_bin_confined() {
    // A K-term cosine-sum window has a discrete spectrum confined to ±(K−1)
    // bins around DC — everything past that is float noise. The −XX dB
    // "sidelobe" figures quoted for these windows are a continuous-frequency
    // property; the discrete signature is exact confinement.
    let n = 256;
    let highest_nonzero_bin = |kind: WindowKind| -> usize {
        let w = Window::new(kind, n, true, 0.0).unwrap();
        let fft = Fft::new(n).unwrap();
        let mut buf: Vec<f64> = w.as_slice().iter().flat_map(|v| [*v, 0.0]).collect();
        fft.forward(&mut buf).unwrap();
        let mags: Vec<f64> = (0..=n / 2)
            .map(|k| (buf[2 * k] * buf[2 * k] + buf[2 * k + 1] * buf[2 * k + 1]).sqrt())
            .collect();
        let dc = mags[0];
        (0..=n / 2)
            .rev()
            .find(|k| mags[*k] > dc * 1.0e-12)
            .unwrap_or(0)
    };
    assert_eq!(highest_nonzero_bin(WindowKind::Rectangular), 0);
    assert_eq!(highest_nonzero_bin(WindowKind::Hann), 1);
    assert_eq!(highest_nonzero_bin(WindowKind::Hamming), 1);
    assert_eq!(highest_nonzero_bin(WindowKind::Blackman), 2);
    assert_eq!(highest_nonzero_bin(WindowKind::BlackmanHarris), 3);
}
