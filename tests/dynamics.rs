#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    missing_docs
)]
//! Dynamics integration: limiter ceiling/unity behavior, latency, and
//! compressor/gate steady states.

use dsp_core::dynamics::{Compressor, CompressorParams, Gate, GateParams, Limiter, LimiterParams};
use dsp_core::math::{db_to_linear, linear_to_db};

const FS: f64 = 48_000.0;

/// Render `n` samples of a sine through `f`.
fn render(n: usize, freq: f64, amplitude: f64, f: impl FnMut(f64) -> f64) -> Vec<f64> {
    let mut f = f;
    (0..n)
        .map(|k| {
            let t = k as f64 / FS;
            f(amplitude * (std::f64::consts::TAU * freq * t).sin())
        })
        .collect()
}

#[test]
fn limiter_ceiling_and_unity() {
    let ceiling_db = -6.0;
    let mut lim = Limiter::new(LimiterParams {
        sample_rate: FS,
        ceiling_db,
        lookahead_ms: 5.0,
        attack_ms: 0.2,
        release_ms: 60.0,
    })
    .unwrap();

    // Full-scale sine → peaks settle exactly at the ceiling.
    let ceiling = db_to_linear(ceiling_db);
    let mut peak = 0.0f64;
    for (k, out) in render(24_000, 997.0, 1.0, |s| lim.process(s))
        .into_iter()
        .enumerate()
    {
        if k > 12_000 {
            peak = peak.max(out.abs());
            assert!(out.abs() <= ceiling * 1.001, "overshoot {out}");
        }
    }
    assert!((linear_to_db(peak) - ceiling_db).abs() < 0.1);

    // Quiet sine → unity after release.
    let mut peak = 0.0f64;
    for (k, out) in render(24_000, 997.0, 0.05, |s| lim.process(s))
        .into_iter()
        .enumerate()
    {
        if k > 12_000 {
            peak = peak.max(out.abs());
        }
    }
    assert!(
        (linear_to_db(peak) - linear_to_db(0.05)).abs() < 0.2,
        "unity failed: {} dB",
        linear_to_db(peak)
    );
}

#[test]
fn limiter_lookahead_prevents_overshoot_on_transient() {
    // A full-scale burst entering an already-limited chain must never
    // exceed the ceiling at the output, thanks to the lookahead window.
    let mut lim = Limiter::new(LimiterParams {
        sample_rate: FS,
        ceiling_db: -12.0,
        lookahead_ms: 3.0,
        attack_ms: 0.0, // instant duck — lookahead does the anticipation
        release_ms: 200.0,
    })
    .unwrap();
    let ceiling = db_to_linear(-12.0);
    for k in 0..9_600 {
        let input = if (2_000..7_000).contains(&k) {
            1.0
        } else {
            0.0
        };
        let out = lim.process(input);
        assert!(
            out.abs() <= ceiling * 1.000_000_1,
            "sample {k} exceeded ceiling: {out}"
        );
    }
}

#[test]
fn compressor_gain_reduction_curve() {
    let mut comp = Compressor::new(CompressorParams {
        sample_rate: FS,
        threshold_db: -18.0,
        ratio: 3.0,
        knee_db: 0.0,
        attack_ms: 0.5,
        release_ms: 100.0,
        makeup_db: 0.0,
    })
    .unwrap();
    // −6 dBFS sine: 12 dB over threshold at 3:1 → 8 dB reduction.
    let mut peak = 0.0f64;
    for (k, out) in render(48_000, 997.0, db_to_linear(-6.0), |s| comp.process(s))
        .into_iter()
        .enumerate()
    {
        if k > 24_000 {
            peak = peak.max(out.abs());
        }
    }
    let out_db = linear_to_db(peak);
    let expected_db = -18.0 + 12.0 / 3.0;
    assert!(
        (out_db - expected_db).abs() < 0.3,
        "{out_db} vs {expected_db}"
    );
}

#[test]
fn gate_hysteresis_window() {
    let mut gate = Gate::new(GateParams {
        sample_rate: FS,
        threshold_db: -30.0,
        hysteresis_db: 10.0,
        attack_ms: 2.0,
        release_ms: 50.0,
    })
    .unwrap();
    // Open with −20 dB signal.
    for out in render(9_600, 997.0, db_to_linear(-20.0), |s| gate.process(s)) {
        let _ = out;
    }
    assert!(gate.is_open());
    // −35 dB: below open threshold (−30) but above close threshold (−40):
    // hysteresis must keep it open.
    for out in render(9_600, 997.0, db_to_linear(-35.0), |s| gate.process(s)) {
        let _ = out;
    }
    assert!(gate.is_open(), "hysteresis failed");
    // −50 dB: below the close threshold → closes.
    for out in render(24_000, 997.0, db_to_linear(-50.0), |s| gate.process(s)) {
        let _ = out;
    }
    assert!(!gate.is_open());
}
