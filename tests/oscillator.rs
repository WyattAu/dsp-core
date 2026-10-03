#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    missing_docs
)]
//! Oscillator integration: frequency accuracy from zero crossings and
//! PolyBLEP aliasing suppression vs the naive saw.

use dsp_core::fft::Fft;
use dsp_core::osc::{Oscillator, SawOsc, SineOsc, SquareOsc, TriangleOsc};

const FS: f64 = 48_000.0;

fn crossings(samples: &[f64]) -> usize {
    samples
        .windows(2)
        .filter(|w| w[0] < 0.0 && w[1] >= 0.0)
        .count()
}

#[test]
fn frequency_accuracy_via_zero_crossings() {
    // One second at several musical frequencies: crossing count must match
    // the expected cycle count within 2.
    let n = 48_000;
    for freq in [20.0, 110.0, 440.0, 1_046.5, 7_000.0] {
        let expected = freq * n as f64 / FS;
        for (name, mut osc) in [
            ("sine", OscillatorBox::Sine(SineOsc::new(FS).unwrap())),
            ("saw", OscillatorBox::Saw(SawOsc::new(FS).unwrap())),
            ("square", OscillatorBox::Square(SquareOsc::new(FS).unwrap())),
            (
                "triangle",
                OscillatorBox::Triangle(TriangleOsc::new(FS).unwrap()),
            ),
        ] {
            let rendered: Vec<f64> = (0..n).map(|_| osc.process(freq)).collect();
            let got = crossings(&rendered);
            assert!(
                (got as f64 - expected).abs() <= 2.0,
                "{name} @ {freq}: {got} crossings vs {expected}"
            );
        }
    }
}

/// Small helper to loop the four oscillator types in one place.
enum OscillatorBox {
    Sine(SineOsc),
    Saw(SawOsc),
    Square(SquareOsc),
    Triangle(TriangleOsc),
}

impl OscillatorBox {
    fn process(&mut self, freq: f64) -> f64 {
        match self {
            OscillatorBox::Sine(o) => o.process(freq),
            OscillatorBox::Saw(o) => o.process(freq),
            OscillatorBox::Square(o) => o.process(freq),
            OscillatorBox::Triangle(o) => o.process(freq),
        }
    }
}

#[test]
fn polyblep_aliasing_suppression() {
    // Saw at a non-bin-aligned frequency: naive scattering into
    // non-harmonic bins vs PolyBLEP.
    const N: usize = 16384;
    let freq = 997.0f64;

    let nonharmonic_energy = |blep: bool| -> f64 {
        let mut phase = 0.0f64;
        let dt = freq / FS;
        let fund_bin = freq * (N as f64) / FS;
        let samples: Vec<f64> = (0..N)
            .map(|i| {
                let naive = 2.0 * phase - 1.0;
                let out = if blep {
                    // Match the crate: subtract the correction.
                    naive - dsp_core::osc::poly_blep(phase, dt)
                } else {
                    naive
                };
                phase += dt;
                phase -= phase.floor();
                out * (0.5 * (1.0 - (std::f64::consts::TAU * i as f64 / N as f64).cos()))
            })
            .collect();
        let fft = Fft::new(N).unwrap();
        let mut buf: Vec<f64> = samples.iter().flat_map(|v| [*v, 0.0]).collect();
        fft.forward(&mut buf).unwrap();
        (1..N / 2)
            .filter(|k| {
                // Exclude bins near any harmonic so we measure aliasing,
                // not the window's leakage.
                let d = (*k as f64 / fund_bin - (*k as f64 / fund_bin).round()) * fund_bin;
                d.abs() > 6.0 && *k > 100
            })
            .map(|k| buf[2 * k] * buf[2 * k] + buf[2 * k + 1] * buf[2 * k + 1])
            .sum()
    };

    let naive = nonharmonic_energy(false);
    let blep = nonharmonic_energy(true);
    let improvement_db = 10.0 * (naive / blep).log10();
    // PolyBLEP reliably buys an order of magnitude of suppression.
    assert!(
        improvement_db >= 10.0,
        "PolyBLEP improvement only {improvement_db:.1} dB (naive {naive}, blep {blep})"
    );
}

#[test]
fn fm_sweep_tracks_instantaneous_frequency() {
    // Exponential-ish sweep 200 → 4 kHz over one second: total cycles from
    // the integral of f(t) must match crossings.
    let mut osc = SineOsc::new(FS).unwrap();
    let n = 48_000;
    let mut expected_cycles = 0.0f64;
    let rendered: Vec<f64> = (0..n)
        .map(|k| {
            let t = k as f64 / FS;
            let freq = 200.0 * (20.0f64).powf(t); // 200 → 4000 Hz log sweep
            expected_cycles += freq / FS;
            osc.process(freq)
        })
        .collect();
    let got = crossings(&rendered) as f64;
    assert!(
        (got - expected_cycles).abs() <= 2.0,
        "sweep: {got} vs {expected_cycles}"
    );
}

#[test]
fn shapes_stay_in_range_long_term() {
    // 10 seconds at an awkward frequency: every shape stays bounded.
    let mut saw = SawOsc::new(FS).unwrap();
    let mut square = SquareOsc::new(FS).unwrap();
    let mut tri = TriangleOsc::new(FS).unwrap();
    for _ in 0..480_000 {
        for (name, s) in [
            ("saw", saw.process(1_237.0)),
            ("square", square.process(1_237.0)),
            ("triangle", tri.process(1_237.0)),
        ] {
            assert!(s.is_finite() && s.abs() <= 1.01, "{name}: {s}");
        }
    }
}
