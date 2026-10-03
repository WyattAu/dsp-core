#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    missing_docs,
    clippy::excessive_precision
)]
//! Biquad integration tests: impulse response vs the analytic transfer
//! function, DC/Nyquist gain identities, and hard-coded RBJ cookbook
//! reference values.

use dsp_core::biquad::{Biquad, Coefficients, Equalizer, Resonant, Shelf};
use dsp_core::fft::Fft;

const FS: f64 = 48_000.0;
const BUTTERWORTH_Q: f64 = core::f64::consts::FRAC_1_SQRT_2;

fn resonant(f: f64, q: f64) -> Resonant {
    Resonant {
        sample_rate: FS,
        frequency: f,
        q,
    }
}

/// Feed an impulse through `filter`, FFT the tail, and compare the complex
/// spectrum against the analytic transfer function evaluated at the same
/// bins.
#[test]
fn impulse_response_matches_transfer_function() {
    let n = 512; // window over the settled IR
    let coeffs = Coefficients::lowpass(resonant(2_000.0, 1.4)).unwrap();
    let mut filter = Biquad::new(coeffs);
    let mut ir = vec![0.0f64; n];
    ir[0] = filter.process(1.0);
    for slot in ir.iter_mut().skip(1) {
        *slot = filter.process(0.0);
    }
    // Analytic H(e^{jw}) at bins 0..n/2, including DC.
    let fft = Fft::new(n).unwrap();
    let mut buf = vec![0.0f64; 2 * n];
    for (dst, src) in buf.chunks_mut(2).zip(ir.iter()) {
        dst[0] = *src;
    }
    fft.forward(&mut buf).unwrap();
    for k in 0..n / 2 {
        let w = std::f64::consts::TAU * k as f64 / n as f64;
        let (cw, sw) = (w.cos(), w.sin());
        let (c2, s2) = ((2.0 * w).cos(), (2.0 * w).sin());
        let num_re = coeffs.b0 + coeffs.b1 * cw + coeffs.b2 * c2;
        let num_im = -(coeffs.b1 * sw + coeffs.b2 * s2);
        let den_re = 1.0 + coeffs.a1 * cw + coeffs.a2 * c2;
        let den_im = -(coeffs.a1 * sw + coeffs.a2 * s2);
        // H = num/den: multiply by conj(den)/|den|².
        let d2 = den_re * den_re + den_im * den_im;
        let h_re = (num_re * den_re + num_im * den_im) / d2;
        let h_im = (num_im * den_re - num_re * den_im) / d2;
        assert!(
            (buf[2 * k] - h_re).abs() < 1e-9 && (buf[2 * k + 1] - h_im).abs() < 1e-9,
            "bin {k}: got ({}, {}), want ({h_re}, {h_im})",
            buf[2 * k],
            buf[2 * k + 1]
        );
    }
}

#[test]
fn dc_and_nyquist_gain_identities() {
    // Lowpass: H(1) = 1 exactly by construction (b0+b1+b2 == 1+a1+a2).
    let lp = Biquad::new(Coefficients::lowpass(resonant(800.0, 0.6)).unwrap());
    let c = lp.coefficients();
    let dc = (c.b0 + c.b1 + c.b2) / (1.0 + c.a1 + c.a2);
    assert!((dc - 1.0).abs() < 1e-12);
    let resp = lp.frequency_response(0.0, FS);
    assert!((resp.magnitude - 1.0).abs() < 1e-12);

    // Highpass: |H(−1)| = 1 (Nyquist).
    let hp = Biquad::new(Coefficients::highpass(resonant(800.0, 0.6)).unwrap());
    let c = hp.coefficients();
    let nyq = (c.b0 - c.b1 + c.b2) / (1.0 - c.a1 + c.a2);
    assert!((nyq.abs() - 1.0).abs() < 1e-12);
    let resp = hp.frequency_response(FS / 2.0 - 1e-9, FS);
    assert!((resp.magnitude - 1.0).abs() < 1e-6);
}

#[test]
fn known_rbj_cookbook_values() {
    // Reference values computed independently from the cookbook formulas
    // (double precision, printed to 17 digits).
    let lp = Coefficients::lowpass(resonant(1_000.0, BUTTERWORTH_Q)).unwrap();
    let want = [
        3.91612666054738309e-03,
        7.83225332109476619e-03,
        3.91612666054738309e-03,
        -1.81534108270456795e+00,
        8.31005589346757612e-01,
    ];
    let got = [lp.b0, lp.b1, lp.b2, lp.a1, lp.a2];
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-15, "{g} vs {w}");
    }

    let peak = Coefficients::peaking_eq(Equalizer {
        sample_rate: FS,
        frequency: 1_000.0,
        q: 2.0,
        gain_db: 6.0,
    })
    .unwrap();
    let want = [
        1.02247276821985822e+00,
        -1.93811658055722291e+00,
        9.32367743910733182e-01,
        -1.93811658055722291e+00,
        9.54840512130591512e-01,
    ];
    let got = [peak.b0, peak.b1, peak.b2, peak.a1, peak.a2];
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-15, "{g} vs {w}");
    }

    let hs = Coefficients::high_shelf(Shelf {
        sample_rate: FS,
        frequency: 4_000.0,
        q: BUTTERWORTH_Q,
        gain_db: 7.0,
    })
    .unwrap();
    let want = [
        1.93493844583659569e+00,
        -2.71339594081437863e+00,
        1.05198704461045534e+00,
        -1.13635525184915109e+00,
        4.09884801481823657e-01,
    ];
    let got = [hs.b0, hs.b1, hs.b2, hs.a1, hs.a2];
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-14, "{g} vs {w}");
    }

    let notch = Coefficients::notch(resonant(3_000.0, 6.0)).unwrap();
    let want = [
        9.69095274505293758e-01,
        -1.79065457833769570e+00,
        9.69095274505293758e-01,
        -1.79065457833769570e+00,
        9.38190549010587516e-01,
    ];
    let got = [notch.b0, notch.b1, notch.b2, notch.a1, notch.a2];
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-15, "{g} vs {w}");
    }
}

#[test]
fn peaking_eq_boosts_center_by_gain() {
    // A +12 dB peaking EQ at 3 kHz shows ≈ +12 dB at 3 kHz.
    let eq = Biquad::new(
        Coefficients::peaking_eq(Equalizer {
            sample_rate: FS,
            frequency: 3_000.0,
            q: 4.0,
            gain_db: 12.0,
        })
        .unwrap(),
    );
    let resp = eq.frequency_response(3_000.0, FS);
    assert!((resp.magnitude_db() - 12.0).abs() < 1e-9);
}

#[test]
fn long_silence_stays_flushed() {
    // 10 minutes of silence through a high-Q filter: the ring decays into
    // the flush threshold and the state lands on exact zeros — it never
    // creeps into the denormal range where FP microcode stalls.
    let mut filter = Biquad::new(Coefficients::lowpass(resonant(50.0, 8.0)).unwrap());
    for _ in 0..500 {
        filter.process(1.0);
    }
    // Discard the audible ring, then require bit-exact silence forever.
    for _ in 0..200_000 {
        filter.process(0.0);
    }
    for _ in 0..480_000 * 10 {
        assert_eq!(filter.process(0.0), 0.0);
    }
}
