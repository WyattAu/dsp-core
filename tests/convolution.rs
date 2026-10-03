#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    missing_docs
)]
//! Convolution integration: impulse passthrough and agreement with direct
//! convolution for both convolvers.

use dsp_core::convolve::{OverlapAdd, PartitionedConvolver};

fn direct(x: &[f64], kernel: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; x.len()];
    for (i, out) in y.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (j, &k) in kernel.iter().enumerate() {
            if i >= j {
                acc += k * x[i - j];
            }
        }
        *out = acc;
    }
    y
}

fn stream<C>(conv: &mut C, x: &[f64], block: usize) -> Vec<f64>
where
    C: FnMut(&[f64], &mut [f64]) -> Result<(), dsp_core::DspError>,
{
    // The convolver consumes exact blocks: pad the stream with zeros to a
    // whole number of blocks, then trim to the input length. (Tests that
    // care about the kernel tail past the input pad their input instead.)
    let remainder = x.len() % block;
    let mut padded = x.to_vec();
    if remainder != 0 {
        padded.extend(std::iter::repeat_n(0.0, block - remainder));
    }
    let mut got = Vec::with_capacity(padded.len());
    for chunk in padded.chunks(block) {
        let mut dst = vec![0.0f64; block];
        conv(chunk, &mut dst).unwrap();
        got.extend_from_slice(&dst);
    }
    got.truncate(x.len());
    got
}

#[test]
fn overlap_add_impulse_is_kernel_passthrough() {
    // x = impulse → y = kernel, laid out over consecutive blocks.
    let kernel: Vec<f64> = [0.5, 0.3, 0.2, -0.1, 0.05].to_vec();
    let block = 8;
    let mut conv = OverlapAdd::new(&kernel, block).unwrap();
    let mut x = vec![0.0f64; 32];
    x[0] = 1.0;
    let got = stream(&mut |i, o| conv.process(i, o), &x, block);
    for (i, k) in kernel.iter().enumerate() {
        assert!((got[i] - k).abs() < 1e-12, "tap {i}: {} vs {k}", got[i]);
    }
    for v in got.iter().skip(kernel.len()) {
        assert!(v.abs() < 1e-12);
    }
}

#[test]
fn overlap_add_matches_direct() {
    let kernel: Vec<f64> = (0..11i32).map(|i| 0.9f64.powi(i)).collect();
    let x: Vec<f64> = (0..128)
        .map(|i| ((i as f64) * 0.19).sin() * (1.0 - i as f64 / 256.0))
        .collect();
    let expected = direct(&x, &kernel);
    let mut conv = OverlapAdd::new(&kernel, 24).unwrap();
    let got = stream(&mut |i, o| conv.process(i, o), &x, 24);
    for (a, b) in expected.iter().zip(got.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn partitioned_impulse_is_kernel_passthrough() {
    let kernel: Vec<f64> = [0.4, 0.3, 0.2, 0.1].to_vec();
    let part = 8;
    let mut conv = PartitionedConvolver::new(&kernel, part).unwrap();
    let mut x = vec![0.0f64; 32];
    x[0] = 1.0;
    let got = stream(&mut |i, o| conv.process(i, o), &x, part);
    for (i, k) in kernel.iter().enumerate() {
        assert!((got[i] - k).abs() < 1e-12, "tap {i}: {} vs {k}", got[i]);
    }
}

#[test]
fn partitioned_matches_direct_over_long_stream() {
    let kernel: Vec<f64> = (0..37)
        .map(|i| ((i as f64 + 1.0) * 0.113).sin() * 0.4)
        .collect();
    let x: Vec<f64> = (0..512).map(|i| ((i as f64) * 0.071).cos() * 0.8).collect();
    let expected = direct(&x, &kernel);
    let mut conv = PartitionedConvolver::new(&kernel, 16).unwrap();
    let got = stream(&mut |i, o| conv.process(i, o), &x, 16);
    let max_err = expected
        .iter()
        .zip(got.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(max_err < 1e-12, "max error {max_err}");
}

#[test]
fn block_size_choices_all_agree() {
    // The same kernel through different block sizes gives identical output.
    let kernel: Vec<f64> = [1.0, -0.5, 0.25, -0.125, 0.0625].to_vec();
    let x: Vec<f64> = (0..96).map(|i| ((i as f64) * 0.23).sin()).collect();
    let expected = direct(&x, &kernel);
    for block in [1usize, 4, 16, 64] {
        let mut conv = OverlapAdd::new(&kernel, block).unwrap();
        let got = stream(&mut |i, o| conv.process(i, o), &x, block);
        for (a, b) in expected.iter().zip(got.iter()) {
            assert!((a - b).abs() < 1e-12, "block {block}");
        }
    }
}
