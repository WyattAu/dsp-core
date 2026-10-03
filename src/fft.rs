//! FFT — radix-2 decimation-in-time, pre-computed twiddle factors, zero
//! allocation on the transform path.
//!
//! [`Fft`] is a stateless transform object: `new(size)` builds the twiddle
//! table (size/2 complex factors, interleaved re/im in a flat `Box<[f64]>`)
//! and the bit-reversal permutation table; every later call transforms a
//! caller-owned interleaved buffer in place (`buffer[2k]` = real part,
//! `buffer[2k + 1]` = imaginary part of bin k) without touching the
//! allocator. Forward scales by 1; [`Fft::inverse`] applies the 1/N
//! normalization, so `inverse(forward(x)) == x`.
//!
//! [`RealFft`] exploits the real-input symmetry to run an N-point real
//! transform through an N/2-point complex FFT (Sorensen-style split): the
//! N real samples are reinterpreted in place as N/2 interleaved complex
//! values, and the N/2+1 useful output bins are written to a separate
//! spectrum buffer. The spectrum is `N + 2` f64 values laid out as a plain
//! interleaved complex array so frequency-domain multiplication needs no
//! special cases: slot 0 holds `(DC, 0)`, slots `2..N` hold bins
//! `1..N/2 − 1` as ordinary complex values, and slot `N` holds
//! `(Nyquist, 0)` — DC and Nyquist are real, their imaginary halves are
//! zero. The transform is exactly inverted by [`RealFft::inverse`].
//!
//! # Accuracy
//!
//! With f64 accumulation and full-precision twiddles the round-trip and
//! direct-DFT comparison errors are at the 1e-13 level for N = 64 (the
//! test-suite gate is 1e-10).
//!
//! # Example
//!
//! ```
//! use dsp_core::fft::Fft;
//!
//! let fft = Fft::new(8).expect("power of two");
//! let mut buf = vec![0.0; 16]; // 8 interleaved complex points
//! buf[0] = 1.0; // impulse
//! fft.forward(&mut buf).expect("length matches");
//! // An impulse transforms to a flat spectrum.
//! assert!(buf.chunks(2).all(|c| (c[0] - 1.0).abs() < 1e-12 && c[1].abs() < 1e-12));
//! ```

use crate::error::DspError;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::f64::consts::TAU;
use libm::{cos, sin};

/// Largest supported transform: 2²⁰ complex points (twiddle table + bit-rev
/// table ≈ 12 MiB), far beyond any audio block size.
const MAX_FFT_SIZE: usize = 1 << 20;
/// Smallest supported complex transform size.
const MIN_FFT_SIZE: usize = 2;

fn validate_fft_size(size: usize, min: usize) -> Result<(), DspError> {
    if size < min {
        return Err(DspError::FftTooSmall { size, min });
    }
    if size > MAX_FFT_SIZE {
        return Err(DspError::FftTooLarge {
            size,
            max: MAX_FFT_SIZE,
        });
    }
    if !size.is_power_of_two() {
        return Err(DspError::NotPowerOfTwo { size });
    }
    Ok(())
}

/// Radix-2 DIT complex FFT with pre-computed tables. Stateless and
/// allocation-free after construction; `forward`/`inverse` take `&self`.
pub struct Fft {
    size: usize,
    /// Twiddle factors e^{−2πik/size}, k in 0..size/2, interleaved re/im.
    twiddles: Box<[f64]>,
    /// Bit-reversal permutation of 0..size.
    bit_rev: Box<[u32]>,
}

impl core::fmt::Debug for Fft {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Fft").field("size", &self.size).finish()
    }
}

impl Fft {
    /// Build tables for a power-of-two transform. `size` is in *complex
    /// points*; the interleaved buffer it transforms is `2·size` f64 values.
    ///
    /// # Errors
    ///
    /// Returns [`DspError::NotPowerOfTwo`], [`DspError::FftTooLarge`], or [`DspError::FftTooSmall`] when `size` is not a supported power of two.
    pub fn new(size: usize) -> Result<Self, DspError> {
        validate_fft_size(size, MIN_FFT_SIZE)?;
        let half = size / 2;
        let twiddles: Vec<f64> = (0..half)
            .flat_map(|k| {
                let angle = TAU * (k as f64) / (size as f64);
                [cos(angle), -sin(angle)]
            })
            .collect();
        let bits = size.trailing_zeros();
        let bit_rev: Vec<u32> = (0..size as u32)
            .map(|i| i.reverse_bits() >> (32 - bits))
            .collect();
        Ok(Self {
            size,
            twiddles: twiddles.into_boxed_slice(),
            bit_rev: bit_rev.into_boxed_slice(),
        })
    }

    /// Complex points per transform (the interleaved buffer is twice this).
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Required interleaved buffer length in f64 values (`2 · size`).
    #[must_use]
    pub fn buffer_len(&self) -> usize {
        self.size * 2
    }

    /// In-place forward FFT over an interleaved complex buffer. Unit gain.
    ///
    /// # Errors
    ///
    /// Returns [`DspError::InvalidLength`] when the buffer is not exactly [`Fft::buffer_len`] values.
    pub fn forward(&self, buffer: &mut [f64]) -> Result<(), DspError> {
        if buffer.len() != self.buffer_len() {
            return Err(DspError::InvalidLength {
                expected: self.buffer_len(),
                got: buffer.len(),
            });
        }
        self.transform(buffer, false);
        Ok(())
    }

    /// In-place inverse FFT over an interleaved complex buffer, scaled by
    /// 1/N so that `inverse(forward(x)) == x`.
    ///
    /// # Errors
    ///
    /// Returns [`DspError::InvalidLength`] when the buffer is not exactly [`Fft::buffer_len`] values.
    pub fn inverse(&self, buffer: &mut [f64]) -> Result<(), DspError> {
        if buffer.len() != self.buffer_len() {
            return Err(DspError::InvalidLength {
                expected: self.buffer_len(),
                got: buffer.len(),
            });
        }
        self.transform(buffer, true);
        let scale = 1.0 / (self.size as f64);
        for value in buffer.iter_mut() {
            *value *= scale;
        }
        Ok(())
    }

    /// Bit-reverse permutation followed by the log2(N) butterfly stages.
    fn transform(&self, buffer: &mut [f64], invert: bool) {
        self.permute_bit_reversed(buffer);
        let n = self.size;
        let mut half = 1; // half-block width in complex points
        while half < n {
            let block_f64 = half * 4; // one butterfly block, in f64 values
            let stride = n / (half * 2); // twiddle table step for this stage
            let mut rest: &mut [f64] = buffer;
            while !rest.is_empty() {
                let (block, tail) = rest.split_at_mut(block_f64);
                rest = tail;
                let (lower, upper) = block.split_at_mut(half * 2);
                for ((a, b), w) in lower
                    .chunks_exact_mut(2)
                    .zip(upper.chunks_exact_mut(2))
                    .zip(self.twiddles.chunks_exact(2).step_by(stride))
                {
                    if let ([a_re, a_im], [b_re, b_im], [w_re, w_im]) = (a, b, w) {
                        // Conjugate the twiddle for the inverse transform.
                        let w_im = if invert { -*w_im } else { *w_im };
                        let (ar, ai) = (*a_re, *a_im);
                        let t_re = *b_re * *w_re - *b_im * w_im;
                        let t_im = *b_re * w_im + *b_im * *w_re;
                        *a_re = ar + t_re;
                        *a_im = ai + t_im;
                        *b_re = ar - t_re;
                        *b_im = ai - t_im;
                    }
                }
            }
            half *= 2;
        }
    }

    /// Swap each complex point with its bit-reversed partner (once).
    fn permute_bit_reversed(&self, buffer: &mut [f64]) {
        for (i, &r) in self.bit_rev.iter().enumerate() {
            let r = r as usize;
            if r > i {
                // Point i is entirely inside `left`; point r starts `right`.
                let (left, right) = buffer.split_at_mut(r * 2);
                let (_, pair) = left.split_at_mut(i * 2);
                let (l0, pair_rest) = pair.split_at_mut(1);
                let (l1, _) = pair_rest.split_at_mut(1);
                let (r0, right_rest) = right.split_at_mut(1);
                let (r1, _) = right_rest.split_at_mut(1);
                if let (Some(l0), Some(l1), Some(r0), Some(r1)) = (
                    l0.first_mut(),
                    l1.first_mut(),
                    r0.first_mut(),
                    r1.first_mut(),
                ) {
                    core::mem::swap(l0, r0);
                    core::mem::swap(l1, r1);
                }
            }
        }
    }
}

/// Real-input FFT of N samples via an N/2-point complex FFT (Sorensen
/// split), with the hermitian-symmetric upper half folded away.
///
/// The spectrum layout for `forward` is `N + 2` f64 values (`N/2 + 1`
/// interleaved complex bins): bin 0 packs `(X[0].re, X[N/2].re)` — DC and
/// Nyquist are real — bins 1..N/2 are the ordinary complex bins, and the
/// final slot is zeroed padding. `inverse` inverts exactly, including the
/// 1/N normalization.
pub struct RealFft {
    size: usize,
    fft: Fft,
    /// Half-length post-twiddles e^{−2πik/N}, k in 0..N/2, interleaved re/im.
    post: Box<[f64]>,
}

impl core::fmt::Debug for RealFft {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RealFft").field("size", &self.size).finish()
    }
}

impl RealFft {
    /// Build tables for an N-point real transform; N must be a power of two
    /// ≥ 4 (the internal complex FFT runs at N/2 points).
    ///
    /// # Errors
    ///
    /// Returns [`DspError::NotPowerOfTwo`], [`DspError::FftTooLarge`], or [`DspError::FftTooSmall`] when `size` is not a supported power of two (minimum 4).
    pub fn new(size: usize) -> Result<Self, DspError> {
        validate_fft_size(size, 4)?;
        let fft = Fft::new(size / 2)?;
        let half = size / 2;
        let post: Vec<f64> = (0..half)
            .flat_map(|k| {
                let angle = TAU * (k as f64) / (size as f64);
                [cos(angle), -sin(angle)]
            })
            .collect();
        Ok(Self {
            size,
            fft,
            post: post.into_boxed_slice(),
        })
    }

    /// Real samples per transform.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Required real-buffer length (`size`).
    #[must_use]
    pub fn real_len(&self) -> usize {
        self.size
    }

    /// Required spectrum-buffer length (`size + 2`).
    #[must_use]
    pub fn spectrum_len(&self) -> usize {
        self.size + 2
    }

    /// Forward transform: `real` (N samples) is consumed as scratch and
    /// left in an unspecified intermediate state; `spectrum` receives the
    /// N/2 + 1 bins as a plain interleaved complex array with `(DC, 0)` in
    /// slot 0 and `(Nyquist, 0)` in slot N (layout in the module docs).
    ///
    /// # Errors
    ///
    /// Returns [`DspError::InvalidLength`] when either buffer length is wrong (see [`RealFft::real_len`] / [`RealFft::spectrum_len`]).
    pub fn forward(&self, real: &mut [f64], spectrum: &mut [f64]) -> Result<(), DspError> {
        if real.len() != self.real_len() {
            return Err(DspError::InvalidLength {
                expected: self.real_len(),
                got: real.len(),
            });
        }
        if spectrum.len() != self.spectrum_len() {
            return Err(DspError::InvalidLength {
                expected: self.spectrum_len(),
                got: spectrum.len(),
            });
        }
        // The even/odd interleave of the real signal is exactly the
        // interleaved complex layout — no data movement needed.
        self.fft.forward(real)?;
        self.untangle_forward(real, spectrum);
        Ok(())
    }

    /// Inverse transform: `spectrum` (N + 2 packed bins) is consumed as
    /// scratch; `real` receives the N time-domain samples, scaled so that
    /// `inverse(forward(x)) == x`.
    ///
    /// # Errors
    ///
    /// Returns [`DspError::InvalidLength`] when either buffer length is wrong (see [`RealFft::real_len`] / [`RealFft::spectrum_len`]).
    pub fn inverse(&self, spectrum: &mut [f64], real: &mut [f64]) -> Result<(), DspError> {
        if spectrum.len() != self.spectrum_len() {
            return Err(DspError::InvalidLength {
                expected: self.spectrum_len(),
                got: spectrum.len(),
            });
        }
        if real.len() != self.real_len() {
            return Err(DspError::InvalidLength {
                expected: self.real_len(),
                got: real.len(),
            });
        }
        self.retangle_inverse(spectrum, real);
        self.fft.inverse(real)?;
        Ok(())
    }

    /// Unpack the N/2-point complex spectrum `z` into the hermitian
    /// half-spectrum: `X[k] = Ze − i·W_k·Zo` with
    /// `Ze = (Z[k] + conj(Z[N/2−k]))/2`, `Zo = (Z[k] − conj(Z[N/2−k]))/2`.
    fn untangle_forward(&self, z: &[f64], spectrum: &mut [f64]) {
        let half = self.size / 2;
        let z0_re = z.first().copied().unwrap_or(0.0);
        let z0_im = z.get(1).copied().unwrap_or(0.0);
        let mut out = spectrum.chunks_exact_mut(2);
        // Bin 0 is DC (real); the Nyquist bin is stored in the trailing
        // slot so the whole spectrum is a plain complex array.
        if let Some([dc_re, dc_im]) = out.next() {
            *dc_re = z0_re + z0_im;
            *dc_im = 0.0;
        }
        // Bins 1..half−1. NOTE: `take` caps the chunk side to exactly the
        // range length — a bare `zip` pulls (and silently drops) one extra
        // chunk on its final step when the range runs out first, which
        // would eat the trailing Nyquist slot.
        for (bin, k) in out.by_ref().take(half - 1).zip(1..half) {
            if let [b_re, b_im] = bin {
                let zk_re = z.get(2 * k).copied().unwrap_or(0.0);
                let zk_im = z.get(2 * k + 1).copied().unwrap_or(0.0);
                let zm_re = z.get(2 * (half - k)).copied().unwrap_or(0.0);
                let zm_im = z.get(2 * (half - k) + 1).copied().unwrap_or(0.0);
                // conj(Z[half − k])
                let (c_re, c_im) = (zm_re, -zm_im);
                let ze_re = (zk_re + c_re) / 2.0;
                let ze_im = (zk_im + c_im) / 2.0;
                let zo_re = (zk_re - c_re) / 2.0;
                let zo_im = (zk_im - c_im) / 2.0;
                // W_k = e^{−2πik/N}; t = W·Zo; the −i rotation maps
                // (t_re, t_im) → (t_im, −t_re).
                let w_re = self.post.get(2 * k).copied().unwrap_or(1.0);
                let w_im = self.post.get(2 * k + 1).copied().unwrap_or(0.0);
                let t_re = w_re * zo_re - w_im * zo_im;
                let t_im = w_re * zo_im + w_im * zo_re;
                *b_re = ze_re + t_im;
                *b_im = ze_im - t_re;
            }
        }
        // Trailing slot carries the real Nyquist bin + a zero pad.
        if let Some([nyq_re, nyq_im]) = out.next() {
            *nyq_re = z0_re - z0_im;
            *nyq_im = 0.0;
        }
    }

    /// Rebuild the N/2-point complex spectrum from the packed half-spectrum:
    /// `Z[k] = (X[k] + conj(X[N/2−k]))/2 + (i/2)·e^{+2πik/N}·(X[k] −
    /// conj(X[N/2−k]))`.
    fn retangle_inverse(&self, spectrum: &[f64], z: &mut [f64]) {
        let half = self.size / 2;
        let x0 = spectrum.first().copied().unwrap_or(0.0);
        // Nyquist lives in the trailing slot.
        let x_nyq = spectrum.get(2 * half).copied().unwrap_or(0.0);
        let mut out = z.chunks_exact_mut(2);
        if let Some([z0_re, z0_im]) = out.next() {
            *z0_re = (x0 + x_nyq) / 2.0;
            *z0_im = (x0 - x_nyq) / 2.0;
        }
        for (bin, k) in out.take(half - 1).zip(1..half) {
            if let [z_re, z_im] = bin {
                let a_re = spectrum.get(2 * k).copied().unwrap_or(0.0);
                let a_im = spectrum.get(2 * k + 1).copied().unwrap_or(0.0);
                // conj(X[half − k])
                let b_re = spectrum.get(2 * (half - k)).copied().unwrap_or(0.0);
                let b_im = -spectrum.get(2 * (half - k) + 1).copied().unwrap_or(0.0);
                let p_re = (a_re + b_re) / 2.0;
                let p_im = (a_im + b_im) / 2.0;
                let d_re = a_re - b_re;
                let d_im = a_im - b_im;
                // (i/2)·conj(W_k)·D with conj(W_k) = (cos, +sin).
                let w_re = self.post.get(2 * k).copied().unwrap_or(1.0);
                let w_im = -self.post.get(2 * k + 1).copied().unwrap_or(0.0);
                let m_re = w_re * d_re - w_im * d_im;
                let m_im = w_re * d_im + w_im * d_re;
                // multiply by i/2: (m_re, m_im) → (−m_im, m_re)/2
                *z_re = p_re - m_im / 2.0;
                *z_im = p_im + m_re / 2.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    // Reference-DFT tests over fixed-size buffers; direct indexing keeps the
    // textbook formulas readable.

    use super::*;
    use alloc::{format, vec, vec::Vec};

    /// Naive O(N²) DFT reference: X[k] = Σ x[n]·e^{−2πink/N}.
    fn dft_reference(x: &[f64]) -> Vec<(f64, f64)> {
        let n = x.len();
        (0..n)
            .map(|k| {
                let (mut re, mut im) = (0.0, 0.0);
                for (j, &v) in x.iter().enumerate() {
                    let angle = TAU * (k * j) as f64 / (n as f64);
                    re += v * cos(angle);
                    im -= v * sin(angle);
                }
                (re, im)
            })
            .collect()
    }

    fn complex_buf(x: &[f64]) -> Vec<f64> {
        // All-real complex buffer (imag parts zero).
        x.iter().flat_map(|&v| [v, 0.0]).collect()
    }

    #[test]
    fn forward_matches_dft_reference_n64() {
        let n = 64;
        let fft = Fft::new(n).unwrap();
        let signal: Vec<f64> = (0..n)
            .map(|i| (0.35 * i as f64).sin() + 0.2 * (1.7 * i as f64).cos())
            .collect();
        let mut buf = complex_buf(&signal);
        fft.forward(&mut buf).unwrap();
        let reference = dft_reference(&signal);
        let mut max_err = 0.0f64;
        for (k, (re, im)) in reference.iter().enumerate() {
            max_err = max_err.max((buf[2 * k] - re).abs());
            max_err = max_err.max((buf[2 * k + 1] - im).abs());
        }
        assert!(max_err < 1e-10, "FFT vs DFT error {max_err}");
    }

    #[test]
    fn impulse_gives_flat_spectrum() {
        for n in [2, 8, 64, 256] {
            let fft = Fft::new(n).unwrap();
            let mut buf = vec![0.0; 2 * n];
            buf[0] = 1.0;
            fft.forward(&mut buf).unwrap();
            for c in buf.chunks(2) {
                assert!((c[0] - 1.0).abs() < 1e-12 && c[1].abs() < 1e-12);
            }
        }
    }

    #[test]
    fn sine_peaks_at_correct_bin() {
        let n = 256;
        let bin = 17;
        let fft = Fft::new(n).unwrap();
        let signal: Vec<f64> = (0..n)
            .map(|i| (TAU * (bin * i) as f64 / n as f64).sin())
            .collect();
        let mut buf = complex_buf(&signal);
        fft.forward(&mut buf).unwrap();
        for (k, c) in buf.chunks(2).enumerate() {
            let mag = libm::sqrt(c[0] * c[0] + c[1] * c[1]);
            if k == bin || k == n - bin {
                assert!((mag - n as f64 / 2.0).abs() < 1e-9, "bin {k}");
            } else {
                assert!(mag < 1e-9, "bin {k} should be empty, got {mag}");
            }
        }
    }

    #[test]
    fn round_trip_is_identity() {
        for n in [2, 8, 64, 1024] {
            let fft = Fft::new(n).unwrap();
            let original: Vec<f64> = (0..2 * n)
                .map(|i| ((7 * i) as f64 * 0.13).sin() * 0.5 + 0.1)
                .collect();
            let mut buf = original.clone();
            fft.forward(&mut buf).unwrap();
            fft.inverse(&mut buf).unwrap();
            for (a, b) in original.iter().zip(buf.iter()) {
                assert!((a - b).abs() < 1e-12, "n={n}");
            }
        }
    }

    #[test]
    fn parseval_theorem_holds() {
        let n = 64;
        let fft = Fft::new(n).unwrap();
        let signal: Vec<f64> = (0..n).map(|i| (0.21 * i as f64).sin()).collect();
        let time_energy: f64 = signal.iter().map(|v| v * v).sum();
        let mut buf = complex_buf(&signal);
        fft.forward(&mut buf).unwrap();
        let freq_energy: f64 = buf.chunks(2).map(|c| c[0] * c[0] + c[1] * c[1]).sum();
        assert!(
            (time_energy - freq_energy / n as f64).abs() < 1e-9,
            "time {time_energy} vs freq {freq_energy}"
        );
    }

    #[test]
    fn inverse_scales_by_one_over_n() {
        let n = 16;
        let fft = Fft::new(n).unwrap();
        let mut buf = complex_buf(&[3.0; 16]);
        fft.forward(&mut buf).unwrap();
        fft.inverse(&mut buf).unwrap();
        for c in buf.chunks(2) {
            assert!((c[0] - 3.0).abs() < 1e-12);
            assert!(c[1].abs() < 1e-12);
        }
    }

    #[test]
    fn fft_size_validation() {
        assert!(matches!(
            Fft::new(0),
            Err(DspError::FftTooSmall {
                size: 0,
                min: MIN_FFT_SIZE
            })
        ));
        assert!(matches!(
            Fft::new(1),
            Err(DspError::FftTooSmall { size: 1, min: 2 })
        ));
        assert!(matches!(
            Fft::new(100),
            Err(DspError::NotPowerOfTwo { size: 100 })
        ));
        assert!(matches!(
            Fft::new(MAX_FFT_SIZE * 2),
            Err(DspError::FftTooLarge { size, max })
                if size == MAX_FFT_SIZE * 2 && max == MAX_FFT_SIZE
        ));
        assert!(Fft::new(MAX_FFT_SIZE).is_ok());
    }

    #[test]
    fn fft_buffer_length_validation() {
        let fft = Fft::new(8).unwrap();
        assert_eq!(fft.size(), 8);
        assert_eq!(fft.buffer_len(), 16);
        let mut short = vec![0.0; 15];
        assert_eq!(
            fft.forward(&mut short),
            Err(DspError::InvalidLength {
                expected: 16,
                got: 15
            })
        );
        assert_eq!(
            fft.inverse(&mut short),
            Err(DspError::InvalidLength {
                expected: 16,
                got: 15
            })
        );
    }

    #[test]
    fn debug_format_hides_tables() {
        let s = format!("{:?}", Fft::new(8).unwrap());
        assert!(s.contains("size: 8"));
        assert!(!s.contains('['), "tables must not be dumped: {s}");
    }

    #[test]
    fn real_fft_round_trip() {
        let n = 128;
        let rfft = RealFft::new(n).unwrap();
        let original: Vec<f64> = (0..n)
            .map(|i| (0.11 * i as f64).sin() + 0.3 * (1.03 * i as f64).cos())
            .collect();
        let mut real = original.clone();
        let mut spectrum = vec![0.0; n + 2];
        rfft.forward(&mut real, &mut spectrum).unwrap();
        let mut out = vec![0.0; n];
        rfft.inverse(&mut spectrum, &mut out).unwrap();
        for (a, b) in original.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn real_fft_matches_complex_fft_of_real_input() {
        let n = 64;
        let rfft = RealFft::new(n).unwrap();
        let signal: Vec<f64> = (0..n)
            .map(|i| (TAU * 5.0 * i as f64 / n as f64).sin() * 0.5 + 0.25)
            .collect();
        let reference = dft_reference(&signal);
        let mut real = signal.clone();
        let mut spectrum = vec![0.0; n + 2];
        rfft.forward(&mut real, &mut spectrum).unwrap();
        // DC in slot 0 (imag half zero), Nyquist in the trailing slot.
        assert!((spectrum[0] - reference[0].0).abs() < 1e-10);
        assert!(spectrum[1].abs() < 1e-10);
        assert!((spectrum[n] - reference[n / 2].0).abs() < 1e-10);
        for k in 1..n / 2 {
            assert!(
                (spectrum[2 * k] - reference[k].0).abs() < 1e-10,
                "bin {k} re"
            );
            assert!(
                (spectrum[2 * k + 1] - reference[k].1).abs() < 1e-10,
                "bin {k} im"
            );
        }
        // Trailing pad half is zeroed.
        assert_eq!(spectrum[n + 1], 0.0);
    }

    #[test]
    fn real_fft_impulse_spectrum_is_flat() {
        let n = 32;
        let rfft = RealFft::new(n).unwrap();
        let mut real = vec![0.0; n];
        real[0] = 1.0;
        let mut spectrum = vec![0.0; n + 2];
        rfft.forward(&mut real, &mut spectrum).unwrap();
        // DC in slot 0 (imag half zero), Nyquist in slot n.
        assert!((spectrum[0] - 1.0).abs() < 1e-12);
        assert!(spectrum[1].abs() < 1e-12);
        assert!((spectrum[n] - 1.0).abs() < 1e-12);
        assert!(spectrum[n + 1].abs() < 1e-12);
        for k in 1..n / 2 {
            assert!((spectrum[2 * k] - 1.0).abs() < 1e-12 && spectrum[2 * k + 1].abs() < 1e-12);
        }
    }

    #[test]
    fn real_fft_size_and_length_validation() {
        assert!(matches!(
            RealFft::new(2),
            Err(DspError::FftTooSmall { size: 2, min: 4 })
        ));
        assert!(matches!(
            RealFft::new(48),
            Err(DspError::NotPowerOfTwo { size: 48 })
        ));
        let rfft = RealFft::new(16).unwrap();
        assert_eq!(rfft.size(), 16);
        assert_eq!(rfft.real_len(), 16);
        assert_eq!(rfft.spectrum_len(), 18);
        let mut real = vec![0.0; 15];
        let mut spectrum = vec![0.0; 18];
        assert!(matches!(
            rfft.forward(&mut real, &mut spectrum),
            Err(DspError::InvalidLength { .. })
        ));
        let mut real = vec![0.0; 16];
        let mut spectrum = vec![0.0; 17];
        assert!(matches!(
            rfft.forward(&mut real, &mut spectrum),
            Err(DspError::InvalidLength { .. })
        ));
        assert!(matches!(
            rfft.inverse(&mut spectrum, &mut real),
            Err(DspError::InvalidLength { .. })
        ));
        let mut spectrum = vec![0.0; 18];
        let mut real = vec![0.0; 17];
        assert!(matches!(
            rfft.inverse(&mut spectrum, &mut real),
            Err(DspError::InvalidLength { .. })
        ));
        assert!(rfft.forward(&mut [0.0; 16], &mut [0.0; 18]).is_ok());
        assert!(rfft.inverse(&mut [0.0; 18], &mut [0.0; 16]).is_ok());
    }

    #[test]
    fn real_fft_debug_format() {
        let s = format!("{:?}", RealFft::new(16).unwrap());
        assert!(s.contains("size: 16"));
    }
}
