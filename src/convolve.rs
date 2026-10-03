//! Convolution — overlap-add block convolution and uniformly partitioned
//! low-latency FFT convolution.
//!
//! Both convolvers are built at construction time (kernel spectra, FFT
//! tables, ring buffers) and their `process` paths never allocate. Blocks
//! are consumed through caller-owned scratch buffers, so the only per-block
//! cost is arithmetic.
//!
//! * [`OverlapAdd`] — classic overlap-add: an arbitrary block size B against
//!   an arbitrary kernel length K, FFT size `next_pow2(B + K − 1)`. Output
//!   block *m* aligns with input block *m*.
//! * [`PartitionedConvolver`] — uniformly partitioned convolution
//!   (Pahlmann/Bruusgaard-style UPOLS): the impulse response is chopped into
//!   `part_len`-sample partitions transformed once up front, and the input
//!   spectra are kept in a frequency-domain delay line. Cost is
//!   O(partitions) FFT-multiply-accumulates per `part_len` samples
//!   regardless of kernel length, with one block (`part_len` samples) of
//!   buffering latency.
//!
//! # Example
//!
//! ```
//! use dsp_core::convolve::OverlapAdd;
//!
//! // 4-sample moving average.
//! let kernel = [0.25f64; 4];
//! let mut conv = OverlapAdd::new(&kernel, 8).expect("valid block size");
//!
//! let mut block = [0.0f64; 8];
//! block[0] = 1.0; // impulse
//! let mut out = [0.0f64; 8];
//! conv.process(&mut block, &mut out).expect("lengths match");
//! assert!((out[0] - 0.25).abs() < 1e-12);
//! ```

use crate::error::DspError;
use crate::fft::RealFft;
use crate::math::flush_denormal;
use alloc::boxed::Box;
use alloc::vec;

fn validate_kernel(kernel: &[f64]) -> Result<usize, DspError> {
    if kernel.is_empty() {
        return Err(DspError::InvalidParameter {
            name: "kernel",
            reason: "must not be empty",
        });
    }
    if kernel.iter().any(|v| !v.is_finite()) {
        return Err(DspError::InvalidParameter {
            name: "kernel",
            reason: "must contain only finite values",
        });
    }
    Ok(kernel.len())
}

fn checked_partitions(kernel_len: usize, part_len: usize) -> Result<usize, DspError> {
    if kernel_len.div_ceil(part_len) == 0 {
        return Err(DspError::InvalidParameter {
            name: "kernel",
            reason: "kernel/partition size overflow",
        });
    }
    Ok(kernel_len.div_ceil(part_len))
}

/// Classic overlap-add streaming convolver.
///
/// Construction is O(K log K); [`OverlapAdd::process`] runs two FFTs and a
/// spectrum multiply per block and never allocates. The convolver emits the
/// output aligned with each input block (no algorithmic delay), but a full
/// block must be buffered first in any real-time chain.
pub struct OverlapAdd {
    block: usize,
    kernel_len: usize,
    fft_size: usize,
    fft: RealFft,
    /// Padded kernel spectrum, `fft_size + 2` values.
    kernel_spec: Box<[f64]>,
    /// Per-block FFT scratch, `fft_size` values.
    work: Box<[f64]>,
    /// Per-block spectrum scratch, `fft_size + 2` values.
    spec: Box<[f64]>,
    /// Overlap tail carried into the next block, `fft_size − block` values.
    tail: Box<[f64]>,
}

impl core::fmt::Debug for OverlapAdd {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OverlapAdd")
            .field("block", &self.block)
            .field("kernel_len", &self.kernel_len)
            .field("fft_size", &self.fft_size)
            .finish()
    }
}

impl OverlapAdd {
    /// Build a convolver for `kernel` processed in `block`-sample blocks.
    /// The FFT size is `next_pow2(block + kernel_len − 1)`.
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when the kernel is empty/non-finite or the block/partition size is not usable as an FFT size.
    pub fn new(kernel: &[f64], block: usize) -> Result<Self, DspError> {
        let kernel_len = validate_kernel(kernel)?;
        if block == 0 {
            return Err(DspError::InvalidParameter {
                name: "block",
                reason: "must be > 0",
            });
        }
        let total = block
            .checked_add(kernel_len)
            .and_then(|sum| sum.checked_sub(1))
            .ok_or(DspError::InvalidParameter {
                name: "block",
                reason: "block/kernel size overflow",
            })?;
        // Round up to a legal real-FFT size (>= 4).
        let fft_size = total.next_power_of_two().max(4);
        let spectrum_len = fft_size + 2;
        let fft = RealFft::new(fft_size)?;
        // Transform the zero-padded kernel once.
        let mut padded = vec![0.0f64; fft_size];
        for (dst, src) in padded.iter_mut().zip(kernel.iter()) {
            *dst = *src;
        }
        let mut kernel_spec = vec![0.0f64; spectrum_len];
        fft.forward(&mut padded, &mut kernel_spec)?;
        Ok(Self {
            block,
            kernel_len,
            fft_size,
            fft,
            kernel_spec: kernel_spec.into_boxed_slice(),
            work: vec![0.0; fft_size].into_boxed_slice(),
            spec: vec![0.0; spectrum_len].into_boxed_slice(),
            tail: vec![0.0; fft_size - block].into_boxed_slice(),
        })
    }

    /// Block size in samples.
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.block
    }

    /// Kernel length in samples.
    #[must_use]
    pub fn kernel_len(&self) -> usize {
        self.kernel_len
    }

    /// Internal FFT size in samples.
    #[must_use]
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Reset the overlap tail (drop any in-flight convolution state).
    pub fn reset(&mut self) {
        self.tail.fill(0.0);
    }

    /// Convolve one block: `input` (`block_size` samples, read-only) →
    /// `output` (`block_size` samples).
    ///
    /// # Errors
    ///
    /// Returns [`DspError::InvalidLength`] when either slice is not exactly [`OverlapAdd::block_size`] samples.
    pub fn process(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), DspError> {
        if input.len() != self.block {
            return Err(DspError::InvalidLength {
                expected: self.block,
                got: input.len(),
            });
        }
        if output.len() != self.block {
            return Err(DspError::InvalidLength {
                expected: self.block,
                got: output.len(),
            });
        }
        // Zero-pad the block into the FFT buffer.
        {
            let (head, rest) = self.work.split_at_mut(self.block);
            head.copy_from_slice(input);
            rest.fill(0.0);
        }
        self.fft.forward(&mut self.work, &mut self.spec)?;
        // Spectrum multiply against the (conjugate-symmetric) kernel, in
        // place.
        for (a, k) in self
            .spec
            .chunks_exact_mut(2)
            .zip(self.kernel_spec.chunks(2))
        {
            if let ([a_re, a_im], [k_re, k_im]) = (a, k) {
                let re = *a_re * *k_re - *a_im * *k_im;
                let im = *a_re * *k_im + *a_im * *k_re;
                *a_re = re;
                *a_im = im;
            }
        }
        self.fft.inverse(&mut self.spec, &mut self.work)?;
        // Add the carried tail, emit `block` samples, then fold this block's
        // tail region into the pending contributions. The tail is a running
        // sum: each block's `r[B..]` adds to (not replaces) whatever older
        // blocks still owe beyond this output block.
        let (head, rest) = self.work.split_at(self.block);
        // `take` caps the pair side to exactly the tail length — a bare
        // `zip` against an empty/short tail would pull and drop one pair.
        let mut pairs = output.iter_mut().zip(head.iter());
        for ((out, value), old) in pairs.by_ref().take(self.tail.len()).zip(self.tail.iter()) {
            *out = flush_denormal(value + old);
        }
        for (out, value) in pairs {
            *out = flush_denormal(*value);
        }
        if self.block < self.tail.len() {
            // Shift the still-pending region (the part not consumed by this
            // block's output) to the front and clear the stale back half
            // before accumulating this block's contribution.
            self.tail.copy_within(self.block.., 0);
            let pending = self.tail.len() - self.block;
            let (_, stale) = self.tail.split_at_mut(pending);
            stale.fill(0.0);
        } else {
            // Everything pending was consumed by this block's output.
            self.tail.fill(0.0);
        }
        for (t, r) in self.tail.iter_mut().zip(rest.iter()) {
            *t = flush_denormal(*t + r);
        }
        Ok(())
    }
}

/// Uniformly partitioned FFT convolver (UPOLS) — low-latency convolution
/// against long impulse responses.
///
/// The kernel is split into `part_len`-sample partitions, each transformed
/// once at construction. Each `process` call consumes one `part_len` block:
/// it is transformed, pushed onto a frequency-domain delay line, and
/// multiply-accumulated against every kernel partition. The IFFT of the
/// accumulator produces a partial block whose head overlaps the previous
/// block's tail; adding them yields the output. Cost per block is constant
/// (2 FFTs of size `2·part_len` + `partitions` spectrum MACs) no matter how
/// long the kernel is.
///
/// The algorithmic output for block *m* aligns with input block *m*; a full
/// block of buffering is inherent, so treat [`PartitionedConvolver::latency`]
/// as the effective delay in a real-time chain.
pub struct PartitionedConvolver {
    part_len: usize,
    fft_size: usize,
    partitions: usize,
    fft: RealFft,
    /// Kernel partition spectra, `partitions × (fft_size + 2)` values.
    kernel_freq: Box<[f64]>,
    /// Frequency-domain input delay line, `partitions × (fft_size + 2)`.
    input_freq: Box<[f64]>,
    /// Index of the slot the current block will occupy.
    hist_pos: usize,
    /// Spectrum scratch, `fft_size + 2` values.
    spec: Box<[f64]>,
    /// Accumulator scratch, `fft_size + 2` values.
    acc: Box<[f64]>,
    /// Time-domain scratch, `fft_size` values.
    work: Box<[f64]>,
    /// Overlap tail carried from the previous partial block, `part_len`.
    tail: Box<[f64]>,
}

/// One spectrum slot in a flat multi-spectrum buffer.
fn spectrum_slot(fft_size: usize) -> usize {
    fft_size + 2
}

impl core::fmt::Debug for PartitionedConvolver {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PartitionedConvolver")
            .field("part_len", &self.part_len)
            .field("fft_size", &self.fft_size)
            .field("partitions", &self.partitions)
            .finish()
    }
}

impl PartitionedConvolver {
    /// Build a convolver for `kernel` with `part_len`-sample partitions.
    /// `part_len` must be a power of two in `2..=2^19` (the internal FFT
    /// runs at `2·part_len` points).
    ///
    /// # Errors
    ///
    /// Returns [`DspError`] when the kernel is empty/non-finite or `part_len` is not a power of two in `2..=2^19`.
    pub fn new(kernel: &[f64], part_len: usize) -> Result<Self, DspError> {
        let kernel_len = validate_kernel(kernel)?;
        if part_len == 0 || !part_len.is_power_of_two() {
            return Err(DspError::NotPowerOfTwo { size: part_len });
        }
        if !(2..=(1 << 19)).contains(&part_len) {
            return Err(DspError::InvalidParameter {
                name: "part_len",
                reason: "must be in 2..=2^19",
            });
        }
        let partitions = checked_partitions(kernel_len, part_len)?;
        let fft_size = part_len * 2;
        let slot = spectrum_slot(fft_size);
        let fft = RealFft::new(fft_size)?;
        let spectrum_len = fft.spectrum_len();

        let mut kernel_freq = vec![0.0f64; partitions * spectrum_len];
        let mut work = vec![0.0f64; fft_size];
        let mut spec = vec![0.0f64; spectrum_len];
        for p in 0..partitions {
            // Zero-pad, copy this partition's kernel slice, transform.
            work.fill(0.0);
            {
                let (head, _) = work.split_at_mut(part_len);
                for (dst, src) in head
                    .iter_mut()
                    .zip(kernel.iter().skip(p * part_len).take(part_len))
                {
                    *dst = *src;
                }
            }
            fft.forward(&mut work, &mut spec)?;
            let dst = kernel_freq.chunks_exact_mut(slot).nth(p);
            if let Some(dst) = dst {
                dst.copy_from_slice(&spec);
            }
        }

        Ok(Self {
            part_len,
            fft_size,
            partitions,
            fft,
            kernel_freq: kernel_freq.into_boxed_slice(),
            input_freq: vec![0.0; partitions * spectrum_len].into_boxed_slice(),
            hist_pos: 0,
            spec: spec.into_boxed_slice(),
            acc: vec![0.0; spectrum_len].into_boxed_slice(),
            work: work.into_boxed_slice(),
            tail: vec![0.0; part_len].into_boxed_slice(),
        })
    }

    /// Partition (block) length in samples — also the buffering latency.
    #[must_use]
    pub fn part_len(&self) -> usize {
        self.part_len
    }

    /// Number of kernel partitions.
    #[must_use]
    pub fn partitions(&self) -> usize {
        self.partitions
    }

    /// Internal FFT size in samples (`2 · part_len`).
    #[must_use]
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Buffering latency in samples (one block).
    #[must_use]
    pub fn latency(&self) -> usize {
        self.part_len
    }

    /// Clear the input delay line and overlap tail (drop in-flight
    /// convolution state).
    pub fn reset(&mut self) {
        self.input_freq.fill(0.0);
        self.tail.fill(0.0);
        self.hist_pos = 0;
    }

    /// Convolve one `part_len` block: `input` (read-only) →
    /// `output`.
    ///
    /// # Errors
    ///
    /// Returns [`DspError::InvalidLength`] when either slice is not exactly [`PartitionedConvolver::part_len`] samples.
    pub fn process(&mut self, input: &[f64], output: &mut [f64]) -> Result<(), DspError> {
        if input.len() != self.part_len {
            return Err(DspError::InvalidLength {
                expected: self.part_len,
                got: input.len(),
            });
        }
        if output.len() != self.part_len {
            return Err(DspError::InvalidLength {
                expected: self.part_len,
                got: output.len(),
            });
        }
        let slot = spectrum_slot(self.fft_size);
        // 1. Zero-pad the block and transform.
        {
            let (head, rest) = self.work.split_at_mut(self.part_len);
            head.copy_from_slice(input);
            rest.fill(0.0);
        }
        self.fft.forward(&mut self.work, &mut self.spec)?;
        // 2. Push the spectrum onto the delay line.
        if let Some(dst) = self.input_freq.chunks_exact_mut(slot).nth(self.hist_pos) {
            dst.copy_from_slice(&self.spec);
        }
        // 3. Multiply-accumulate the delay line against the kernel.
        self.acc.fill(0.0);
        for p in 0..self.partitions {
            let newest = (self.hist_pos + self.partitions - p) % self.partitions;
            let history = self.input_freq.chunks_exact(slot).nth(newest);
            let kernel = self.kernel_freq.chunks_exact(slot).nth(p);
            if let (Some(history), Some(kernel)) = (history, kernel) {
                for (a, (x, k)) in self
                    .acc
                    .chunks_exact_mut(2)
                    .zip(history.chunks_exact(2).zip(kernel.chunks_exact(2)))
                {
                    if let ([a_re, a_im], [x_re, x_im], [k_re, k_im]) = (a, x, k) {
                        *a_re += *x_re * *k_re - *x_im * *k_im;
                        *a_im += *x_re * *k_im + *x_im * *k_re;
                    }
                }
            }
        }
        // 4. Back to time domain; overlap the partial block with the tail
        // carried from the previous one.
        self.fft.inverse(&mut self.acc, &mut self.work)?;
        {
            let (head, second) = self.work.split_at(self.part_len);
            for ((out, value), old) in output.iter_mut().zip(head.iter()).zip(self.tail.iter()) {
                *out = flush_denormal(value + old);
            }
            self.tail.copy_from_slice(second);
        }
        // 5. Advance the delay-line write cursor.
        self.hist_pos += 1;
        if self.hist_pos >= self.partitions {
            self.hist_pos = 0;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        missing_docs
    )]
    // Direct-convolution reference tests over fixed-size buffers.

    use super::*;
    use alloc::{format, vec::Vec};

    /// Naive direct convolution: y[i] = Σ kernel[j]·x[i−j].
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

    #[test]
    fn impulse_kernel_passes_through() {
        let kernel = [1.0f64];
        let mut conv = OverlapAdd::new(&kernel, 8).unwrap();
        assert_eq!(conv.block_size(), 8);
        assert_eq!(conv.kernel_len(), 1);
        assert!(conv.fft_size() >= 8);
        let input = [0.7f64; 8];
        let mut output = [0.0f64; 8];
        conv.process(&input, &mut output).unwrap();
        for (o, i) in output.iter().zip(input.iter()) {
            assert!((o - i).abs() < 1e-12);
        }
    }

    #[test]
    fn overlap_add_matches_direct_convolution() {
        // Block-stream a ramp through a 7-tap kernel; concatenate the block
        // outputs; compare against the direct calculation.
        let kernel: Vec<f64> = [0.5, 0.25, 0.125, -0.25, 0.1, 0.05, -0.02].to_vec();
        let block = 16;
        let mut conv = OverlapAdd::new(&kernel, block).unwrap();
        let x: Vec<f64> = (0..64)
            .map(|i| ((i as f64) * 0.31).sin() * (1.0 - 0.01 * i as f64))
            .collect();
        let expected = direct(&x, &kernel);
        let mut got = vec![0.0f64; x.len()];
        for (start, chunk) in x.chunks(block).enumerate() {
            let input = chunk.to_vec();
            let mut dst = vec![0.0f64; chunk.len()];
            conv.process(&input, &mut dst).unwrap();
            let base = start * block;
            for (j, v) in dst.iter().enumerate() {
                got[base + j] = *v;
            }
        }
        for (a, b) in expected.iter().zip(got.iter()) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn overlap_add_reset_drops_tail() {
        let kernel = [1.0f64, 1.0];
        let mut conv = OverlapAdd::new(&kernel, 8).unwrap();
        let input = [1.0f64; 8];
        let mut output = [0.0f64; 8];
        conv.process(&input, &mut output).unwrap();
        conv.reset();
        // After reset the first sample of the next block has no tail: a
        // zero block in, zero block out.
        let zeros = [0.0f64; 8];
        let mut output = [9.9f64; 8];
        conv.process(&zeros, &mut output).unwrap();
        for v in output.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn overlap_add_rejects_bad_input() {
        assert!(matches!(
            OverlapAdd::new(&[], 8),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(matches!(
            OverlapAdd::new(&[f64::NAN], 8),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(matches!(
            OverlapAdd::new(&[1.0], 0),
            Err(DspError::InvalidParameter { .. })
        ));
        let mut conv = OverlapAdd::new(&[1.0f64, 0.5], 8).unwrap();
        let input = [0.0f64; 7];
        let mut output = [0.0f64; 8];
        assert!(matches!(
            conv.process(&input, &mut output),
            Err(DspError::InvalidLength { .. })
        ));
        let input = [0.0f64; 8];
        let mut output = [0.0f64; 7];
        assert!(matches!(
            conv.process(&input, &mut output),
            Err(DspError::InvalidLength { .. })
        ));
    }

    #[test]
    fn overlap_add_debug_format() {
        let s = format!("{:?}", OverlapAdd::new(&[1.0], 8).unwrap());
        assert!(s.contains("OverlapAdd") && s.contains("block: 8"));
    }

    #[test]
    fn partitioned_impulse_kernel_passes_through() {
        let kernel = [1.0f64];
        let mut conv = PartitionedConvolver::new(&kernel, 8).unwrap();
        assert_eq!(conv.partitions(), 1);
        assert_eq!(conv.part_len(), 8);
        assert_eq!(conv.fft_size(), 16);
        assert_eq!(conv.latency(), 8);
        let input = [0.7f64; 8];
        let mut output = [0.0f64; 8];
        conv.process(&input, &mut output).unwrap();
        for (o, i) in output.iter().zip(input.iter()) {
            assert!((o - i).abs() < 1e-12);
        }
    }

    #[test]
    fn partitioned_matches_direct_convolution() {
        // 20-tap pseudo-random kernel over 3 partitions of 8 → stream 64
        // samples in 8-sample blocks and compare with the direct result.
        let kernel: Vec<f64> = (0..20)
            .map(|i| ((i as f64 + 1.0) * 0.173).sin() * 0.5)
            .collect();
        let part_len = 8;
        let mut conv = PartitionedConvolver::new(&kernel, part_len).unwrap();
        assert_eq!(conv.partitions(), 3);
        let x: Vec<f64> = (0..64)
            .map(|i| ((i as f64) * 0.27).sin() * (1.0 - 0.008 * i as f64))
            .collect();
        let expected = direct(&x, &kernel);
        let mut got = vec![0.0f64; x.len()];
        for (start, chunk) in x.chunks(part_len).enumerate() {
            let input = chunk.to_vec();
            let mut dst = vec![0.0f64; chunk.len()];
            conv.process(&input, &mut dst).unwrap();
            let base = start * part_len;
            for (j, v) in dst.iter().enumerate() {
                got[base + j] = *v;
            }
        }
        for (a, b) in expected.iter().zip(got.iter()) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn partitioned_reset_drops_delay_line() {
        let kernel = [1.0f64, 1.0, 1.0];
        let mut conv = PartitionedConvolver::new(&kernel, 4).unwrap();
        let input = [1.0f64; 4];
        let mut output = [0.0f64; 4];
        conv.process(&input, &mut output).unwrap();
        conv.reset();
        let zeros = [0.0f64; 4];
        let mut output = [7.7f64; 4];
        conv.process(&zeros, &mut output).unwrap();
        for v in output.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn partitioned_rejects_bad_input() {
        assert!(matches!(
            PartitionedConvolver::new(&[], 8),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(matches!(
            PartitionedConvolver::new(&[1.0], 7),
            Err(DspError::NotPowerOfTwo { size: 7 })
        ));
        assert!(matches!(
            PartitionedConvolver::new(&[1.0], 1),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(matches!(
            PartitionedConvolver::new(&[1.0], 1 << 20),
            Err(DspError::InvalidParameter { .. })
        ));
        assert!(PartitionedConvolver::new(&[1.0], 2).is_ok());
        let mut conv = PartitionedConvolver::new(&[1.0f64; 4], 4).unwrap();
        let input = [0.0f64; 3];
        let mut output = [0.0f64; 4];
        assert!(matches!(
            conv.process(&input, &mut output),
            Err(DspError::InvalidLength { .. })
        ));
        let input = [0.0f64; 4];
        let mut output = [0.0f64; 5];
        assert!(matches!(
            conv.process(&input, &mut output),
            Err(DspError::InvalidLength { .. })
        ));
    }

    #[test]
    fn partitioned_debug_format() {
        let s = format!("{:?}", PartitionedConvolver::new(&[1.0; 5], 4).unwrap());
        assert!(s.contains("PartitionedConvolver") && s.contains("partitions: 2"));
    }

    #[test]
    fn partitioned_multiblock_wraparound() {
        // Stream enough blocks to wrap the delay line several times and
        // verify every block against the direct convolution.
        let kernel: Vec<f64> = (0..16i32).map(|i| 0.5f64.powi(i + 1)).collect();
        let part_len = 4;
        let mut conv = PartitionedConvolver::new(&kernel, part_len).unwrap();
        let x: Vec<f64> = (0..96).map(|i| ((i as f64) * 0.53).cos() * 0.4).collect();
        let expected = direct(&x, &kernel);
        for (start, chunk) in x.chunks(part_len).enumerate() {
            let input = chunk.to_vec();
            let mut dst = vec![0.0f64; chunk.len()];
            conv.process(&input, &mut dst).unwrap();
            for (j, v) in dst.iter().enumerate() {
                let e = expected[start * part_len + j];
                assert!((e - *v).abs() < 1e-11, "sample {}", start * part_len + j);
            }
        }
    }
}
