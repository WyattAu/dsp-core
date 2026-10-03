//! The crate-wide error type.
//!
//! Every fallible constructor validates its parameters up front and returns
//! [`DspError`] instead of panicking; process-path methods are infallible by
//! design and never allocate.

use core::fmt;

/// Errors returned by constructors and length-checked process methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DspError {
    /// An FFT size (or size derived from a block/partition length) was not a
    /// power of two.
    NotPowerOfTwo {
        /// The offending size.
        size: usize,
    },
    /// An FFT size exceeded the pre-computed-table budget.
    FftTooLarge {
        /// The offending size.
        size: usize,
        /// The inclusive maximum this crate accepts.
        max: usize,
    },
    /// An FFT size fell below the minimum this crate accepts.
    FftTooSmall {
        /// The offending size.
        size: usize,
        /// The inclusive minimum this crate accepts.
        min: usize,
    },
    /// A buffer length did not match what the transform or block requires.
    InvalidLength {
        /// The required length.
        expected: usize,
        /// The length that was supplied.
        got: usize,
    },
    /// A numeric parameter was outside its valid range (non-finite,
    /// non-positive frequency, `Q <= 0`, `ratio < 1`, negative window
    /// length, and so on).
    InvalidParameter {
        /// The parameter name.
        name: &'static str,
        /// What was wrong with it.
        reason: &'static str,
    },
}

impl fmt::Display for DspError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DspError::NotPowerOfTwo { size } => {
                write!(f, "size {size} is not a power of two")
            }
            DspError::FftTooLarge { size, max } => {
                write!(
                    f,
                    "size {size} exceeds the maximum supported FFT size {max}"
                )
            }
            DspError::FftTooSmall { size, min } => {
                write!(
                    f,
                    "size {size} is below the minimum supported FFT size {min}"
                )
            }
            DspError::InvalidLength { expected, got } => {
                write!(f, "buffer length {got}, expected {expected}")
            }
            DspError::InvalidParameter { name, reason } => {
                write!(f, "invalid parameter `{name}`: {reason}")
            }
        }
    }
}

impl core::error::Error for DspError {}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    // Unit tests over known-size buffers; assertions instead of panicking
    // unwraps would obscure the error-shape contract under test here.

    use crate::error::DspError;
    use alloc::string::ToString;

    #[test]
    fn display_is_stable() {
        let e = DspError::InvalidLength {
            expected: 128,
            got: 64,
        };
        assert_eq!(e.to_string(), "buffer length 64, expected 128");

        let e = DspError::NotPowerOfTwo { size: 100 };
        assert_eq!(e.to_string(), "size 100 is not a power of two");

        let e = DspError::FftTooLarge {
            size: 1 << 30,
            max: 1 << 20,
        };
        assert_eq!(
            e.to_string(),
            "size 1073741824 exceeds the maximum supported FFT size 1048576"
        );

        let e = DspError::FftTooSmall { size: 1, min: 2 };
        assert_eq!(
            e.to_string(),
            "size 1 is below the minimum supported FFT size 2"
        );

        let e = DspError::InvalidParameter {
            name: "q",
            reason: "must be finite and > 0",
        };
        assert_eq!(
            e.to_string(),
            "invalid parameter `q`: must be finite and > 0"
        );
    }

    #[test]
    fn is_std_error() {
        fn assert_error<E: core::error::Error>(_: &E) {}
        assert_error(&DspError::NotPowerOfTwo { size: 3 });
    }
}
