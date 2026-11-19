# Pattern adoption: dsp-core

Declared against the WyattAu pattern canon (WyattAu/pattern-standards,
valid IDs T1-T50, P1-P119, X1-X7). Format and statuses follow the
adoption-format reference in that canon; this table is validated by its
adoption-check script.

dsp-core is an L0 numerical kernel crate: single-threaded, no I/O, no
telemetry, no concurrency. Patterns outside that domain are omitted
(adoption-format rule 6). Time-discipline patterns (P8, P20, P13) live
in consumer crates; this crate owns no clocks, only sample counts.

| Pattern | Status | Evidence |
|---|---|---|
| T13 | adopted | src/lib.rs:16 (process paths are pure arithmetic over caller-owned memory; constructors own allocation and validation), src/biquad.rs:109 (constructors validate and return DspError), src/biquad.rs:439 (process is infallible scalar arithmetic) |
| P14 | partial | src/fft.rs:1 (fixed radix-2 DIT topology: precomputed twiddles and a bit-reversal table pin the reduction tree to logical index), src/convolve.rs:1 (uniform partitioning fixes per-block cost); the crate is single-threaded, so arrival-order drift cannot arise and the multi-core half of the mechanism is out of scope |
| P15 | partial | src/biquad.rs:120 (is_finite trap at the originating constructor; 19 such checks across five kernel modules), src/math.rs:19 (flush_denormal passes NaN and Inf through so origin stays visible); no Finite wrapper type yet |
| P64 | planned | tests/fft.rs:47 (conservation invariants are hand-written fixed-vector tests today: Parseval, DFT reference, analytic transfer functions); proptest port plus nightly soak workflow per the adoption-format soak clause, target 2027-Q1 |
