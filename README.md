# Fast Ternary Matrix Multiplication

This repository is a collection of various approaches to archieve the fastest possible matrix multiplication of ternary matrices.

For now, the focus is on CPU model inference to pave the way for fast 2-bit model inference on edge devices running ARM processors. It will be extended to include x86 cpus and multiprocessing for more general consumer inference.

## Core ideas

Most approaches have been structured like BLAS/BLIS with a microkernel and repacking a and b to be contiguous in memory.

The ternary microkernel is currently only a little less than 2x as efficient as the base int8 case, which we should be able to get closer to the theoretical limit of 4x.

It works like this:

First we compress a ternary matrix `A(m, k)` into two separete matrixes, `A_val(m, k / 8)` and `A_sign(m, k / 8)`, where `A_val` contains all val bits (i.e. 1 if the ternary value is either 1 or -1) and `A_sign` contains all sign bits (i.e. 1 if the ternary value is -1).

Then we calculate a part of C by dotting a compressed vector from `a` and `b` using purely bit manipulation (no multiplication):

| step | description                                                              | pseudocode                           |
| ---- | ------------------------------------------------------------------------ | ------------------------------------ |
| 1    | Calc value bits with bitwise AND                                         | `c_val = a_val & b_val`              |
| 2    | Calc sign bits with bitwise xor between signs and bitwise and with value | `c_sign = (a_sign ^ b_sign) & c_val` |
| 3    | Count bits in values                                                     | `c_val_count = popcount(c_val)`      |
| 4    | Count bits in signs                                                      | `c_sign_count = popcount(c_sign)`    |
| 5    | Subtract 2x sign count (with bitshift) from val count                    | `c = val_count - sign_count << 1`    |

## Structure

Each attempt of matrix multiplication is contained in its own file in the `muls/` directory. Each attempt is purposefully verbose - the goal is to discover a fast way, not make production-level code.

To create/test a new attempt, copy on of the files and increment the counter. Test the approach (against `ndarray`) with `cargo test mm[num]`.

`constants.rs` contains the size of the given (square) matrix problem.

Note: All matrices are column-major order (row stride = 1, col stride = # rows).

# Performance

Run `rustup run nightly cargo bench` to benchmark the different approaches. You can include the `ndarray` benchmark or BLAS as well.

This is the performance on my M1 macbook current for a `m=k=n=1024` problem:

```
bench_mm1      ... bench:  84,751,291 ns/iter (+/- 2,997,778)
bench_mm2      ... bench:   5,638,622 ns/iter (+/- 321,324)
bench_mm3      ... bench:   3,285,750 ns/iter (+/- 140,154)
bench_mm4      ... bench:   1,656,936 ns/iter (+/- 76,470)
bench_mm5      ... bench:   1,534,104 ns/iter (+/- 97,869)
bench_mm6      ... bench:   1,581,695 ns/iter (+/- 91,915)
bench_mm7      ... bench:     904,520 ns/iter (+/- 23,409)
bench_mm8      ... bench:     874,420 ns/iter (+/- 18,969)
bench_mm9      ... bench:     858,866 ns/iter (+/- 6,606)

References:

bench_blas_f32 ... bench:   2,802,070 ns/iter (+/- 105,245)
bench_ndarray  ... bench:   2,767,095 ns/iter (+/- 111,917)
```

# Todos

- Multithreading
- Alt packing (pack 5 trits into a byte)
- CUDA/Triton kernel
- x86_64 core
  - test avx2-based 32x32 kernel
