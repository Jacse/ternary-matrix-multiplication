#![feature(portable_simd)]
pub mod muls;

pub mod constants;
pub mod test_util;

// BLAS library for benchmarking
extern crate blas;
extern crate blis_src;
// extern crate accelerate_src;
