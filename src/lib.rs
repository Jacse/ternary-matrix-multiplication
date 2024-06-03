#![feature(portable_simd)]
#![feature(stdarch_aarch64_prefetch)]
#![feature(core_intrinsics)]
pub mod muls;

pub mod constants;
pub mod test_util;

// BLAS library for benchmarking
extern crate blas;
extern crate blis_src;
// extern crate accelerate_src;
