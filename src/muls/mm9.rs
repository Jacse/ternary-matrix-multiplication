use std::{cmp::min, simd::Simd};

use crate::constants::SIZE;

// col-major order with stride SIZE
fn c_at(r: usize, c: usize) -> usize {
    r + SIZE * c
}
// col-major with stride SIZE / 8 (compression factor) (compressed in k/rows)
fn b_at(r: usize, c: usize) -> usize {
    r + SIZE / 8 * c
}
// col-major with stride STRIDE (compressed in k, not m)
fn a_at(r: usize, c: usize) -> usize {
    r + SIZE * c
}

/// Counts bits in SIMD lanes.
/// Had to go unsafe because popcount is missing in portable SIMD
/// # Safety
/// The bit count of a u8 can never be outside the range of 0-8
/// which is safe for both unsigned and signed ints so we can safely cast
#[cfg(target_arch = "aarch64")]
#[cfg(target_feature = "neon")]
unsafe fn popcount(input: Simd<u8, 16>) -> Simd<i8, 16> {
    use core::arch::aarch64::*;
    use std::simd::num::SimdUint;
    let count = vcntq_u8(uint8x16_t::from(input));
    Simd::from(count).cast::<i8>()
}

#[inline(always)]
fn dot(
    a_val: Simd<u8, 16>,
    a_sign: Simd<u8, 16>,
    b_val: Simd<u8, 16>,
    b_sign: Simd<u8, 16>,
) -> Simd<i8, 16> {
    // Calc val bits between all A1 rows and first B[i] col
    let val = b_val & a_val;
    // We can maybe speed this up using bitwise insert if true / other instrinsics
    let or = b_sign ^ a_sign;
    let sign = val & or;

    // Add vals and counts
    unsafe {
        let val_count = popcount(val);
        let sign_count = popcount(sign);
        let sign_countx2 = sign_count << 1;

        val_count - sign_countx2
    }
}

// Computes an 16x16 block of C
fn dot16x16(k: usize, a_vals: &[u8], a_signs: &[u8], b_vals: &[u8], b_signs: &[u8], c: &mut [i8]) {
    // 16 rows of ab/c, with each 16 items
    // Load initial values from c
    let mut ab = vec![Simd::<i8, 16>::splat(0); 16];

    ab[0] = Simd::from_slice(&c[c_at(0, 0)..(c_at(0, 0) + 16)]);
    ab[1] = Simd::from_slice(&c[c_at(0, 1)..(c_at(0, 1) + 16)]);
    ab[2] = Simd::from_slice(&c[c_at(0, 2)..(c_at(0, 2) + 16)]);
    ab[3] = Simd::from_slice(&c[c_at(0, 3)..(c_at(0, 3) + 16)]);
    ab[4] = Simd::from_slice(&c[c_at(0, 4)..(c_at(0, 4) + 16)]);
    ab[5] = Simd::from_slice(&c[c_at(0, 5)..(c_at(0, 5) + 16)]);
    ab[6] = Simd::from_slice(&c[c_at(0, 6)..(c_at(0, 6) + 16)]);
    ab[7] = Simd::from_slice(&c[c_at(0, 7)..(c_at(0, 7) + 16)]);
    ab[8] = Simd::from_slice(&c[c_at(0, 8)..(c_at(0, 8) + 16)]);
    ab[9] = Simd::from_slice(&c[c_at(0, 9)..(c_at(0, 9) + 16)]);
    ab[10] = Simd::from_slice(&c[c_at(0, 10)..(c_at(0, 10) + 16)]);
    ab[11] = Simd::from_slice(&c[c_at(0, 11)..(c_at(0, 11) + 16)]);
    ab[12] = Simd::from_slice(&c[c_at(0, 12)..(c_at(0, 12) + 16)]);
    ab[13] = Simd::from_slice(&c[c_at(0, 13)..(c_at(0, 13) + 16)]);
    ab[14] = Simd::from_slice(&c[c_at(0, 14)..(c_at(0, 14) + 16)]);
    ab[15] = Simd::from_slice(&c[c_at(0, 15)..(c_at(0, 15) + 16)]);

    for ki in (0..k).step_by(4) {
        // Load one col of 16 rows of A (=> 8*16=128 ternary values )
        let a1_val = Simd::<u8, 16>::from_slice(&a_vals[(ki * 16)..(ki * 16 + 16)]);
        let a1_sign = Simd::<u8, 16>::from_slice(&a_signs[(ki * 16)..(ki * 16 + 16)]);
        let a2_val = Simd::<u8, 16>::from_slice(&a_vals[(ki * 16 + 16)..(ki * 16 + 32)]);
        let a2_sign = Simd::<u8, 16>::from_slice(&a_signs[(ki * 16 + 16)..(ki * 16 + 32)]);

        let a3_val = Simd::<u8, 16>::from_slice(&a_vals[(ki * 16 + 32)..(ki * 16 + 48)]);
        let a3_sign = Simd::<u8, 16>::from_slice(&a_signs[(ki * 16 + 32)..(ki * 16 + 48)]);
        let a4_val = Simd::<u8, 16>::from_slice(&a_vals[(ki * 16 + 48)..(ki * 16 + 64)]);
        let a4_sign = Simd::<u8, 16>::from_slice(&a_signs[(ki * 16 + 48)..(ki * 16 + 64)]);

        // Load one row of 16 cols of B
        let b1_val = Simd::<u8, 16>::from_slice(&b_vals[(ki * 16)..(ki * 16 + 16)]);
        let b1_sign = Simd::<u8, 16>::from_slice(&b_signs[(ki * 16)..(ki * 16 + 16)]);
        let b2_val = Simd::<u8, 16>::from_slice(&b_vals[(ki * 16 + 16)..(ki * 16 + 32)]);
        let b2_sign = Simd::<u8, 16>::from_slice(&b_signs[(ki * 16 + 16)..(ki * 16 + 32)]);

        let b3_val = Simd::<u8, 16>::from_slice(&b_vals[(ki * 16 + 32)..(ki * 16 + 48)]);
        let b3_sign = Simd::<u8, 16>::from_slice(&b_signs[(ki * 16 + 32)..(ki * 16 + 48)]);
        let b4_val = Simd::<u8, 16>::from_slice(&b_vals[(ki * 16 + 48)..(ki * 16 + 64)]);
        let b4_sign = Simd::<u8, 16>::from_slice(&b_signs[(ki * 16 + 48)..(ki * 16 + 64)]);

        // Compute
        macro_rules! set_c {
            ($i:expr) => {
                // Broadcast ith lane (col) in B to a full 16-col simd
                let b1_val_i = Simd::splat(b1_val[$i]); // vdupq_laneq_u8 in aarch64 asm
                let b1_sign_i = Simd::splat(b1_sign[$i]);
                let b2_val_i = Simd::splat(b2_val[$i]);
                let b2_sign_i = Simd::splat(b2_sign[$i]);

                let b3_val_i = Simd::splat(b3_val[$i]); // vdupq_laneq_u8 in aarch64 asm
                let b3_sign_i = Simd::splat(b3_sign[$i]);
                let b4_val_i = Simd::splat(b4_val[$i]);
                let b4_sign_i = Simd::splat(b4_sign[$i]);

                // output 16 rows to C (16 rows of A x ith col of B)
                ab[$i] += dot(a1_val, a1_sign, b1_val_i, b1_sign_i);
                ab[$i] += dot(a2_val, a2_sign, b2_val_i, b2_sign_i);
                ab[$i] += dot(a3_val, a3_sign, b3_val_i, b3_sign_i);
                ab[$i] += dot(a4_val, a4_sign, b4_val_i, b4_sign_i);
            };
        }

        set_c!(0);
        set_c!(1);
        set_c!(2);
        set_c!(3);
        set_c!(4);
        set_c!(5);
        set_c!(6);
        set_c!(7);
        set_c!(8);
        set_c!(9);
        set_c!(10);
        set_c!(11);
        set_c!(12);
        set_c!(13);
        set_c!(14);
        set_c!(15);
    }

    // TODO: Pack c?
    c[c_at(0, 0)..(c_at(0, 0) + 16)].copy_from_slice(&ab[0].to_array());
    c[c_at(0, 1)..(c_at(0, 1) + 16)].copy_from_slice(&ab[1].to_array());
    c[c_at(0, 2)..(c_at(0, 2) + 16)].copy_from_slice(&ab[2].to_array());
    c[c_at(0, 3)..(c_at(0, 3) + 16)].copy_from_slice(&ab[3].to_array());
    c[c_at(0, 4)..(c_at(0, 4) + 16)].copy_from_slice(&ab[4].to_array());
    c[c_at(0, 5)..(c_at(0, 5) + 16)].copy_from_slice(&ab[5].to_array());
    c[c_at(0, 6)..(c_at(0, 6) + 16)].copy_from_slice(&ab[6].to_array());
    c[c_at(0, 7)..(c_at(0, 7) + 16)].copy_from_slice(&ab[7].to_array());

    c[c_at(0, 8)..(c_at(0, 8) + 16)].copy_from_slice(&ab[8].to_array());
    c[c_at(0, 9)..(c_at(0, 9) + 16)].copy_from_slice(&ab[9].to_array());
    c[c_at(0, 10)..(c_at(0, 10) + 16)].copy_from_slice(&ab[10].to_array());
    c[c_at(0, 11)..(c_at(0, 11) + 16)].copy_from_slice(&ab[11].to_array());
    c[c_at(0, 12)..(c_at(0, 12) + 16)].copy_from_slice(&ab[12].to_array());
    c[c_at(0, 13)..(c_at(0, 13) + 16)].copy_from_slice(&ab[13].to_array());
    c[c_at(0, 14)..(c_at(0, 14) + 16)].copy_from_slice(&ab[14].to_array());
    c[c_at(0, 15)..(c_at(0, 15) + 16)].copy_from_slice(&ab[15].to_array());
}

fn pack_b(k: usize, n: usize, b: &[u8], packed: &mut [u8]) {
    let mut offset = 0;
    // Loop over all 16 col vertical sections
    for ni in (0..n).step_by(16) {
        // Loop over all rows in the 16 col section
        for ki in 0..k {
            // Loop over rows of B, ensure 8 cols are contiguous in memory
            packed[offset] = b[b_at(ki, ni + 0)];
            packed[offset + 1] = b[b_at(ki, ni + 1)];
            packed[offset + 2] = b[b_at(ki, ni + 2)];
            packed[offset + 3] = b[b_at(ki, ni + 3)];
            packed[offset + 4] = b[b_at(ki, ni + 4)];
            packed[offset + 5] = b[b_at(ki, ni + 5)];
            packed[offset + 6] = b[b_at(ki, ni + 6)];
            packed[offset + 7] = b[b_at(ki, ni + 7)];
            packed[offset + 8] = b[b_at(ki, ni + 8)];
            packed[offset + 9] = b[b_at(ki, ni + 9)];
            packed[offset + 10] = b[b_at(ki, ni + 10)];
            packed[offset + 11] = b[b_at(ki, ni + 11)];
            packed[offset + 12] = b[b_at(ki, ni + 12)];
            packed[offset + 13] = b[b_at(ki, ni + 13)];
            packed[offset + 14] = b[b_at(ki, ni + 14)];
            packed[offset + 15] = b[b_at(ki, ni + 15)];

            offset += 16;
        }
    }
}

fn pack_a(k: usize, m: usize, a: &[u8], packed: &mut [u8]) {
    let mut offset = 0;
    // Loop over all 16 row horizontal section
    for mi in (0..m).step_by(16) {
        // loop over all cols in section
        for ki in 0..k {
            // Ensure 8 rows are contiguous in memory
            // Since a is col-major we can get all 8 rows by copying directly
            let start = a_at(mi, ki);
            packed[offset..(offset + 16)].copy_from_slice(&a[start..start + 16]);

            offset += 16;
        }
    }
}

// Expects and and b to be contiguous
fn inner_kernel(
    m: usize,
    k: usize,
    n: usize,
    packed_a_vals: &[u8],
    packed_a_signs: &[u8],
    packed_b_vals: &[u8],
    packed_b_signs: &[u8],
    c: &mut [i8],
) {
    for ni in (0..n).step_by(16) {
        for mi in (0..m).step_by(16) {
            dot16x16(
                k,
                &packed_a_vals[(mi * k)..],
                &packed_a_signs[(mi * k)..],
                &packed_b_vals[(ni * k)..],
                &packed_b_signs[(ni * k)..],
                &mut c[c_at(mi, ni)..],
            );
        }
    }
}

// Compresses a vec of ternary i8s into val and sign u8s
fn compress(input: &[i8]) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(input.len() % 8, 0);

    let mut vals = Vec::with_capacity(input.len() / 8);
    let mut signs = Vec::with_capacity(input.len() / 8);

    for i in (0..input.len()).step_by(8) {
        let mut val = 0_u8;
        let mut sign = 0_u8;
        for j in 0..8 {
            let (v, s) = match &input[i + j] {
                -1 => (1, 1),
                1 => (1, 0),
                0 => (0, 0),
                _ => unreachable!(),
            };
            val |= v << j;
            sign |= s << j;
        }
        vals.push(val);
        signs.push(sign);
    }

    (vals, signs)
}

fn col_major_to_row_major<T>(size: (usize, usize), input: &[T]) -> Vec<T>
where
    T: Copy,
{
    let (rows, cols) = size;
    let mut out = Vec::with_capacity(input.len());
    for ri in 0..rows {
        for ci in 0..cols {
            out.push(input[ci * rows + ri]);
        }
    }
    out
}
fn row_major_to_col_major<T>(size: (usize, usize), input: &[T]) -> Vec<T>
where
    T: Copy,
{
    let (rows, cols) = size;
    let mut out = Vec::with_capacity(input.len());
    for ci in 0..cols {
        for ri in 0..rows {
            out.push(input[ri * cols + ci]);
        }
    }
    out
}

pub fn prep9(a: &[i8], b: &[i8]) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    // one u16 contains 8 ternary values (u8 vals + u8 signs)
    // So the resulting matrix is 4x smaller than the original

    // transform a from col-major to row-major, compress and transform back to col-major
    let a_row = col_major_to_row_major((SIZE, SIZE), &a);
    let (a_vals_row, a_signs_row) = compress(&a_row);
    let a_vals = row_major_to_col_major((SIZE, SIZE / 8), &a_vals_row);
    let a_signs = row_major_to_col_major((SIZE, SIZE / 8), &a_signs_row);

    let (b_vals, b_signs) = compress(&b);
    (a_vals, a_signs, b_vals, b_signs)
}

pub fn matmul9(a_vals: &[u8], a_signs: &[u8], b_vals: &[u8], b_signs: &[u8]) -> Vec<i8> {
    let (m, k, n) = (SIZE, SIZE / 8, SIZE);
    let mut c = vec![0; m * n];

    let nc = 256;
    let kc = 512;
    let mc = 256;

    let mut packed_a_vals = vec![0_u8; mc * kc];
    let mut packed_a_signs = vec![0_u8; mc * kc];
    let mut packed_b_vals = vec![0_u8; kc * nc];
    let mut packed_b_signs = vec![0_u8; kc * nc];

    // LOOP 5: Split B and C on the n-dimension into parts of nc size
    for ni in (0..n).step_by(nc) {
        let tile_n = min(n - ni, nc);

        // LOOP 4: Split A and B on the k-dimension into parts of kc size
        for ki in (0..k).step_by(kc) {
            let tile_k = min(k - ki, kc);

            pack_b(
                tile_k,
                tile_n,
                &b_vals[(b_at(ki, ni))..],
                &mut packed_b_vals,
            );
            pack_b(
                tile_k,
                tile_n,
                &b_signs[(b_at(ki, ni))..],
                &mut packed_b_signs,
            );

            // LOOP 3: Split A and C on the m-dimension into parts of mc
            for mi in (0..m).step_by(mc) {
                let tile_m = min(m - mi, mc);

                pack_a(
                    tile_k,
                    tile_m,
                    &a_vals[(a_at(mi, ki))..],
                    &mut packed_a_vals,
                );
                pack_a(
                    tile_k,
                    tile_m,
                    &a_signs[(a_at(mi, ki))..],
                    &mut packed_a_signs,
                );

                inner_kernel(
                    tile_m,
                    tile_k,
                    tile_n,
                    &packed_a_vals,
                    &packed_a_signs,
                    &packed_b_vals,
                    &packed_b_signs,
                    &mut c[(c_at(mi, ni))..],
                );
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use crate::{constants::SIZE, test_util::test_util::test_matmul};

    use super::{matmul9, prep9};

    #[test]
    fn test() {
        test_matmul(|a, b| {
            let (av, asi, bv, bs) = prep9(&a, &b);
            matmul9(&av, &asi, &bv, &bs)
        })
    }
}
