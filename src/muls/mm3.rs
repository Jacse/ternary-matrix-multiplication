use std::{cmp::min, simd::Simd};

use crate::constants::SIZE;

// all matrices in col-major order with stride SIZE
fn at(r: usize, c: usize) -> usize {
    r + SIZE * c
}

// Computes an 8x8 block of C
fn dot8x8(k: usize, a: &[i8], b: &[i8], c: &mut [i8]) {
    // 8 rows of ab/c, with each 8 items
    // Load initial values from c
    let mut ab = vec![Simd::<i8, 8>::splat(0); 8];

    ab[0] = Simd::from_slice(&c[at(0, 0)..(at(0, 0) + 8)]);
    ab[1] = Simd::from_slice(&c[at(0, 1)..(at(0, 1) + 8)]);
    ab[2] = Simd::from_slice(&c[at(0, 2)..(at(0, 2) + 8)]);
    ab[3] = Simd::from_slice(&c[at(0, 3)..(at(0, 3) + 8)]);
    ab[4] = Simd::from_slice(&c[at(0, 4)..(at(0, 4) + 8)]);
    ab[5] = Simd::from_slice(&c[at(0, 5)..(at(0, 5) + 8)]);
    ab[6] = Simd::from_slice(&c[at(0, 6)..(at(0, 6) + 8)]);
    ab[7] = Simd::from_slice(&c[at(0, 7)..(at(0, 7) + 8)]);

    for ki in 0..k {
        // Load 8 columns of b (b0-b7) and duplicate the value on that row to all rows
        let b0 = Simd::<i8, 8>::splat(b[ki * 8 + 0]);
        let b1 = Simd::<i8, 8>::splat(b[ki * 8 + 1]);
        let b2 = Simd::<i8, 8>::splat(b[ki * 8 + 2]);
        let b3 = Simd::<i8, 8>::splat(b[ki * 8 + 3]);
        let b4 = Simd::<i8, 8>::splat(b[ki * 8 + 4]);
        let b5 = Simd::<i8, 8>::splat(b[ki * 8 + 5]);
        let b6 = Simd::<i8, 8>::splat(b[ki * 8 + 6]);
        let b7 = Simd::<i8, 8>::splat(b[ki * 8 + 7]);

        // Load one col of 8 rows of A
        let a0 = Simd::<i8, 8>::from_slice(&a[(ki * 8)..(ki * 8 + 8)]);

        // Calculate c
        ab[0] += a0 * b0;
        ab[1] += a0 * b1;
        ab[2] += a0 * b2;
        ab[3] += a0 * b3;
        ab[4] += a0 * b4;
        ab[5] += a0 * b5;
        ab[6] += a0 * b6;
        ab[7] += a0 * b7;
    }

    // TODO: Pack c?
    c[at(0, 0)..(at(0, 0) + 8)].copy_from_slice(&ab[0].to_array());
    c[at(0, 1)..(at(0, 1) + 8)].copy_from_slice(&ab[1].to_array());
    c[at(0, 2)..(at(0, 2) + 8)].copy_from_slice(&ab[2].to_array());
    c[at(0, 3)..(at(0, 3) + 8)].copy_from_slice(&ab[3].to_array());
    c[at(0, 4)..(at(0, 4) + 8)].copy_from_slice(&ab[4].to_array());
    c[at(0, 5)..(at(0, 5) + 8)].copy_from_slice(&ab[5].to_array());
    c[at(0, 6)..(at(0, 6) + 8)].copy_from_slice(&ab[6].to_array());
    c[at(0, 7)..(at(0, 7) + 8)].copy_from_slice(&ab[7].to_array());
}

fn pack_b(k: usize, b: &[i8], packed: &mut [i8]) {
    let mut offset = 0;
    for ki in 0..k {
        // Loop over rows of B, ensure 8 cols are contiguous in memory
        packed[offset] = b[at(ki, 0)];
        packed[offset + 1] = b[at(ki, 1)];
        packed[offset + 2] = b[at(ki, 2)];
        packed[offset + 3] = b[at(ki, 3)];
        packed[offset + 4] = b[at(ki, 4)];
        packed[offset + 5] = b[at(ki, 5)];
        packed[offset + 6] = b[at(ki, 6)];
        packed[offset + 7] = b[at(ki, 7)];

        offset += 8;
    }
}

fn pack_a(k: usize, a: &[i8], packed: &mut [i8]) {
    let mut offset = 0;
    for ki in 0..k {
        // Loop over cols of A, ensure 8 rows are contiguous in memory
        // Since a is col-major we can get all 8 rows by copying directly
        let start = at(0, ki);
        packed[offset..(offset + 8)].copy_from_slice(&a[start..start + 8]);

        offset += 8;
    }
}

fn inner_kernel(m: usize, k: usize, n: usize, a: &[i8], b: &[i8], c: &mut [i8]) {
    let mut packed_a = vec![0_i8; m * k];
    let mut packed_b = vec![0_i8; k * n];

    for ni in (0..n).step_by(8) {
        // Pack B to be contiguous
        // TODO: Use the same small buffer instead of populating a full-sized matrix
        // if should_pack_b {
        pack_b(k, &b[(at(0, ni))..], &mut packed_b[(ni * k)..]);
        // }

        for mi in (0..m).step_by(8) {
            // Pack A
            if ni == 0 {
                pack_a(k, &a[(at(mi, 0))..], &mut packed_a[(mi * k)..]);
            }

            dot8x8(
                k,
                &packed_a[(mi * k)..],
                &packed_b[(ni * k)..],
                &mut c[at(mi, ni)..],
            );
        }
    }
}

pub fn matmul3(a: &[i8], b: &[i8]) -> Vec<i8> {
    let (m, k, n) = (SIZE, SIZE, SIZE);
    let mut c = vec![0; m * n];

    let nc = 256;
    let kc = 512;
    let mc = 256;

    // Splitting B and C on the n-dimension into parts of nc size
    // Splitting A and C on the m-dimension into parts of mc
    // Each inner_kernel computes an mc x nc block of C

    for ni in (0..n).step_by(nc) {
        let tile_n = min(n - ni, nc);

        for ki in (0..k).step_by(kc) {
            let tile_k = min(k - ki, kc);

            for mi in (0..m).step_by(mc) {
                let tile_m = min(m - mi, mc);

                inner_kernel(
                    tile_m,
                    tile_k,
                    tile_n,
                    &a[at(mi, ki)..],
                    &b[at(ki, ni)..],
                    &mut c[(at(mi, ni))..],
                );
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use crate::test_util::test_util::test_matmul;

    use super::matmul3;

    #[test]
    fn test() {
        test_matmul(matmul3)
    }
}
