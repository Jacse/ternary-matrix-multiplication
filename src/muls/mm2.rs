use std::simd::Simd;

use crate::constants::SIZE;

// all matrices in col-major order with stride SIZE
fn at(r: usize, c: usize) -> usize {
    r + SIZE * c
}

// Computes an 8x8 block of C
fn dot8x8(k: usize, a: &[i8], b: &[i8], c: &mut [i8]) {
    // 8 rows of ab/c, with each 8 items
    let mut ab = vec![Simd::<i8, 8>::splat(0); 8];

    for ki in 0..k {
        // Load 8 columns of b (b0-b7) and duplicate the value on that row to all lanes
        let b0 = Simd::<i8, 8>::splat(b[at(ki, 0)]);
        let b1 = Simd::<i8, 8>::splat(b[at(ki, 1)]);
        let b2 = Simd::<i8, 8>::splat(b[at(ki, 2)]);
        let b3 = Simd::<i8, 8>::splat(b[at(ki, 3)]);
        let b4 = Simd::<i8, 8>::splat(b[at(ki, 4)]);
        let b5 = Simd::<i8, 8>::splat(b[at(ki, 5)]);
        let b6 = Simd::<i8, 8>::splat(b[at(ki, 6)]);
        let b7 = Simd::<i8, 8>::splat(b[at(ki, 7)]);

        // Load one col of 8 rows of A
        let a0 = Simd::<i8, 8>::from_slice(&a[at(0, ki)..at(0, ki) + 8]);

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

    c[at(0, 0)..(at(0, 0) + 8)].copy_from_slice(&ab[0].to_array());
    c[at(0, 1)..(at(0, 1) + 8)].copy_from_slice(&ab[1].to_array());
    c[at(0, 2)..(at(0, 2) + 8)].copy_from_slice(&ab[2].to_array());
    c[at(0, 3)..(at(0, 3) + 8)].copy_from_slice(&ab[3].to_array());
    c[at(0, 4)..(at(0, 4) + 8)].copy_from_slice(&ab[4].to_array());
    c[at(0, 5)..(at(0, 5) + 8)].copy_from_slice(&ab[5].to_array());
    c[at(0, 6)..(at(0, 6) + 8)].copy_from_slice(&ab[6].to_array());
    c[at(0, 7)..(at(0, 7) + 8)].copy_from_slice(&ab[7].to_array());
}

pub fn matmul2(a: &[i8], b: &[i8]) -> Vec<i8> {
    let (m, k, n) = (SIZE, SIZE, SIZE);
    let mut c = vec![0; m * n];

    // Loop over cols in c
    for ni in (0..n).step_by(8) {
        // Loop over rows in c
        for mi in (0..m).step_by(8) {
            dot8x8(k, &a[at(mi, 0)..], &b[at(0, ni)..], &mut c[at(mi, ni)..]);
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use crate::test_util::test_util::test_matmul;

    use super::matmul2;

    #[test]
    fn test() {
        test_matmul(matmul2)
    }
}
