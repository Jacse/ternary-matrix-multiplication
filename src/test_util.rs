// #[cfg(test)]
pub mod test_util {
    use ndarray::{Array2, ShapeBuilder};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use crate::{constants::SIZE, test_util::print_matrix};

    pub fn test_matmul(matmul: fn(a: &[i8], b: &[i8]) -> Vec<i8>) -> () {
        let m = SIZE;
        let k = SIZE;
        let n = SIZE;

        let (a, b) = rand_vecs();

        let a_array = Array2::from_shape_vec(
            (m, k).f(),
            // Convert to f32 to make it faster
            a.clone().into_iter().map(|e| e as f32).collect(),
        )
        .unwrap()
        .to_owned();

        let b_array = Array2::from_shape_vec(
            (k, n).f(),
            b.clone().into_iter().map(|e| e as f32).collect(),
        )
        .unwrap();

        let res_true = a_array.dot(&b_array).map(|f| *f as i8);

        let res = matmul(&a, &b);

        let res_clone = res.clone();

        let res_array = Array2::from_shape_vec((m, n).f(), res).unwrap();

        println!("Res:\n");
        print_matrix(&res_clone, 32, 32, 32, 1);
        println!("Res true:\n");
        print_matrix(&res_true.clone().into_raw_vec(), 32, 32, 32, 1);

        assert_eq!(res_array, res_true);
    }

    pub fn rand_vecs() -> (Vec<i8>, Vec<i8>) {
        let mut rng = StdRng::seed_from_u64(1337);
        let a = rand_ternary_vec(&mut rng, SIZE * SIZE);
        let b = rand_ternary_vec(&mut rng, SIZE * SIZE);
        (a, b)
    }

    fn rand_ternary_vec(rng: &mut impl Rng, length: usize) -> Vec<i8> {
        let mut vec = vec![0; length];
        for i in 0..vec.len() {
            vec[i] = rng.gen_range(-1..2);
        }
        vec
    }
}

pub fn print_matrix(inp: &[i8], rows: usize, cols: usize, cstride: usize, rstride: usize) -> () {
    for ri in 0..rows {
        for ci in 0..cols {
            let val = inp[cstride * ci + ri * rstride];
            print!(" {:0<2} ", val);
        }
        println!("");
    }
}

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
