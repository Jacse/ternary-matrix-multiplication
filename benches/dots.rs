#![feature(test)]

extern crate test;

#[cfg(test)]
mod tests {
    use matmul::dots::{dot_masks, dot_naive, dot_shifted_mask_multiplication};
    use test::{black_box, Bencher};

    #[bench]
    fn bench_dot_naive(bench: &mut Bencher) {
        let a = [100, -50, 50, 75, -74, 25, 25, 0].repeat(2);
        let b = [-1, -1, -1, -1, 1, 0, -1, 1].repeat(2);

        bench.iter(|| {
            black_box(dot_naive(&a, &b));
        });
    }

    #[bench]
    fn bench_dot_shifted_mask_multiplication(bench: &mut Bencher) {
        let a = [100, -50, 50, 75, -74, 25, 25, 0].repeat(2);
        let b_vals = 0b1111101111111011;
        let b_signs = 0b1111001011110010;

        bench.iter(|| {
            black_box(unsafe { dot_shifted_mask_multiplication(&a, b_vals, b_signs) });
        });
    }

    #[bench]
    fn bench_dot_masks(bench: &mut Bencher) {
        let a = [100, -50, 50, 75, -74, 25, 25, 0].repeat(2);
        let b_vals = 0b1111101111111011;
        let b_signs = 0b1111001011110010;

        bench.iter(|| {
            black_box(unsafe { dot_masks(&a, b_vals, b_signs) });
        });
    }
}
