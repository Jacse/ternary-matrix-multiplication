#![feature(test)]

extern crate test;

#[cfg(test)]
mod tests {
    use blas::sgemm;
    use matmul::{
        constants::SIZE,
        muls::{
            mm1::matmul1,
            mm2::matmul2,
            mm3::matmul3,
            mm4::matmul4,
            mm5::matmul5,
            mm6::matmul6,
            mm7::{matmul7, prep7},
            mm8::{matmul8, prep8},
            mm9::{matmul9, prep9},
        },
        test_util::test_util::rand_vecs,
    };
    use ndarray::Array2;
    use test::{black_box, Bencher};

    macro_rules! bench {
        ($num: literal) => {
            paste::item! {
                #[bench]
                fn [< bench_mm $num >](bench: &mut Bencher) {
                    let (a, b) = rand_vecs();
                    bench.iter(|| {
                        black_box([< matmul $num >](&a, &b));
                    });
                }
            }
        };
    }
    macro_rules! bench_w_prep {
        ($num: literal) => {
            paste::item! {
                #[bench]
                fn [< bench_mm $num >](bench: &mut Bencher) {
                    let (a, b) = rand_vecs();
                    let (av, asi, bv, bs) = [< prep $num >](&a, &b);

                    bench.iter(|| {
                        black_box([< matmul $num >](&av, &asi, &bv, &bs));
                    });
                }
            }
        };
    }

    bench!(1);
    bench!(2);
    bench!(3);
    bench!(4);
    bench!(5);
    bench!(6);

    // From now on, we compress ternary matrices
    bench_w_prep!(7);
    bench_w_prep!(8);
    bench_w_prep!(9);

    #[bench]
    fn bench_blas_f32(bench: &mut Bencher) {
        let (a_i8, b_i8) = rand_vecs();
        let a_f: Vec<f32> = a_i8.iter().map(|e| *e as f32).collect();
        let b_f: Vec<f32> = b_i8.iter().map(|e| *e as f32).collect();

        let size_32 = SIZE.try_into().unwrap();
        let mut c = vec![0_f32; SIZE * SIZE];

        bench.iter(|| {
            black_box(unsafe {
                sgemm(
                    b'N', b'N', size_32, size_32, size_32, 1.0, &a_f, size_32, &b_f, size_32, 1.0,
                    &mut c, size_32,
                )
            });
        });
    }

    #[bench]
    fn bench_ndarray(bench: &mut Bencher) {
        let (a, b) = rand_vecs();

        let array_a =
            Array2::from_shape_vec((SIZE, SIZE), a.iter().map(|e| *e as f32).collect()).unwrap();
        let array_b =
            Array2::from_shape_vec((SIZE, SIZE), b.iter().map(|e| *e as f32).collect()).unwrap();

        bench.iter(|| {
            black_box(array_a.dot(&array_b));
        });
    }
}
