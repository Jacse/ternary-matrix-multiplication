use crate::constants::SIZE;

// all matrices in col-major order with stride SIZE
fn at(r: usize, c: usize) -> usize {
    r + SIZE * c
}

fn dot(k: usize, a: &[i8], b: &[i8], c: &mut i8) {
    for ki in 0..k {
        // Since a is a row we need to use col stride to loop through contents
        *c += a[SIZE * ki] * b[ki];
    }
}

pub fn matmul1(a: &[i8], b: &[i8]) -> Vec<i8> {
    let (m, k, n) = (SIZE, SIZE, SIZE);
    let mut c = vec![0; m * n];

    // Loop over cols in c
    for ni in 0..n {
        // Loop over rows in c
        for mi in 0..m {
            dot(k, &a[at(mi, 0)..], &b[at(0, ni)..], &mut c[at(mi, ni)]);
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use crate::test_util::test_util::test_matmul;

    use super::matmul1;

    #[test]
    fn test() {
        test_matmul(matmul1)
    }
}
