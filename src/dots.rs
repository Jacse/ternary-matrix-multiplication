use std::{
    arch::aarch64::{uint8x16_t, uint8x8_t, vcombine_u8, vtstq_u8},
    simd::{num::SimdUint, Simd},
};

pub unsafe fn dot_shifted_mask_multiplication(a: &[i8], b_vals: u16, b_signs: u16) -> Vec<i8> {
    let mut output = Simd::from_slice(&a);

    let negatives_mask = Simd::from_array([
        (b_signs & 0b1000000000000000) >> 15,
        (b_signs & 0b0100000000000000) >> 14,
        (b_signs & 0b0010000000000000) >> 13,
        (b_signs & 0b0001000000000000) >> 12,
        (b_signs & 0b0000100000000000) >> 11,
        (b_signs & 0b0000010000000000) >> 10,
        (b_signs & 0b0000001000000000) >> 9,
        (b_signs & 0b0000000100000000) >> 8,
        (b_signs & 0b0000000010000000) >> 7,
        (b_signs & 0b0000000001000000) >> 6,
        (b_signs & 0b0000000000100000) >> 5,
        (b_signs & 0b0000000000010000) >> 4,
        (b_signs & 0b0000000000001000) >> 3,
        (b_signs & 0b0000000000000100) >> 2,
        (b_signs & 0b0000000000000010) >> 1,
        (b_signs & 0b0000000000000001),
    ])
    .cast::<i8>();

    output = (output ^ -negatives_mask) + negatives_mask;

    let ones_mask = Simd::from_array([
        (b_vals & 0b1000000000000000) >> 15,
        (b_vals & 0b0100000000000000) >> 14,
        (b_vals & 0b0010000000000000) >> 13,
        (b_vals & 0b0001000000000000) >> 12,
        (b_vals & 0b0000100000000000) >> 11,
        (b_vals & 0b0000010000000000) >> 10,
        (b_vals & 0b0000001000000000) >> 9,
        (b_vals & 0b0000000100000000) >> 8,
        (b_vals & 0b0000000010000000) >> 7,
        (b_vals & 0b0000000001000000) >> 6,
        (b_vals & 0b0000000000100000) >> 5,
        (b_vals & 0b0000000000010000) >> 4,
        (b_vals & 0b0000000000001000) >> 3,
        (b_vals & 0b0000000000000100) >> 2,
        (b_vals & 0b0000000000000010) >> 1,
        (b_vals & 0b0000000000000001),
    ])
    .cast::<i8>();

    output = output * ones_mask;

    output.to_array().to_vec()
}

pub unsafe fn dot_masks(a: &[i8], b_vals: u16, b_signs: u16) -> Vec<i8> {
    let simd = Simd::from_slice(&a);
    let negatives = -simd;
    let mut output = Simd::splat(0);

    let shifter = uint8x16_t::from(Simd::from_array([
        0b10000000, 0b01000000, 0b00100000, 0b00010000, 0b00001000, 0b00000100, 0b00000010,
        0b00000001, 0b10000000, 0b01000000, 0b00100000, 0b00010000, 0b00001000, 0b00000100,
        0b00000010, 0b00000001,
    ]));

    let b_signs_first = uint8x8_t::from(Simd::splat((b_signs >> 8) as u8));
    let b_signs_second = uint8x8_t::from(Simd::splat(b_signs as u8));
    let b_signs_vec = vcombine_u8(b_signs_first, b_signs_second);

    let negatives_mask = Simd::from(vtstq_u8(b_signs_vec, shifter)).cast::<i8>();

    output |= negatives & negatives_mask;

    let b_vals_first = uint8x8_t::from(Simd::splat((b_vals >> 8) as u8));
    let b_vals_second = uint8x8_t::from(Simd::splat(b_vals as u8));
    let b_vals_vec = vcombine_u8(b_vals_first, b_vals_second);

    let ones_mask = Simd::from(vtstq_u8(b_vals_vec, shifter)).cast::<i8>();

    output |= simd & (ones_mask ^ negatives_mask);

    output.to_array().to_vec()
}

pub fn dot_naive(a: &[i8], b: &[i8]) -> Vec<i8> {
    let simd_a: Simd<i8, 16> = Simd::from_slice(&a);
    let simd_b = Simd::from_slice(&b);

    (simd_a * simd_b).to_array().to_vec()
}

#[cfg(test)]
mod tests {
    use crate::dots::{dot_masks, dot_naive, dot_shifted_mask_multiplication};

    #[test]
    fn test_dot_naive() {
        let a = [100, -50, 50, 75, -74, 25, 25, 0].repeat(2);
        let b = [-1, -1, -1, -1, 1, 0, -1, 1].repeat(2);

        let res = dot_naive(&a, &b);

        assert_eq!(res, [-100, 50, -50, -75, -74, 0, -25, 0].repeat(2));
    }

    #[test]
    fn test_dot_masks() {
        unsafe {
            let a = [100, -50, 50, 75, -74, 25, 25, 0].repeat(2);
            let b_vals = 0b1111101111111011;
            let b_signs = 0b1111001011110010;

            let res = dot_masks(&a, b_vals, b_signs);

            assert_eq!(res, [-100, 50, -50, -75, -74, 0, -25, 0].repeat(2));
        }
    }

    #[test]
    fn test_dot_shifted_mask_multiplication() {
        unsafe {
            let a = [100, -50, 50, 75, -74, 25, 25, 0].repeat(2);
            let b_vals = 0b1111101111111011;
            let b_signs = 0b1111001011110010;

            let res = dot_shifted_mask_multiplication(&a, b_vals, b_signs);

            assert_eq!(res, [-100, 50, -50, -75, -74, 0, -25, 0].repeat(2));
        }
    }
}
