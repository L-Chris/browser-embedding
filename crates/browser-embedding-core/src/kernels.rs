use half::f16;

use crate::format::TernaryMatrix;

#[derive(Debug)]
pub struct RuntimeTernaryMatrix {
    pub weights: Vec<f32>,
    pub scale: f32,
    pub rows: usize,
    pub columns: usize,
}

impl RuntimeTernaryMatrix {
    pub fn decode(bytes: &[u8], matrix: &TernaryMatrix) -> Self {
        let mut weights = Vec::with_capacity(matrix.rows * matrix.columns);
        for weight_index in 0..matrix.rows * matrix.columns {
            let packed = bytes[matrix.packed_offset + weight_index / 4];
            let code = (packed >> (2 * (weight_index % 4))) & 0b11;
            weights.push(match code {
                1 => 1.0,
                2 => -1.0,
                _ => 0.0,
            });
        }
        Self {
            weights,
            scale: read_f32(bytes, matrix.scale_offset),
            rows: matrix.rows,
            columns: matrix.columns,
        }
    }
}

#[derive(Debug)]
pub struct LayerNormWeights {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
}

impl LayerNormWeights {
    pub fn decode(bytes: &[u8], offset: usize, width: usize) -> Self {
        Self {
            gamma: (0..width)
                .map(|column| read_f16(bytes, offset + column * 2))
                .collect(),
            beta: (0..width)
                .map(|column| read_f16(bytes, offset + (width + column) * 2))
                .collect(),
        }
    }
}

pub fn read_f16(bytes: &[u8], offset: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
}

pub fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

pub fn layer_norm(input: &[f32], rows: usize, width: usize, params: &LayerNormWeights) -> Vec<f32> {
    debug_assert_eq!(params.gamma.len(), width);
    debug_assert_eq!(params.beta.len(), width);
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let values = &input[row * width..(row + 1) * width];
        let mean = values.iter().sum::<f32>() / width as f32;
        let variance = values
            .iter()
            .map(|value| {
                let centered = value - mean;
                centered * centered
            })
            .sum::<f32>()
            / width as f32;
        let inverse_std = 1.0 / (variance + 1e-5).sqrt();
        for column in 0..width {
            output[row * width + column] =
                (values[column] - mean) * inverse_std * params.gamma[column] + params.beta[column];
        }
    }
    output
}

pub fn ternary_linear(matrix: &RuntimeTernaryMatrix, input: &[f32], rows: usize) -> Vec<f32> {
    let mut output = vec![0.0; rows * matrix.rows];
    for batch_row in 0..rows {
        let input_row = &input[batch_row * matrix.columns..(batch_row + 1) * matrix.columns];
        for output_column in 0..matrix.rows {
            let weight_base = output_column * matrix.columns;
            let weights = &matrix.weights[weight_base..weight_base + matrix.columns];
            output[batch_row * matrix.rows + output_column] =
                dot_product(input_row, weights) * matrix.scale;
        }
    }
    output
}

#[cfg(target_arch = "wasm32")]
#[inline]
fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    use core::arch::wasm32::*;

    debug_assert_eq!(left.len(), right.len());
    let mut accumulator = f32x4_splat(0.0);
    let mut index = 0;
    while index + 4 <= left.len() {
        // SAFETY: the loop condition proves that both unaligned four-lane loads
        // remain within their slices. WASM v128 loads do not require alignment.
        let left_vector = unsafe { v128_load(left.as_ptr().add(index) as *const v128) };
        let right_vector = unsafe { v128_load(right.as_ptr().add(index) as *const v128) };
        accumulator = f32x4_add(accumulator, f32x4_mul(left_vector, right_vector));
        index += 4;
    }
    let mut lanes = [0.0_f32; 4];
    // SAFETY: `lanes` is exactly one v128 wide.
    unsafe { v128_store(lanes.as_mut_ptr() as *mut v128, accumulator) };
    let mut total = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    while index < left.len() {
        total += left[index] * right[index];
        index += 1;
    }
    total
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

pub fn gelu_tanh(value: f32) -> f32 {
    const COEFFICIENT: f32 = 0.797_884_6;
    0.5 * value * (1.0 + (COEFFICIENT * (value + 0.044_715 * value.powi(3))).tanh())
}
