use half::f16;

use crate::format::TernaryMatrix;

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

pub fn layer_norm(input: &[f32], rows: usize, width: usize, params: &[u8]) -> Vec<f32> {
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
            let gamma = read_f16(params, column * 2);
            let beta = read_f16(params, (width + column) * 2);
            output[row * width + column] = (values[column] - mean) * inverse_std * gamma + beta;
        }
    }
    output
}

pub fn ternary_linear(
    bytes: &[u8],
    matrix: &TernaryMatrix,
    input: &[f32],
    rows: usize,
) -> Vec<f32> {
    let scale = read_f32(bytes, matrix.scale_offset);
    let mut output = vec![0.0; rows * matrix.rows];
    for batch_row in 0..rows {
        for output_column in 0..matrix.rows {
            let weight_base = output_column * matrix.columns;
            let mut accumulator = 0.0;
            for input_column in 0..matrix.columns {
                let weight_index = weight_base + input_column;
                let packed = bytes[matrix.packed_offset + weight_index / 4];
                let code = (packed >> (2 * (weight_index % 4))) & 0b11;
                let input_value = input[batch_row * matrix.columns + input_column];
                accumulator += match code {
                    1 => input_value,
                    2 => -input_value,
                    _ => 0.0,
                };
            }
            output[batch_row * matrix.rows + output_column] = accumulator * scale;
        }
    }
    output
}

pub fn gelu_tanh(value: f32) -> f32 {
    const COEFFICIENT: f32 = 0.797_884_6;
    0.5 * value * (1.0 + (COEFFICIENT * (value + 0.044_715 * value.powi(3))).tanh())
}
