use crate::format::{EmbeddingFormat, FormatError, ModelHeader, ModelLayout, ParsedModel};
use crate::kernels::{
    LayerNormWeights, RuntimeTernaryMatrix, gelu_tanh, layer_norm, read_f16, ternary_linear,
};

#[derive(Debug)]
struct RuntimeWeights {
    embedding_scales: Option<Vec<f32>>,
    position_embedding: Vec<f32>,
    embedding_norm: LayerNormWeights,
    embedding_projection: RuntimeTernaryMatrix,
    attention_norm: LayerNormWeights,
    attention_qkv: RuntimeTernaryMatrix,
    attention_output: RuntimeTernaryMatrix,
    ffn_norm: LayerNormWeights,
    ffn_up: RuntimeTernaryMatrix,
    ffn_down: RuntimeTernaryMatrix,
    final_norm: LayerNormWeights,
    output_projection: Vec<f32>,
    output_bias: Vec<f32>,
}

impl RuntimeWeights {
    fn decode(bytes: &[u8], header: &ModelHeader, layout: &ModelLayout) -> Self {
        let embedding_scales = layout.token_embedding_scales_offset.map(|offset| {
            (0..header.vocab_size)
                .map(|row| read_f16(bytes, offset + row * 2))
                .collect()
        });
        let position_embedding = (0..header.max_sequence_length * header.embedding_dim)
            .map(|index| read_f16(bytes, layout.position_embedding_offset + index * 2))
            .collect();
        let output_projection = (0..header.output_dim * header.hidden_dim)
            .map(|index| read_f16(bytes, layout.output_projection_offset + index * 2))
            .collect();
        let bias_offset =
            layout.output_projection_offset + header.output_dim * header.hidden_dim * 2;
        let output_bias = (0..header.output_dim)
            .map(|row| read_f16(bytes, bias_offset + row * 2))
            .collect();
        Self {
            embedding_scales,
            position_embedding,
            embedding_norm: LayerNormWeights::decode(
                bytes,
                layout.embedding_norm_offset,
                header.embedding_dim,
            ),
            embedding_projection: RuntimeTernaryMatrix::decode(bytes, &layout.embedding_projection),
            attention_norm: LayerNormWeights::decode(
                bytes,
                layout.attention_norm_offset,
                header.hidden_dim,
            ),
            attention_qkv: RuntimeTernaryMatrix::decode(bytes, &layout.attention_qkv),
            attention_output: RuntimeTernaryMatrix::decode(bytes, &layout.attention_output),
            ffn_norm: LayerNormWeights::decode(bytes, layout.ffn_norm_offset, header.hidden_dim),
            ffn_up: RuntimeTernaryMatrix::decode(bytes, &layout.ffn_up),
            ffn_down: RuntimeTernaryMatrix::decode(bytes, &layout.ffn_down),
            final_norm: LayerNormWeights::decode(
                bytes,
                layout.final_norm_offset,
                header.hidden_dim,
            ),
            output_projection,
            output_bias,
        }
    }
}

pub struct Encoder {
    bytes: Vec<u8>,
    header: ModelHeader,
    layout: ModelLayout,
    weights: RuntimeWeights,
}

impl Encoder {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FormatError> {
        let model = ParsedModel::parse(bytes)?;
        let weights = RuntimeWeights::decode(bytes, &model.header, &model.layout);
        Ok(Self {
            bytes: bytes.to_vec(),
            header: model.header,
            layout: model.layout,
            weights,
        })
    }

    pub fn header(&self) -> &crate::ModelHeader {
        &self.header
    }

    pub fn encode_tokens(
        &self,
        input_ids: &[u32],
        attention_mask: &[u8],
        output_dimension: usize,
    ) -> Result<Vec<f32>, FormatError> {
        let header = &self.header;
        if input_ids.is_empty()
            || input_ids.len() != attention_mask.len()
            || input_ids.len() > header.max_sequence_length
        {
            return Err(FormatError::InvalidShape("token input"));
        }
        if !header.matryoshka_dims.contains(&output_dimension) {
            return Err(FormatError::InvalidShape("output dimension"));
        }
        if input_ids.iter().any(|id| *id as usize >= header.vocab_size) {
            return Err(FormatError::InvalidShape("token id"));
        }
        if !attention_mask.contains(&1) {
            return Err(FormatError::InvalidShape("attention mask"));
        }
        let sequence = input_ids.len();
        let bytes = &self.bytes;
        let layout = &self.layout;
        let weights = &self.weights;
        let mut embedding = vec![0.0; sequence * header.embedding_dim];
        for (position, token_id) in input_ids.iter().enumerate() {
            let token = *token_id as usize;
            let scale = weights
                .embedding_scales
                .as_ref()
                .map(|scales| scales[token])
                .unwrap_or(1.0);
            for column in 0..header.embedding_dim {
                let token_value = match header.embedding_format {
                    EmbeddingFormat::Int4 => {
                        let row_bytes = header.embedding_dim / 2;
                        let packed =
                            bytes[layout.token_embedding_offset + token * row_bytes + column / 2];
                        let nibble = if column % 2 == 0 {
                            packed & 0x0f
                        } else {
                            packed >> 4
                        };
                        let signed = if nibble < 8 {
                            nibble as i8
                        } else {
                            nibble as i8 - 16
                        };
                        signed as f32 * scale
                    }
                    EmbeddingFormat::Int8 => {
                        let offset =
                            layout.token_embedding_offset + token * header.embedding_dim + column;
                        bytes[offset] as i8 as f32 * scale
                    }
                    EmbeddingFormat::Fp16 => {
                        let offset = layout.token_embedding_offset
                            + (token * header.embedding_dim + column) * 2;
                        read_f16(bytes, offset)
                    }
                };
                embedding[position * header.embedding_dim + column] = token_value
                    + weights.position_embedding[position * header.embedding_dim + column];
            }
        }
        let embedding_normalized = layer_norm(
            &embedding,
            sequence,
            header.embedding_dim,
            &weights.embedding_norm,
        );
        let mut hidden = ternary_linear(
            &weights.embedding_projection,
            &embedding_normalized,
            sequence,
        );
        for _ in 0..header.num_repeats {
            hidden = self.shared_block(hidden, attention_mask);
        }
        hidden = layer_norm(&hidden, sequence, header.hidden_dim, &weights.final_norm);
        let real_tokens = attention_mask.iter().filter(|value| **value != 0).count() as f32;
        let mut pooled = vec![0.0; header.hidden_dim];
        for row in 0..sequence {
            if attention_mask[row] == 0 {
                continue;
            }
            for column in 0..header.hidden_dim {
                pooled[column] += hidden[row * header.hidden_dim + column] / real_tokens;
            }
        }
        let mut output = vec![0.0; output_dimension];
        for (row, output_value) in output.iter_mut().enumerate() {
            let mut value = weights.output_bias[row];
            for (column, pooled_value) in pooled.iter().enumerate() {
                value += pooled_value * weights.output_projection[row * header.hidden_dim + column];
            }
            *output_value = value;
        }
        let norm = output
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
            .max(1e-12);
        output.iter_mut().for_each(|value| *value /= norm);
        Ok(output)
    }

    fn shared_block(&self, hidden: Vec<f32>, attention_mask: &[u8]) -> Vec<f32> {
        let header = &self.header;
        let weights = &self.weights;
        let sequence = attention_mask.len();
        let normalized = layer_norm(
            &hidden,
            sequence,
            header.hidden_dim,
            &weights.attention_norm,
        );
        let qkv = ternary_linear(&weights.attention_qkv, &normalized, sequence);
        let attention = self.attention(&qkv, attention_mask);
        let attention_output = ternary_linear(&weights.attention_output, &attention, sequence);
        let residual = hidden
            .iter()
            .zip(attention_output)
            .map(|(left, right)| left + right)
            .collect::<Vec<_>>();
        let ffn_input = layer_norm(&residual, sequence, header.hidden_dim, &weights.ffn_norm);
        let mut expanded = ternary_linear(&weights.ffn_up, &ffn_input, sequence);
        expanded
            .iter_mut()
            .for_each(|value| *value = gelu_tanh(*value));
        let contracted = ternary_linear(&weights.ffn_down, &expanded, sequence);
        residual
            .into_iter()
            .zip(contracted)
            .map(|(left, right)| left + right)
            .collect()
    }

    fn attention(&self, qkv: &[f32], attention_mask: &[u8]) -> Vec<f32> {
        let header = &self.header;
        let sequence = attention_mask.len();
        let head_dim = header.hidden_dim / header.num_heads;
        let mut output = vec![0.0; sequence * header.hidden_dim];
        let scale = (head_dim as f32).sqrt();
        for head in 0..header.num_heads {
            for query_position in 0..sequence {
                let mut scores = vec![f32::NEG_INFINITY; sequence];
                for key_position in 0..sequence {
                    if attention_mask[key_position] == 0 {
                        continue;
                    }
                    let mut dot = 0.0;
                    for component in 0..head_dim {
                        let q_index =
                            query_position * 3 * header.hidden_dim + head * head_dim + component;
                        let k_index = key_position * 3 * header.hidden_dim
                            + header.hidden_dim
                            + head * head_dim
                            + component;
                        dot += qkv[q_index] * qkv[k_index];
                    }
                    scores[key_position] = dot / scale;
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denominator = scores
                    .iter_mut()
                    .map(|score| {
                        *score = (*score - maximum).exp();
                        *score
                    })
                    .sum::<f32>();
                for component in 0..head_dim {
                    let mut value = 0.0;
                    for (key_position, probability) in scores.iter().enumerate() {
                        let v_index = key_position * 3 * header.hidden_dim
                            + 2 * header.hidden_dim
                            + head * head_dim
                            + component;
                        value += probability / denominator * qkv[v_index];
                    }
                    output[query_position * header.hidden_dim + head * head_dim + component] =
                        value;
                }
            }
        }
        output
    }
}
