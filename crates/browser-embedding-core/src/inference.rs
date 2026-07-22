use crate::format::{EmbeddingFormat, FormatError, ParsedModel};
use crate::kernels::{gelu_tanh, layer_norm, read_f16, ternary_linear};

pub struct Encoder<'a> {
    model: ParsedModel<'a>,
}

impl<'a> Encoder<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self, FormatError> {
        Ok(Self {
            model: ParsedModel::parse(bytes)?,
        })
    }

    pub fn header(&self) -> &crate::ModelHeader {
        &self.model.header
    }

    pub fn encode_tokens(
        &self,
        input_ids: &[u32],
        attention_mask: &[u8],
        output_dimension: usize,
    ) -> Result<Vec<f32>, FormatError> {
        let header = &self.model.header;
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
        let bytes = self.model.bytes;
        let layout = &self.model.layout;
        let mut embedding = vec![0.0; sequence * header.embedding_dim];
        for (position, token_id) in input_ids.iter().enumerate() {
            let token = *token_id as usize;
            let scale = layout
                .token_embedding_scales_offset
                .map(|offset| read_f16(bytes, offset + token * 2))
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
                let position_offset = layout.position_embedding_offset
                    + (position * header.embedding_dim + column) * 2;
                embedding[position * header.embedding_dim + column] =
                    token_value + read_f16(bytes, position_offset);
            }
        }
        let embedding_normalized = layer_norm(
            &embedding,
            sequence,
            header.embedding_dim,
            &bytes[layout.embedding_norm_offset..],
        );
        let mut hidden = ternary_linear(
            bytes,
            &layout.embedding_projection,
            &embedding_normalized,
            sequence,
        );
        for _ in 0..header.num_repeats {
            hidden = self.shared_block(hidden, attention_mask);
        }
        hidden = layer_norm(
            &hidden,
            sequence,
            header.hidden_dim,
            &bytes[layout.final_norm_offset..],
        );
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
        let projection = layout.output_projection_offset;
        let bias = projection + header.output_dim * header.hidden_dim * 2;
        let mut output = vec![0.0; output_dimension];
        for (row, output_value) in output.iter_mut().enumerate() {
            let mut value = read_f16(bytes, bias + row * 2);
            for (column, pooled_value) in pooled.iter().enumerate() {
                value += pooled_value
                    * read_f16(bytes, projection + (row * header.hidden_dim + column) * 2);
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
        let header = &self.model.header;
        let layout = &self.model.layout;
        let bytes = self.model.bytes;
        let sequence = attention_mask.len();
        let normalized = layer_norm(
            &hidden,
            sequence,
            header.hidden_dim,
            &bytes[layout.attention_norm_offset..],
        );
        let qkv = ternary_linear(bytes, &layout.attention_qkv, &normalized, sequence);
        let attention = self.attention(&qkv, attention_mask);
        let attention_output =
            ternary_linear(bytes, &layout.attention_output, &attention, sequence);
        let residual = hidden
            .iter()
            .zip(attention_output)
            .map(|(left, right)| left + right)
            .collect::<Vec<_>>();
        let ffn_input = layer_norm(
            &residual,
            sequence,
            header.hidden_dim,
            &bytes[layout.ffn_norm_offset..],
        );
        let mut expanded = ternary_linear(bytes, &layout.ffn_up, &ffn_input, sequence);
        expanded
            .iter_mut()
            .for_each(|value| *value = gelu_tanh(*value));
        let contracted = ternary_linear(bytes, &layout.ffn_down, &expanded, sequence);
        residual
            .into_iter()
            .zip(contracted)
            .map(|(left, right)| left + right)
            .collect()
    }

    fn attention(&self, qkv: &[f32], attention_mask: &[u8]) -> Vec<f32> {
        let header = &self.model.header;
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
