use serde::Serialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const MAGIC: &[u8; 4] = b"BEM2";
pub const FORMAT_VERSION: u16 = 2;
pub const HEADER_SIZE: usize = 64;
pub const SHA256_SIZE: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[repr(u8)]
pub enum EmbeddingFormat {
    Int4 = 1,
    Int8 = 2,
    Fp16 = 3,
}

impl TryFrom<u8> for EmbeddingFormat {
    type Error = FormatError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Int4),
            2 => Ok(Self::Int8),
            3 => Ok(Self::Fp16),
            other => Err(FormatError::UnknownEmbeddingFormat(other)),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelHeader {
    pub vocab_size: usize,
    pub max_sequence_length: usize,
    pub embedding_dim: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_repeats: usize,
    pub ffn_dim: usize,
    pub output_dim: usize,
    pub embedding_format: EmbeddingFormat,
    pub padding_idx: usize,
    pub matryoshka_dims: Vec<usize>,
    pub body_length: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Section {
    pub name: &'static str,
    pub offset: usize,
    pub length: usize,
}

#[derive(Debug, Clone)]
pub struct TernaryMatrix {
    pub packed_offset: usize,
    pub packed_length: usize,
    pub scale_offset: usize,
    pub rows: usize,
    pub columns: usize,
}

#[derive(Debug, Clone)]
pub struct ModelLayout {
    pub sections: Vec<Section>,
    pub token_embedding_offset: usize,
    pub token_embedding_scales_offset: Option<usize>,
    pub position_embedding_offset: usize,
    pub embedding_norm_offset: usize,
    pub embedding_projection: TernaryMatrix,
    pub attention_norm_offset: usize,
    pub attention_qkv: TernaryMatrix,
    pub attention_output: TernaryMatrix,
    pub ffn_norm_offset: usize,
    pub ffn_up: TernaryMatrix,
    pub ffn_down: TernaryMatrix,
    pub final_norm_offset: usize,
    pub output_projection_offset: usize,
}

#[derive(Debug)]
pub struct ParsedModel<'a> {
    pub header: ModelHeader,
    pub layout: ModelLayout,
    pub bytes: &'a [u8],
}

#[derive(Debug, Error)]
pub enum FormatError {
    #[error("model file is truncated")]
    Truncated,
    #[error("bad BEM2 magic")]
    BadMagic,
    #[error("unsupported model format version {0}")]
    UnsupportedVersion(u16),
    #[error("invalid header size {0}")]
    InvalidHeaderSize(u16),
    #[error("unknown embedding format {0}")]
    UnknownEmbeddingFormat(u8),
    #[error("invalid model shape: {0}")]
    InvalidShape(&'static str),
    #[error("model size does not match its header")]
    SizeMismatch,
    #[error("model checksum does not match")]
    ChecksumMismatch,
    #[error("section layout does not consume the complete body")]
    LayoutMismatch,
}

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

impl<'a> ParsedModel<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, FormatError> {
        if bytes.len() < HEADER_SIZE + SHA256_SIZE {
            return Err(FormatError::Truncated);
        }
        if &bytes[..4] != MAGIC {
            return Err(FormatError::BadMagic);
        }
        let version = u16_at(bytes, 4);
        if version != FORMAT_VERSION {
            return Err(FormatError::UnsupportedVersion(version));
        }
        let header_size = u16_at(bytes, 6);
        if usize::from(header_size) != HEADER_SIZE {
            return Err(FormatError::InvalidHeaderSize(header_size));
        }
        let body_length = u32_at(bytes, 28) as usize;
        if bytes.len() != HEADER_SIZE + body_length + SHA256_SIZE {
            return Err(FormatError::SizeMismatch);
        }
        let expected = &bytes[bytes.len() - SHA256_SIZE..];
        let actual = Sha256::digest(&bytes[..bytes.len() - SHA256_SIZE]);
        if actual.as_slice() != expected {
            return Err(FormatError::ChecksumMismatch);
        }

        let matryoshka_count = usize::from(bytes[34]);
        if matryoshka_count == 0 || matryoshka_count > 8 {
            return Err(FormatError::InvalidShape("matryoshka dimension count"));
        }
        let matryoshka_dims = (0..matryoshka_count)
            .map(|index| usize::from(u16_at(bytes, 35 + 2 * index)))
            .collect::<Vec<_>>();
        let header = ModelHeader {
            vocab_size: u32_at(bytes, 8) as usize,
            max_sequence_length: usize::from(u16_at(bytes, 12)),
            embedding_dim: usize::from(u16_at(bytes, 14)),
            hidden_dim: usize::from(u16_at(bytes, 16)),
            num_heads: usize::from(u16_at(bytes, 18)),
            num_repeats: usize::from(u16_at(bytes, 20)),
            ffn_dim: usize::from(u16_at(bytes, 22)),
            output_dim: usize::from(u16_at(bytes, 24)),
            embedding_format: EmbeddingFormat::try_from(bytes[26])?,
            body_length,
            padding_idx: usize::from(u16_at(bytes, 32)),
            matryoshka_dims,
        };
        header.validate()?;
        let layout = ModelLayout::from_header(&header)?;
        Ok(Self {
            header,
            layout,
            bytes,
        })
    }
}

impl ModelHeader {
    fn validate(&self) -> Result<(), FormatError> {
        if self.hidden_dim == 0
            || self.num_heads == 0
            || !self.hidden_dim.is_multiple_of(self.num_heads)
        {
            return Err(FormatError::InvalidShape("attention heads"));
        }
        if !self.embedding_dim.is_multiple_of(4)
            || !self.hidden_dim.is_multiple_of(4)
            || !self.ffn_dim.is_multiple_of(4)
        {
            return Err(FormatError::InvalidShape("packed dimensions"));
        }
        if self.padding_idx >= self.vocab_size {
            return Err(FormatError::InvalidShape("padding index"));
        }
        if self.matryoshka_dims.last().copied() != Some(self.output_dim) {
            return Err(FormatError::InvalidShape("matryoshka output"));
        }
        Ok(())
    }
}

impl ModelLayout {
    fn from_header(header: &ModelHeader) -> Result<Self, FormatError> {
        let mut offset = HEADER_SIZE;
        let mut sections = Vec::with_capacity(12);
        let mut take = |name: &'static str, length: usize| {
            let start = offset;
            offset += length;
            sections.push(Section {
                name,
                offset: start,
                length,
            });
            start
        };
        let embedding_values = match header.embedding_format {
            EmbeddingFormat::Int4 => header.vocab_size * (header.embedding_dim / 2),
            EmbeddingFormat::Int8 => header.vocab_size * header.embedding_dim,
            EmbeddingFormat::Fp16 => header.vocab_size * header.embedding_dim * 2,
        };
        let embedding_scales = match header.embedding_format {
            EmbeddingFormat::Int4 | EmbeddingFormat::Int8 => header.vocab_size * 2,
            EmbeddingFormat::Fp16 => 0,
        };
        let token_embedding_offset = take("token_embedding", embedding_values + embedding_scales);
        let token_embedding_scales_offset =
            (embedding_scales > 0).then_some(token_embedding_offset + embedding_values);
        let position_embedding_offset = take(
            "position_embedding",
            header.max_sequence_length * header.embedding_dim * 2,
        );
        let embedding_norm_offset = take("embedding_norm", header.embedding_dim * 4);

        fn take_ternary(
            take: &mut impl FnMut(&'static str, usize) -> usize,
            name: &'static str,
            rows: usize,
            columns: usize,
        ) -> TernaryMatrix {
            let packed_length = rows * columns / 4;
            let packed_offset = take(name, packed_length + 4);
            TernaryMatrix {
                packed_offset,
                packed_length,
                scale_offset: packed_offset + packed_length,
                rows,
                columns,
            }
        }

        let embedding_projection = take_ternary(
            &mut take,
            "embedding_projection",
            header.hidden_dim,
            header.embedding_dim,
        );
        let attention_norm_offset = take("attention_norm", header.hidden_dim * 4);
        let attention_qkv = take_ternary(
            &mut take,
            "attention_qkv",
            3 * header.hidden_dim,
            header.hidden_dim,
        );
        let attention_output = take_ternary(
            &mut take,
            "attention_output",
            header.hidden_dim,
            header.hidden_dim,
        );
        let ffn_norm_offset = take("ffn_norm", header.hidden_dim * 4);
        let ffn_up = take_ternary(&mut take, "ffn_up", header.ffn_dim, header.hidden_dim);
        let ffn_down = take_ternary(&mut take, "ffn_down", header.hidden_dim, header.ffn_dim);
        let final_norm_offset = take("final_norm", header.hidden_dim * 4);
        let output_projection_offset = take(
            "output_projection",
            (header.output_dim * header.hidden_dim + header.output_dim) * 2,
        );
        if offset != HEADER_SIZE + header.body_length {
            return Err(FormatError::LayoutMismatch);
        }
        Ok(Self {
            sections,
            token_embedding_offset,
            token_embedding_scales_offset,
            position_embedding_offset,
            embedding_norm_offset,
            embedding_projection,
            attention_norm_offset,
            attention_qkv,
            attention_output,
            ffn_norm_offset,
            ffn_up,
            ffn_down,
            final_norm_offset,
            output_projection_offset,
        })
    }
}
