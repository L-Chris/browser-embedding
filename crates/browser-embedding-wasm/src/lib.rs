use browser_embedding_core::Encoder;
use tokenizers::{Tokenizer, TruncationParams};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct BrowserModel {
    model_bytes: Vec<u8>,
    tokenizer: Tokenizer,
}

#[wasm_bindgen]
impl BrowserModel {
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: &[u8], tokenizer_json: &[u8]) -> Result<BrowserModel, JsError> {
        let encoder =
            Encoder::from_bytes(model_bytes).map_err(|error| JsError::new(&error.to_string()))?;
        let mut tokenizer = Tokenizer::from_bytes(tokenizer_json)
            .map_err(|error| JsError::new(&error.to_string()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: encoder.header().max_sequence_length,
                ..TruncationParams::default()
            }))
            .map_err(|error| JsError::new(&error.to_string()))?;
        Ok(Self {
            model_bytes: model_bytes.to_vec(),
            tokenizer,
        })
    }

    pub fn model_info(&self) -> Result<String, JsError> {
        let encoder = Encoder::from_bytes(&self.model_bytes)
            .map_err(|error| JsError::new(&error.to_string()))?;
        let header = encoder.header();
        Ok(format!(
            "BEM2 vocab={} hidden={} output={} repeats={}",
            header.vocab_size, header.hidden_dim, header.output_dim, header.num_repeats
        ))
    }

    pub fn embed(&self, text: &str, role: &str, dimension: usize) -> Result<Vec<f32>, JsError> {
        let prefix = match role {
            "query" => "[QRY] ",
            "document" => "[DOC] ",
            _ => return Err(JsError::new("role must be 'query' or 'document'")),
        };
        let encoder = Encoder::from_bytes(&self.model_bytes)
            .map_err(|error| JsError::new(&error.to_string()))?;
        let encoded = self
            .tokenizer
            .encode(format!("{prefix}{text}"), true)
            .map_err(|error| JsError::new(&error.to_string()))?;
        let length = encoded
            .get_ids()
            .len()
            .min(encoder.header().max_sequence_length);
        let input_ids = &encoded.get_ids()[..length];
        let attention_mask = vec![1_u8; length];
        encoder
            .encode_tokens(input_ids, &attention_mask, dimension)
            .map_err(|error| JsError::new(&error.to_string()))
    }
}
