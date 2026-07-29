use browser_embedding_core::Encoder;
use tokenizers::{Tokenizer, TruncationParams};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct BrowserModel {
    encoder: Encoder,
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
        Ok(Self { encoder, tokenizer })
    }

    pub fn model_info(&self) -> String {
        let header = self.encoder.header();
        format!(
            "BEM2 vocab={} hidden={} output={} repeats={}",
            header.vocab_size, header.hidden_dim, header.output_dim, header.num_repeats
        )
    }

    pub fn embed(&self, text: &str, role: &str, dimension: usize) -> Result<Vec<f32>, JsError> {
        let prefix = match role {
            "query" => "[QRY] ",
            "document" => "[DOC] ",
            _ => return Err(JsError::new("role must be 'query' or 'document'")),
        };
        let encoded = self
            .tokenizer
            .encode(format!("{prefix}{text}"), true)
            .map_err(|error| JsError::new(&error.to_string()))?;
        let length = encoded
            .get_ids()
            .len()
            .min(self.encoder.header().max_sequence_length);
        let input_ids = &encoded.get_ids()[..length];
        let attention_mask = vec![1_u8; length];
        self.encoder
            .encode_tokens(input_ids, &attention_mask, dimension)
            .map_err(|error| JsError::new(&error.to_string()))
    }
}
