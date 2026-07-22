//! Portable BEM2 parser and reference inference graph.
//!
//! The crate deliberately has no JavaScript or filesystem dependency. Native
//! tests and the WASM binding execute exactly the same byte reader and graph.

mod format;
mod inference;
mod kernels;

pub use format::{EmbeddingFormat, FormatError, ModelHeader, ModelLayout, ParsedModel, Section};
pub use inference::Encoder;
