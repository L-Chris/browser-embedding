use std::{env, fs, process};

use browser_embedding_core::Encoder;

fn parse_list<T: std::str::FromStr>(value: &str, label: &str) -> Vec<T> {
    value
        .split(',')
        .map(|item| {
            item.parse::<T>().unwrap_or_else(|_| {
                eprintln!("invalid {label} value: {item}");
                process::exit(2);
            })
        })
        .collect()
}

fn main() {
    let arguments = env::args().collect::<Vec<_>>();
    if arguments.len() != 5 {
        eprintln!("usage: embed <model.bem> <ids-csv> <mask-csv> <dimension>");
        process::exit(2);
    }
    let bytes = fs::read(&arguments[1]).expect("read model");
    let input_ids = parse_list::<u32>(&arguments[2], "token id");
    let attention_mask = parse_list::<u8>(&arguments[3], "mask");
    let dimension = arguments[4].parse::<usize>().expect("dimension");
    let encoder = Encoder::from_bytes(&bytes).expect("parse model");
    let output = encoder
        .encode_tokens(&input_ids, &attention_mask, dimension)
        .expect("run encoder");
    println!(
        "{}",
        output
            .iter()
            .map(|value| format!("{value:.9}"))
            .collect::<Vec<_>>()
            .join(",")
    );
}
