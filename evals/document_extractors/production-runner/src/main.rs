use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use serde::Serialize;

#[derive(Serialize)]
struct Segment {
    text: String,
    heading: String,
    page: Option<u32>,
}

#[derive(Serialize)]
struct Output {
    candidate: &'static str,
    format: String,
    status: &'static str,
    text: String,
    segments: Vec<Segment>,
    elapsed_ns: u128,
    error: Option<String>,
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: document-extractor-production-runner FORMAT INPUT");
        return ExitCode::from(2);
    }
    let format = args[1].clone();
    let path = Path::new(&args[2]);
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("could not read {}: {error}", path.display());
            return ExitCode::from(2);
        }
    };
    let started = Instant::now();
    let result =
        lode_core::ingestion::extract::extract_document(&bytes, &format_from_name(&format));
    let elapsed_ns = started.elapsed().as_nanos();
    let output = match result {
        Ok(Some(segments)) => Output {
            candidate: "lode_core",
            format,
            status: "ok",
            text: segments
                .iter()
                .map(|segment| segment.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n"),
            segments: segments
                .into_iter()
                .map(|segment| Segment {
                    text: segment.text,
                    heading: segment.heading,
                    page: segment.page,
                })
                .collect(),
            elapsed_ns,
            error: None,
        },
        Ok(None) => Output {
            candidate: "lode_core",
            format,
            status: "error",
            text: String::new(),
            segments: Vec::new(),
            elapsed_ns,
            error: Some("unsupported extraction format".to_owned()),
        },
        Err(error) => Output {
            candidate: "lode_core",
            format,
            status: "error",
            text: String::new(),
            segments: Vec::new(),
            elapsed_ns,
            error: Some(error.to_string()),
        },
    };
    println!(
        "{}",
        serde_json::to_string(&output).expect("output is serializable")
    );
    ExitCode::SUCCESS
}

fn format_from_name(format: &str) -> String {
    if format.starts_with('.') {
        format.to_owned()
    } else {
        format!(".{format}")
    }
}
