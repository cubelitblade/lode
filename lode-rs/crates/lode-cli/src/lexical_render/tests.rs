use super::*;

use lode_core::fts::MatchExpr;
use lode_core::index::records::{FileRecord, FileStatus};
use lode_core::index::store::Store;
use lode_core::ingestion::types::Chunk;
use lode_core::relpath::WorkspacePath;

fn matched_details() -> LexicalDetails {
    let directory = tempfile::tempdir().unwrap();
    let mut store =
        Store::open(&directory.path().join("index.db"), "test", 4, "unicode61").unwrap();
    store
        .replace_file(
            &FileRecord {
                path: WorkspacePath::from_posix("notes.txt"),
                digest: "blake3:document".into(),
                mtime: 1.0,
                size: 30,
                status: FileStatus::Fresh,
            },
            "text",
            &[Chunk {
                digest: "blake3:chunk".into(),
                text: "alpha beta alpha beta".into(),
                seq: 0,
                heading: String::new(),
                page: None,
                span: None,
            }],
            None,
        )
        .unwrap();
    let expression = "\"alpha beta\" OR \"alpha beta\" OR \"missing\" OR \"alph\"*";
    let diagnostic = store
        .explain_fts(&MatchExpr::Prebuilt(expression.into()), 1)
        .unwrap();
    LexicalDetails {
        status: LexicalDetailsStatus::Matched,
        in_candidate_pool: true,
        strategy: "unicode61".into(),
        tokenize: "unicode61".into(),
        match_expression: Some(diagnostic.match_expression),
        matched: diagnostic.matched,
    }
}

fn output(details: &LexicalDetails) -> Vec<String> {
    let mut lines = Vec::new();
    render(details, |_, text| lines.push(text.to_owned()));
    lines
}

#[test]
fn real_phrases_keep_multi_tokens_repeats_zero_branches_and_unknown_prefix_tokens() {
    let details = matched_details();
    let lines = output(&details);
    let phrases: Vec<_> = lines
        .iter()
        .filter(|line| line.contains("phrase #"))
        .collect();
    assert_eq!(phrases.len(), 4);
    for (index, line) in phrases.iter().enumerate() {
        assert!(line.contains(&format!("phrase #{index}:")));
    }
    for phrase in &phrases[..2] {
        assert!(phrase.contains(r#"query tokens=["alpha","beta"]"#));
        assert!(phrase.contains("hits=2"));
        assert!(
            phrase.contains("e-6"),
            "small BM25 contributions must remain visible: {phrase}"
        );
    }
    assert!(phrases[2].contains("hits=0"));
    let zero_score = phrases[2]
        .rsplit('=')
        .next()
        .unwrap()
        .parse::<f64>()
        .unwrap();
    assert!(zero_score.abs() < f64::EPSILON);
    assert!(
        lines
            .iter()
            .any(|line| line.contains(r#"document tokens=["alpha","beta"]"#))
    );
    assert!(lines.iter().any(|line| line.contains("token offset=2")));
    assert!(
        lines
            .iter()
            .any(|line| line.contains("document tokens=[null] (null = unknown document token)"))
    );
    assert!(
        lines
            .iter()
            .any(|line| line.contains("signed raw BM25 only"))
    );
    assert!(lines.iter().any(|line| line.contains("including RRF")));
    assert!(
        lines
            .iter()
            .any(|line| line.contains("not text byte offsets"))
    );
}

#[test]
fn external_metadata_and_tokens_cannot_inject_terminal_controls_or_extra_lines() {
    let mut details = matched_details();
    details.strategy = "strategy\u{1b}[31m\n".into();
    details.tokenize = "tokenizer\u{0}\t".into();
    details.match_expression = Some("\"alpha\"\nOR\t\"beta\"\r\u{1b}".into());
    let matched = details.matched.as_mut().unwrap();
    matched.phrases[0].query_tokens = vec!["alpha\n\u{1b}\u{85}\"".into()];
    matched.phrases[0].hits[0].document_tokens = vec![Some("beta\t\r\u{0}\u{85}".into())];
    let lines = output(&details);
    for line in &lines {
        assert!(!line.chars().any(char::is_control), "{line:?}");
    }
    let text = lines.join("\n");
    assert!(text.contains(r"\u{1b}"));
    assert!(text.contains(r"\n"));
    assert!(text.contains(r"\t"));
    assert!(text.contains(r"\u{85}"));
}

#[test]
fn pool_out_and_skipped_statuses_explain_participation_without_invented_statistics() {
    let mut details = matched_details();
    details.in_candidate_pool = false;
    let text = output(&details).join("\n");
    assert!(text.contains("in lexical candidate pool: no"));
    assert!(text.contains("BM25 score did not participate in fusion"));
    assert!(text.contains("raw BM25 (display rounded):"));
    for (status, expression, reason) in [
        (
            LexicalDetailsStatus::Disabled,
            None,
            "Lexical retrieval is disabled",
        ),
        (
            LexicalDetailsStatus::EmptyQuery,
            Some(""),
            "lexical expression is empty",
        ),
        (
            LexicalDetailsStatus::NotMatched,
            Some("\"missing\""),
            "does not satisfy the full MATCH",
        ),
    ] {
        details.status = status;
        details.match_expression = expression.map(str::to_owned);
        details.matched = None;
        let text = output(&details).join("\n");
        assert!(text.contains(reason));
        assert!(!text.contains("phrase #"));
        assert!(!text.contains("corpus rows:"));
        assert!(!text.contains("raw BM25 (display rounded):"));
    }
}
