#![warn(clippy::pedantic)]

use super::fts_details::lexical_config;
use super::*;

#[test]
fn strong_bm25_match_survives_pool_cutoff_and_has_positive_fusion_contribution() {
    for norm in ["minmax", "softmax"] {
        let settings = format!(
            "{}[retrieval.norm]\ntype = \"{norm}\"\n",
            lexical_config("unicode61")
        );
        let fixture = fixture(&settings, "alpha alpha alpha alpha");
        let mut store = Store::open_existing(&fixture.workspace.join(".lode/index.db")).unwrap();
        for index in 0..5 {
            let text = format!("alpha {}{index}", "filler ".repeat(100));
            let source = fixture.workspace.join(format!("weak-{index}.txt"));
            fs::write(&source, &text).unwrap();
            store
                .replace_file(
                    &record_for(&source, &fixture.workspace),
                    "text",
                    &[Chunk {
                        digest: lode_core::ingestion::digest::chunk_digest(&text),
                        text,
                        seq: 0,
                        heading: String::new(),
                        page: None,
                        span: None,
                    }],
                    Some(&[vec![1.0, 0.0, 0.0, 0.0]]),
                )
                .unwrap();
        }
        drop(store);
        let output = run_lode(&fixture, "json", &["prospect", "alpha", "--top-k", "1"]);
        assert!(output.status.success(), "{output:?}");
        let prospect = json(&output);
        assert_eq!(prospect["hits"][0]["digest"], fixture.digest);
        let output = run_lode(
            &fixture,
            "json",
            &["assay", "why", &fixture.digest, "alpha", "--top-k", "1"],
        );
        assert!(output.status.success(), "{output:?}");
        let plain = json(&output);
        let output = run_lode(
            &fixture,
            "json",
            &[
                "assay",
                "why",
                &fixture.digest,
                "alpha",
                "--top-k",
                "1",
                "--fts-details",
            ],
        );
        assert!(output.status.success(), "{output:?}");
        let mut diagnostic = json(&output);
        let details = diagnostic
            .as_object_mut()
            .unwrap()
            .remove("lexical_details")
            .unwrap();
        assert_eq!(plain, diagnostic);
        assert_eq!(plain["rank"], 1);
        assert_eq!(plain["sources"]["lexical"]["pool_rank"], 1);
        assert_eq!(details["in_candidate_pool"], true);
        let raw = plain["sources"]["lexical"]["raw_score"].as_f64().unwrap();
        let prepared = plain["sources"]["lexical"]["prepared_score"]
            .as_f64()
            .unwrap();
        let combined = plain["combined"].as_f64().unwrap();
        assert!(raw < 0.0 && prepared < 0.0 && combined > 0.0);
        assert_eq!(details["bm25"].as_f64(), Some(raw));
        assert!((combined + prepared).abs() < 1e-12);
        assert!(
            (plain["ranking_factors"]["lexical"]["value"]
                .as_f64()
                .unwrap()
                - combined)
                .abs()
                < 1e-12
        );
        assert!((prospect["hits"][0]["score"].as_f64().unwrap() - combined).abs() < 1e-12);
        let output = run_lode(
            &fixture,
            "extended",
            &[
                "assay",
                "why",
                &fixture.digest,
                "alpha",
                "--top-k",
                "1",
                "--fts-details",
            ],
        );
        assert!(output.status.success(), "{output:?}");
        let text = String::from_utf8(output.stdout).unwrap();
        assert!(text.contains("(-lexical) × 1"));
        assert!(text.contains("in lexical candidate pool: yes"));
    }
}
