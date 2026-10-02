#![warn(clippy::pedantic)]

use crate::fts::explanation::FtsExplanationError;

use super::Store;

impl Store {
    pub(crate) fn with_read_snapshot<T>(
        &self,
        read: impl FnOnce(&Self) -> crate::Result<T>,
    ) -> crate::Result<T> {
        let transaction = self
            .conn
            .unchecked_transaction()
            .map_err(FtsExplanationError::from)?;
        let result = read(self)?;
        transaction.commit().map_err(FtsExplanationError::from)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use rusqlite::hooks::{AuthAction, AuthContext, Authorization};

    use super::*;
    use crate::embeddings::testing::{FakeEmbedder, file_record, make_chunks};
    use crate::index::ranking::{Fusion, RetrievalPlan};
    use crate::index::records::Source;
    use crate::index::search::explain_with_fts;

    #[test]
    fn disabled_diagnostics_skip_fts_and_failure_rolls_back() {
        let directory = tempfile::tempdir().unwrap();
        let mut store = Store::open(
            &directory.path().join("index.db"),
            "test-model",
            4,
            "unicode61",
        )
        .unwrap();
        let (chunks, vectors) = make_chunks(&["alpha"], None);
        store
            .replace_file(
                &file_record("notes.txt", "blake3:notes", 1),
                "text",
                &chunks,
                Some(&vectors),
            )
            .unwrap();
        store
            .conn
            .authorizer(Some(|context: AuthContext<'_>| match context.action {
                AuthAction::Read {
                    table_name: "chunks_fts",
                    ..
                } => Authorization::Deny,
                _ => Authorization::Allow,
            }));
        let plan = RetrievalPlan {
            norm: None,
            fusion: Fusion::Linear {
                weights: [(Source::Semantic, 1.0), (Source::Lexical, 0.0)].into(),
            },
        };
        let result =
            explain_with_fts(&store, &FakeEmbedder::default(), "alpha", 1, &plan, 1).unwrap();
        assert!(result.lexical_details.matched.is_none());
        assert!(store.conn.is_autocommit());
        store
            .conn
            .authorizer(Some(|context: AuthContext<'_>| match context.action {
                AuthAction::Function {
                    function_name: "lode_fts_details",
                } => Authorization::Deny,
                _ => Authorization::Allow,
            }));
        let plan = RetrievalPlan {
            norm: None,
            fusion: Fusion::Rrf { k: 60 },
        };
        let error =
            explain_with_fts(&store, &FakeEmbedder::default(), "alpha", 1, &plan, 1).unwrap_err();
        assert!(matches!(
            error,
            crate::Error::FtsExplanation(FtsExplanationError::Sqlite(_))
        ));
        assert!(std::error::Error::source(&error).is_some());
        assert!(store.conn.is_autocommit());
        store
            .conn
            .authorizer(None::<fn(AuthContext<'_>) -> Authorization>);
        assert!(explain_with_fts(&store, &FakeEmbedder::default(), "alpha", 1, &plan, 1).is_ok());
    }
}
