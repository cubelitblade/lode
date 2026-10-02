#![warn(clippy::pedantic)]

//! Original-text ranges from actual DOCUMENT positions, never query guesses.

use std::ops::Range;

use lode_simple_native::{TOKEN_COLOCATED, Token};

use super::explanation::{
    FtsHit, FtsHitMapping, FtsMappingUnavailable, FtsMatchDetails, FtsSourceSpan,
};

struct Position {
    range: Range<usize>,
    variants: Vec<String>,
}

pub(crate) fn apply(
    details: &mut FtsMatchDetails,
    text: &str,
    tokens: Result<Vec<Token>, FtsMappingUnavailable>,
) {
    let positions = tokens
        .and_then(|tokens| positions(text, tokens))
        .and_then(|positions| {
            if usize::try_from(details.document_length).ok() == Some(positions.len()) {
                Ok(positions)
            } else {
                Err(FtsMappingUnavailable::TokenCountMismatch)
            }
        });
    for phrase in &mut details.phrases {
        for hit in &mut phrase.hits {
            hit.mapping = match &positions {
                Ok(positions) => map_hit(text, positions, hit, phrase.query_tokens.len())
                    .map_or_else(FtsHitMapping::Unavailable, FtsHitMapping::Mapped),
                Err(reason) => FtsHitMapping::Unavailable(*reason),
            };
        }
    }
}

fn positions(text: &str, tokens: Vec<Token>) -> Result<Vec<Position>, FtsMappingUnavailable> {
    let mut positions: Vec<Position> = Vec::new();
    for token in tokens {
        if token.flags & !TOKEN_COLOCATED != 0
            || token.text.is_empty()
            || token.start >= token.end
            || text.get(token.start..token.end).is_none()
        {
            return Err(FtsMappingUnavailable::InvalidTokenStream);
        }
        if token.flags & TOKEN_COLOCATED != 0 {
            let previous = positions
                .last_mut()
                .ok_or(FtsMappingUnavailable::InvalidTokenStream)?;
            if previous.range != (token.start..token.end) {
                return Err(FtsMappingUnavailable::InvalidTokenStream);
            }
            previous.variants.push(token.text);
        } else {
            if positions.last().is_some_and(|previous| {
                token.start < previous.range.start || token.end < previous.range.end
            }) {
                return Err(FtsMappingUnavailable::InvalidTokenStream);
            }
            positions.push(Position {
                range: token.start..token.end,
                variants: vec![token.text],
            });
        }
    }
    Ok(positions)
}

fn map_hit(
    text: &str,
    positions: &[Position],
    hit: &FtsHit,
    phrase_length: usize,
) -> Result<FtsSourceSpan, FtsMappingUnavailable> {
    if hit.column != 0 {
        return Err(FtsMappingUnavailable::UnsupportedColumn);
    }
    let offset =
        usize::try_from(hit.token_offset).map_err(|_| FtsMappingUnavailable::HitOutsideDocument)?;
    let end = offset
        .checked_add(phrase_length)
        .ok_or(FtsMappingUnavailable::HitOutsideDocument)?;
    let matched = positions
        .get(offset..end)
        .filter(|positions| !positions.is_empty())
        .ok_or(FtsMappingUnavailable::HitOutsideDocument)?;
    if hit.document_tokens.len() != phrase_length
        || hit
            .document_tokens
            .iter()
            .zip(matched)
            .any(|(token, position)| {
                token
                    .as_ref()
                    .is_some_and(|token| !position.variants.contains(token))
            })
    {
        return Err(FtsMappingUnavailable::DocumentTokenMismatch);
    }
    let start_byte = matched[0].range.start;
    let end_byte = matched[matched.len() - 1].range.end;
    let surface = text
        .get(start_byte..end_byte)
        .ok_or(FtsMappingUnavailable::InvalidTokenStream)?;
    Ok(FtsSourceSpan {
        start_byte,
        end_byte,
        surface: surface.to_owned(),
    })
}

/// Verify the whole fixed definition, rather than guessing tokenizer options.
/// Whitespace and keyword case may differ; quoted parameter bytes may not.
pub(crate) fn same_definition(actual: &str, expected: &str) -> bool {
    canonical_sql(actual) == canonical_sql(expected)
}

fn canonical_sql(sql: &str) -> String {
    let mut quote = None;
    let mut output = String::new();
    for character in sql.chars() {
        if let Some(delimiter) = quote {
            output.push(character);
            if character == delimiter {
                quote = None;
            }
        } else if matches!(character, '\'' | '"' | '\u{60}') {
            quote = Some(character);
            output.push(character);
        } else if !character.is_ascii_whitespace() && character != ';' {
            output.push(character.to_ascii_lowercase());
        }
    }
    output
}

#[cfg(test)]
mod tests;
