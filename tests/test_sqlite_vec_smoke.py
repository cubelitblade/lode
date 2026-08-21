"""Smoke tests for sqlite-vec availability and the vec0 virtual table.

These lock in the sqlite-vec version/extension contract (see PLAN risk
register: "sqlite-vec version/build compatibility"). Tests are hermetic:
an in-memory database, no network, no external services.
"""

from __future__ import annotations

import sqlite3

import pytest

# The sqlite_vec package ships no type stubs; the ignore is scoped to this
# import line and should be revisited after a dependency upgrade.
import sqlite_vec  # pyright: ignore[reportMissingTypeStubs]

VEC_VERSION_PREFIX = "v0.1"


@pytest.fixture
def db() -> sqlite3.Connection:
    """In-memory connection with the sqlite-vec extension loaded."""
    connection = sqlite3.connect(":memory:")
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    return connection


def test_extension_loads_and_version_is_available(db: sqlite3.Connection) -> None:
    version = db.execute("select vec_version()").fetchone()[0]
    assert version.startswith(VEC_VERSION_PREFIX)


def test_vec0_roundtrip_and_knn_ordering(db: sqlite3.Connection) -> None:
    db.execute("CREATE VIRTUAL TABLE v USING vec0(embedding float[4])")
    db.executemany(
        "INSERT INTO v(rowid, embedding) VALUES (?, ?)",
        [
            (1, "[1.0, 0.0, 0.0, 0.0]"),
            (2, "[0.0, 1.0, 0.0, 0.0]"),
            (3, "[0.9, 0.1, 0.0, 0.0]"),
        ],
    )
    db.commit()

    rows = db.execute(
        """
        SELECT rowid, distance FROM v
        WHERE embedding MATCH '[1.0, 0.0, 0.0, 0.0]' AND k = 3
        ORDER BY distance
        """
    ).fetchall()

    assert [rowid for rowid, _ in rows] == [1, 3, 2]
    distances = [distance for _, distance in rows]
    assert distances == sorted(distances)
