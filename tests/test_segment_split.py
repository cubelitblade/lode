"""Tests for the heading-aware segment splitter.

`RecursiveSegmentSplitter` windows each segment independently (heading
boundaries are hard chunk boundaries) and tags the resulting chunks with the
segment's heading/page. A plain format arrives as a single unstyled segment,
so its text and seq match the plain `RecursiveTextSplitter` exactly. Note
that chunk ids are *not* asserted here as a strong regression condition —
content text and sequence are the contract.
"""

from __future__ import annotations

from lode.ingestion import Segment
from lode.ingestion.split import RecursiveSegmentSplitter, RecursiveTextSplitter


def test_single_segment_matches_recursive_window_text_and_seq() -> None:
    text = "alpha beta gamma\n\ndelta epsilon zeta"
    got = RecursiveSegmentSplitter(chunk_size=50, chunk_overlap=5).split_segments([Segment(text=text)])
    expected = RecursiveTextSplitter(chunk_size=50, chunk_overlap=5).split(text)

    assert [c.text for c in got] == [c.text for c in expected]
    assert [c.seq for c in got] == list(range(len(got)))
    assert all(c.heading == "" for c in got)
    assert all(c.page is None for c in got)


def test_heading_propagated_to_every_chunk() -> None:
    chunks = RecursiveSegmentSplitter(chunk_size=20, chunk_overlap=5).split_segments(
        [Segment(text="word " * 30, heading="报告 / 第三章")]
    )
    assert chunks
    assert all(c.heading == "报告 / 第三章" for c in chunks)
    assert all(c.page is None for c in chunks)
    assert [c.seq for c in chunks] == list(range(len(chunks)))


def test_page_propagated_to_chunk() -> None:
    chunks = RecursiveSegmentSplitter(chunk_size=1000).split_segments([Segment(text="hello", page=3)])
    assert len(chunks) == 1
    assert chunks[0].page == 3


def test_seq_numbered_globally_across_segments() -> None:
    chunks = RecursiveSegmentSplitter(chunk_size=1000, chunk_overlap=0).split_segments(
        [Segment(text="first"), Segment(text="second", heading="a")]
    )
    assert [c.text for c in chunks] == ["first", "second"]
    assert [c.seq for c in chunks] == [0, 1]
    assert chunks[1].heading == "a"


def test_empty_input_yields_no_chunks() -> None:
    assert RecursiveSegmentSplitter().split_segments([]) == []
    assert RecursiveSegmentSplitter().split_segments([Segment(text="")]) == []
