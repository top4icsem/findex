"""Three property families required for Lab 04."""

from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import given, settings
from hypothesis import strategies as st

from findex.index import build_index
from findex.search import merge_and
from findex.storage import load_index, save_index

small_words = st.lists(
    st.sampled_from(["python", "search", "java", "memory", "event", "loop"]),
    min_size=1,
    max_size=20,
)


@settings(max_examples=20, deadline=None)
@given(small_words)
def test_roundtrip_invariant(words: list[str]) -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        corpus = root / "docs"
        corpus.mkdir()
        (corpus / "a.txt").write_text(" ".join(words), encoding="utf-8")
        index = build_index(corpus, with_positions=True)
        for fmt, name in (("pickle", "test.bin"), ("json", "test.json")):
            dest = root / name
            save_index(index, dest, fmt)
            assert load_index(dest, fmt) == index


@settings(max_examples=20, deadline=None)
@given(st.lists(small_words, min_size=1, max_size=5))
def test_postings_sorted(documents: list[list[str]]) -> None:
    with TemporaryDirectory() as temporary:
        corpus = Path(temporary)
        for i, words in enumerate(documents):
            (corpus / f"{i:03d}.txt").write_text(" ".join(words), encoding="utf-8")
        index = build_index(corpus, with_positions=True)
        for posting_list in index.postings.values():
            ids = [posting.doc_id for posting in posting_list]
            assert ids == sorted(set(ids))


@settings(max_examples=80, deadline=None)
@given(
    st.sets(st.integers(min_value=0, max_value=200)),
    st.sets(st.integers(min_value=0, max_value=200)),
)
def test_merge_matches_set(left: set[int], right: set[int]) -> None:
    assert merge_and(sorted(left), sorted(right)) == sorted(left & right)
