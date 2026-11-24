"""Stream the crawl JSONL into an inverted index, outside asyncio's event loop."""

from collections import Counter, defaultdict
from pathlib import Path

from findex.corpus import iter_documents
from findex.index import DocMeta, Index, Posting
from findex.tokenize import tokenize


def build_jsonl_index(source: Path, *, with_positions: bool = True) -> Index:
    """One-pass JSONL indexing, bounded by one document's text in memory."""
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    documents: dict[int, DocMeta] = {}
    for doc_id, doc in enumerate(iter_documents(source)):
        terms = list(tokenize(doc.text))
        if with_positions:
            positions: defaultdict[str, list[int]] = defaultdict(list)
            for position, term in enumerate(terms):
                positions[term].append(position)
            frequencies = {term: len(offsets) for term, offsets in positions.items()}
        else:
            frequencies = dict(Counter(terms))
            positions = defaultdict(list)
        documents[doc_id] = DocMeta(doc_id, doc.doc_id, len(terms), doc.text[:200])
        for term, frequency in frequencies.items():
            postings[term].append(
                Posting(
                    doc_id, frequency, tuple(positions[term]) if with_positions else ()
                )
            )
    return Index(dict(postings), documents, with_positions, str(source.resolve()))
