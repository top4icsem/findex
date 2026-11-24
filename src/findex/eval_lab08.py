"""Lab 08 labeled Precision@5 for BM25, semantic, hybrid retrieval.

Uses 10 *human-labeled* Lab 03 query sets plus 5 new manually labeled
paraphrases. Refuses to manufacture or silently infer relevance judgments.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from findex.open_index import open_index
from findex.semantic import EmbeddingStore
from findex.tokenize import tokenize
from findex.web.service import SearchService


def query_for_lexical(query: str) -> str:
    from findex.query_ast import parse_query

    try:
        parse_query(query)
        return query
    except ValueError:
        words = list(dict.fromkeys(tokenize(query)))
        return " OR ".join(words)


def load_labels(original: Path, paraphrases: Path) -> list[tuple[str, str, set[int]]]:
    original_labels: dict[tuple[str, str], set[int]] = defaultdict(set)
    with original.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["query_id"], row["query"])
            if row["relevant"].strip() == "1":
                original_labels[key].add(int(row["doc_id"]))
            else:
                original_labels[key] |= set()
    if len(original_labels) < 10:
        raise ValueError("Require 10 original labeled Lab 03 queries")
    queries = [
        ("lab03-" + key[0], key[1], refs) for key, refs in original_labels.items()
    ]
    with paraphrases.open(encoding="utf-8", newline="") as stream:
        fresh = list(csv.DictReader(stream))
    if len(fresh) < 5:
        raise ValueError("Require five manually labeled paraphrased queries")
    for row in fresh:
        value = row.get("relevant_doc_ids", "").strip()
        if not value:
            raise ValueError(
                "Five paraphrase labels must be completed by a person: " + row["query"]
            )
        refs = {int(item.strip()) for item in value.split(";") if item.strip()}
        if not refs:
            raise ValueError("No document relevance IDs for " + row["query"])
        queries.append((row["id"], row["query"], refs))
    return queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("index.bin"))
    parser.add_argument("--embeddings", type=Path, default=Path("embeddings.npy"))
    parser.add_argument(
        "--lab03", type=Path, default=Path("evaluation/lab03_judgments.csv")
    )
    parser.add_argument(
        "--paraphrases", type=Path, default=Path("evaluation/lab08_paraphrases.csv")
    )
    parser.add_argument("--out", type=Path, default=Path("docs/lab08_precision.csv"))
    args = parser.parse_args()
    queries = load_labels(args.lab03, args.paraphrases)
    rows: list[dict[str, str]] = []
    with open_index(args.index) as index:
        store = EmbeddingStore.load(args.embeddings, index)
        service = SearchService(index, store)
        for ident, query, relevant in queries:
            for mode in ("lexical", "semantic", "hybrid"):
                q = query_for_lexical(query) if mode == "lexical" else query
                # RRF gracefully treats natural language as an OR query.
                hits = service.ranked(q, 5, "bm25", mode)
                precision = len(relevant & {hit.doc_id for hit in hits}) / 5
                rows.append(
                    {
                        "id": ident,
                        "query": query,
                        "mode": mode,
                        "p_at_5": f"{precision:.3f}",
                    }
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved human-labeled P@5 results:", args.out)
    for mode in ("lexical", "semantic", "hybrid"):
        subset = [float(row["p_at_5"]) for row in rows if row["mode"] == mode]
        print(
            f"{mode}: mean P@5 = {sum(subset) / len(subset):.3f} over {len(subset)} queries"
        )


if __name__ == "__main__":
    main()
