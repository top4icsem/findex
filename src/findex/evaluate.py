"""Manual Precision@5: generate a judgment CSV, then compute scores."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from findex.open_index import open_index
from findex.ranking import RankedSearcher
from findex.scoring import BM25, Tfidf

QUERIES = [
    "python search",
    "memory optimization",
    "information retrieval",
    "machine learning",
    "unicode normalization",
    "search engine",
    "document tokens",
    "data structures",
    "generators python",
    "performance evaluation",
]
COLUMNS = (
    "query_id",
    "query",
    "scorer",
    "rank",
    "doc_id",
    "path",
    "score",
    "snippet",
    "relevant",
)


def create_judgments(index_file: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    with (
        open_index(index_file) as index,
        output.open("w", encoding="utf-8", newline="") as file,
    ):
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()

        for name, scorer in (("tfidf", Tfidf()), ("bm25", BM25())):
            engine = RankedSearcher(index, scorer)

            for query_id, query in enumerate(QUERIES, start=1):
                results = engine.search(query, top_k=5)

                for rank in range(1, 6):
                    found = results[rank - 1] if rank <= len(results) else None

                    writer.writerow(
                        {
                            "query_id": query_id,
                            "query": query,
                            "scorer": name,
                            "rank": rank,
                            "doc_id": found.doc_id if found else "",
                            "path": found.path if found else "",
                            "score": f"{found.score:.6f}" if found else "",
                            "snippet": found.snippet if found else "",
                            "relevant": "" if found else "0",
                        }
                    )

    print(f"Created {output}: label every document row with relevant=1 or 0.")
    print("1 = truly relevant to that query, 0 = not relevant.")


def score_judgments(path: Path) -> None:
    per_query: defaultdict[tuple[int, str, str], list[int]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            key = (int(row["query_id"]), row["query"], row["scorer"])
            if row["relevant"] not in {"0", "1"}:
                raise ValueError(
                    f"Missing manual relevance label for {key}, rank {row['rank']}"
                )
            per_query[key].append(int(row["relevant"]))

    if len(per_query) != 20 or any(len(v) != 5 for v in per_query.values()):
        raise ValueError("Expected 10 queries x 2 scorers x 5 ranks = 100 rows")
    averages: dict[str, list[float]] = {"tfidf": [], "bm25": []}
    print("\nPrecision@5 (manual 0/1 relevance)")
    print("query_id  tfidf  bm25  query")
    for i, query in enumerate(QUERIES, start=1):
        t = sum(per_query[i, query, "tfidf"]) / 5
        b = sum(per_query[i, query, "bm25"]) / 5
        averages["tfidf"].append(t)
        averages["bm25"].append(b)
        print(f"{i:>8}  {t:.2f}   {b:.2f}  {query}")
    print("Mean P@5 TF-IDF:", round(sum(averages["tfidf"]) / 10, 4))
    print("Mean P@5 BM25:  ", round(sum(averages["bm25"]) / 10, 4))


def main():
    parser = argparse.ArgumentParser(description="Manual Precision@5 evaluation")
    parser.add_argument("action", choices=("create", "score"))
    parser.add_argument("--index", type=Path, default=Path("index.bin"))
    parser.add_argument(
        "--file", type=Path, default=Path("evaluation/lab03_judgments.csv")
    )
    args = parser.parse_args()
    if args.action == "create":
        create_judgments(args.index, args.file)
    else:
        score_judgments(args.file)


if __name__ == "__main__":
    main()
