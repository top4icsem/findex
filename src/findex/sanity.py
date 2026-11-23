"""Three deterministic ranking sanity checks based on actual corpus statistics."""

import argparse
from pathlib import Path

from findex.index import Index
from findex.open_index import open_index
from findex.scoring import BM25, Scorer, Tfidf


def check_ranking(index: Index) -> None:
    words = [term for term in index if term.isalpha()]
    rare = min(words, key=lambda term: (index.df(term), term))
    common = max(words, key=lambda term: (index.df(term), term))
    smallest = min(index.documents.values(), key=lambda meta: meta.length)
    largest = max(index.documents.values(), key=lambda meta: meta.length)

    print("Ranking sanity checks (real corpus df and doc lengths)")
    print(f"rare={rare} df={index.df(rare)}; common={common} df={index.df(common)}")
    print(f"short={smallest.length} tokens; long={largest.length} tokens")
    for scorer in (Tfidf(), BM25()):
        evaluator: Scorer = scorer

        def score(
            tf: int, df: int, length: int, evaluator: Scorer = evaluator
        ) -> float:
            return evaluator.score(
                tf=tf,
                df=df,
                doc_length=length,
                avg_doc_length=index.avg_doc_length,
                num_docs=index.num_docs,
            )

        length = max(1, round(index.avg_doc_length))
        rare_score = score(1, index.df(rare), length)
        common_score = score(1, index.df(common), length)
        delta_first = score(2, index.df(rare), length) - score(
            1, index.df(rare), length
        )
        delta_20th = score(20, index.df(rare), length) - score(
            19, index.df(rare), length
        )
        short_score = score(1, index.df(rare), smallest.length)
        long_score = score(1, index.df(rare), largest.length)
        checks = {
            "rare > frequent": rare_score > common_score,
            "20th increment < 2nd increment": delta_20th < delta_first,
            "short doc > long doc (same tf)": short_score > long_score,
        }
        print(f"\n{type(scorer).__name__}:")
        print(f"  rare={rare_score:.8f}; frequent={common_score:.8f}")
        print(f"  second_gain={delta_first:.8f}; twentieth_gain={delta_20th:.8f}")
        print(f"  short={short_score:.8f}; long={long_score:.8f}")
        for description, passed in checks.items():
            print(f"  {'PASS' if passed else 'FAIL'} {description}")
        if not all(checks.values()):
            raise AssertionError("One or more sanity checks failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ranking sanity checks")
    parser.add_argument("index_file", type=Path)
    args = parser.parse_args()
    with open_index(args.index_file) as index:
        check_ranking(index)


if __name__ == "__main__":
    main()
