import gc
import sys
import tracemalloc
from collections import Counter
from pathlib import Path
from time import perf_counter

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def lazy_stats(root: Path):
    counts = Counter()
    documents = 0
    tokens = 0

    for document in iter_documents(root):
        documents += 1

        for token in tokenize(document.text):
            counts[token] += 1
            tokens += 1

    return documents, tokens, len(counts)


def eager_stats(root: Path):
    documents_list = list(iter_documents(root))
    all_tokens = []

    for document in documents_list:
        all_tokens.extend(list(tokenize(document.text)))

    counts = Counter(all_tokens)

    return len(documents_list), len(all_tokens), len(counts)


def measure(name, function, root):
    gc.collect()
    tracemalloc.start()
    start = perf_counter()

    try:
        result = function(root)
        elapsed = perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    print(f"\n{name}")
    print(f"Documents: {result[0]}")
    print(f"Tokens: {result[1]}")
    print(f"Vocabulary: {result[2]}")
    print(f"Time: {elapsed:.4f} seconds")
    print(f"Peak memory: {peak / 1024 / 1024:.4f} MB")

    return result


if __name__ == "__main__":
    root = Path("data")

    print(f"Python version: {sys.version.split()[0]}")

    lazy_result = measure("LAZY", lazy_stats, root)
    eager_result = measure("EAGER", eager_stats, root)

    print("\nResults match:", lazy_result == eager_result)
