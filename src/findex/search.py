import argparse
import re
import tracemalloc
from pathlib import Path
from time import perf_counter

from findex.index import InvertedIndex
from findex.storage import load_index
from findex.tokenize import tokenize

OPERATORS = {"AND", "OR", "NOT"}
PRECEDENCE = {"OR": 1, "AND": 2, "NOT": 3}


def merge_and(left: list[int], right: list[int]) -> list[int]:
    """Intersect two sorted document ID lists."""
    result: list[int] = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            i += 1
        else:
            j += 1

    return result


def merge_or(left: list[int], right: list[int]) -> list[int]:
    """Union of two sorted document ID lists."""
    result: list[int] = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result


def merge_not(universe: list[int], excluded: list[int]) -> list[int]:
    """Return document IDs not present in excluded."""
    result: list[int] = []
    i = j = 0

    while i < len(universe):
        if j >= len(excluded) or universe[i] < excluded[j]:
            result.append(universe[i])
            i += 1
        elif universe[i] == excluded[j]:
            i += 1
            j += 1
        else:
            j += 1

    return result


def parse_query(query: str) -> list[str]:
    """Convert a Boolean query into postfix notation."""
    raw_parts = re.findall(r"\(|\)|[^\s()]+", query)
    parts: list[str] = []

    for raw in raw_parts:
        if raw in ("(", ")"):
            token = raw
        elif raw.upper() in OPERATORS:
            token = raw.upper()
        else:
            normalized = list(tokenize(raw))
            if len(normalized) != 1:
                raise ValueError(f"Invalid search term: {raw}")
            token = normalized[0]

        if parts:
            previous = parts[-1]
            previous_is_value = previous == ")" or (
                previous not in OPERATORS and previous != "("
            )
            next_is_value = (
                token == "("
                or token == "NOT"
                or (token not in OPERATORS and token not in ("(", ")"))
            )

            if previous_is_value and next_is_value:
                parts.append("AND")

        parts.append(token)

    output: list[str] = []
    operators: list[str] = []

    for token in parts:
        if token == "(":
            operators.append(token)
        elif token == ")":
            while operators and operators[-1] != "(":
                output.append(operators.pop())

            if not operators:
                raise ValueError("Unmatched closing parenthesis")

            operators.pop()
        elif token in OPERATORS:
            while operators and operators[-1] != "(":
                previous = operators[-1]
                if PRECEDENCE[previous] > PRECEDENCE[token] or (
                    PRECEDENCE[previous] == PRECEDENCE[token] and token != "NOT"
                ):
                    output.append(operators.pop())
                else:
                    break

            operators.append(token)
        else:
            output.append(token)

    while operators:
        operator = operators.pop()
        if operator == "(":
            raise ValueError("Unmatched opening parenthesis")
        output.append(operator)

    return output


def search(
    index: InvertedIndex,
    query: str,
    engine: str = "merge",
) -> list[int]:
    """Search documents using Boolean operators."""
    if engine not in ("merge", "set"):
        raise ValueError("Engine must be 'merge' or 'set'")

    postfix = parse_query(query)
    if not postfix:
        return []

    universe = sorted(index.documents)
    stack: list[list[int]] = []

    for token in postfix:
        if token == "NOT":
            if not stack:
                raise ValueError("NOT requires an operand")

            operand = stack.pop()

            if engine == "merge":
                stack.append(merge_not(universe, operand))
            else:
                stack.append(sorted(set(universe) - set(operand)))

        elif token in ("AND", "OR"):
            if len(stack) < 2:
                raise ValueError(f"{token} requires two operands")

            right = stack.pop()
            left = stack.pop()

            if engine == "merge":
                result = (
                    merge_and(left, right) if token == "AND" else merge_or(left, right)
                )
            else:
                result_set = (
                    set(left) & set(right) if token == "AND" else set(left) | set(right)
                )
                result = sorted(result_set)

            stack.append(result)

        else:
            postings = index.postings.get(token, [])
            stack.append([posting.doc_id for posting in postings])

    if len(stack) != 1:
        raise ValueError("Invalid Boolean query")

    return stack[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a saved findex index.")
    parser.add_argument("index_file", type=Path)
    parser.add_argument(
        "query",
        help="Boolean query: AND, OR, NOT, parentheses",
    )
    parser.add_argument(
        "--engine",
        choices=("merge", "set"),
        default="merge",
    )
    parser.add_argument(
        "--format",
        choices=("pickle", "json"),
        default=None,
    )
    parser.add_argument("--top", type=int, default=10)

    args = parser.parse_args()

    if args.top < 0:
        parser.error("--top must not be negative")

    file_format = args.format or (
        "json" if args.index_file.suffix == ".json" else "pickle"
    )

    tracemalloc.start()
    start = perf_counter()

    try:
        index = load_index(args.index_file, file_format)
        loaded = perf_counter()

        doc_ids = search(index, args.query, args.engine)
        finished = perf_counter()

        _, peak = tracemalloc.get_traced_memory()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    finally:
        tracemalloc.stop()

    print(f"Query: {args.query}")
    print(f"Engine: {args.engine}")
    print(f"Matches: {len(doc_ids)}")

    for doc_id in doc_ids[: args.top]:
        meta = index.documents[doc_id]
        snippet = " ".join(meta.preview.split())[:100]
        print(f"  {doc_id} | {meta.path} | {meta.length} tokens | {snippet}")

    print(f"Load time: {loaded - start:.4f} s")
    print(f"Search time: {finished - loaded:.6f} s")
    print(f"Peak memory: {peak / 1024**2:.4f} MiB")


if __name__ == "__main__":
    main()
