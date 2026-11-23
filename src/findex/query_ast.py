"""Recursive-descent Boolean query parser with positional phrases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from findex.tokenize import tokenize

if TYPE_CHECKING:
    from findex.index import Index


class Query:
    """Base query node supporting Python's &, | and ~ operators."""

    def __and__(self, other: Query) -> Query:
        return And(self, other)

    def __or__(self, other: Query) -> Query:
        return Or(self, other)

    def __invert__(self) -> Query:
        return Not(self)

    def evaluate(self, index: Index) -> set[int]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Term(Query):
    value: str

    def evaluate(self, index: Index) -> set[int]:
        return {p.doc_id for p in index.postings.get(self.value, ())}


@dataclass(frozen=True, slots=True)
class Phrase(Query):
    words: tuple[str, ...]

    def evaluate(self, index: Index) -> set[int]:
        if not index.with_positions:
            raise ValueError(
                'Phrase search requires positions. Rebuild with "--positions".'
            )
        if not self.words:
            return set()

        candidate_ids = {p.doc_id for p in index.postings.get(self.words[0], ())}
        positions_by_term: list[dict[int, set[int]]] = []
        for word in self.words:
            positions = {
                p.doc_id: set(p.positions) for p in index.postings.get(word, ())
            }
            positions_by_term.append(positions)
            candidate_ids &= positions.keys()

        result = set()
        for doc_id in candidate_ids:
            starts = positions_by_term[0][doc_id]
            if any(
                all(
                    start + offset in positions_by_term[offset][doc_id]
                    for offset in range(1, len(self.words))
                )
                for start in starts
            ):
                result.add(doc_id)
        return result


@dataclass(frozen=True, slots=True)
class And(Query):
    left: Query
    right: Query

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)


@dataclass(frozen=True, slots=True)
class Or(Query):
    left: Query
    right: Query

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)


@dataclass(frozen=True, slots=True)
class Not(Query):
    child: Query

    def evaluate(self, index: Index) -> set[int]:
        return set(index.documents) - self.child.evaluate(index)


def positive_terms(query: Query, negated: bool = False) -> set[str]:
    """Terms that provide a positive relevance signal (not exclusions)."""
    if isinstance(query, Term):
        return set() if negated else {query.value}
    if isinstance(query, Phrase):
        return set() if negated else set(query.words)
    if isinstance(query, Not):
        return positive_terms(query.child, not negated)
    if isinstance(query, (And, Or)):
        return positive_terms(query.left, negated) | positive_terms(
            query.right, negated
        )
    raise TypeError(f"Unknown query node: {type(query)!r}")


def _lex(source: str) -> list[str | Query]:
    tokens: list[str | Query] = []
    cursor = 0
    while cursor < len(source):
        ch = source[cursor]
        if ch.isspace():
            cursor += 1
        elif ch in "()":
            tokens.append(ch)
            cursor += 1
        elif ch == '"':
            end = source.find('"', cursor + 1)
            if end < 0:
                raise ValueError("Unclosed quoted phrase")
            words = tuple(tokenize(source[cursor + 1 : end]))
            if not words:
                raise ValueError("Empty quoted phrase")
            tokens.append(Phrase(words))
            cursor = end + 1
        else:
            end = cursor
            while (
                end < len(source)
                and not source[end].isspace()
                and source[end] not in '()"'
            ):
                end += 1
            raw = source[cursor:end]
            if raw.upper() in {"AND", "OR", "NOT"}:
                tokens.append(raw.upper())
            else:
                words = tuple(tokenize(raw))
                if len(words) != 1:
                    raise ValueError(f"Invalid term: {raw!r}")
                tokens.append(Term(words[0]))
            cursor = end
    return tokens


class _Parser:
    def __init__(self, tokens: list[str | Query]):
        self.tokens = tokens
        self.position = 0

    def peek(self) -> str | Query | None:
        if self.position >= len(self.tokens):
            return None
        return self.tokens[self.position]

    def pop(self) -> str | Query:
        current = self.peek()
        if current is None:
            raise ValueError("Unexpected end of query")
        self.position += 1
        return current

    def parse_or(self) -> Query:
        node = self.parse_and()
        while self.peek() == "OR":
            self.pop()
            node = Or(node, self.parse_and())
        return node

    def parse_and(self) -> Query:
        node = self.parse_not()
        while self.peek() is not None and self.peek() not in (")", "OR"):
            if self.peek() == "AND":
                self.pop()
            # Otherwise two adjacent operands imply AND.
            node = And(node, self.parse_not())
        return node

    def parse_not(self) -> Query:
        if self.peek() == "NOT":
            self.pop()
            return Not(self.parse_not())
        return self.parse_atom()

    def parse_atom(self) -> Query:
        token = self.pop()
        if token == "(":
            node = self.parse_or()
            if self.pop() != ")":
                raise ValueError("Missing closing parenthesis")
            return node
        if isinstance(token, Query):
            return token
        raise ValueError(f"Expected a term or phrase, got {token!r}")


def parse_query(query: str) -> Query:
    """Parse Boolean query; NOT > AND > OR; adjacent operands imply AND."""
    tokens = _lex(query)
    if not tokens:
        raise ValueError("Query is empty")
    parser = _Parser(tokens)
    result = parser.parse_or()
    if parser.peek() is not None:
        raise ValueError(f"Unexpected token: {parser.peek()!r}")
    return result
