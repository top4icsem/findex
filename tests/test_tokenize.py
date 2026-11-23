import pytest

from findex.tokenize import tokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello WORLD", ["hello", "world"]),
        ("Привіт, Україно!", ["привіт", "україно"]),
        ("Python 3.13", ["python", "3", "13"]),
        ("don't stop", ["don't", "stop"]),
        ("п'ять об'єктів", ["п'ять", "об'єктів"]),
        ("rock’n’roll", ["rock’n’roll"]),
        ("state-of-the-art", ["state", "of", "the", "art"]),
        ("hello_world", ["hello", "world"]),
        ("", []),
        ("!!! ... ???", []),
        ("CAFÉ café", ["café", "café"]),
        ("Straße STRASSE", ["strasse", "strasse"]),
        ("2026 рік", ["2026", "рік"]),
    ],
)
def test_tokenize(text, expected):
    assert list(tokenize(text)) == expected


def test_unicode_normalization():
    composed = "café"
    decomposed = "cafe\u0301"

    assert list(tokenize(composed)) == list(tokenize(decomposed))


def test_tokenize_is_generator():
    result = tokenize("Hello world")

    assert iter(result) is result
    assert next(result) == "hello"
    assert next(result) == "world"

    with pytest.raises(StopIteration):
        next(result)
