
import re
import unicodedata
from collections.abc import Iterator

TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def tokenize(text: str) -> Iterator[str]:
    """Normalize text and yield tokens one by one.

    Rules:
    - Unicode is normalized to NFC.
    - Case is normalized using casefold().
    - Apostrophes inside words are preserved.
    - Hyphens split words into separate tokens.
    - Numbers are preserved as tokens.
    - Underscores and punctuation are separators.
    """
    normalized = unicodedata.normalize("NFC", text).casefold()

    for match in TOKEN_PATTERN.finditer(normalized):
        yield match.group()
