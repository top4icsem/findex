import random
from pathlib import Path

OUTPUT_DIR = Path("data")
DOCUMENT_COUNT = 1000
PARAGRAPHS_PER_DOCUMENT = 100

PARAGRAPHS = [
    "Python is a powerful programming language used for web development, data analysis and automation.",
    "A search engine processes documents, extracts tokens and builds an index for fast information retrieval.",
    "Generators in Python allow lazy evaluation and help reduce memory consumption when processing large datasets.",
    "Unicode normalization ensures that text from different languages can be processed consistently.",
    "Iterators provide an efficient way to read large collections without loading everything into memory.",
    "Software testing helps developers detect errors and verify the correctness of their programs.",
    "Data structures such as dictionaries, lists and sets are essential for efficient algorithms.",
    "Machine learning and artificial intelligence use Python for training models and analyzing information.",
    "Search systems calculate term frequencies and maintain vocabularies for document retrieval.",
    "Performance optimization involves measuring execution time and peak memory usage.",
]

random.seed(42)
OUTPUT_DIR.mkdir(exist_ok=True)

for index in range(DOCUMENT_COUNT):
    path = OUTPUT_DIR / f"document_{index:04d}.txt"

    with path.open("w", encoding="utf-8") as file:
        file.write(f"Document {index}: Python Search Engine\n\n")

        for _ in range(PARAGRAPHS_PER_DOCUMENT):
            file.write(random.choice(PARAGRAPHS) + "\n")

print(f"Generated {DOCUMENT_COUNT} documents in {OUTPUT_DIR}/")
