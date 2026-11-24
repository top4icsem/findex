"""Preserve Labs 1-7 README; add an honest Lab 08 landing section + metrics hub."""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SECTION = ROOT / "LAB08_README_SECTION.md"
SCORING = ROOT / "docs/lab08_scoring.csv"
HEAD = "<!-- LAB08_LANDING_START -->"
TAIL = "<!-- LAB08_LANDING_END -->"


def speedup() -> str:
    if SCORING.exists():
        with SCORING.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            best = max(rows, key=lambda row: float(row["speedup"]))
            return f"{float(best['speedup']):.2f}× (NumPy; {best['query_type']}, {best['scorer']})"
    return "ще не виміряно — запустити `findex.bench_lab08`"


def main() -> None:
    original = README.read_text(encoding="utf-8")
    gif = ROOT / "docs/images/findex_demo.gif"
    gif_note = (
        "![Живий демопошук](docs/images/findex_demo.gif)"
        if gif.is_file()
        else "**GIF пошуку:** потрібно записати реальне демо у `docs/images/findex_demo.gif`."
    )
    block = f"""{HEAD}
# Findex — пошук документів: від токенізатора до API й NumPy

[Findex Live / Render](https://findex-web-search.onrender.com/) · [API](https://findex-web-search.onrender.com/docs)

**Корпус:** 1002 локальних документи (демо на Render: 18) · **p95:** 726.22 мс (Render, ЛР7; нове вимірювання потрібне) · **Найбільше прискорення:** {speedup()}.

{gif_note}

[Схема потоку даних](docs/lab08_architecture.md) · [Зведені таблиці ЛР1–8](docs/ALL_METRICS.md) · [Обмеження й відтворення](#лабораторна-8---профілювання-numpy-та-семантичний-пошук)

{TAIL}
"""
    if HEAD in original and TAIL in original:
        original = re.sub(
            re.escape(HEAD) + r".*?" + re.escape(TAIL),
            block.strip(),
            original,
            count=1,
            flags=re.DOTALL,
        )
    else:
        original = block + "\n" + original
    section = SECTION.read_text(encoding="utf-8").strip()
    start, end = "<!-- LAB08_START -->", "<!-- LAB08_END -->"
    if start in original and end in original:
        original = re.sub(
            re.escape(start) + r".*?" + re.escape(end),
            section,
            original,
            count=1,
            flags=re.DOTALL,
        )
    else:
        original = original.rstrip() + "\n\n" + section + "\n"
    README.write_text(original, encoding="utf-8")
    print("Updated README landing and Lab 08 section without deleting prior labs")

    # Collect existing Markdown tables in one location, grouped by lab heading.
    # This is a non-destructive extraction, not fabricated metrics.
    rows = original.splitlines()
    chunks: list[tuple[str, list[str]]] = []
    lab = "Загальний розділ / Findex"
    pos = 0
    while pos < len(rows):
        line = rows[pos]
        if line.startswith("#") and re.search(
            r"лабораторна\s+(?:робота\s+)?(?:№\s*)?[1-8]\b", line, flags=re.IGNORECASE
        ):
            lab = line.strip("# ")
        if (
            line.startswith("|")
            and pos + 1 < len(rows)
            and re.match(r"^\|[\s:|\-]+\|\s*$", rows[pos + 1])
        ):
            table = []
            while pos < len(rows) and rows[pos].startswith("|"):
                table.append(rows[pos])
                pos += 1
            chunks.append((lab, table))
            continue
        pos += 1
    hub = ROOT / "docs/ALL_METRICS.md"
    hub.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Зведені таблиці ЛР1–8",
        "",
        "Таблиці збережені з наявного README без зміни чисел. Дані ЛР8, якщо ще не згенеровані, не вигадуються.",
        "",
    ]
    for i, (lab, table) in enumerate(chunks, 1):
        header += [f"## {i}. {lab}", "", *table, ""]
    if not chunks:
        header += [
            "Таблиць у поточному README не знайдено — їх потрібно додати вручну."
        ]
    lab08_scores = ROOT / "docs/lab08_scoring.md"
    if lab08_scores.is_file():
        header.extend(
            [
                "## ЛР8 — вимірювання NumPy (власна машина)",
                "",
                lab08_scores.read_text(encoding="utf-8").strip(),
                "",
            ]
        )
    lab08_eval = ROOT / "docs/lab08_precision.csv"
    if lab08_eval.is_file():
        grouped: dict[str, list[float]] = {}
        with lab08_eval.open(encoding="utf-8", newline="") as stream:
            for record in csv.DictReader(stream):
                grouped.setdefault(record["mode"], []).append(float(record["p_at_5"]))
        header.extend(
            [
                "## ЛР8 — Precision@5 (ручна розмітка 15 запитів)",
                "",
                "| Режим | Mean P@5 | Кількість запитів |",
                "|---|---:|---:|",
            ]
        )
        for mode, values in grouped.items():
            header.append(
                f"| {mode} | {sum(values) / len(values):.3f} | {len(values)} |"
            )
        header.append("")
    hub.write_text("\n".join(header) + "\n", encoding="utf-8")
    print(f"Collected {len(chunks)} existing Markdown tables into {hub}")


if __name__ == "__main__":
    main()
