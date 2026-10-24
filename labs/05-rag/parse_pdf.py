"""5.2 文档解析：文本层 PDF 的正文与表格抽取。

只测能测的：
  - 正文完整性：与 corpus/raw 里已有的 full.txt（pypdf 抽取）比 4-gram 覆盖；
  - 表格：pymupdf 的 find_tables 结果，数表/行/格，并对两张人工转录的小表算单元格精确率；
  - 版面顺序：没有双栏样本，不做结论（见 corpus/GAPS.md）。

不测 OCR、不测扫描件。产物：benchmarks/rag/parse.csv；表格预览落在 .cache/，不入库（含真实企业名与地址）。
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import raglib as r  # noqa: E402


def grams(text: str, size: int = 4) -> set[str]:
    text = re.sub(r"\s+", "", text)
    return {text[i:i + size] for i in range(max(len(text) - size + 1, 0))}


def normalize_cell(cell: str | None) -> str:
    return re.sub(r"\s+", "", cell or "")


def suspect(cell: str) -> bool:
    """文本层表格最常见的问题：小数点被挤到行首/行尾。"""
    compact = normalize_cell(cell)
    return bool(re.fullmatch(r"\d+\.", compact) or compact.startswith(".") or re.search(r"\d+\.\d*$", compact) is None and re.fullmatch(r"\d+", compact) and len(compact) > 2)


def table_accuracy(extracted: list[list[str]], truth: list[list[str]]) -> float:
    if not truth:
        return 0.0
    total = sum(len(row) for row in truth)
    hit = 0
    for row_index, truth_row in enumerate(truth):
        for col_index, truth_cell in enumerate(truth_row):
            try:
                got = extracted[row_index][col_index]
            except IndexError:
                continue
            if normalize_cell(got) == normalize_cell(truth_cell):
                hit += 1
    return hit / total


def main() -> int:
    truth = json.loads((r.BENCH / "parse-ground-truth.json").read_text(encoding="utf-8"))
    truth_by_page = {item["page"]: item for item in truth["tables"]}
    out: list[dict] = []
    previews: list[dict] = []

    for doc_id in ("eia-injection-01", "gov-ledger-guide"):
        folder = r.ROOT / "corpus/raw" / doc_id
        pdf = next(path for path in folder.iterdir() if path.suffix.lower() == ".pdf")
        full = folder / "full.txt"
        reference = full.read_text(encoding="utf-8", errors="ignore") if full.exists() else ""
        start = time.perf_counter()
        with pymupdf.open(pdf) as document:
            pages = [page.get_text("text") for page in document]
            tables_by_page = {}
            for page_number, page in enumerate(document, start=1):
                tables = page.find_tables().tables
                if tables:
                    tables_by_page[page_number] = [table.extract() for table in tables]
        elapsed = time.perf_counter() - start

        text = "\n".join(pages)
        rows = sum(len(table) for tables in tables_by_page.values() for table in tables)
        cells = sum(len(row) for tables in tables_by_page.values() for table in tables for row in table)
        flat = [cell for tables in tables_by_page.values() for table in tables for row in table for cell in row]
        empty = sum(1 for cell in flat if not normalize_cell(cell))
        multiline = sum(1 for cell in flat if "\n" in (cell or ""))
        dotted = sum(1 for cell in flat if suspect(cell))
        overlap = len(grams(reference) & grams(text)) / max(len(grams(reference)), 1) if reference else 0.0

        for page_number, item in truth_by_page.items():
            if page_number - 1 >= len(pages):
                continue
            tables = tables_by_page.get(page_number, [])
            if not tables:
                continue
            accuracy = table_accuracy(tables[0], item["cells"])
            out.append(
                {
                    "doc": doc_id,
                    "metric": f"table_accuracy_p{page_number}",
                    "value": round(accuracy, 3),
                }
            )

        out += [
            {"doc": doc_id, "metric": "pages", "value": len(pages)},
            {"doc": doc_id, "metric": "pages_with_text", "value": sum(1 for page in pages if page.strip())},
            {"doc": doc_id, "metric": "chars", "value": len(text)},
            {"doc": doc_id, "metric": "tables", "value": sum(len(tables) for tables in tables_by_page.values())},
            {"doc": doc_id, "metric": "table_rows", "value": rows},
            {"doc": doc_id, "metric": "table_cells", "value": cells},
            {"doc": doc_id, "metric": "empty_cell_ratio", "value": round(empty / max(cells, 1), 3)},
            {"doc": doc_id, "metric": "multiline_cell_ratio", "value": round(multiline / max(cells, 1), 3)},
            {"doc": doc_id, "metric": "suspect_decimal_cells", "value": dotted},
            {"doc": doc_id, "metric": "text_4gram_overlap_vs_pypdf", "value": round(overlap, 3)},
            {"doc": doc_id, "metric": "extract_s", "value": round(elapsed, 1)},
        ]

        for page_number, tables in list(tables_by_page.items())[:4]:
            previews.append({"doc": doc_id, "page": page_number, "tables": tables})

    r.write_csv(r.BENCH / "parse.csv", out)
    (r.CACHE / "parse-tables.json").write_text(json.dumps(previews, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in out:
        r.log(f"{row['doc']} {row['metric']} = {row['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
