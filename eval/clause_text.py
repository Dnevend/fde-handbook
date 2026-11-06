#!/usr/bin/env python3
"""从 corpus/raw/ 还原条款正文，供 eval 校验「答案有没有原文支撑」。

corpus/index.json 只存条款号与 40 字预览，核对答案需要正文；正文在 raw/ 里、
不入库。本模块零依赖，缺 raw 时由调用方降级为警告。

条款切分规则与 corpus/index.py 保持一致，但更严格：只认行首的「第X条」，
不认正文里「依照本法第X条」这类引用；政府文件没有条号的，退回行首的
「1」「8.1」编号。切不出来的条目直接缺席，调用方按「无法核对」处理。

    python3 eval/clause_text.py --doc reg-safety-law --clause 第二十四条
    python3 eval/clause_text.py --stats
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "corpus/raw"
INDEX = ROOT / "corpus/index.json"

CN = "一二三四五六七八九十百零〇"
CLAUSE_LINE = re.compile(rf"^\s*第([{CN}]+)条(?=\s|$)", re.M)
NUMBERED_LINE = re.compile(r"^\s*(\d{1,3}(?:\.\d{1,2}){0,2})\s*[.、]?\s*(?=[\u4e00-\u9fff])", re.M)
HTML_EXT = {".html", ".htm"}
TEXT_EXT = {".txt", ".md"}


def read_document(doc_dir: Path) -> str:
    parts: list[str] = []
    for path in sorted(doc_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in HTML_EXT | TEXT_EXT:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix.lower() in HTML_EXT:
            text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
            text = re.sub(r"(?s)<[^>]+>", "\n", text)
            text = html.unescape(text)
        parts.append(text)
    text = "\n".join(parts)
    text = re.sub(r"[\u00a0\u3000\t ]+", " ", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _split(text: str, starts: list[re.Match]) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    for i, match in enumerate(starts):
        no = match.group(0).strip()
        if match.re is NUMBERED_LINE:
            no = match.group(1)
        body = text[match.end() : starts[i + 1].start() if i + 1 < len(starts) else len(text)]
        out.setdefault(no, []).append(body.strip())
    return {no: "\n".join(bodies) for no, bodies in out.items()}


def load_clause_occurrences(doc_ids: list[str] | None = None) -> dict[tuple[str, str], list[str]]:
    """{(doc, 条款号): [该条号在正文里的每一处]}。

    汇编类文档（如工作指南）会把不同法规的「第二十八条」排在一起，同一个条号
    对应多段正文。调用方需要判断答案落在哪一段，不能只看合并后的文本。
    """
    if doc_ids is None:
        if not INDEX.exists():
            return {}
        payload = json.loads(INDEX.read_text(encoding="utf-8"))
        doc_ids = [source["id"] for source in payload["sources"]]
    occurrences: dict[tuple[str, str], list[str]] = {}
    for doc in doc_ids:
        doc_dir = RAW / doc
        if not doc_dir.is_dir():
            continue
        text = read_document(doc_dir)
        starts = list(CLAUSE_LINE.finditer(text))
        if not starts:
            starts = list(NUMBERED_LINE.finditer(text))
        for i, match in enumerate(starts):
            no = match.group(0).strip()
            if match.re is NUMBERED_LINE:
                no = match.group(1)
            end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
            occurrences.setdefault((doc, no), []).append(text[match.end() : end].strip())
    return occurrences


def load_clause_texts(doc_ids: list[str] | None = None) -> dict[tuple[str, str], str]:
    """{(doc, 条款号): 合并后的正文}。重复条号会拼成一段，只用于宽松核对。"""
    return {key: "\n".join(bodies) for key, bodies in load_clause_occurrences(doc_ids).items()}


def lookup(doc: str, clause: str) -> str | None:
    return load_clause_texts([doc]).get((doc, clause))


def main() -> int:
    parser = argparse.ArgumentParser(description="查看 raw/ 里的条款正文")
    parser.add_argument("--doc")
    parser.add_argument("--clause")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.doc and args.clause:
        body = lookup(args.doc, args.clause)
        if body is None:
            raise SystemExit(f"没有找到 {args.doc} {args.clause}（raw 未下载或条款号不对）")
        print(body)
        return 0

    texts = load_clause_texts()
    by_doc: dict[str, int] = {}
    for doc, _ in texts:
        by_doc[doc] = by_doc.get(doc, 0) + 1
    for doc, count in sorted(by_doc.items()):
        print(f"{doc:24s} {count:4d} 条")
    print(f"合计 {len(texts)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
