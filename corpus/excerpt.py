#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml", "pypdf", "beautifulsoup4", "lxml"]
# ///
"""把 excerpt 档的语料切成节选，落到 raw/<id>/__excerpt.md。

fetch.py 只管 full 档，excerpt 要人工决定"取哪几节"，这个脚本就是那把刀：
章节边界写在 sources.yaml 的 ranges 里（正则，成对出现），切完报告占原文的比例，
超过 excerpt_ratio 就报错，逼你回去收窄。

为什么不在切的时候就脱敏：raw/ 已被 .gitignore 覆盖，节选留在本机用于实验；
真实企业名与个人信息只在这一层存在，写到正文时按 redaction.yaml 替换。把脱敏
放在"进正文"这一步，而不是"切语料"这一步，少一道可能把证据洗掉的工序。

用法：
    uv run corpus/excerpt.py --id acc-fangsu-0518
    uv run corpus/excerpt.py --id acc-fangsu-0518,acc-yisheng-0527
    uv run corpus/excerpt.py --all
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
MANIFEST = ROOT / "sources.yaml"
RAW_DIR = ROOT / "raw"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124 Safari/537.36 fde-handbook-corpus/1"
)


def load_sources() -> list[dict]:
    import yaml

    with MANIFEST.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)["sources"]


def local_path(item: dict) -> pathlib.Path:
    name = pathlib.PurePosixPath(urllib.parse.unquote(urllib.parse.urlparse(item["url"]).path)).name
    return RAW_DIR / item["id"] / name


def ensure_file(item: dict) -> pathlib.Path:
    dest = local_path(item)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if not item.get("url"):
        raise ValueError("没有 url，先按 resolve 检索后补进 sources.yaml")
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载 {item['id']} {item['url']}")
    request = urllib.request.Request(item["url"], headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=90) as response, dest.open("wb") as fh:
        fh.write(response.read())
    return dest


def extract_text(path: pathlib.Path) -> str:
    suffix = path.suffix.lower()
    head = path.open("rb").read(2048).lower()
    is_pdf = suffix == ".pdf" or head.startswith(b"%pdf")
    is_html = suffix in {".html", ".htm"} or b"<html" in head or b"<!doctype" in head

    if is_pdf:
        from pypdf import PdfReader

        reader = PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if is_html:
        from bs4 import BeautifulSoup

        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return re.sub(r"\n{2,}", "\n", soup.get_text("\n"))
    return path.read_text(encoding="utf-8", errors="ignore")


def slice_ranges(text: str, ranges: list[dict]) -> list[tuple[str, str]]:
    out = []
    for spec in ranges:
        start = re.search(spec["start"], text)
        if not start:
            raise ValueError(f"找不到起始锚点：{spec['start']}")
        end = re.search(spec["end"], text[start.end():])
        if not end:
            raise ValueError(f"在「{spec['start']}」之后找不到结束锚点：{spec['end']}")
        stop = start.end() + end.start()
        out.append((spec.get("label", ""), text[start.start():stop].strip()))
    return out


def build_excerpt(item: dict, full: str) -> tuple[str, float]:
    ranges = item.get("ranges")
    if not ranges:
        raise ValueError("sources.yaml 条目缺 ranges，先补章节锚点")

    chunks = slice_ranges(full, ranges)
    body = "\n\n".join(
        f"<!-- {label} -->\n\n{chunk}" if label else chunk for label, chunk in chunks
    )
    total = sum(len(chunk) for _, chunk in chunks)
    ratio = total / max(len(full), 1)

    header = (
        f"<!-- source: {item['url']}\n"
        f"     title: {item['title']}\n"
        f"     license: {item['license']}\n"
        f"     excerpt_ratio: {ratio:.3f}（上限 {item.get('excerpt_ratio')}）\n"
        f"     注意：本节选含真实企业名与个人信息，仅本机实验使用，不入库。\n"
        f"     写入正文前按 corpus/redaction.yaml 替换。 -->\n\n"
    )
    return header + body + "\n", ratio


def process(item: dict, force: bool) -> str:
    if item.get("ocr_sample"):
        return f"跳过 {item['id']}（扫描件，无文本层，留作 5.2 的 OCR 样本）"

    out = RAW_DIR / item["id"] / "__excerpt.md"
    if out.exists() and not force:
        return f"跳过 {item['id']}（已存在 {out.relative_to(ROOT)}）"

    path = ensure_file(item)
    full = extract_text(path)
    (path.parent / "full.txt").write_text(full, encoding="utf-8")

    if len(full.strip()) < 500:
        return f"失败 {item['id']}（抽出 {len(full)} 字，疑为扫描件；在 sources.yaml 标记 ocr_sample: true）"

    text, ratio = build_excerpt(item, full)
    limit = item.get("excerpt_ratio")
    if isinstance(limit, (int, float)) and ratio > limit:
        raise ValueError(
            f"节选占原文 {ratio:.1%}，超过上限 {limit:.0%}；收窄 ranges 再跑"
        )

    out.write_text(text, encoding="utf-8")
    return f"ok   {item['id']}  {len(text)} 字  占原文 {ratio:.1%}  → {out.relative_to(ROOT)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="切 excerpt 档语料")
    parser.add_argument("--id", help="逗号分隔的 id")
    parser.add_argument("--all", action="store_true", help="处理所有 excerpt 档")
    parser.add_argument("--force", action="store_true", help="已存在也重切")
    args = parser.parse_args()

    sources = load_sources()
    if args.all:
        selected = [s for s in sources if s["redistributable"] == "excerpt"]
    elif args.id:
        wanted = {part.strip() for part in args.id.split(",") if part.strip()}
        selected = [s for s in sources if s["id"] in wanted]
        unknown = wanted - {s["id"] for s in selected}
        if unknown:
            raise SystemExit(f"清单里没有这些 id：{', '.join(sorted(unknown))}")
    else:
        parser.error("要么 --id，要么 --all")

    failures = 0
    for item in selected:
        if item["redistributable"] != "excerpt":
            print(f"跳过 {item['id']}（不是 excerpt 档）")
            continue
        try:
            print(process(item, args.force))
        except (ValueError, urllib.error.URLError, TimeoutError, OSError) as exc:
            failures += 1
            print(f"失败 {item['id']}：{exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
