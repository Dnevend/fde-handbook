#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml"]
# ///
"""按 corpus/sources.yaml 抓取可直接入库的语料，并打印需要人工处理的清单。

只下载 redistributable: full 且 url 非空的条目。excerpt 与 none 一律不自动下载——
前者要人工挑章节，后者根本不能进仓库。这个限制是刻意的：让"顺手 git add -f"多一道手续。

用法：
    uv run corpus/fetch.py --list          # 只看清单，不联网
    uv run corpus/fetch.py --dry-run       # 打印将要做什么
    uv run corpus/fetch.py                 # 执行
    uv run corpus/fetch.py --only reg-mem-order-13,gov-checklist-hs
    uv run corpus/fetch.py --force         # 已存在也重下

产出：
    corpus/raw/<id>/<文件名>
    corpus/raw/<id>/.meta.json             # url、retrieved_at、sha256、bytes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
import unicodedata
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
LEVELS = ("full", "excerpt", "none")


def load_manifest(path: pathlib.Path) -> dict:
    import yaml

    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "sources" not in data:
        raise SystemExit(f"{path} 缺少 sources 列表")
    return data


def select(sources: list[dict], only: str | None) -> list[dict]:
    if not only:
        return sources
    wanted = {part.strip() for part in only.split(",") if part.strip()}
    unknown = wanted - {s["id"] for s in sources}
    if unknown:
        raise SystemExit(f"清单里没有这些 id：{', '.join(sorted(unknown))}")
    return [s for s in sources if s["id"] in wanted]


def target_name(item: dict, url: str) -> str:
    path = urllib.parse.unquote(urllib.parse.urlparse(url).path)
    name = pathlib.PurePosixPath(path).name
    if not name or "." not in name:
        name = f"{item['id']}.bin"
    return name


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: pathlib.Path) -> tuple[str, int]:
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=90) as response, dest.open("wb") as fh:
        while chunk := response.read(1 << 16):
            fh.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def display_width(text: str) -> int:
    """终端里中文占两列，len() 会把表头对歪。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text: str, width: int) -> str:
    return text + " " * max(0, width - display_width(text))


def report_table(sources: list[dict]) -> None:
    width_id = max(display_width(s["id"]) for s in sources)
    width_level = max(display_width(s["redistributable"]) for s in sources)
    print(f"{pad('id', width_id)}  {pad('档位', width_level)}  状态  标题")
    for s in sources:
        level = s["redistributable"]
        if s.get("url"):
            status = "可下载"
        elif s.get("resolve"):
            status = "待检索"
        else:
            status = "缺地址"
        print(f"{pad(s['id'], width_id)}  {pad(level, width_level)}  {status}  {s['title']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="按 sources.yaml 抓取语料")
    parser.add_argument("--list", action="store_true", help="只列清单，不联网")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不下载")
    parser.add_argument("--only", help="逗号分隔的 id，只处理这些")
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = parser.parse_args()

    manifest = load_manifest(MANIFEST)
    sources = select(manifest["sources"], args.only)

    bad_levels = {s["id"] for s in sources if s["redistributable"] not in LEVELS}
    if bad_levels:
        raise SystemExit(f"redistributable 取值非法：{', '.join(sorted(bad_levels))}")

    if args.list:
        report_table(sources)
        return 0

    counts = {level: 0 for level in LEVELS}
    downloaded: list[tuple[str, str, str, int]] = []
    failures: list[tuple[str, str]] = []
    manual: list[dict] = []

    for item in sources:
        level = item["redistributable"]
        counts[level] += 1
        url = item.get("url")

        if level != "full" or not url:
            manual.append(item)
            continue

        name = target_name(item, url)
        folder = RAW_DIR / item["id"]
        dest = folder / name

        if dest.exists() and dest.stat().st_size > 0 and not args.force:
            expected = item.get("sha256")
            if expected and sha256_of(dest) != expected:
                print(f"重下  {item['id']}（本地文件与清单 sha256 不符）")
            else:
                print(f"跳过  {item['id']}（已存在 {dest.relative_to(ROOT)}）")
                continue

        print(f"下载  {item['id']}  {url}")
        if args.dry_run:
            continue

        folder.mkdir(parents=True, exist_ok=True)
        try:
            sha, size = download(url, dest)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            failures.append((item["id"], str(exc)))
            print(f"  失败：{exc}", file=sys.stderr)
            continue

        expected = item.get("sha256")
        if expected and expected != sha:
            failures.append((item["id"], f"sha256 不匹配：清单 {expected[:12]}… 实际 {sha[:12]}…"))
            print(f"  sha256 不匹配：清单 {expected[:12]}… 实际 {sha[:12]}…", file=sys.stderr)
            dest.unlink(missing_ok=True)
            continue

        meta = {
            "id": item["id"],
            "url": url,
            "file": name,
            "sha256": sha,
            "bytes": size,
            "retrieved_at": time.strftime("%Y-%m-%d"),
        }
        (folder / ".meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        downloaded.append((item["id"], name, sha, size))
        print(f"  ok  {size} 字节  sha256 {sha[:12]}…")

    if manual:
        print("\n需要人工处理：")
        for item in manual:
            level = item["redistributable"]
            print(f"\n[{level}] {item['id']} — {item['title']}")
            if item.get("url"):
                print(f"  下载：{item['url']}")
            elif item.get("resolve"):
                print(f"  检索：{item['resolve']}")
            else:
                print("  下载：（清单里没有地址，需补 url 或 resolve）")
            sections = item.get("sections") or []
            if sections:
                print(f"  只取：{'；'.join(sections)}")
            if level == "excerpt":
                ratio = item.get("excerpt_ratio")
                hint = f"（不超过原文 {ratio:.0%}）" if isinstance(ratio, float) else ""
                print(f"  产出：corpus/raw/{item['id']}/__excerpt.md{hint}")
            if level == "none":
                print("  注意：不下载，只记链接与条款号")
            drop = item.get("redaction") or []
            if drop:
                print(f"  脱敏：{'、'.join(drop)}（见 corpus/redaction.yaml）")

    print("\n汇总：")
    print(f"  可全文入库 {counts['full']} 份，已下载 {len(downloaded)} 份")
    print(f"  仅节选     {counts['excerpt']} 份")
    print(f"  仅存链接   {counts['none']} 份")

    if downloaded:
        print("\n把 sha256 回填到 sources.yaml 对应条目（可选，但建议）：")
        for item_id, name, sha, _size in downloaded:
            print(f"  {item_id}: sha256: {sha}")

    if failures:
        print(f"\n{len(failures)} 份下载失败：", file=sys.stderr)
        for item_id, reason in failures:
            print(f"  {item_id}: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
