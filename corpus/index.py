#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml", "pypdf", "beautifulsoup4", "lxml"]
# ///
"""给 full 档语料建条款索引，产出 corpus/index.json。

5.9 的权限过滤要回答"这条规则有没有效力、属哪一级"，光有全文不够，得先知道
哪一条是第几条、在第几章、来自哪一级文件。这个脚本把三件事抽出来：

  条款号   —— 第X条、第X章；政府文件没有条号的，退回 8 / 8.1 这类节号
  预览     —— 条款开头 40 字，够人工核对，不复制全文
  效力层级 —— 法律 / 部门规章 / 规范性文件 / 标准 / 内部制度 / 企业披露 …

索引只存定位与摘要，不存条款正文：正文在 raw/ 里，索引是可以入库的元数据。

用法：
    uv run corpus/index.py
    uv run corpus/index.py --out corpus/index.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

from excerpt import MANIFEST, extract_text, local_path  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent

# 效力层级不写在 sources.yaml 里，而是集中在这里：新增语料忘了登记时，
# 脚本会报"未定义效力层级"，比默默给个默认值安全。
AUTHORITY = {
    "reg-safety-law": "法律",
    "reg-mem-order-13": "部门规章",
    "gov-checklist-hs": "规范性文件（市级）",
    "gov-responsibility-zs": "规范性文件（市级）",
    "gov-ledger-guide": "规范性文件（市级）",
    "acc-fangsu-0518": "事故调查技术文档",
    "acc-meizhu-0116": "事故调查技术文档",
    "acc-yisheng-0527": "事故调查技术文档",
    "ipo-hitech": "企业公开披露",
    "ipo-borch": "企业公开披露",
    "eia-injection-01": "行政审批公示",
    "std-gbt-12555": "推荐性国家标准",
    "pat-inj-01": "专利文献",
    "pat-inj-02": "专利文献",
    "inst-equip-repair": "内部制度",
    "vendor-injection-manual": "厂商文档",
}

CLAUSE = re.compile(r"第([一二三四五六七八九十百零〇]+)条")
CHAPTER = re.compile(r"^\s*第([一二三四五六七八九十百零〇]+)[章节]", re.M)
GOV_CHAPTER = re.compile(r"^\s*([一二三四五六七八九十]+、[^\n]{0,20})", re.M)
NUMBERED = re.compile(r"^\s*(\d+(?:\.\d+){0,2})(?:[.、]|\s)\s*(\S[^\n]{0,30})")


def chapter_at(text: str, offset: int) -> str:
    """条款前面最近的一个章/节标题；政府文件没有章号时退回"一、xxx"。"""
    best_pos, best_title = -1, ""
    for pattern in (CHAPTER, GOV_CHAPTER):
        for match in pattern.finditer(text, 0, offset):
            if match.start() > best_pos:
                best_pos, best_title = match.start(), match.group(0).strip()
    return best_title


def clause_index(text: str) -> list[dict]:
    clauses = []
    seen = set()
    for match in CLAUSE.finditer(text):
        tail = text[match.end() : match.end() + 60]
        preview = re.sub(r"\s+", " ", tail).strip()[:40]
        key = (f"第{match.group(1)}条", preview[:12])
        if key in seen:
            continue
        seen.add(key)
        clauses.append(
            {
                "no": f"第{match.group(1)}条",
                "chapter": chapter_at(text, match.start()),
                "preview": preview,
            }
        )
    if clauses:
        return clauses

    # 政府文件（工作指南、检查清单）没有"第X条"，退回 8 / 8.1 / 1. 这类编号。
    for line in text.splitlines():
        if "...." in line:  # 目录的点导引，不是正文条目
            continue
        match = NUMBERED.match(line)
        if not match:
            continue
        key = (match.group(1), match.group(2)[:12])
        if key in seen:
            continue
        seen.add(key)
        clauses.append({"no": match.group(1), "chapter": "", "preview": match.group(2).strip()})
    return clauses


def main() -> int:
    parser = argparse.ArgumentParser(description="给 full 档语料建条款索引")
    parser.add_argument("--out", type=pathlib.Path, default=ROOT / "index.json")
    args = parser.parse_args()

    import yaml

    sources = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["sources"]
    targets = [s for s in sources if s["redistributable"] == "full"]

    missing_authority = [s["id"] for s in targets if s["id"] not in AUTHORITY]
    if missing_authority:
        raise SystemExit(f"这些语料没登记效力层级：{', '.join(missing_authority)}（改 index.py 的 AUTHORITY）")

    payload = {"generated_from": "corpus/sources.yaml", "sources": []}
    for item in targets:
        path = local_path(item)
        if not path.exists():
            print(f"跳过 {item['id']}（未下载，先跑 corpus/fetch.sh）", file=sys.stderr)
            continue
        text = extract_text(path)
        clauses = clause_index(text)
        payload["sources"].append(
            {
                "id": item["id"],
                "title": item["title"],
                "authority": AUTHORITY[item["id"]],
                "doc_type": item["doc_type"],
                "clauses": clauses,
                "clause_count": len(clauses),
            }
        )
        print(f"{item['id']:24s} {AUTHORITY[item['id']]:12s} {len(clauses):4d} 条")

    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    total = sum(s["clause_count"] for s in payload["sources"])
    print(f"\n{len(payload['sources'])} 份，{total} 条 → {args.out.relative_to(ROOT.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
