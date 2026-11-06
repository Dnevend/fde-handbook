#!/usr/bin/env python3
"""8.7 交付包自检：文件齐不齐、关键章节在不在、有没有漏填/密钥。

它不判断“写得好不好”，只做交付前的最低门槛：
- 8 个核心文档存在且不是空壳；
- 每份文档有约定的小节（按标题行匹配，不匹配代码块里的注释）；
- 告警/许可证/验收等表格至少有一行数据；
- 没有未填的【占位符】/TODO/中文 <...>/待填/XXX；
- 没有明显的密钥/密码值；
- 交付包有 MANIFEST.sha256，清单覆盖包内所有文件，且逐文件哈希能对上。

用法：
    python3 labs/08-delivery/delivery/check_delivery.py \
      --allow-placeholders --allow-missing-manifest     # 模板自检
    python3 labs/08-delivery/delivery/check_delivery.py --dir dist/delivery-xxx
    python3 labs/08-delivery/delivery/check_delivery.py --dir ... --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parent / "templates"

REQUIRED = {
    "00-README.md": ["版本", "范围", "联系人", "目录", "变更记录"],
    "01-architecture.md": ["组件", "数据流", "信任边界", "版本与依赖", "容量与 SLO", "已知限制"],
    "02-deploy.md": ["前置条件", "安装", "配置项", "验证", "回滚", "离线"],
    "03-operations.md": ["启停", "健康检查", "日志与追踪", "常见告警", "备份与恢复", "升级", "故障处理"],
    "04-acceptance.md": ["验收标准", "测试用例", "证据", "未决问题", "签字"],
    "05-training.md": ["培训对象", "议程", "练习", "考核", "材料与记录"],
    "06-ai-usage-policy.md": ["数据分类", "允许的用法", "禁止的用法", "引用与人工复核", "事件上报"],
    "07-licenses.md": ["模型", "语料", "软件", "合规检查", "签字"],
}

TABLE_ROWS = {
    "04-acceptance.md": [r"^\|\s*A\d+\s*\|", r"^\|\s*直接问答\s*\|"],
}
TABLE_SECTIONS = {
    "03-operations.md": ["常见告警"],
    "07-licenses.md": ["模型", "语料", "软件"],
}

FENCE = re.compile(r"```.*?```", re.S)
PLACEHOLDER = re.compile(
    r"【[^】]+】|TODO|TBD|<[^>\n]*[\u4e00-\u9fff][^>\n]*>|<[A-Z][A-Z0-9_]{2,}>|待填|待补|XXX"
)
SECRETS = [
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |PRIVATE )?PRIVATE KEY-----"),
    re.compile(
        r"(?i)(?:api[_-]?key|secret|token|passwd|password)\s*[:=]\s*"
        r"(?!\$\{|\$[A-Z_]|os\.environ|<|【|不|无|见)[^\s【]{6,}"
    ),
    re.compile(
        r"(?:密码|口令)\s*[:：=]\s*(?!\$\{|\$[A-Z_]|os\.environ|<|【|不|无|见)[^\s【]{2,}"
    ),
]
TEXT_SUFFIXES = {".md", ".txt", ".sh", ".py", ".yaml", ".yml", ".json", ".toml", ".ini", ".env"}
SKIP_DIRS = {"models", "wheels", ".git", "__pycache__", "node_modules"}


def strip_fences(text: str) -> str:
    return FENCE.sub("", text)


def section_has_table_row(text: str, heading: str) -> bool:
    match = re.search(
        rf"(?ms)^#+[^\n]*{re.escape(heading)}[^\n]*\n(.*?)(?=^#+\s|\Z)",
        text,
    )
    if not match:
        return False
    rows = [line for line in match.group(1).splitlines() if line.strip().startswith("|")]
    # 表头 + 分隔行 + 至少一行数据
    return len(rows) >= 3


def check_file(path: Path, headings: list[str], allow_placeholders: bool) -> list[str]:
    problems: list[str] = []
    if not path.is_file():
        return [f"缺文件：{path.name}"]
    text = path.read_text(encoding="utf-8")
    if len(text.strip()) < 200:
        problems.append(f"{path.name}: 内容太短（{len(text.strip())} 字符）")

    heading_text = strip_fences(text)
    for heading in headings:
        if not re.search(rf"(?m)^#{{1,6}}\s*.*{re.escape(heading)}", heading_text):
            problems.append(f"{path.name}: 缺标题「{heading}」")
    for pattern in TABLE_ROWS.get(path.name, []):
        if not re.search(rf"(?m)^{pattern}", text):
            problems.append(f"{path.name}: 缺数据行（{pattern}）")
    for heading in TABLE_SECTIONS.get(path.name, []):
        if not section_has_table_row(text, heading):
            problems.append(f"{path.name}: 「{heading}」表缺数据行")

    if not allow_placeholders:
        for match in PLACEHOLDER.finditer(text):
            problems.append(f"{path.name}: 未填占位符「{match.group(0)[:40]}」")
    for pattern in SECRETS:
        match = pattern.search(text)
        if match:
            problems.append(f"{path.name}: 疑似密钥/密码「{match.group(0)[:40]}」")
    return problems


def check_manifest(base: Path) -> list[str]:
    manifest = base / "MANIFEST.sha256"
    if not manifest.is_file():
        return ["缺 MANIFEST.sha256（真实交付包必须有逐文件校验清单）"]

    problems: list[str] = []
    listed: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            problems.append(f"MANIFEST.sha256: 行格式不对：{line[:60]}")
            continue
        digest, rel = parts
        listed.add(rel)
        path = base / rel
        if not path.is_file():
            problems.append(f"MANIFEST.sha256: 缺文件 {rel}")
            continue
        actual = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                actual.update(block)
        if actual.hexdigest() != digest:
            problems.append(f"MANIFEST.sha256: 哈希不匹配 {rel}")

    for name in REQUIRED:
        if name not in listed:
            problems.append(f"MANIFEST.sha256: 没列核心文档 {name}")
    for path in base.rglob("*"):
        if path.is_file() and path.name != "MANIFEST.sha256":
            rel = path.relative_to(base).as_posix()
            if rel not in listed:
                problems.append(f"MANIFEST.sha256: 没列文件 {rel}")
    return problems


def scan_extra(base: Path, checked: set[Path], allow_placeholders: bool) -> list[str]:
    """扫交付包里的 config.env/run.sh/install.sh 等文本文件，避免只查 8 份文档。"""
    problems: list[str] = []
    for path in base.rglob("*"):
        if not path.is_file() or path in checked:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name == "MANIFEST.sha256" or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.stat().st_size > 2 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if not allow_placeholders:
            match = PLACEHOLDER.search(text)
            if match:
                problems.append(f"{path.relative_to(base)}: 未填占位符「{match.group(0)[:40]}」")
        for pattern in SECRETS:
            match = pattern.search(text)
            if match:
                problems.append(f"{path.relative_to(base)}: 疑似密钥/密码「{match.group(0)[:40]}」")
                break
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="8.7 交付包自检")
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help="交付包目录（默认 templates）")
    parser.add_argument("--allow-placeholders", action="store_true", help="模板自检：允许【占位符】")
    parser.add_argument(
        "--allow-missing-manifest",
        action="store_true",
        help="模板自检：允许没有 MANIFEST.sha256（不影响占位符检查）",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    base = Path(args.dir)
    problems: list[str] = []
    checked: set[Path] = set()
    for name, headings in REQUIRED.items():
        path = base / name
        checked.add(path)
        problems.extend(check_file(path, headings, args.allow_placeholders))
    problems.extend(scan_extra(base, checked, args.allow_placeholders))

    manifest = base / "MANIFEST.sha256"
    if manifest.is_file():
        problems.extend(check_manifest(base))
    elif not args.allow_missing_manifest:
        problems.append("缺 MANIFEST.sha256（真实交付包必须有逐文件校验清单）")

    result = {
        "dir": str(base),
        "files": len(REQUIRED),
        "allow_placeholders": args.allow_placeholders,
        "allow_missing_manifest": args.allow_missing_manifest,
        "manifest": manifest.is_file(),
        "problems": problems,
        "ok": not problems,
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"交付包：{base}")
        print(f"核心文档：{len(REQUIRED)} 份；MANIFEST：{'有' if manifest.is_file() else '无'}")
        if problems:
            print(f"FAIL：{len(problems)} 个问题")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print("PASS：文件齐全、关键小节齐全、没有未填占位符/明显密钥")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
