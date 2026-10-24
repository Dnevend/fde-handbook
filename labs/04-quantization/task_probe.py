#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=2,<3"]
# ///
"""任务正确率探针：在合成工单上做「从文档里取一个字段」的抽取题，自动判分。

题目从 corpus/raw/synthetic 生成，答案就是文档表格里的字段值，所以判分不需要模型。
用途不是证明模型多强，而是量出**这个指标本身有多粗**：60 道题、正确率 95% 时，
95% 置信区间仍有约 ±11 个百分点，比我们想检测的差异大一个数量级。

用法：
  source labs/env.sh
  uv run labs/04-quantization/task_probe.py                 # 起服务、跑三遍、关服务
  uv run labs/04-quantization/task_probe.py --passes 1 --temperature 0

产出：
  benchmarks/quantization/probe-items.csv    题目与标准答案（可入库）
  benchmarks/quantization/task-probe.csv     每次运行的逐题结果
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
import re
import socket
import subprocess
import sys
import time
import urllib.request

from openai import OpenAI

ROOT = pathlib.Path(__file__).resolve().parents[2]
SYNTH = ROOT / "corpus" / "raw" / "synthetic"

# 每种单据取一个字段：既唯一、又必须真的读表才能答对。
RULES = {
    "process_card": (re.compile(r"\|\s*模温\(℃\)\s*\|"), 0),
    "daily_inspection": (re.compile(r"日期：([0-9-]{10})"), 1),
    "maintenance_plan": (re.compile(r"下次执行：([0-9-]{10})"), 1),
    "repair_order": (re.compile(r"停机：([0-9.]+ h)"), 1),
    "mold_change": (re.compile(r"\|\s*换模结束\s*\|\s*([^|]+)\|"), 1),
    "quality_incident": (re.compile(r"\|\s*批次\s*\|\s*([^|]+)\|"), 1),
}

QUESTIONS = {
    "process_card": "这张工艺参数卡里，保压时间与冷却时间相加是多少秒？只回答数字。",
    "daily_inspection": "这张点检表的日期是哪一天？只回答 YYYY-MM-DD。",
    "maintenance_plan": "这份保养计划的下次执行日期是哪一天？只回答 YYYY-MM-DD。",
    "repair_order": "这张维修工单的停机时间是多少？只回答数字加 h，例如 2 h。",
    "mold_change": "这份换模记录的换模结束日期是哪一天？只回答 YYYY-MM-DD。",
    "quality_incident": "这张质量异常单的批次号是什么？只回答批次号。",
}


def parse_doc(text: str, kind: str) -> str | None:
    pattern, mode = RULES[kind]
    if kind == "process_card":
        # 列宽随表格变化，按表头定位比数第几列稳
        header = next((l for l in text.splitlines() if "模温(℃)" in l and "产品名称" in l), None)
        if not header:
            return None
        cols = [c.strip() for c in header.strip().strip("|").split("|")]
        rows = [l for l in text.splitlines() if l.startswith("|") and "---" not in l]
        data = next((r for r in rows if not r.startswith("| 产品名称")), None)
        if not data:
            return None
        cells = [c.strip() for c in data.strip().strip("|").split("|")]
        try:
            total = float(cells[cols.index("保压时间(s)")]) + float(cells[cols.index("冷却时间(s)")])
        except (ValueError, IndexError):
            return None
        return f"{total:g}"
    m = pattern.search(text)
    return m.group(mode).strip() if m else None


def build_items() -> list[dict[str, str]]:
    items = []
    for kind in RULES:
        folder = SYNTH / kind
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            answer = parse_doc(text, kind)
            if not answer:
                continue
            # 只给正文，不给 frontmatter，避免模型从元数据里抄答案
            body = text.split("---", 2)[-1].strip()
            items.append(
                {
                    "doc_id": path.stem,
                    "kind": kind,
                    "question": QUESTIONS[kind],
                    "answer": answer,
                    "body": body,
                }
            )
    return items


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(model: str, port: int, ctx: int, parallel: int, ctk: str, ctv: str) -> subprocess.Popen:
    cmd = [
        "llama-server",
        "-m", model,
        "-c", str(ctx),
        "-fa", "on",
        "-ctk", ctk,
        "-ctv", ctv,
        "--port", str(port),
        "-np", str(parallel),
        "--no-webui",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                if r.status == 200:
                    return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise SystemExit("llama-server 启动失败，先手动跑一次看看报错")
        time.sleep(1)
    proc.kill()
    raise SystemExit("llama-server 180 秒内没就绪")


def normalize(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def wilson(k: int, n: int) -> tuple[float, float]:
    """二项分布的 95% Wilson 区间。"""
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL"))
    ap.add_argument("--out-dir", default=str(ROOT / "benchmarks" / "quantization"))
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--passes", type=int, default=3, help="跑几遍；第一遍温度 0，其余用 --temperature")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--label", default="f16-f16", help="写进 CSV，标记这轮用的 KV 类型")
    ap.add_argument("--ctk", default="f16")
    ap.add_argument("--ctv", default="f16")
    ap.add_argument("--url", default="", help="已有服务地址时直接用它，不再自己起")
    args = ap.parse_args()

    if not args.model and not args.url:
        print("缺少 --model，先 source labs/env.sh", file=sys.stderr)
        return 1

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = build_items()
    if not items:
        print("没找到合成语料，先跑 uv run corpus/synthetic/generate.py", file=sys.stderr)
        return 1

    with (out / "probe-items.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "kind", "question", "answer"])
        for it in items:
            w.writerow([it["doc_id"], it["kind"], it["question"], it["answer"]])
    print(f"题目 {len(items)} 道，写出 {out / 'probe-items.csv'}")

    proc = None
    base_url = args.url
    if not base_url:
        port = free_port()
        base_url = f"http://127.0.0.1:{port}/v1"
        print(f"起服务：ctx={args.ctx} kv={args.ctk}/{args.ctv} port={port}")
        proc = start_server(args.model, port, args.ctx, args.parallel, args.ctk, args.ctv)

    client = OpenAI(base_url=base_url, api_key="sk-no-key")

    rows = []
    try:
        for p in range(args.passes):
            temp = 0.0 if p == 0 else args.temperature
            seed = 0 if p == 0 else p
            correct = 0
            started = time.time()
            for it in items:
                resp = client.chat.completions.create(
                    model="local",
                    temperature=temp,
                    seed=seed,
                    max_tokens=24,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    messages=[
                        {
                            "role": "system",
                            "content": "你在做文档字段抽取。只根据用户给的文档回答，只输出答案本身，不要解释。",
                        },
                        {"role": "user", "content": f"{it['body']}\n\n问题：{it['question']}"},
                    ],
                )
                got = (resp.choices[0].message.content or "").strip()
                ok = normalize(it["answer"]) in normalize(got)
                correct += int(ok)
                rows.append([args.label, f"pass{p}", f"{temp}", it["doc_id"], it["kind"], it["answer"], got.replace("\n", " "), int(ok)])
            lo, hi = wilson(correct, len(items))
            print(
                f"{args.label} pass{p} temp={temp}: {correct}/{len(items)} = {correct / len(items):.1%}"
                f"（95% CI {lo:.1%}–{hi:.1%}），{time.time() - started:.0f}s"
            )
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=30)

    path = out / "task-probe.csv"
    fresh = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["label", "pass", "temperature", "doc_id", "kind", "answer", "got", "correct"])
        w.writerows(rows)
    print(f"追加 {len(rows)} 行到 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
