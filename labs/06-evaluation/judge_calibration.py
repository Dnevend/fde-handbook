#!/usr/bin/env python3
"""6.3 LLM-as-judge：用本地 9B 当裁判，和 80 条人工标签对齐。

三组实验，全部在第五部分的 80 条带人工标签的回答上做（这是本书唯一有人工
标签的生成数据）：

  1. 逐条判 correct / partial / wrong，两种提示词：分类 rubric 与 0-100 打分；
     算与人工标签的准确率、Cohen's κ、混淆矩阵；
  2. 偏置：给正确回答补一句无害的废话，看标签/分数是否变化；把两个回答
     交换 A/B 顺序，看胜负是否翻转；
  3. 拒绝题单独判：明确拒答算对，编造算错。

用法（在 labs/ 下）：
    uv run 06-evaluation/judge_calibration.py
    uv run 06-evaluation/judge_calibration.py --limit 20   # 先抽 20 条试跑

产物：
    benchmarks/evaluation/judge-pointwise.csv   逐条的人类/裁判标签与耗时
    benchmarks/evaluation/judge-bias.csv        废话与顺序偏置
    benchmarks/evaluation/judge-confusion.csv   三分类混淆矩阵
    benchmarks/evaluation/judge-summary.csv     准确率、κ、F1、耗时
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "labs/05-rag"))

import raglib as r  # noqa: E402

GEN = ROOT / "benchmarks/rag/generation.jsonl"
AUDIT = ROOT / "benchmarks/rag/generation-audit.csv"
OUT = ROOT / "benchmarks/evaluation"
LABELS = ["correct", "partial", "wrong"]

STRICT_PROMPT = """你是严格的事实评估员。给你一个问题、一个标准答案、一个模型回答。
判断模型回答属于哪一类：

- correct：关键事实、数字、适用条件与标准答案一致；允许措辞不同，允许补充不冲突的细节。
- partial：部分正确，但遗漏关键条件或数字，或含与标准答案冲突的次要细节。
- wrong：关键事实错误、答非所问，或编造了标准答案没有的内容。

只输出 JSON，不要解释：{"label":"correct|partial|wrong","reason":"不超过 30 字"}"""

SCORE_PROMPT = """你是严格的事实评估员。给你一个问题、一个标准答案、一个模型回答。
给回答打 0-100 分：
- 90-100：关键事实、数字、条件全对；
- 60-89：主体正确，但遗漏或含糊了关键条件/数字；
- 30-59：只对了一部分，或混入了错误细节；
- 0-29：关键事实错误、答非所问，或编造。

只输出 JSON，不要解释：{"score": 0-100 的整数, "reason": "不超过 30 字"}"""

REFUSE_PROMPT = """你是严格的事实评估员。资料里没有答案，模型应当明确拒答。
判断模型回答：
- correct：明确说明资料中没有相关信息或无法回答，且没有给出具体事实；
- wrong：编造了具体信息、给出猜测，或含糊其辞但没有明确拒答。

只输出 JSON，不要解释：{"label":"correct|wrong","reason":"不超过 30 字"}"""

PAIR_PROMPT = """你是评估员。给你一个问题、一个标准答案、两个回答 A 和 B。
哪个回答更符合标准答案？允许措辞不同；关键事实错误、遗漏关键条件或编造内容的回答更差。
只输出 JSON，不要解释：{"winner":"A|B|tie","reason":"不超过 30 字"}"""

REFUSAL = ["资料中没有相关信息", "无法回答", "没有相关信息", "资料不足", "未提供相关信息"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def judge(system: str, user: str, max_tokens: int = 160) -> tuple[dict, float, str]:
    start = time.perf_counter()
    raw = r.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
    )
    elapsed = time.perf_counter() - start
    return parse_json(raw), elapsed, raw.strip()


def strict_label(payload: dict) -> str:
    label = str(payload.get("label", "")).lower().strip()
    return label if label in LABELS else ""


def score_label(payload: dict) -> str:
    try:
        score = int(payload.get("score"))
    except (TypeError, ValueError):
        return ""
    if score >= 90:
        return "correct"
    if score >= 30:
        return "partial"
    return "wrong"


def is_refusal(text: str) -> bool:
    return any(marker in text for marker in REFUSAL)


def condition(payload: dict, fallback: str = "") -> str:
    return str(payload.get("winner", fallback)).strip().upper()


def kappa(human: list[str], judge_labels: list[str]) -> float:
    n = len(human)
    if not n:
        return float("nan")
    observed = sum(1 for a, b in zip(human, judge_labels) if a == b) / n
    human_dist = Counter(human)
    judge_dist = Counter(judge_labels)
    expected = sum(human_dist[label] / n * judge_dist[label] / n for label in set(human) | set(judge_labels))
    return (observed - expected) / (1 - expected) if expected < 1 else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM 裁判校准")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（试跑）")
    parser.add_argument("--no-bias", action="store_true", help="跳过偏置实验")
    args = parser.parse_args()

    outputs = {row["id"]: row for row in read_jsonl(GEN) if row["mode"] == "rag"}
    audit = {row["id"]: row for row in csv.DictReader(AUDIT.open(encoding="utf-8"))}
    ids = [qid for qid in audit if qid in outputs]
    if args.limit:
        ids = ids[: args.limit]

    pointwise = []
    for number, qid in enumerate(ids, start=1):
        output = outputs[qid]
        human = audit[qid]["manual_label"]
        question, gold, answer = output["question"], output["gold"], output["answer"]
        if output["category"] == "refuse":
            payload, elapsed, raw = judge("你只输出 JSON。", f"{REFUSE_PROMPT}\n\n问题：{question}\n模型回答：{answer}")
            strict = "correct" if str(payload.get("label", "")).lower() == "correct" else "wrong"
            score = ""
            score_cls = ""
            reason = str(payload.get("reason", ""))[:60]
        else:
            user = f"{STRICT_PROMPT}\n\n问题：{question}\n标准答案：{gold}\n模型回答：{answer}"
            payload, elapsed, raw = judge("你只输出 JSON。", user)
            strict = strict_label(payload)
            reason = str(payload.get("reason", ""))[:60]

            user_score = f"{SCORE_PROMPT}\n\n问题：{question}\n标准答案：{gold}\n模型回答：{answer}"
            payload_score, elapsed_score, _ = judge("你只输出 JSON。", user_score)
            score = payload_score.get("score", "")
            score_cls = score_label(payload_score)
            elapsed += elapsed_score

        pointwise.append(
            {
                "id": qid,
                "category": output["category"],
                "human": human,
                "strict": strict,
                "score_class": score_cls,
                "score": score,
                "reason": reason,
                "latency_s": round(elapsed, 2),
                "answer_chars": len(answer),
            }
        )
        r.log(f"{number}/{len(ids)} {qid} human={human} strict={strict} score={score_cls}")

    write_csv(OUT / "judge-pointwise.csv", pointwise)

    answerable = [row for row in pointwise if row["category"] != "refuse" and row["strict"]]
    scored = [row for row in pointwise if row["category"] != "refuse" and row["score_class"]]
    refusal = [row for row in pointwise if row["category"] == "refuse" and row["strict"]]

    strict_acc = sum(row["strict"] == row["human"] for row in answerable) / len(answerable) if answerable else 0
    score_acc = sum(row["score_class"] == row["human"] for row in scored) / len(scored) if scored else 0
    strict_kappa = kappa([row["human"] for row in answerable], [row["strict"] for row in answerable])
    score_kappa = kappa([row["human"] for row in scored], [row["score_class"] for row in scored])
    prompt_agree = (
        sum(row["strict"] == row["score_class"] for row in scored) / len(scored) if scored else 0
    )

    predicted_correct = [row["strict"] for row in answerable]
    tp = sum(1 for row in answerable if row["human"] == "correct" and row["strict"] == "correct")
    fp = sum(1 for row in answerable if row["human"] != "correct" and row["strict"] == "correct")
    fn = sum(1 for row in answerable if row["human"] == "correct" and row["strict"] != "correct")
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    confusion = Counter((row["human"], row["strict"]) for row in answerable)
    confusion_rows = [
        {"human": human, "judge": judge_label, "count": confusion[(human, judge_label)]}
        for human in LABELS
        for judge_label in LABELS
    ]
    write_csv(OUT / "judge-confusion.csv", confusion_rows)

    summary = [
        ("questions", len(pointwise)),
        ("answerable", len(answerable)),
        ("strict_accuracy", round(strict_acc, 3)),
        ("score_accuracy", round(score_acc, 3)),
        ("strict_kappa", round(strict_kappa, 3)),
        ("score_kappa", round(score_kappa, 3)),
        ("strict_vs_score_agreement", round(prompt_agree, 3)),
        ("correct_precision", round(precision, 3)),
        ("correct_recall", round(recall, 3)),
        ("correct_f1", round(f1, 3)),
        ("refusal_accuracy", round(sum(row["strict"] == "correct" for row in refusal) / len(refusal), 3) if refusal else ""),
        ("mean_latency_s", round(sum(row["latency_s"] for row in pointwise) / len(pointwise), 2)),
        ("p95_latency_s", round(sorted(row["latency_s"] for row in pointwise)[int(0.95 * (len(pointwise) - 1))], 2)),
    ]

    if not args.no_bias:
        # 废话偏置：正确回答后补一句无害的套话
        verbose = [
            row for row in pointwise if row["category"] != "refuse" and row["human"] == "correct"
        ][:15]
        bias_rows = []
        for row in verbose:
            output = outputs[row["id"]]
            padded = output["answer"] + " 以上内容供参考，具体执行时请结合本单位实际情况。"
            user = f"{STRICT_PROMPT}\n\n问题：{output['question']}\n标准答案：{output['gold']}\n模型回答：{padded}"
            payload, elapsed, _ = judge("你只输出 JSON。", user)
            label = strict_label(payload)
            bias_rows.append(
                {
                    "test": "verbosity",
                    "id": row["id"],
                    "variant": "padded",
                    "human": row["human"],
                    "label": label,
                    "score": "",
                    "note": f"{row['strict']} -> {label}",
                }
            )
        # 顺序偏置：同一道题的 rag 回答与闭卷回答配对，交换 A/B
        pairs = []
        closed = {row["id"]: row for row in read_jsonl(GEN) if row["mode"] == "closed_book"}
        for qid in ids:
            output = outputs[qid]
            if output["category"] == "refuse" or qid not in closed:
                continue
            if audit[qid]["manual_label"] == "correct":
                pairs.append(qid)
            if len(pairs) >= 15:
                break
        for qid in pairs:
            rag_answer = outputs[qid]["answer"]
            closed_answer = closed[qid]["answer"]
            question, gold = outputs[qid]["question"], outputs[qid]["gold"]
            user = f"{PAIR_PROMPT}\n\n问题：{question}\n标准答案：{gold}\n回答 A：{rag_answer}\n回答 B：{closed_answer}"
            first, _, _ = judge("你只输出 JSON。", user)
            user = f"{PAIR_PROMPT}\n\n问题：{question}\n标准答案：{gold}\n回答 A：{closed_answer}\n回答 B：{rag_answer}"
            second, _, _ = judge("你只输出 JSON。", user)
            w1 = condition(first)
            w2 = condition(second)
            # 内容一致只有两种：先判 A 后判 B（都选 rag），或先判 B 后判 A（都选闭卷）；
            # A/A、B/B、TIE/A、A/TIE 都算判定随顺序变了。
            flipped = (w1, w2) not in {("A", "B"), ("B", "A")}
            bias_rows.append(
                {
                    "test": "position",
                    "id": qid,
                    "variant": "swap",
                    "human": audit[qid]["manual_label"],
                    "label": f"{w1}/{w2}",
                    "score": "",
                    "note": "flip" if flipped else "consistent",
                }
            )
        # 覆盖写：把顺序偏置追加到同一份文件
        write_csv(OUT / "judge-bias.csv", bias_rows)

        verbosity_flip = sum(
            1 for row in bias_rows if row["test"] == "verbosity" and row["note"].split(" -> ")[0] != row["note"].split(" -> ")[1]
        ) / max(1, sum(1 for row in bias_rows if row["test"] == "verbosity"))
        position_flip = sum(1 for row in bias_rows if row["test"] == "position" and row["note"] == "flip") / max(
            1, sum(1 for row in bias_rows if row["test"] == "position")
        )
        summary += [
            ("verbosity_flip_rate", round(verbosity_flip, 3)),
            ("position_flip_rate", round(position_flip, 3)),
        ]

    write_csv(OUT / "judge-summary.csv", [{"metric": key, "value": value} for key, value in summary])
    for key, value in summary:
        print(f"{key:28s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
