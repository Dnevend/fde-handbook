#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml"]
# ///
"""按四层拼装生成合成语料：骨架（招股书/环评）+ 表壳（政府台账范本）+ 数值（专利区间）+ 空缺。

刻意不调模型：正文由模板 + 固定种子拼出来，同一 seed 永远得到同一份文件。
书里凡是引用合成语料的实验，读者能一条命令复现。需要更自然的行文时，再用模型
改写，并把 prompt_sha256 记进元数据（spec.md 里有要求）。

产出落在 corpus/raw/synthetic/（已被 .gitignore 覆盖），不入库。

用法：
    uv run corpus/synthetic/generate.py
    uv run corpus/synthetic/generate.py --seed 42 --out /tmp/syn
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent
FIELDS = ROOT / "fields.yaml"
DEFAULT_OUT = ROOT.parent / "raw" / "synthetic"

PLAN = {
    "process_card": 20,
    "daily_inspection": 15,
    "maintenance_plan": 8,
    "repair_order": 10,
    "mold_change": 5,
    "quality_incident": 2,
}

EQUIPMENT = [
    ("ZJ-01", "注塑机", "海天 MA2100II"),
    ("ZJ-02", "注塑机", "震雄 SM1500"),
    ("ZJ-03", "注塑机", "博创 BU900"),
    ("MW-01", "模温机", "奥德 AOS-120"),
    ("LS-01", "冷水机", "台洪 10HP"),
    ("JS-01", "机械手", "天行 TSI-800"),
    ("KY-01", "空压机", "阿特拉斯 GA37"),
]

MOLDS = [f"M-{2000 + i}" for i in range(1, 13)]
PRODUCTS = ["前保险杠支架", "仪表台饰板", "门板卡扣", "电池托盘上盖", "格栅本体", "空调出风口"]
MATERIALS = ["PP-T20", "ABS-750", "PC-1250", "PA66-GF30", "POM-M90"]
SHIFTS = ["白班", "夜班"]

# 同一部件的三种叫法。术语不一致是 5.4/5.6 的负样本来源，不是笔误。
INJECTION_UNIT = {"process_card": "射台", "daily_inspection": "注射座", "repair_order": "料筒座"}

# 数值区间取自注塑工艺专利实施例的常见窗口（pat-inj-01，不在正文复述原文）。
PARAM_WINDOW = {
    "干燥温度(℃)": (80, 120),
    "干燥时间(h)": (2, 4),
    "料筒温度(℃)": (190, 280),
    "模温(℃)": (40, 90),
    "注射压力(MPa)": (60, 140),
    "保压压力(MPa)": (25, 70),
    "保压时间(s)": (2, 12),
    "冷却时间(s)": (8, 30),
    "成型周期(s)": (25, 60),
}

BASE_DATE = dt.date(2026, 3, 1)


def rng_for(seed: int, doc_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{doc_id}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def pick_date(rng: random.Random, span_days: int = 240) -> str:
    return (BASE_DATE + dt.timedelta(days=rng.randint(0, span_days))).isoformat()


def table(fields: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(fields) + " |"
    rule = "|" + "|".join("---" for _ in fields) + "|"
    body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return f"{head}\n{rule}\n{body}"


def meta(doc_id: str, kind: str, fields_cfg: dict, seed: int, dept: str, level: str) -> str:
    template = "gov-ledger-guide" if fields_cfg["provenance"] == "verbatim" else "derived"
    return (
        "---\n"
        "synthetic: true\n"
        "generator: corpus/synthetic/generate.py\n"
        f"seed: {seed}\n"
        f"doc_id: {doc_id}\n"
        f"doc_kind: {kind}\n"
        f"base_template: {template}\n"
        f"dept: {dept}\n"
        f"level: {level}\n"
        "---\n"
    )


def build_card_values(rng: random.Random, mold_bonus: int = 0) -> dict[str, str]:
    values: dict[str, str] = {
        "产品名称": rng.choice(PRODUCTS),
        "模具编号": rng.choice(MOLDS),
        "机台编号": rng.choice(EQUIPMENT[:3])[0],
        "材料牌号": rng.choice(MATERIALS),
    }
    for field, (lo, hi) in PARAM_WINDOW.items():
        if field == "模温(℃)":
            lo, hi = lo + mold_bonus, hi + mold_bonus
        if field.endswith("(h)"):
            values[field] = f"{rng.uniform(lo, hi):.1f}"
        elif "料筒" in field:
            values[field] = f"{lo}-{hi}（分段）"
        else:
            values[field] = str(rng.randint(lo, hi))
    return values


def gen_process_card(n: int, seed: int, out: list[dict]) -> None:
    columns = [
        "产品名称", "模具编号", "机台编号", "材料牌号",
        "干燥温度(℃)", "干燥时间(h)", "料筒温度(℃)", "模温(℃)",
        "注射压力(MPa)", "保压压力(MPa)", "保压时间(s)", "冷却时间(s)", "成型周期(s)",
    ]
    for i in range(n):
        doc_id = f"PC-{i + 1:03d}"
        level = "受限" if i == 6 else "内部"
        # 第 7 份是 PC-001 的受控变体：同产品、同模具、同机台，只有模温高 10℃。
        # 5.9 的权限过滤实验就靠这一对——检索到哪一份，答案就不同。
        if i == 6:
            values = build_card_values(rng_for(seed, "PC-001"), mold_bonus=10)
        else:
            values = build_card_values(rng_for(seed, doc_id))

        body = (
            f"# 注塑工艺参数卡 {doc_id}\n\n"
            + table(columns, [[values[c] for c in columns]])
            + f"\n\n备注：{INJECTION_UNIT['process_card']}温度以实际显示为准。\n"
        )
        out.append({"id": doc_id, "kind": "process_card", "dept": "注塑", "level": level, "body": body})


def gen_inspection(n: int, seed: int, out: list[dict]) -> None:
    checks = [
        ("液压系统", "油位在视窗上下限之间", "目视"),
        ("液压系统", "油温 ≤ 55℃", "测温枪"),
        ("锁模机构", "哥林柱无拉伤、润滑正常", "目视"),
        ("顶出机构", "顶针无断裂、动作到位", "试运行"),
        ("安全门", "联锁开关动作可靠", "手动试验"),
        (INJECTION_UNIT["daily_inspection"], "无异响、无渗漏", "听诊/目视"),
    ]
    for i in range(n):
        doc_id = f"DI-{i + 1:03d}"
        rng = rng_for(seed, doc_id)
        eq = EQUIPMENT[i % len(EQUIPMENT)]
        # 第 5 份刻意漏掉"安全门"一项：事故调查报告里真实出现过的漏项形态。
        items = checks if i != 4 else [c for c in checks if c[0] != "安全门"]
        rows = []
        for part, standard, method in items:
            bad = rng.random() < 0.12
            rows.append([eq[0], eq[1], part, standard, method, rng.choice(SHIFTS),
                         "异常" if bad else "正常", "油温偏高，已开冷却" if bad else "",
                         "已报修" if bad else ""])
        body = (
            f"# 设备日常点检表 {doc_id}\n\n"
            f"设备：{eq[1]}（{eq[0]}，{eq[2]}）　日期：{pick_date(rng, 30)}\n\n"
            + table(
                ["设备编号", "设备名称", "点检部位", "点检项目", "点检方法", "班次", "点检结果", "异常描述", "处理措施"],
                rows,
            )
            + "\n\n点检人：　　　　　班组长确认：　　　　　\n"
        )
        out.append({"id": doc_id, "kind": "daily_inspection", "dept": "设备", "level": "内部", "body": body})


def gen_maintenance(n: int, seed: int, out: list[dict]) -> None:
    for i in range(n):
        doc_id = f"MP-{i + 1:03d}"
        rng = rng_for(seed, doc_id)
        eq = EQUIPMENT[0] if i < 2 else EQUIPMENT[i % len(EQUIPMENT)]
        # 第 1、2 份是同一台机器的两个版本：换油周期不同，日期更新但没有修订说明。
        if i == 0:
            period, version, level = "2000 h", "A/0", "内部"
        elif i == 1:
            period, version, level = "3000 h", "A/1", "内部"
        else:
            period, version, level = rng.choice(["500 h", "1000 h", "2000 h"]), "A/0", "内部"
        rows = [
            ["润滑", period, "按润滑点表加注", "设备说明书"],
            ["液压油", period, "检测清洁度，超标更换", "NAS 9 级"],
            ["冷却水路", "每季度", "反冲洗，检查流量", "进出水温差 ≤ 5℃"],
        ]
        body = (
            f"# 设备保养计划 {doc_id}\n\n"
            f"设备：{eq[1]}（{eq[0]}）　版本：{version}\n\n"
            + table(["保养项目", "保养周期", "作业内容", "执行标准"], rows)
            + f"\n\n下次执行：{pick_date(rng)}　编制：设备部\n"
        )
        out.append({"id": doc_id, "kind": "maintenance_plan", "dept": "设备", "level": level, "body": body})


def gen_repair(n: int, seed: int, out: list[dict]) -> None:
    faults = [
        ("合模不到位，报警 ALM-213", "锁模力不足", "更换密封圈，重新设定锁模力"),
        ("料筒温度波动 ±15℃", "加热圈老化", "更换加热圈，校准热电偶"),
        ("顶出异响", "顶针板导柱缺油", "清洁并加注润滑脂"),
        ("液压油温高", "冷却器结垢", "清洗冷却器"),
        ("机械手取件掉落", "真空吸盘老化", "更换吸盘"),
    ]
    for i in range(n):
        doc_id = f"RO-{i + 1:03d}"
        rng = rng_for(seed, doc_id)
        eq = EQUIPMENT[i % len(EQUIPMENT)]
        symptom, cause, action = rng.choice(faults)
        # 第 3 份引用已废止的旧规程编号，且文中没有任何标记：过时条款样本。
        clause = "QG-2016-07《设备维护规程》（已废止）" if i == 2 else "QG-2024-02《设备维护规程》"
        hours = rng.randint(1, 16)
        body = (
            f"# 设备维修工单 {doc_id}\n\n"
            f"设备：{eq[1]}（{eq[0]}）　报修：{pick_date(rng)}　停机：{hours} h\n\n"
            + table(
                ["项目", "内容"],
                [
                    ["故障现象", symptom],
                    ["涉及部位", INJECTION_UNIT["repair_order"]],
                    ["原因分析", cause],
                    ["处理措施", action],
                    ["依据", clause],
                    ["更换备件", "密封圈 / 加热圈" if rng.random() < 0.5 else "无"],
                    ["验收", "已验收" if i != 7 else "待验收"],
                ],
            )
            + f"\n\n维修人：　　　　　使用部门确认：　　　　　\n"
        )
        out.append({"id": doc_id, "kind": "repair_order", "dept": "设备", "level": "内部", "body": body})


def gen_mold_change(n: int, seed: int, out: list[dict]) -> None:
    for i in range(n):
        doc_id = f"MC-{i + 1:03d}"
        rng = rng_for(seed, doc_id)
        mold = MOLDS[i % len(MOLDS)]
        src, dst = rng.sample(EQUIPMENT[:3], 2)
        drift = rng.randint(-8, 8)
        body = (
            f"# 换模记录 {doc_id}\n\n"
            + table(
                ["项目", "内容"],
                [
                    ["模具编号", mold],
                    ["原机台", f"{src[0]}（{src[2]}）"],
                    ["新机台", f"{dst[0]}（{dst[2]}）"],
                    ["换模开始", pick_date(rng, 120)],
                    ["换模结束", pick_date(rng, 120)],
                    ["首件确认", "合格" if i != 3 else "尺寸超差，复调"],
                    ["参数偏移", f"模温 {drift:+d}℃，保压时间 +{abs(drift) // 3}s"],
                ],
            )
            + "\n\n操作工：　　　　　工艺员确认：　　　　　\n"
        )
        out.append({"id": doc_id, "kind": "mold_change", "dept": "注塑", "level": "内部", "body": body})


def gen_quality(n: int, seed: int, out: list[dict]) -> None:
    for i in range(n):
        doc_id = f"QI-{i + 1:03d}"
        rng = rng_for(seed, doc_id)
        product = rng.choice(PRODUCTS)
        body = (
            f"# 质量异常单 {doc_id}\n\n"
            + table(
                ["项目", "内容"],
                [
                    ["发现日期", pick_date(rng)],
                    ["批次", f"B{rng.randint(10000, 99999)}"],
                    ["产品名称", product],
                    ["缺陷描述", "批量缩痕，集中在加强筋根部"],
                    ["关联点检", f"DI-{rng.randint(1, 15):03d}"],
                    ["关联工单", f"RO-{rng.randint(1, 10):03d}"],
                    ["初步原因", "保压压力不足，模温偏低"],
                    ["处置措施", "调高保压 5 MPa，模温 +8℃；隔离返工 320 件"],
                    ["结论", "已闭环" if i == 0 else "跟踪中"],
                ],
            )
            + "\n\n质量工程师：　　　　　生产确认：　　　　　\n"
        )
        out.append({"id": doc_id, "kind": "quality_incident", "dept": "质量", "level": "内部", "body": body})


GENERATORS = {
    "process_card": gen_process_card,
    "daily_inspection": gen_inspection,
    "maintenance_plan": gen_maintenance,
    "repair_order": gen_repair,
    "mold_change": gen_mold_change,
    "quality_incident": gen_quality,
}

KEY_VALUE_HEADER = ("项目", "内容")
# 字段白名单：表头里的结构列不算"字段"，不然每张 key-value 表都要把这两列声明一遍。
STRUCTURAL = {"项目", "内容"}


def build_docs(seed: int) -> list[dict]:
    docs: list[dict] = []
    for kind, count in PLAN.items():
        GENERATORS[kind](count, seed, docs)
    return docs


def write_docs(docs: list[dict], out: pathlib.Path, forms: dict, seed: int) -> None:
    for doc in docs:
        folder = out / doc["kind"]
        folder.mkdir(parents=True, exist_ok=True)
        head = meta(doc["id"], doc["kind"], forms[doc["kind"]], seed, doc["dept"], doc["level"])
        (folder / f"{doc['id']}.md").write_text(head + "\n" + doc["body"], encoding="utf-8")


def table_blocks(text: str) -> list[list[list[str]]]:
    blocks, current = [], []
    for line in text.splitlines():
        if line.startswith("|"):
            current.append([cell.strip() for cell in line.strip().strip("|").split("|")])
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def used_labels(text: str) -> list[str]:
    labels: list[str] = []
    for block in table_blocks(text):
        if len(block) < 2:
            continue
        header, body = block[0], block[1:]
        body = [row for row in body if not all(cell and set(cell) <= set("-:") for cell in row)]
        if tuple(header) == KEY_VALUE_HEADER:
            labels += [row[0] for row in body if row]
        else:
            labels += header
    return labels


def data_row(text: str) -> list[str]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|---") and i + 1 < len(lines):
            return [cell.strip() for cell in lines[i + 1].strip().strip("|").split("|")]
    return []


def validate(docs: list[dict], out: pathlib.Path, forms: dict, seed: int) -> list[str]:
    problems: list[str] = []

    for kind, expected in PLAN.items():
        folder = out / kind
        got = len(list(folder.glob("*.md"))) if folder.is_dir() else 0
        if got != expected:
            problems.append(f"{kind} 份数 {got}，应为 {expected}")

    again = build_docs(seed)
    if [(d["id"], d["body"]) for d in docs] != [(d["id"], d["body"]) for d in again]:
        problems.append("同一 seed 两次生成结果不同（有用到全局随机？）")

    for doc in docs:
        path = out / doc["kind"] / f"{doc['id']}.md"
        if not path.exists():
            problems.append(f"{doc['id']} 缺文件")
            continue
        text = path.read_text(encoding="utf-8")
        for token in ("synthetic: true", "generator:", f"seed: {seed}", "base_template:", "dept:", "level:"):
            if token not in text:
                problems.append(f"{doc['id']} 元数据缺 {token}")
        allowed = set(forms[doc["kind"]]["fields"]) | STRUCTURAL
        for label in used_labels(text):
            if label and label not in allowed:
                problems.append(f"{doc['id']} 出现未声明字段「{label}」")
        if len(doc["body"]) < 100:
            problems.append(f"{doc['id']} 正文过短")

    # ---- 四类「脏」逐项验，缺一个就说明实验材料没了 ----
    def read(doc_id: str, kind: str) -> str:
        path = out / kind / f"{doc_id}.md"
        if not path.exists():
            problems.append(f"{doc_id} 缺文件，四类脏检查无法进行")
            return ""
        return path.read_text(encoding="utf-8")

    mp1, mp2 = read("MP-001", "maintenance_plan"), read("MP-002", "maintenance_plan")
    if "ZJ-01" not in mp1 or "ZJ-01" not in mp2:
        problems.append("版本冲突：MP-001/MP-002 不是同一台设备")
    if "2000 h" not in mp1 or "3000 h" not in mp2:
        problems.append("版本冲突：保养周期不是 2000 h / 3000 h")

    pc1, pc7 = read("PC-001", "process_card"), read("PC-007", "process_card")
    row1, row7 = data_row(pc1), data_row(pc7)
    if len(row1) < 13 or len(row7) < 13:
        problems.append("权限分层：PC-001/PC-007 数据行不完整")
    elif row1[0] != row7[0] or row1[1] != row7[1]:
        problems.append("权限分层：PC-007 与 PC-001 不是同产品同模具")
    elif int(row7[7]) - int(row1[7]) != 10:
        problems.append("权限分层：模温差不是 10℃")

    if "已废止" not in read("RO-003", "repair_order"):
        problems.append("过时条款：RO-003 未引用已废止规程")
    if "安全门" in read("DI-005", "daily_inspection"):
        problems.append("漏项：DI-005 不应包含安全门")

    if "射台" not in pc1:
        problems.append("术语不一致：工艺卡未用「射台」")
    if "注射座" not in read("DI-001", "daily_inspection"):
        problems.append("术语不一致：点检表未用「注射座」")
    if "料筒座" not in read("RO-001", "repair_order"):
        problems.append("术语不一致：工单未用「料筒座」")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="生成合成语料")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--clean", action="store_true", help="生成前清空输出目录")
    parser.add_argument("--validate", action="store_true", help="自检：份数、字段、四类脏、确定性")
    args = parser.parse_args()

    import yaml

    forms = yaml.safe_load(FIELDS.read_text(encoding="utf-8"))["forms"]
    if args.clean and args.out.exists():
        for path in args.out.rglob("*.md"):
            path.unlink()

    docs = build_docs(args.seed)
    write_docs(docs, args.out, forms, args.seed)

    by_kind = {kind: sum(1 for d in docs if d["kind"] == kind) for kind in PLAN}
    total = sum(len(d["body"]) for d in docs)
    print(f"生成 {len(docs)} 份，约 {total / 1000:.1f} 千字 → {args.out}")
    for kind, count in by_kind.items():
        print(f"  {kind:20s} {count:3d}")

    if args.validate:
        problems = validate(docs, args.out, forms, args.seed)
        if problems:
            print(f"\n自检失败 {len(problems)} 项：", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print("\n自检通过：份数、字段、四类脏、确定性")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
