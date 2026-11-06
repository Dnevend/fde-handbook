#!/usr/bin/env python3
"""7.9 容量规划计算器：并发 × 上下文 → 显存、速度、TTFT、每百万 token 成本。

这是**上限估算**，不是承诺。公式来自 1.6（KV/显存）、7.1（带宽/算力 roofline）
和 vLLM 的预分配行为；真实数字用 7.4 的压测替换。

用法：
    uv run labs/07-gpu-inference/capacity.py \
      --model qwen3.5-9b-awq --vram 24 --ctx 8192 --concurrency 50 --gpu-price 1.92

    # 带 SLO 判定
    uv run labs/07-gpu-inference/capacity.py \
      --model qwen3.5-9b-awq --ctx 8192 --concurrency 50 \
      --tpot-ms 50 --ttft-ms 2000

    # 输出 JSON，给画图脚本用
    uv run labs/07-gpu-inference/capacity.py --json

口径：
- KV 按 f16 估；FP8 KV 把 --kv-kib 减半。
- 权重按“文件大小”估，不用参数量反推。
- decode 效率默认 0.5（工程估算），prefill MFU 默认 0.3；
  想看理论上限就把 --decode-eff 设为 1.0。
"""

from __future__ import annotations

import argparse
import json

PRESETS = {
    "qwen3.5-9b-awq": {"weight_gb": 5.7, "kv_kib": 32, "params_b": 9.0},
    "qwen3.5-9b-fp8": {"weight_gb": 9.0, "kv_kib": 32, "params_b": 9.0},
    "qwen3.5-9b-bf16": {"weight_gb": 18.0, "kv_kib": 32, "params_b": 9.0},
    "qwen3-14b-awq": {"weight_gb": 8.5, "kv_kib": 160, "params_b": 14.0},
}


def build_payload(args: argparse.Namespace) -> dict:
    preset = PRESETS.get(args.model, {})
    weight_gb = args.weight_gb or preset.get("weight_gb")
    kv_kib = args.kv_kib or preset.get("kv_kib")
    params_b = args.params_b or preset.get("params_b")
    if not all((weight_gb, kv_kib, params_b)):
        raise SystemExit(
            f"未知模型 {args.model}；用 --weight-gb/--kv-kib/--params-b 显式给参数，"
            f"或选 {', '.join(PRESETS)}。"
        )

    kv_per_token_gb = kv_kib * 1024 / 1e9
    kv_per_request_gb = kv_per_token_gb * args.ctx
    kv_budget_gb = args.vram * args.util - weight_gb - args.overhead_gb
    max_c_vram = max(0, int(kv_budget_gb // kv_per_request_gb)) if kv_per_request_gb > 0 else 0

    def step_ms(concurrency: int, decode_eff: float) -> tuple[float, float, float]:
        bandwidth_ms = (weight_gb + concurrency * kv_per_request_gb) / args.bandwidth * 1000
        compute_ms = (
            2 * params_b * 1e9 * concurrency / (args.compute_tflops * 1e12 * decode_eff) * 1000
        )
        return bandwidth_ms, compute_ms, max(bandwidth_ms, compute_ms)

    bandwidth_ms, compute_ms, t_step_ms = step_ms(args.concurrency, args.decode_eff)
    _, _, t_step_theory_ms = step_ms(args.concurrency, 1.0)
    per_req_tok_s = 1000 / t_step_ms if t_step_ms > 0 else 0.0
    aggregate_tok_s = args.concurrency * per_req_tok_s
    per_req_tok_s_theory = 1000 / t_step_theory_ms if t_step_theory_ms > 0 else 0.0
    aggregate_tok_s_theory = args.concurrency * per_req_tok_s_theory

    ttft_ms = (
        2 * params_b * 1e9 * args.prompt_tokens / (args.compute_tflops * 1e12 * args.mfu) * 1000
    )
    cost_per_m = args.gpu_price / (aggregate_tok_s * 3600) * 1e6 if aggregate_tok_s > 0 else 0.0

    # 交叉点用「每个请求的 KV」而不是「每 token 的 KV」：
    # t_bw = (W + B*KV_request)/BW，t_compute = 2*P*B/(F*eff)
    denominator = (
        2 * params_b * 1e9 / (args.compute_tflops * 1e12 * args.decode_eff)
        - kv_per_request_gb / args.bandwidth
    )
    b_star = (weight_gb / args.bandwidth) / denominator if denominator > 0 else None

    return {
        "model": args.model,
        "assumptions": {
            "weight_gb": weight_gb,
            "kv_kib_per_token": kv_kib,
            "params_b": params_b,
            "vram_gb": args.vram,
            "util": args.util,
            "overhead_gb": args.overhead_gb,
            "bandwidth_gbps": args.bandwidth,
            "compute_tflops": args.compute_tflops,
            "decode_eff": args.decode_eff,
            "prefill_mfu": args.mfu,
            "gpu_price_per_hour": args.gpu_price,
        },
        "kv": {
            "per_token_gb": round(kv_per_token_gb, 6),
            "per_request_gb": round(kv_per_request_gb, 4),
            "budget_gb": round(kv_budget_gb, 2),
        },
        "concurrency": {
            "requested": args.concurrency,
            "max_by_vram": max_c_vram,
            "fits_vram": args.concurrency <= max_c_vram,
            "crossover_b_star": round(b_star, 1) if b_star else None,
        },
        "speed": {
            "bandwidth_step_ms": round(bandwidth_ms, 2),
            "compute_step_ms": round(compute_ms, 2),
            "step_ms": round(t_step_ms, 2),
            "per_request_tok_s": round(per_req_tok_s, 1),
            "aggregate_tok_s": round(aggregate_tok_s, 1),
            "theory_step_ms": round(t_step_theory_ms, 2),
            "theory_per_request_tok_s": round(per_req_tok_s_theory, 1),
            "theory_aggregate_tok_s": round(aggregate_tok_s_theory, 1),
        },
        "ttft": {
            "prompt_tokens": args.prompt_tokens,
            "estimate_ms": round(ttft_ms, 1),
        },
        "cost": {
            "gpu_price_per_hour": args.gpu_price,
            "cny_per_million_tokens": round(cost_per_m, 3),
        },
        "slo": {
            "tpot_ms": args.tpot_ms or None,
            "tpot_ok": (t_step_ms <= args.tpot_ms) if args.tpot_ms else None,
            "ttft_ms": args.ttft_ms or None,
            "ttft_ok": (ttft_ms <= args.ttft_ms) if args.ttft_ms else None,
        },
    }


def print_report(payload: dict) -> None:
    a = payload["assumptions"]
    kv = payload["kv"]
    c = payload["concurrency"]
    s = payload["speed"]
    t = payload["ttft"]
    cost = payload["cost"]

    print("容量估算（上限，不是承诺）")
    print(f"模型: {payload['model']}")
    print(
        f"假设: 权重 {a['weight_gb']:.1f} GB；KV {a['kv_kib_per_token']} KiB/token；"
        f"参数 {a['params_b']:.1f}B；显存 {a['vram_gb']} GB × {a['util']:.2f}；"
        f"开销 {a['overhead_gb']:.1f} GB"
    )
    print(
        f"     带宽 {a['bandwidth_gbps']:.0f} GB/s；算力 {a['compute_tflops']:.1f} TFLOPS；"
        f"decode 效率 {a['decode_eff']:.2f}；prefill MFU {a['prefill_mfu']:.2f}"
    )
    print()
    print(f"KV/token: {a['kv_kib_per_token']:.0f} KiB；KV/请求: {kv['per_request_gb']:.2f} GB")
    print(f"KV 预算: {kv['budget_gb']:.1f} GB；显存上限并发: {c['max_by_vram']}")
    print(f"请求并发: {c['requested']}；显存放得下: {'是' if c['fits_vram'] else '否'}")
    if c["crossover_b_star"]:
        print(f"带宽/算力交叉点 B*: 约 {c['crossover_b_star']:.0f}")
    else:
        print("带宽/算力交叉点: 无（当前上下文下带宽主导）")
    print()
    print(f"带宽项: {s['bandwidth_step_ms']:.2f} ms/step")
    print(f"算力项: {s['compute_step_ms']:.2f} ms/step（eff={a['decode_eff']:.2f}）")
    print(f"综合: {s['step_ms']:.2f} ms/step")
    print(f"单请求: {s['per_request_tok_s']:.1f} token/s；聚合: {s['aggregate_tok_s']:.0f} token/s")
    print(
        f"理论上限（eff=1.0）: {s['theory_step_ms']:.2f} ms/step，"
        f"聚合 {s['theory_aggregate_tok_s']:.0f} token/s"
    )
    print()
    print(f"TTFT @ {t['prompt_tokens']} prompt tokens: 约 {t['estimate_ms']:.0f} ms")
    print(f"每百万 token 成本: 约 ¥{cost['cny_per_million_tokens']:.3f}")
    if payload["slo"]["tpot_ms"]:
        print(f"TPOT SLO {payload['slo']['tpot_ms']} ms: {'达标' if payload['slo']['tpot_ok'] else '不达标'}")
    if payload["slo"]["ttft_ms"]:
        print(f"TTFT SLO {payload['slo']['ttft_ms']} ms: {'达标' if payload['slo']['ttft_ok'] else '不达标'}")
    print()
    print("注意：这是按公式算的上限；实际以 7.4 的压测为准，尤其要看 KV 精度、块大小和长上下文。")


def main() -> int:
    parser = argparse.ArgumentParser(description="容量规划计算器（估算）")
    parser.add_argument("--model", default="qwen3.5-9b-awq", help=f"预设：{', '.join(PRESETS)}")
    parser.add_argument("--weight-gb", type=float, default=0.0, help="覆盖权重文件大小")
    parser.add_argument("--kv-kib", type=float, default=0.0, help="覆盖每 token KV（KiB）")
    parser.add_argument("--params-b", type=float, default=0.0, help="覆盖参数量（B）")
    parser.add_argument("--vram", type=float, default=24.0, help="显存 GB")
    parser.add_argument("--util", type=float, default=0.90, help="gpu_memory_utilization")
    parser.add_argument("--overhead-gb", type=float, default=1.5, help="运行时/激活开销 GB")
    parser.add_argument("--bandwidth", type=float, default=1008.0, help="显存带宽 GB/s")
    parser.add_argument("--compute-tflops", type=float, default=165.2, help="FP16 稠密算力 TFLOPS")
    parser.add_argument("--ctx", type=int, default=8192, help="上下文长度")
    parser.add_argument("--concurrency", type=int, default=50, help="目标并发")
    parser.add_argument("--prompt-tokens", type=int, default=8192, help="TTFT 估算用的 prompt token 数")
    parser.add_argument("--gpu-price", type=float, default=1.92, help="GPU 每小时价格（元）")
    parser.add_argument("--mfu", type=float, default=0.30, help="prefill MFU")
    parser.add_argument("--decode-eff", type=float, default=0.50, help="decode 算力效率")
    parser.add_argument("--tpot-ms", type=float, default=0.0, help="TPOT SLO（ms），0 表示不判定")
    parser.add_argument("--ttft-ms", type=float, default=0.0, help="TTFT SLO（ms），0 表示不判定")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    payload = build_payload(args)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_report(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
