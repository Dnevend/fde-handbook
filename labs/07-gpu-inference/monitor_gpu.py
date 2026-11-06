#!/usr/bin/env python3
"""7.3 监控：按固定间隔采 nvidia-smi，输出 CSV，并在首尾记录拓扑和详细状态。

为什么不用 nvidia-smi -l：
- 我们要同时记录时间戳、功耗、温度、时钟和显存；
- 采完要能算均值/峰值，CSV 比终端滚动输出好处理；
- 采集中断时还能保留已经写入的行。

用法（GPU 机器上）：
    uv run labs/07-gpu-inference/monitor_gpu.py \
      --duration 120 --interval 1 --label prefill-heavy

    # 一直采到 Ctrl-C
    uv run labs/07-gpu-inference/monitor_gpu.py --duration 0 --label manual

产物：
    benchmarks/gpu/monitor/<label>-<timestamp>.csv
    benchmarks/gpu/monitor/<label>-<timestamp>-topo.txt
    benchmarks/gpu/monitor/<label>-<timestamp>-snapshot.txt
    benchmarks/gpu/monitor/<label>-<timestamp>-env.txt
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpulib import RESULTS_DIR, env_note, gpu_snapshot, have_nvidia_smi  # noqa: E402

# nvidia-smi --query-gpu 只接受点号字段；CSV 表头用可读的 snake_case。
QUERY_FIELDS = [
    "timestamp",
    "index",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "temperature.gpu",
    "clocks.current.sm",
    "clocks.current.mem",
]
HEADERS = [
    "timestamp",
    "index",
    "name",
    "gpu_util_pct",
    "mem_util_pct",
    "mem_used_mb",
    "mem_total_mb",
    "power_w",
    "temp_c",
    "sm_clock_mhz",
    "mem_clock_mhz",
]
QUERY = ",".join(QUERY_FIELDS)


def sample() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi 失败")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="nvidia-smi 采样器")
    parser.add_argument("--duration", type=float, default=60, help="秒；0 表示一直采到 Ctrl-C")
    parser.add_argument("--interval", type=float, default=1.0, help="采样间隔（秒）")
    parser.add_argument("--label", default="monitor", help="输出文件前缀")
    parser.add_argument("--output", default="", help="CSV 路径；默认 benchmarks/gpu/monitor/<label>-<时间>.csv")
    args = parser.parse_args()

    if not have_nvidia_smi():
        print("nvidia-smi 不可用：这个脚本只在 NVIDIA GPU 机器上跑。", file=sys.stderr)
        return 2

    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = Path(args.output) if args.output else RESULTS_DIR / "monitor" / f"{args.label}-{stamp}.csv"
    base.parent.mkdir(parents=True, exist_ok=True)
    stem = base.with_suffix("")

    snapshot = gpu_snapshot()
    (stem.with_name(stem.name + "-topo.txt")).write_text(snapshot["topo"] + "\n", encoding="utf-8")
    (stem.with_name(stem.name + "-snapshot.txt")).write_text(snapshot["detail"] + "\n", encoding="utf-8")
    (stem.with_name(stem.name + "-env.txt")).write_text(env_note() + "\n", encoding="utf-8")

    started = time.monotonic()
    print(f"采样中：{base}（间隔 {args.interval}s，时长 {args.duration or '∞'}s）")
    try:
        with base.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADERS)
            while True:
                loop_started = time.monotonic()
                for line in sample():
                    writer.writerow([cell.strip() for cell in next(csv.reader([line]))])
                    handle.flush()
                if args.duration and time.monotonic() - started >= args.duration:
                    break
                time.sleep(max(0.0, args.interval - (time.monotonic() - loop_started)))
    except KeyboardInterrupt:
        print("\n收到 Ctrl-C，停止采样。")
    except Exception as exc:  # noqa: BLE001
        print(f"采样失败：{exc}", file=sys.stderr)
        return 1

    print(f"完成：{base}")
    print(f"拓扑：{stem.name}-topo.txt")
    print(f"快照：{stem.name}-snapshot.txt")
    print(f"环境：{stem.name}-env.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
