"""提示词实验室 · 主运行器。

用法：
  python run_lab.py                 # 跑全部任务 × 全部变体
  python run_lab.py --task classify # 只跑分类任务
  python run_lab.py --variant v2    # 只跑 id 包含 v2 的变体（按子串匹配）
  python run_lab.py --mock          # 不调用 API，打印将要发送的 messages（强烈建议先跑这个）
  python run_lab.py --limit 1       # 每个任务只跑第 1 条样本（快速试跑）
  python run_lab.py --temperature 0.7

真实模式下，每条结果会保存到 results/raw/<任务>/<变体>/<样本id>.txt，
方便你逐条对比不同变体的输出质量。
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# 强制 UTF-8 输出，避免中文在 Windows 管道/重定向下乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import config  # noqa: F401  先加载 .env（见 config.py）
from llm import chat, get_api_key
from tasks import TASKS

RESULTS_DIR = Path(__file__).parent / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="提示词实验室 · 运行器")
    p.add_argument("--task", choices=list(TASKS), help="只跑指定任务")
    p.add_argument("--variant", help="只跑 id 包含该子串的变体")
    p.add_argument("--mock", action="store_true", help="不调用 API，只打印将发送的 messages")
    p.add_argument("--limit", type=int, default=0, help="每个任务只跑前 N 条样本（0 = 全部）")
    p.add_argument("--temperature", type=float, default=0.3, help="采样温度，默认 0.3")
    p.add_argument("--model", default=None, help="模型名，默认 deepseek-v4-flash（旧名 deepseek-chat 已停用）")
    return p.parse_args()


def print_messages(messages: list[dict]) -> None:
    """把 messages 按 role 打印出来，方便阅读。"""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            print(f"  [system] {content}")
        elif role == "user":
            print(f"  [user] {content}")
        elif role == "assistant":
            print(f"  [assistant] {content}")


def run_mock(task_id: str, task: dict, variant_filter: str | None) -> None:
    """mock 模式：不花钱、不联网，先把每个变体的 prompt 完整看一遍。"""
    for variant in task["variants"]:
        if variant_filter and variant_filter not in variant["id"]:
            continue
        print(f"\n{'=' * 64}")
        print(f"任务：{task['title']}（{task_id}） 变体：{variant['id']} {variant['name']}")
        print(f"设计意图：{variant['note']}")
        print(f"{'-' * 64}")
        sample = task["data"][0]
        print(f"样本 {sample['id']}：{sample['text'][:60]}...")
        print(f"{'-' * 64}")
        print_messages(variant["build"](sample["text"]))
        print(f"{'=' * 64}")


def run_real(args: argparse.Namespace, task_id: str, task: dict) -> dict:
    """真实模式：调用 DeepSeek，逐条保存结果。返回 {variant_id: 结果列表}。"""
    out = {}
    samples = task["data"]
    if args.limit > 0:
        samples = samples[: args.limit]

    for variant in task["variants"]:
        if args.variant and args.variant not in variant["id"]:
            continue
        out[variant["id"]] = []
        print(f"\n>>> 任务：{task['title']}（{task_id}） 变体：{variant['id']} {variant['name']}")
        print(f"    设计意图：{variant['note']}")

        for i, sample in enumerate(samples, 1):
            messages = variant["build"](sample["text"])
            print(f"  [{i}/{len(samples)}] 样本 {sample['id']} 调用 API ...", flush=True)
            try:
                kwargs: dict = {"temperature": args.temperature}
                if args.model:  # 没传 --model 时交给 llm.py 用默认模型，避免 model=None
                    kwargs["model"] = args.model
                reply = chat(messages, **kwargs)
            except Exception as e:  # noqa: BLE001  API 错误不影响其他样本
                reply = f"[调用失败] {e}"
                print(f"    ✗ 失败：{e}")
            else:
                print(f"    ✓ 完成")

            record = {"sample_id": sample["id"], "output": reply}
            out[variant["id"]].append(record)

            save_dir = RESULTS_DIR / "raw" / task_id / variant["id"]
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / f"{sample['id']}.txt").write_text(
                f"== 样本原文 ==\n{sample['text']}\n\n== 模型输出 ==\n{reply}\n",
                encoding="utf-8",
            )
    return out


def main() -> None:
    args = parse_args()
    task_ids = [args.task] if args.task else list(TASKS)

    if args.mock:
        for tid in task_ids:
            run_mock(tid, TASKS[tid], args.variant)
        print("\n[提示] mock 模式只展示 prompt，未调用 API。"
              "配置好 .env 里的 key 后去掉 --mock 即可真实运行。")
        return

    # fail-fast：key 没配置就不必白跑一遍
    try:
        get_api_key()
    except RuntimeError as e:
        print(e, file=sys.stderr)
        print("提示：可以先运行 python run_lab.py --mock 预览所有 prompt。", file=sys.stderr)
        sys.exit(1)

    all_results: dict[str, dict] = {}
    for tid in task_ids:
        print(f"\n########## 任务：{TASKS[tid]['title']}（{tid}）##########")
        all_results[tid] = run_real(args, tid, TASKS[tid])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = RESULTS_DIR / f"summary_{stamp}.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[完成] 所有结果已保存到 results/raw/ 目录，汇总见 {summary_file}")
    print("[下一步] 运行 python evaluate.py --help 了解如何用评测脚本对比变体优劣。")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
