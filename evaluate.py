"""评测脚本：对比同一个任务下不同 prompt 变体的效果。

评测方式（每种任务用最适合的方式）：
  classify   -> 从输出中解析出标签，与数据里的 expected 精确比对，算准确率
  summarize  -> LLM 当裁判，按 忠实度/完整性/简洁性/格式 打分（1-5）
  extract    -> LLM 当裁判，对照期望字段核对准确率与幻觉情况

前提：先运行 run_lab.py 生成 results/raw/ 下的模型输出。

用法：
  python evaluate.py                  # 评测全部已有结果
  python evaluate.py --task summarize # 只评测摘要
  python evaluate.py --variant v3     # 只评测 id 含 v3 的变体
  python evaluate.py --judge-model deepseek-chat
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# 强制 UTF-8 输出，避免中文在 Windows 管道/重定向下乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import config  # noqa: F401
from llm import chat
from tasks import TASKS

RESULTS_DIR = Path(__file__).parent / "results"

# 分类任务可解析的候选标签（与 prompts/classify_variants.py 保持一致）
LABELS = ["物流问题", "商品质量问题", "退换货", "咨询", "投诉", "待人工处理","正面", "负面", "中性"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="提示词实验室 · 评测")
    p.add_argument("--task", choices=list(TASKS), help="只评测指定任务")
    p.add_argument("--variant", help="只评测 id 包含该子串的变体")
    p.add_argument("--judge-model", default=None, help="裁判模型，默认与 llm.py 相同")
    p.add_argument("--judge-max-tokens", type=int, default=8000,
                   help="裁判输出上限（默认 8000）。V4 是推理模型，推理会先消耗 token，"
                        "上限太小会导致裁判 JSON 被截断、无法解析")
    p.add_argument("--no-llm", action="store_true",
                   help="不调用 LLM 裁判（classify 仍可算准确率；summarize/extract 跳过）")
    return p.parse_args()


# ---------- 工具 ----------

def parse_label(output: str) -> str:
    """从模型输出中尽力解析出标签。"""
    output = output.strip()
    # 优先 JSON 的 label 字段
    m = re.search(r'"label"\s*:\s*"([^"]+)"', output)
    if m:
        return m.group(1)
    # 否则看输出里是否直接包含某个标签（按长度降序匹配，避免"咨询"误吞"投诉咨询"等）
    for label in sorted(LABELS, key=len, reverse=True):
        if label in output:
            return label
    return "(无法解析)"


def extract_json(text: str) -> dict | None:
    """从文本中提取第一个 JSON 对象（容错：模型偶尔会包一层 ```json）。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ---------- 各任务评测 ----------

def eval_classify(variant_id: str, results_dir: Path, samples: list[dict], no_llm: bool = False) -> dict:
    total = correct = 0
    for rec_file in sorted(results_dir.glob("*.txt")):
        raw = rec_file.read_text(encoding="utf-8")
        output = raw.split("== 模型输出 ==")[-1].strip()
        sample_id = rec_file.stem
        sample = next(s for s in samples if s["id"] == sample_id)
        pred = parse_label(output)
        total += 1
        if pred == sample["expected"]:
            correct += 1
        else:
            print(f"    ✗ {sample_id}: 期望[{sample['expected']}] 实际[{pred}]")
    accuracy = correct / total if total else 0
    print(f"    准确率：{correct}/{total} = {accuracy:.0%}")
    return {"type": "accuracy", "correct": correct, "total": total, "accuracy": accuracy}



JUDGE_SUMMARIZE = (
    "你是一名严格的摘要质量评审员。下面是原文和一篇候选摘要。\n"
    "请从四个维度分别打分（每项 1-5 分，5 为最好）：\n"
    "- 忠实度：摘要是否只包含原文信息，没有编造或添加。3分=有个别小出入，1分=明显编造；\n"
    "- 完整性：是否覆盖了原文的核心信息。5分=核心信息全部覆盖，3分=遗漏1条次要信息，1分=遗漏关键信息；\n"
    "- 简洁性：是否精炼、无冗余。5分=每句都有信息量，3分=有1-2处啰嗦，1分=大量凑字；\n"
    "- 格式：是否严格满足要点式、恰好3条、每条不超过30字。只要违反其中任意一条，格式分直接给1分。\n"
    "总分按加权计算：total = 忠实度×0.3 + 简洁性×0.4 + 完整性×0.2 + 格式×0.1（保留1位小数）。\n"
    "只输出 JSON：{{\"scores\": {{\"faithfulness\": n, \"completeness\": n, "
    "\"conciseness\": n, \"format\": n}}, \"total\": 加权总分(保留1位小数), "
    "\"reason\": \"先写明各项分数，再一句话点评\"}}\n\n"
    "【原文】\n{source}\n\n【候选摘要】\n{output}"
)

def eval_summarize(variant_id: str, results_dir: Path, judge_model: str | None,
                   no_llm: bool, judge_max_tokens: int) -> dict:
    if no_llm:
        print("    [跳过] --no-llm 模式下摘要评测需要 LLM 裁判")
        return {"type": "judge", "skipped": True}
    scores = []
    for rec_file in sorted(results_dir.glob("*.txt")):
        raw = rec_file.read_text(encoding="utf-8")
        source = raw.split("== 样本原文 ==")[1].split("== 模型输出 ==")[0].strip()
        output = raw.split("== 模型输出 ==")[-1].strip()
        judge_msg = [
            {"role": "user", "content": JUDGE_SUMMARIZE.format(source=source, output=output)}
        ]
        verdict = chat(judge_msg, temperature=0, model=judge_model,
                       max_tokens=judge_max_tokens)
        parsed = extract_json(verdict)
        if parsed and "total" in parsed:
            scores.append(float(parsed["total"]))
            print(f"    {rec_file.stem}: 总分 {parsed['total']} ｜ {parsed.get('reason', '')}")
        else:
            print(f"    {rec_file.stem}: 裁判输出无法解析，原文：{verdict[:80]}")
    avg = sum(scores) / len(scores) if scores else 0
    print(f"    平均分：{avg:.2f}（{len(scores)} 条）")
    return {"type": "judge", "avg": avg, "count": len(scores)}


JUDGE_EXTRACT = (
    "你是信息提取质检员。下面给出原文、期望提取结果和模型实际输出。\n"
    "请核对：\n"
    "1. 字段值是否正确（与期望语义一致即可，格式不必完全相同；期望为 null 表示原文缺失，"
    "模型也应为 null 或合理留空）；\n"
    "2. 是否有幻觉：模型是否输出了原文中不存在的字段值；\n"
    "3. 输出是否为合法 JSON。\n"
    "只输出 JSON：{{\"total\": 1-5分, \"hallucination\": \"有/无+说明\", "
    "\"reason\": \"逐字段核对结论\"}}\n\n"
    "【原文】\n{source}\n\n【期望结果】\n{expected}\n\n【模型输出】\n{output}"
)


def eval_extract(variant_id: str, results_dir: Path, judge_model: str | None,
                 no_llm: bool, judge_max_tokens: int) -> dict:
    if no_llm:
        print("    [跳过] --no-llm 模式下提取评测需要 LLM 裁判")
        return {"type": "judge", "skipped": True}
    scores = []
    for rec_file in sorted(results_dir.glob("*.txt")):
        raw = rec_file.read_text(encoding="utf-8")
        source = raw.split("== 样本原文 ==")[1].split("== 模型输出 ==")[0].strip()
        output = raw.split("== 模型输出 ==")[-1].strip()
        sample_id = rec_file.stem
        expected = next(s for s in TASKS["extract"]["data"] if s["id"] == sample_id)["expected"]
        judge_msg = [{
            "role": "user",
            "content": JUDGE_EXTRACT.format(
                source=source, expected=json.dumps(expected, ensure_ascii=False), output=output),
        }]
        verdict = chat(judge_msg, temperature=0, model=judge_model,
                       max_tokens=judge_max_tokens)
        parsed = extract_json(verdict)
        if parsed and "total" in parsed:
            scores.append(float(parsed["total"]))
            print(f"    {sample_id}: {parsed['total']} 分 ｜ 幻觉:{parsed.get('hallucination', '?')} ｜ "
                  f"{parsed.get('reason', '')}")
        else:
            print(f"    {sample_id}: 裁判输出无法解析，原文：{verdict[:80]}")
    avg = sum(scores) / len(scores) if scores else 0
    print(f"    平均分：{avg:.2f}（{len(scores)} 条）")
    return {"type": "judge", "avg": avg, "count": len(scores)}


# ---------- 主流程 ----------

def main() -> None:
    args = parse_args()
    task_ids = [args.task] if args.task else list(TASKS)
    report: dict = {"generated_at": datetime.now().isoformat(timespec="seconds")}

    for tid in task_ids:
        task = TASKS[tid]
        print(f"\n########## 任务：{task['title']}（{tid}）##########")
        report[tid] = {}
        for variant in task["variants"]:
            if args.variant and args.variant not in variant["id"]:
                continue
            rdir = RESULTS_DIR / "raw" / tid / variant["id"]
            if not rdir.exists() or not list(rdir.glob("*.txt")):
                print(f"  [跳过] 变体 {variant['id']} 没有结果，先运行：python run_lab.py --task {tid}")
                continue
            print(f"\n  ◆ 变体：{variant['id']} {variant['name']}")
            if tid in ("classify", "emotion"):
                report[tid][variant["id"]] = eval_classify(
                    variant["id"], rdir, TASKS[tid]["data"], args.no_llm)
            elif tid == "summarize":
                report[tid][variant["id"]] = eval_summarize(
                    variant["id"], rdir, args.judge_model, args.no_llm, args.judge_max_tokens)         
            else:
                report[tid][variant["id"]] = eval_extract(
                    variant["id"], rdir, args.judge_model, args.no_llm, args.judge_max_tokens)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"eval_{stamp}.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[完成] 评测报告已保存：{out_file}")


if __name__ == "__main__":
    main()
