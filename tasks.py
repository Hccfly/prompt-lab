"""任务注册表：把"数据 + prompt 变体"组装成三个可运行任务。"""
import json
from pathlib import Path

from prompts import classify_variants, extract_variants, summarize_variants,emotion_variants

BASE = Path(__file__).parent


def _load_samples(name: str) -> list[dict]:
    with open(BASE / "data" / name, encoding="utf-8") as f:
        return json.load(f)


TASKS: dict[str, dict] = {
    "summarize": {
        "title": "文本摘要",
        "data": _load_samples("samples_summarize.json"),
        "variants": summarize_variants.VARIANTS,
    },
    "classify": {
        "title": "客服消息分类",
        "data": _load_samples("samples_classify.json"),
        "variants": classify_variants.VARIANTS,
    },
    "extract": {
        "title": "信息提取",
        "data": _load_samples("samples_extract.json"),
        "variants": extract_variants.VARIANTS,
    },
        "emotion": {
        "title": "情感分析",
        "data": _load_samples("samples_emotion.json"),
        "variants": emotion_variants.VARIANTS,
    },
}


def get_task(task_id: str) -> dict:
    if task_id not in TASKS:
        raise SystemExit(f"未知任务：{task_id}。可选：{', '.join(TASKS)}")
    return TASKS[task_id]
