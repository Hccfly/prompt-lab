"""DeepSeek API 封装。

只做一件事：把 messages 发给模型，返回回复文本。
messages 的格式与 OpenAI 一致：
    [{"role": "system", "content": "..."},
     {"role": "user", "content": "..."}]

练习提示词工程时，建议直接读这个文件里的 payload，
理解"你写的 prompt 最终是以什么形态发给模型的"。
"""
import os

import requests

BASE_URL = "https://api.deepseek.com"
# 注意：deepseek-chat / deepseek-reasoner 旧模型名已于 2026-07-24 停用，
# 请使用 V4 新模型名：deepseek-v4-flash（快、便宜）或 deepseek-v4-pro（更强）。
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")


def get_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key or key == "sk-你的key填这里":
        raise RuntimeError(
            "未找到有效的 DEEPSEEK_API_KEY。\n"
            "请复制 .env.example 为 .env，填入你的 key（申请地址 https://platform.deepseek.com/）。\n"
            "也可以先运行：python run_lab.py --mock  来不调用 API、只看 prompt 本身。"
        )
    return key


def chat(
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 2000,
    model: str | None = None,
) -> str:
    """调用 DeepSeek chat completions，返回回复文本（已去除首尾空白）。"""
    # 防御：调用方显式传 None 时回落默认模型，避免 payload 里 model 为 null 被 API 拒绝
    model = model or DEFAULT_MODEL
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code >= 400:
        # 把 API 返回的具体错误信息透出来，方便排查（比如模型名/参数不合法）
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text[:500]
        raise RuntimeError(
            f"DeepSeek API 返回 HTTP {resp.status_code}: {detail}"
        )
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()
