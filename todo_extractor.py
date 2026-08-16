"""邮件待办提取器：把 inbox/ 文件夹里的 .txt 邮件，批量提取成待办清单（邮件专用 prompt 版）。

用法：
    python todo_extractor.py                    # 处理 inbox/ 下所有邮件
    python todo_extractor.py inbox/meeting.txt  # 只处理指定文件
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")   # 中文输出不乱码

import config          # noqa: F401  加载 .env 里的 key
from llm import chat

INBOX = Path("inbox")
REPORT = Path("待办清单.md")


# ① 新增：邮件专用提取 prompt（字段贴合邮件场景：主题/发件人/日期/截止时间/行动项）
def email_messages(text: str) -> list[dict]:
    system = (
        "你是行政助理，负责从邮件中提取待办。请输出 JSON：\n"
        '{"主题": "邮件主题，一句话", "发件人": "发件人姓名", '
        '"日期": "涉及的关键日期(YYYY-MM-DD)，无则null", '
        '"截止时间": "需要回复或行动的截止时间，无则null", '
        '"行动项": ["要求你做的事", ...]}\n'
        "规则：行动项必须能在原文找到依据；找不到的字段填 null，行动项为 []；只输出 JSON。"
    )
    examples = [
        (
            "主题：产品评审会通知\n发件人：张经理\n\n本周四（2026-08-20）下午3点在A栋501开评审会，请准备PPT，周三18:00前发到review@example.com。",
            '{"主题": "产品评审会通知", "发件人": "张经理", "日期": "2026-08-20", '
            '"截止时间": "2026-08-19 18:00", "行动项": ["准备评审PPT", "发到review@example.com"]}',
        ),
        (
            "主题：报价需求\n发件人：李总\n\n请提供XX-2000报价单、交货周期和技术方案，最迟周五下班前回复。",
            '{"主题": "报价需求", "发件人": "李总", "日期": null, '
            '"截止时间": "周五下班前", "行动项": ["提供报价单", "说明交货周期", "提供技术方案"]}',
        ),
    ]
    messages = [{"role": "system", "content": system}]
    for msg, ans in examples:
        messages.append({"role": "user", "content": msg})
        messages.append({"role": "assistant", "content": ans})
    messages.append({"role": "user", "content": text})
    return messages


# ② 解析工具：从模型输出里抠出第一个 JSON 对象
def extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ③ 单封邮件提取：调用 API，失败不中断整批
def parse_email(f: Path) -> dict | None:
    text = f.read_text(encoding="utf-8")
    print(f"  ⏳ 提取中：{f.name} ...", flush=True)
    try:
        reply = chat(email_messages(text), temperature=0)   # ← 切换点：通用 prompt → 邮件专用 prompt
    except Exception as e:  # noqa: BLE001  一封失败不影响其他邮件
        print(f"  ✗ {f.name} 调用失败：{e}")
        return None
    data = extract_json(reply)
    if data is None:
        print(f"  ✗ {f.name} 输出不是合法 JSON，原文：{reply[:120]}")
        return None
    # 空结果判定：模型对乱文也常输出"全是 null 的合法 JSON"，算无效
    if not data.get("事项") and not data.get("主题") and not data.get("行动项"):
        print(f"  ✗ {f.name} 提取结果为空（可能是无效邮件），已跳过")
        return None
    return data


# ④ 渲染报告：结构化数据 → Markdown 待办清单
def render_markdown(items: list[tuple[Path, dict]]) -> str:
    lines = [f"# 邮件待办清单（{datetime.now():%Y-%m-%d %H:%M} 生成）", ""]
    for f, data in items:
        title = data.get("主题") or data.get("事项") or f.stem
        lines.append(f"## {title}　（来源：{f.name}）")
        if data.get("发件人"):
            lines.append(f"- **发件人**：{data['发件人']}")
        if data.get("日期"):
            lines.append(f"- **日期**：{data['日期']}")
        if data.get("截止时间"):
            lines.append(f"- **截止时间**：{data['截止时间']}")
        actions = data.get("行动项") or []
        if actions:
            lines.append("- **行动项**：")
            for a in actions:
                lines.append(f"  - [ ] {a}")
        lines.append("")
    return "\n".join(lines)


# ⑤ 主流程
def main() -> None:
    args = sys.argv[1:]
    files = [Path(a) for a in args] if args else sorted(INBOX.glob("*.txt"))
    if not files:
        print(f"没有找到邮件。请把 .txt 邮件放进 {INBOX}/ 文件夹。")
        return
    print(f"共 {len(files)} 封邮件，开始提取……")

    items: list[tuple[Path, dict]] = []
    for f in files:
        data = parse_email(f)
        if data:
            items.append((f, data))

    # 控制台摘要（打印全部字段）
    print("\n===== 控制台摘要 =====")
    for f, data in items:
        title = data.get("主题") or data.get("事项") or f.stem
        line = f"• {title}（{f.name}）"
        if data.get("发件人"):
            line += f"｜发件人：{data['发件人']}"
        if data.get("日期"):
            line += f"｜日期：{data['日期']}"
        if data.get("截止时间"):
            line += f"｜截止：{data['截止时间']}"
        actions = data.get("行动项") or []
        if actions:
            line += f"｜{len(actions)} 个待办：{'；'.join(actions)}"
        print(line)

    REPORT.write_text(render_markdown(items), encoding="utf-8")
    print(f"\n✅ 成功 {len(items)}/{len(files)} 封，报告已生成：{REPORT}")
    if len(items) < len(files):
        print("⚠️ 有邮件提取失败，请检查上面的 ✗ 提示。")


if __name__ == "__main__":
    main()