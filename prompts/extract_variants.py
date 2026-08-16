"""信息提取任务的 4 个 prompt 变体。

任务：从活动通知/邮件中提取结构化字段，缺失的填 null。
字段：事项 / 日期 / 地点 / 截止时间 / 行动项

技巧演进路线：
  v1_baseline     零样本，让模型自己决定输出什么
  v2_schema       明确 schema + 类型 + null 规则
  v3_few_shot     完整"原文 → JSON"映射示例
  v4_self_check   自检机制（每字段必须能在原文找到依据），防幻觉
"""
from __future__ import annotations

SCHEMA_DESC = (
    "输出字段定义：\n"
    '- "事项"：本次通知的核心事件，字符串；\n'
    '- "日期"：事件发生的日期（YYYY-MM-DD），字符串；\n'
    '- "地点"：事件地点，字符串；\n'
    '- "截止时间"：需要行动的截止时间，字符串；\n'
    '- "行动项"：要求接收者去做的事情，字符串数组，可为空数组。\n'
    "规则：字段在原文中找不到明确信息时填 null（行动项为数组时填 []）。"
)


def v1_baseline(text: str) -> list[dict]:
    return [
        {"role": "user", "content": f"从下面这段通知中提取信息：\n\n{text}"},
    ]


def v2_schema(text: str) -> list[dict]:
    """明确 schema：字段名、含义、缺失时的处理规则。

    提取任务最大的坑是"模型自由发挥字段"，
    先把字段和 null 规则定死，输出才可被程序消费。
    """
    system = (
        "你是信息提取助手。请从用户提供的通知/邮件中提取信息，"
        "只输出一个 JSON 对象，不要输出其他文字。\n" + SCHEMA_DESC
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


def v3_few_shot(text: str) -> list[dict]:
    """少样本：给出完整的"原文 → 目标 JSON"映射。

    注意第二个示例故意缺了地点，演示"缺失填 null"的处理方式。
    """
    system = (
        "你是信息提取助手。请模仿示例，从用户提供的通知中提取 JSON 对象。\n"
        "字段：事项 / 日期 / 地点 / 截止时间 / 行动项；缺失的填 null，行动项为空数组。"
    )
    examples = [
        (
            "各位同事：本周五（2025-03-14）下午3点，在三楼会议室召开季度评审会，"
            "请各项目负责人准备汇报材料，并于周四下班前把PPT发到行政邮箱。",
            '{"事项": "季度评审会", "日期": "2025-03-14", "地点": "三楼会议室", '
            '"截止时间": "周四下班前", "行动项": ["准备汇报材料", "把PPT发到行政邮箱"]}',
        ),
        (
            "温馨提示：本周六（2025-03-15）上午10点进行消防演练，请大家提前关闭电源，"
            "演练时按疏散路线到操场集合。",
            '{"事项": "消防演练", "日期": "2025-03-15", "地点": null, '
            '"截止时间": "2025-03-15 上午10点", "行动项": ["提前关闭电源", "按疏散路线到操场集合"]}',
        ),
    ]
    messages = [{"role": "system", "content": system}]
    for msg, answer in examples:
        messages.append({"role": "user", "content": msg})
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": text})
    return messages


def v4_self_check(text: str) -> list[dict]:
    """自检机制：每个字段必须能在原文中找到依据，找不到就填 null。

    信息提取类任务最怕模型"脑补"不存在的字段值（幻觉），
    这个指令相当于给模型加了一道校验闸门。
    """
    system = (
        "你是信息提取助手，提取时必须严格忠实于原文。\n"
        "步骤：\n"
        "1. 通读全文，定位与各字段相关的句子；\n"
        "2. 逐一提取字段值，每个值必须能在原文中找到依据；找不到就填 null（行动项为 []）；\n"
        "3. 输出 JSON 对象，并在最后加上 \"checked\": true 表示已完成自检。\n"
        "字段定义：\n" + SCHEMA_DESC + "\n只输出 JSON，不要输出其他文字。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


VARIANTS = [
    {
        "id": "v1_baseline",
        "name": "零样本基线",
        "note": "无字段定义，模型自由发挥，字段名和格式都不可控",
        "build": v1_baseline,
    },
    {
        "id": "v2_schema",
        "name": "明确Schema",
        "note": "字段名/类型/null规则写死，输出稳定可解析",
        "build": v2_schema,
    },
    {
        "id": "v3_few_shot",
        "name": "少样本示例",
        "note": "'原文→JSON'完整映射，含缺失字段的示范",
        "build": v3_few_shot,
    },
    {
        "id": "v4_self_check",
        "name": "自检防幻觉",
        "note": "要求每个字段都有原文依据，找不到就填 null",
        "build": v4_self_check,
    },
]
