"""情感分析任务的 4 个 prompt 变体（对应任务B2，仿照 classify_variants.py）。

标签体系（互斥，一条评论只归一类）：
  正面 / 负面 / 中性

技巧演进路线：
  emotion_v1_baseline          零样本基线
  emotion_v2_label_def         标签定义 + 输出约束
  emotion_v3_few_shot          少样本示例
  emotion_v4_cot_structured    思维链 + JSON + 兜底分支
"""
from __future__ import annotations

LABELS = "正面 / 负面 / 中性"


def emotion_v1_baseline(text: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": f"请判断下面这条用户评论的情感倾向：\n\n{text}\n\n"
            f"分类标签：{LABELS}",
        },
    ]


def emotion_v2_label_def(text: str) -> list[dict]:
    """给每个标签定义判断要点，并约束输出格式。

    情感分析里最容易混的是"中性"：很多评论既有好话又有抱怨，
    需要明确的规则告诉模型什么时候算中性。
    """
    system = (
        "你是电商平台的用户评论情感分析员。请把评论归入且仅归入一个标签。\n"
        "标签定义：\n"
        "- 正面：整体满意，表达认可、感谢、推荐、好评，或虽有轻微瑕疵但语气积极；\n"
        "- 负面：表达不满、抱怨、失望、愤怒，或明确指出问题并要求解决；\n"
        "- 中性：客观陈述事实或提问，无明显的情绪倾向，或好话与抱怨大致抵消、无法判断倾向。\n"
        "输出要求：只输出一个标签词，不要输出任何解释或其他文字。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


def emotion_v3_few_shot(text: str) -> list[dict]:
    """少样本：示例示范"看语气 → 下结论"的判断方式，重点演示"中性"的判断。"""
    system = (
        "你是电商平台的用户评论情感分析员。请把评论归入且仅归入一个标签。\n"
        "标签：正面 / 负面 / 中性\n"
        "下面给出几个示例（评论 → 判断依据 → 标签），请模仿其判断方式。"
    )
    examples = [
        (
            "质量太好了！客服态度也棒，下次还来这家买，必须五星好评。",
            "依据：明确认可和推荐，语气积极 → 正面",
        ),
        (
            "收到货发现屏幕有条裂纹，联系客服三天没人回，太失望了。",
            "依据：明确抱怨问题、表达失望 → 负面",
        ),
        (
            "请问这款手机支持无线充电吗？和旧款比有什么升级？",
            "依据：客观提问，无情绪倾向 → 中性",
        ),
        (
            "东西还行吧，价格便宜是便宜，就是包装有点简陋，无所谓了。",
            "依据：好坏大致抵消，无明显倾向 → 中性",
        ),
    ]
    messages = [{"role": "system", "content": system}]
    for msg, answer in examples:
        messages.append({"role": "user", "content": msg})
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": text})
    return messages


def emotion_v4_cot_structured(text: str) -> list[dict]:
    """思维链 + JSON + 兜底：先说明判断依据再给结论；信息不足时交给人工。"""
    system = (
        "你是电商平台的用户评论情感分析员。\n"
        "标签：正面 / 负面 / 中性。\n"
        "请以 JSON 格式输出，结构如下：\n"
        '{"reasoning": "先简要说明判断依据，不超过50字", "label": "标签名"}\n'
        "规则：label 必须取自上述标签之一；如果评论信息不足、无法可靠判断，"
        'label 填"待人工处理"；只输出 JSON，不要输出其他文字。'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


VARIANTS = [
    {
        "id": "emotion_v1_baseline",
        "name": "零样本基线",
        "note": "只列标签名，观察模型对中性/矛盾评论的判断",
        "build": emotion_v1_baseline,
    },
    {
        "id": "emotion_v2_label_def",
        "name": "标签定义+输出约束",
        "note": "重点定义'中性'的判断规则，消除歧义",
        "build": emotion_v2_label_def,
    },
    {
        "id": "emotion_v3_few_shot",
        "name": "少样本示例",
        "note": "示例演示'好坏抵消→中性'这类易错场景",
        "build": emotion_v3_few_shot,
    },
    {
        "id": "emotion_v4_cot_structured",
        "name": "思维链+JSON+兜底",
        "note": "CoT 提升判断准确率；JSON 便于解析；无法判断时交给人工",
        "build": emotion_v4_cot_structured,
    },
]