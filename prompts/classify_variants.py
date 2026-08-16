"""客服消息分类任务的 4 个 prompt 变体。

标签体系（互斥，一条消息只归一类）：
  物流问题 / 商品质量问题 / 退换货 / 咨询 / 投诉

技巧演进路线：
  v1_baseline      零样本，标签只列名字，模型会自由发挥
  v2_label_def     给每个标签下定义 + 输出约束（只输出标签词）
  v3_few_shot      示例同时示范"判断理由 + 结论"
  v4_cot_structured  思维链 + JSON 输出 + 兜底分支（无法判断时）
"""
from __future__ import annotations

LABELS = "物流问题 / 商品质量问题 / 退换货 / 咨询 / 投诉"


def v1_baseline(text: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": f"请对下面这条客服消息进行分类：\n\n{text}\n\n"
            f"分类标签：{LABELS}",
        },
    ]


def v2_label_def(text: str) -> list[dict]:
    """每个标签给出定义和判断要点，并约束输出格式。

    标签名字本身有歧义（比如"快递弄丢了"是物流问题还是投诉？），
    定义清楚后，模型才不会凭感觉选。
    """
    system = (
        "你是电商平台客服工单分类员。请把用户消息归入且仅归入一个标签。\n"
        "标签定义：\n"
        "- 物流问题：涉及快递配送、运输延误、丢件、地址错误等；\n"
        "- 商品质量问题：商品破损、缺件、功能故障、与描述不符等；\n"
        "- 退换货：用户明确要求退货、换货、退款、返修；\n"
        "- 咨询：询问价格、库存、使用方法、优惠活动等，无抱怨情绪；\n"
        "- 投诉：表达强烈不满，要求追究责任或补偿，或包含辱骂。\n"
        "输出要求：只输出一个标签词，不要输出任何解释或其他文字。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


def v3_few_shot(text: str) -> list[dict]:
    """少样本：示例不仅给答案，还示范"先看依据、再下结论"。

    这能教会模型在模糊场景下如何取舍。
    """
    system = (
        "你是电商平台客服工单分类员。请把用户消息归入且仅归入一个标签。\n"
        "标签：物流问题 / 商品质量问题 / 退换货 / 咨询 / 投诉\n"
        "下面给出几个示例（消息 → 判断依据 → 标签），请模仿其判断方式。"
    )
    examples = [
        (
            "我的包裹显示签收但我没收到，打电话也没人接，太气人了！",
            "依据：涉及快递配送与丢件 → 物流问题",
        ),
        (
            "请问这个保温杯现在有活动价吗？",
            "依据：询问价格，无抱怨 → 咨询",
        ),
        (
            "你们这破店，客服永远不回消息，我要投诉你们！",
            "依据：强烈不满并要求追责 → 投诉",
        ),
    ]
    messages = [{"role": "system", "content": system}]
    for msg, answer in examples:
        messages.append({"role": "user", "content": msg})
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": text})
    return messages


def v4_cot_structured(text: str) -> list[dict]:
    """思维链 + 结构化输出 + 兜底分支。

    - reasoning 字段让模型先说明判断依据（CoT 提升复杂判断准确率）
    - label 字段保证程序可直接解析
    - "待人工处理" 兜底：模型没把握时不许硬猜
    """
    system = (
        "你是电商平台客服工单分类员。\n"
        "标签：物流问题 / 商品质量问题 / 退换货 / 咨询 / 投诉。\n"
        "请以 JSON 格式输出，结构如下：\n"
        '{"reasoning": "先简要说明判断依据，不超过50字", "label": "标签名"}\n'
        "规则：label 必须取自上述标签之一；如果消息信息不足、无法可靠判断，"
        'label 填"待人工处理"；只输出 JSON，不要输出其他文字。'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


VARIANTS = [
    {
        "id": "v1_baseline",
        "name": "零样本基线",
        "note": "只列标签名，观察模型对歧义样本的判断",
        "build": v1_baseline,
    },
    {
        "id": "v2_label_def",
        "name": "标签定义+输出约束",
        "note": "定义消歧义；'只输出标签词'保证结果可直接解析",
        "build": v2_label_def,
    },
    {
        "id": "v3_few_shot",
        "name": "少样本示例",
        "note": "示例示范'依据→结论'的判断过程",
        "build": v3_few_shot,
    },
    {
        "id": "v4_cot_structured",
        "name": "思维链+JSON+兜底",
        "note": "CoT 提升准确率；结构化便于解析；信息不足时不硬猜",
        "build": v4_cot_structured,
    },
]
