"""摘要任务的 4 个 prompt 变体。

每个变体演示一种提示词技巧：
  v1_baseline          零样本基线（最常见的"菜鸟写法"）
  v2_system_constraints  系统提示词 + 角色 + 明确约束（格式/长度/忠实度）
  v3_few_shot          少样本示例，示范理想输出形态
  v4_structured        结构化输出（JSON），便于程序直接消费
"""
from __future__ import annotations


def v1_baseline(text: str) -> list[dict]:
    """最朴素的写法：一句话丢给模型，无任何约束。"""
    return [
        {"role": "user", "content": f"请总结下面这篇文章：\n\n{text}"},
    ]


def v2_system_constraints(text: str) -> list[dict]:
    """角色 + 输出格式 + 三条硬约束。

    约束写清楚后，模型不再自由发挥：
    1. 格式：要点式、固定 3 条
    2. 长度：每条 ≤ 30 字
    3. 忠实度：只保留原文事实，不添加、不评价
    """
    system = (
        "你是一名专业的新闻编辑，擅长从长文中提炼核心信息。\n"
        "请阅读用户提供的文章，用中文输出一份摘要，要求：\n"
        "1. 输出为要点式（每行一条），共 3 条；\n"
        "2. 每条不超过 30 个字；\n"
        "3. 只保留最重要的事实，去掉修辞、背景和细节铺垫；\n"
        "4. 不要添加原文没有的信息，不要发表观点。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


def v3_few_shot(text: str) -> list[dict]:
    """少样本：先给 2 个"原文 → 理想摘要"的例子，再给目标文本。

    示例本身就在教模型"什么算好的摘要"，比口头描述更有效。
    """
    system = (
        "你是一名专业的新闻编辑，擅长提炼核心信息。\n"
        "下面给出两个例子，请模仿例子的风格和格式，为用户提供的文章输出 3 条要点摘要。"
    )
    examples = [
        {
            "article": (
                "全球知名咖啡连锁品牌星巴克今天宣布，将在未来三年内在中国新增3000家门店，"
                "重点布局三四线城市。公司表示，下沉市场是未来增长的主要引擎，"
                "同时将继续推进数字化点单和会员体系建设。"
            ),
            "summary": "要点：\n- 星巴克三年内将在华新增3000家门店\n- 重点布局三四线城市下沉市场\n- 同步推进数字化点单与会员体系",
        },
        {
            "article": (
                "教育部昨日发布通知，明确2025年秋季学期起，全国中小学将全面开设人工智能通识课程。"
                "课程以体验和启蒙为主，不设考试，旨在培养学生的数字素养。"
                "首批试点城市包括北京、上海、深圳等10个城市。"
            ),
            "summary": "要点：\n- 2025年秋全国中小学开设AI通识课\n- 课程重体验启蒙、不设考试\n- 首批在北上深等10城试点",
        },
    ]
    messages = [{"role": "system", "content": system}]
    for ex in examples:
        messages.append({"role": "user", "content": ex["article"]})
        messages.append({"role": "assistant", "content": ex["summary"]})
    messages.append({"role": "user", "content": text})
    return messages


def v4_structured(text: str) -> list[dict]:
    """结构化输出：要求模型返回 JSON，程序可以直接解析。

    实战价值：摘要结果要被下游系统使用（比如喂给报表、推送给用户），
    固定 JSON 结构比自由文本可靠得多。
    """
    system = (
        "你是信息提炼助手。请阅读用户提供的文章，输出一个 JSON 对象，结构如下：\n"
        '{"summary": "一句话概括全文，不超过50字", '
        '"key_points": ["要点1", "要点2", "要点3"]}\n'
        "要求：key_points 恰好 3 条；内容必须忠于原文且全面包含原文内容，去除冗余信息；只输出 JSON，不要输出其他文字。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]


def v5_conclusion_first(text: str) -> list[dict]:
    system = (
        "你是一名资深新闻编辑。请严格按两步输出：\n"
        "1. 第一行：用不超过 20 个字的一句话概括全文核心结论；\n"
        "2. 第二行起：用 3 条要点展开支撑结论的事实，每条不超过 30 字；\n"
        "只写事实，不评价，不添加原文没有的信息。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]

def v6_for_boss(text: str) -> list[dict]:
    system = (
        "你是一名资深新闻编辑,读者是公司高管，只保留与决策相关的事实，去掉行业背景。请严格按两步输出：\n"
        "1. 第一行：用不超过 20 个字的一句话概括全文核心结论；\n"
        "2. 第二行起：用 3 条要点展开支撑结论的事实，每条不超过 30 字；\n"
        "只写事实，不评价，不添加原文没有的信息。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]

def v6_for_newbie(text: str) -> list[dict]:
    system = (
        "你是一名资深新闻编辑,读者是刚入职的新人，需要保留必要的背景解释。请严格按两步输出：\n"
        "1. 第一行：用不超过 20 个字的一句话概括全文核心结论；\n"
        "2. 第二行起：用 3 条要点展开支撑结论的事实，每条不超过 30 字；\n"
        "只写事实，不评价，不添加原文没有的信息。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": text},
    ]
VARIANTS = [
    {
        "id": "v1_baseline",
        "name": "零样本基线",
        "note": "无约束写法，观察模型自由发挥的常见问题（太长、重点偏、格式随意）",
        "build": v1_baseline,
    },
    {
        "id": "v2_system_constraints",
        "name": "系统提示词+约束",
        "note": "角色 + 明确格式/长度/忠实度约束，输出显著变稳",
        "build": v2_system_constraints,
    },
    {
        "id": "v3_few_shot",
        "name": "少样本示例",
        "note": "用例子示范理想输出，减少风格漂移",
        "build": v3_few_shot,
    },
    {
        "id": "v4_structured",
        "name": "结构化JSON",
        "note": "固定 JSON 结构，可直接被程序解析消费",
        "build": v4_structured,
    },
    {
        "id": "v5_conclusion_first",
        "name": "结论先行",
        "note": "先一句话结论再要点展开——适合给领导汇报的形态",
        "build": v5_conclusion_first,
    },
    {
            "id": "v6_for_boss",
            "name": "面向高管的摘要",
            "note": "针对公司高管的简洁摘要，只保留与决策相关的信息",
            "build": v6_for_boss,
        },
    {
            "id": "v6_for_newbie",
            "name": "面向新人的摘要",
            "note": "针对刚入职的新人，需要保留必要的背景解释",
            "build": v6_for_newbie,
        },
]
