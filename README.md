# 提示词实验室 Prompt Lab

一个入门级的 **Prompt Engineering 实战项目**：用 DeepSeek API 跑「摘要 / 分类 / 信息提取」
三个任务，每个任务提供 4 个由浅入深的 prompt 变体，通过真实调用和评测脚本，直观感受
**提示词的写法如何影响模型输出质量**。

## 为什么用这个项目练手

- **零成本起步**：所有 prompt 变体都在 `prompts/` 里，先跑 `--mock` 模式就能读懂、改写，不花钱也能学；
- **对比出真知**：同一输入、不同写法的输出并列对比，效果差异一目了然；
- **可量化评测**：分类任务算准确率，摘要/提取任务用「LLM 裁判」打分，改完 prompt 立刻知道有没有变好；
- **覆盖核心技能**：系统提示词、few-shot、结构化输出、思维链（CoT）、防幻觉、LLM-as-judge 全都能练到。

## 快速开始

```bash
# 1. 安装依赖（只需要 requests）
pip install -r requirements.txt

# 2. 配置 API key（申请地址 https://platform.deepseek.com/）
copy .env.example .env
#    然后用编辑器打开 .env，把 DEEPSEEK_API_KEY 换成你自己的 key

# 3. 先不花钱看一遍所有 prompt 长什么样（强烈建议第一步做这个）
python run_lab.py --mock

# 4. 真实跑起来（可加 --task / --variant / --limit 缩小范围）
python run_lab.py --limit 1        # 先每条样本试 1 条，确认没问题再全量跑
python run_lab.py                  # 全量跑

# 5. 评测：对比哪个变体效果最好
python evaluate.py                 # 需要先生成结果
```

> 费用参考：deepseek-chat 很便宜，全量跑一遍（3 任务 × 4 变体 × 3~6 样本 + 评测）通常不到 0.1 元。
> 真实验证前，可以先 `python run_lab.py --mock --task summarize` 检查 prompt 内容。

## 目录结构

```
prompt-lab/
├── run_lab.py                 # 主运行器：跑任务、保存结果、mock 预览
├── evaluate.py                # 评测脚本：准确率 / LLM 裁判打分
├── llm.py                     # DeepSeek API 封装（读读它，理解 payload）
├── config.py                  # .env 加载
├── tasks.py                   # 任务注册表
├── prompts/                   # ★ 三个任务各自的 prompt 变体（重点改造对象）
│   ├── summarize_variants.py  #   摘要：4 个变体
│   ├── classify_variants.py   #   分类：4 个变体
│   └── extract_variants.py    #   提取：4 个变体
├── data/                      # 示例数据（含 expected 期望答案）
├── results/                   # 运行结果（自动生成，已 gitignore）
└── .env / .env.example        # API key 配置
```

## 三个任务与技巧对照

| 任务 | 练什么技能 | 变体演进 |
|---|---|---|
| 文本摘要 | 系统提示词、格式/长度约束、忠实度、结构化输出 | 零样本 → 角色+约束 → few-shot → JSON |
| 客服消息分类 | 标签定义消歧义、输出约束、思维链、兜底分支 | 零样本 → 标签定义 → few-shot → CoT+JSON |
| 信息提取 | Schema 设计、null 处理、防幻觉自检 | 零样本 → Schema → few-shot → 自检 |

## 建议的练习路线（由浅入深）

1. **读懂阶段**：跑 `python run_lab.py --mock`，把 12 个变体的 prompt 通读一遍，
   对照每个文件的注释理解"为什么这样写"。
2. **基线阶段**：真实跑一遍 `python run_lab.py`，打开 `results/raw/` 对比，
   观察 v1（零样本）暴露的问题——摘要太长、分类凭感觉、提取字段乱。
3. **改进阶段**（核心练习）：挑一个任务，把 v1 改进成你自己的 v5：
   - 摘要：试试"先写一句话结论，再给要点"；或加"受众"设定（给领导看 vs 给新人看）；
   - 分类：给更细的标签定义，或加一个你遇到的真实歧义样本；
   - 提取：给一个故意刁钻的文本（信息矛盾/口语化/中英混杂），看模型怎么处理。
   每改一次就重新跑，用 `evaluate.py` 验证是否变好，把观察记下来。
4. **评测阶段**：打开 `evaluate.py` 里 `JUDGE_SUMMARIZE` / `JUDGE_EXTRACT`，
   改一改裁判 prompt 的评分维度，体会"评测 prompt 本身也是 prompt 工程"。
5. **迁移阶段**：用自己的真实素材替换 `data/` 里的样本（比如你的工作邮件、真实客服记录），
   把三个任务改造成你手头能用的工具——这时你就完成了一个真实项目。

## 常见问题

- **报错 `未找到有效的 DEEPSEEK_API_KEY`**：确认 `.env` 已创建且 key 填对；也可以先 `--mock` 模式。
- **报错 `model: invalid type: null`（HTTP 400）**：说明请求里 `model` 字段是空值。本项目已在 `llm.py`/`run_lab.py` 里防御修复；如果你自己改代码，注意别把 `None` 传进 `chat()` 的 `model` 参数。
- **报错 `Model Not Exist` / 迁移提示**：`deepseek-chat` / `deepseek-reasoner` 旧模型名已于 2026-07-24 停用，改用 `deepseek-v4-flash`（默认）或 `deepseek-v4-pro`，可通过 `--model` 或 `.env` 的 `DEEPSEEK_MODEL` 指定。
- **V4 模型是推理模型**：响应里有 `reasoning_content`（推理过程）和 `content`（最终答案），本项目只取 `content`；如果 `max_tokens` 太小，token 可能全被推理耗尽导致 `content` 为空，调大即可。
- **想让输出更"有创意"**：调高 `--temperature`（0.7+）；想要稳定一致则用 0~0.3。
- **裁判打分不准**：LLM 裁判有随机性，可把 `temperature` 固定为 0（评测脚本已默认），
  或加大样本量、修改裁判评分维度。
- **想换模型**：`llm.py` 里的 `DEEPSEEK_MODEL`，或运行时 `--model` 参数。

## 下一步可以玩的花样

- 加一个 `--repeat 3` 参数，同一个变体跑 3 次观察稳定性；
- 给每个变体加"成本"统计（token 用量），练"效果 vs 成本"的权衡；
- 把评测结果画成对比表格（输出 Markdown 表格），形成自己的实验报告。
