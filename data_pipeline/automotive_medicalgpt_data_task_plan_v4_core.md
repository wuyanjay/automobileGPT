# 汽车领域 MedicalGPT 数据工程计划（V4：核心构想版）

> 定位：个人项目可实施的中等复杂度方案  
> 环境：Windows 本地 + Google Colab  
> 核心：以 AMCK 为问答基底，用 D2/D3/D4 的维修证据改写 AMCK，并构造 PT/SFT 数据

## 1. 项目目标

构造两类 MedicalGPT 数据：

1. SFT：回答保持 AMCK 的用户友好风格，同时增加真实、按顺序的维修排查步骤；
2. PT：由 D2/D3/D4 中筛选出的汽车维修技术文本组成。

目标回答应满足：

- 用户信息充分时，根据相关维修证据给出具体排查路径；
- 只找到跨车型相似案例时，明确以“类似案例”方式提供参考；
- 没有相关证据时，优化 AMCK 原回答，但不添加具体数值、DTC、车型配置或确定故障结论；
- 用户信息不足时，给出首轮检查，并询问真正有助于缩小范围的信息。

## 2. 对 `task.md` 初步构想的实现

`task.md` 的核心流程在本方案中直接实现为：

```text
D1 AMCK：instruction + input + 原 output
                    ↓
              生成 query embedding
                    ↓
从 D2 / D3 / D4 Evidence 中召回 Top-K
                    ↓
简单系统/动力类型过滤，保留 Top-3
                    ↓
一次 LLM 调用同时完成：
  相关性判断 + 路由选择 + 最终答案改写
                    ↓
        matched / analogy / no_evidence
                    ↓
            Literal Validator
                    ↓
              最终 SFT JSONL
```

对应关系：

- 匹配到强相关数据：用 D2/D3/D4 的检查过程增强 AMCK；
- 只匹配到同系统案例：以“类似案例”方式迁移检查思路；
- 没有匹配：保留 AMCK 的有用部分，删除过度联想，增加信息不足提示；
- D2/D3/D4 中没有被 AMCK 匹配使用的优质内容，另外生成 Evidence→QA，避免浪费。

## 3. 控制复杂度的边界

本方案保留必要环节：

- 一个多语言 embedding 模型；
- 一次向量召回；
- 少量 `powertrain/system` 标签；
- 一次 LLM 判断并改写；
- 一个基于正则的 literal 校验器；
- 小规模人工抽检。

首版不做：

- BM25 + dense + reranker 多级检索；
- 独立 A/B/C Judge 模型调用；
- 复杂案例卡和逐句 claim provenance；
- 全量 Stack Exchange 帖子溯源恢复；
- DPO/RM/PPO；
- 专用 embedding 微调；
- 全量 43,962 条 AMCK 改写。

## 4. 当前数据角色

| 来源 | 当前情况 | 在流程中的角色 |
|---|---:|---|
| D1 AMCK | 43,962 条，61 条空答案 | 用户 query、友好回答基底、SFT 主体 |
| D2 Mechanics Exchange | 12,827 条英文 QA | 真实维修经验、跨语言 Evidence、Evidence→QA、PT |
| D3 故障数据 | 1,441 条，混有非汽车文本 | 中文诊断过程 Evidence、PT |
| D3 派生专业 SFT | 4,302 条 | 过滤后作为 train-only 补充 SFT |
| D4 新能源案例 | 82 个有效 docx + 1 个 doc | 新能源维修 Evidence、Evidence→QA、PT |

D2 不需要先全量翻译。使用多语言 embedding 直接完成中文 AMCK 与英文 D2 的跨语言召回；只把最终选中的英文 Evidence 交给改写模型处理。

## 5. 工程目录

```text
data_pipeline/
├── data/
│   ├── raw/                         # 不修改
│   ├── work/
│   │   ├── normalized/
│   │   ├── evidence/
│   │   ├── embeddings/
│   │   ├── retrieval/
│   │   ├── generated/
│   │   └── rejected/
│   └── final/v1/
│       ├── pt/{train,validation}/
│       └── sft/{train,validation,test}/
├── scripts/
│   ├── 01_inventory_extract.py
│   ├── 02_normalize_filter.py
│   ├── 03_build_evidence.py
│   ├── 04_embed_retrieve.py
│   ├── 05_rewrite_amck.py
│   ├── 06_generate_evidence_qa.py
│   ├── 07_validate_export.py
│   └── 08_build_pt.py
├── configs/
│   └── pipeline.yaml
└── notebooks/
    ├── build_embeddings_colab.ipynb
    └── train_medicalgpt_colab.ipynb
```

脚本支持 `--limit`、`--seed 42` 和 `--resume`。模型结果按输入哈希缓存，不能因为 Colab 断线重复生成。

## 6. 统一数据格式

### 6.1 AMCK 查询记录

```json
{
  "query_id": "d1_000001",
  "instruction": "以下为汽车品牌和车型信息……",
  "query": "方向机重，助力泵、方向机都换了还是一样",
  "base_answer": "AMCK 原 output",
  "powertrain": "unknown",
  "system": "steering",
  "literals": [],
  "split": "train"
}
```

### 6.2 Evidence 记录

```json
{
  "evidence_id": "d3_000123",
  "source": "d3",
  "text": "完整维修问答或案例片段",
  "language": "zh",
  "powertrain": "ice",
  "system": "hvac",
  "literals": ["90-98°C"],
  "source_ref": "spo_0.json:123",
  "split": "train"
}
```

只保留两个结构化标签：

- `powertrain`：`ice / hev / phev / bev / unknown`；
- `system`：`engine / transmission / brake / steering / hvac / electrical / battery / charging / chassis / body / other`。

标签主要使用关键词规则；只有无法判断的记录才批量交给 LLM 分类。不要提取十几个槽位。

### 6.3 改写结果

```json
{
  "sample_id": "rewrite_d1_000001",
  "query_id": "d1_000001",
  "route": "analogy",
  "selected_evidence_ids": ["d3_000123"],
  "answer": "最终中文回答",
  "unsupported_literals": [],
  "passed": true,
  "split": "train"
}
```

## 7. Phase 1：读取、过滤与切分

### 7.1 D1 AMCK

1. 读取 JSON 数组；
2. 删除 61 条空答案；
3. 规范化空白并按 `instruction+input` 去精确重复；
4. 排除选购、价格、纯闲聊等明显非维修/保养问题；
5. 抽取 `powertrain/system` 和显式 DTC、数值+单位；
6. 保留原答案，但不把其中内容自动当作事实。

第一轮只取 1,000 条 AMCK 做端到端 Pilot。通过后扩到 5,000-10,000 条，不在首版处理全部数据。

### 7.2 D2 Mechanics Exchange

1. 读取 12,827 条 JSONL；
2. 先用关键词排除 PCB、自行车、法规纠纷、虫害等明显非目标内容；
3. 模糊记录以 50 条标题为一批让 LLM 判断是否属于汽车维修/保养；
4. 标记依赖图片、评论、其他答案或外链的记录，首版不进入 Evidence；
5. 标记制动、举升、燃油、气囊等安全高风险回答；
6. Evidence 文本使用 `input + output` 英文原文，不先全量翻译；
7. 个人内部实验保留现有 metadata；如果以后发布，再补逐条署名和许可处理。

### 7.3 D3 故障数据

1. `spo_0.json` 按 JSONL 读取；
2. 使用关键词排除电网、绕组、抽油机等非汽车文本；
3. 使用原 `input` 作为 Evidence，不使用三元组 `output` 生成维修回答；
4. 只有包含检查、处理或原因分析的记录进入 Evidence；
5. 只有故障现象而没有处理过程的记录可用于 PT，但不用于增强回答。

### 7.4 D4 新能源 Word 文档

1. 忽略 `~$*.docx`；
2. 唯一 `.doc` 用 Word/LibreOffice 转成 `.docx`；
3. 用 `python-docx` 提取标题和段落；
4. 故障案例按完整文档作为一个 Evidence；
5. 超过模型上下文的文档按“现象/诊断/排除”分成 2-3 段；
6. 关键内容依赖图片的文档首版跳过，不做 OCR；
7. 技术文章可用于 PT，但不一定用于故障回答增强。

### 7.5 简单切分

在 embedding 和生成之前切分：

- D1、D2：按稳定 ID 做 90% train、5% validation、5% test；
- D4：按文件做 80% train、10% validation、10% test；
- D3 原文和 4,302 条专业 SFT 在 V1 全部只进入 train，因为现成专业 SFT 无父案例 ID，暂不使用 D3 构造 validation/test；
- 相同规范化问题放在同一 split；
- 检索时 train query 只能使用 train Evidence，validation/test 同理。

validation/test 只从 D1、D2、D4 构造。这样不需要追查 4,302 条专业 SFT 的父案例，也能避免它与 D3 测试样本发生明显泄漏。

## 8. Phase 2：构造 Evidence Corpus

实现 `03_build_evidence.py`。

Evidence 单元：

- D2：一条英文问题标题 + 最高票答案；
- D3：一条汽车故障 `input`；
- D4：一篇完整故障案例或一个诊断段；
- 不把 AMCK output 放入 Evidence Corpus。

简单质量门：

- 文本非空；
- 内容属于汽车维修/保养；
- 不是纯广告、法规纠纷或闲聊；
- 脱离缺失图片/评论仍能理解；
- 不含明显危险且无说明的操作；
- 长度不超过配置上限，超长 D4 才做分段。

输出：

```text
data/work/evidence/evidence_train.jsonl
data/work/evidence/evidence_validation.jsonl
data/work/evidence/evidence_test.jsonl
data/work/evidence/evidence_stats.json
```

## 9. Phase 3：向量化与匹配

实现 `04_embed_retrieve.py`，主要在 Colab 运行。

### 9.1 Embedding

首版只使用一个支持中英文的模型，例如 `BAAI/bge-m3`：

- AMCK 向量文本：`instruction + input`；
- D2 向量文本：英文 `input + output`；
- D3/D4 向量文本：中文 Evidence；
- embedding 保存为 NumPy 数组，ID 单独保存为 JSON；
- 不使用向量数据库。

### 9.2 召回

每条 AMCK：

1. 计算与同 split Evidence 的余弦相似度；
2. 在取候选池前删除与 query 明确 `powertrain` 冲突的 Evidence；
3. 从剩余 Evidence 中取原始 Top-50 候选池；
4. 双方 `system` 已知且相同时加 `0.02`，已知且不同时减 `0.03`，不再硬删除；
5. query 中有 DTC 时，包含相同 DTC 的候选加 `0.05`；
6. 使用原始余弦分数 `0.55` 清理低分尾部，不用元数据加分救回低分候选；
7. 去除同 source record、相同文本和极高向量相似的重复候选；
8. 最终保留 Top-3 交给改写模型。

`0.55` 是根据 Pilot V1 的 100 条 Top-3 抽检得到的保守起点，只负责清理低分尾部。
每次调整召回规则后继续使用相同 query ID 做 A/B 检查：

- 相关案例进入 Top-3 的比例；
- 明显错误系统匹配的比例；
- 完全没有可用证据的比例。

阈值不负责判断最终能否迁移；该判断仍由下一阶段的 route 完成。

输出：

```json
{
  "query_id": "d1_000001",
  "candidates": [
    {"evidence_id": "d3_000123", "score": 0.78},
    {"evidence_id": "d2_000456", "score": 0.74}
  ]
}
```

## 10. Phase 4：一次调用完成判断与改写

实现 `05_rewrite_amck.py`。

每个请求包含：

- AMCK instruction/input；
- AMCK 原答案；
- Top-3 Evidence 原文及 ID；
- 固定改写规则。

要求模型输出：

```json
{
  "route": "matched | analogy | no_evidence | reject",
  "selected_evidence_ids": [],
  "answer": "最终回答",
  "reason": "一句内部说明"
}
```

### 10.1 `matched`

适用：至少一个 Evidence 与当前问题的系统、主要现象和工况较一致。

改写要求：

- 保留 AMCK 友好表达；
- 使用 Evidence 中真实出现的检查动作组织排查顺序；
- 写清“先做什么、看什么结果、下一步怎么分支”；
- 除非用户输入已有确认结果，否则不能直接宣布 Evidence 的最终故障就是当前车辆故障。

### 10.2 `analogy`

适用：系统和症状类似，但车型、动力类型、工况或最终故障点不同。

改写要求：

- 使用“在同类系统/类似案例中……”；
- 只迁移检查顺序和可验证方向；
- 不迁移特定车型的最终部件结论、DTC、测量值和工具型号。

### 10.3 `no_evidence`

适用：Top-3 都不相关，或没有候选超过阈值。

改写要求：

- 保留 AMCK 中通用、合理的检查方向；
- 删除无来源的具体数值、DTC、零件号、价格、车型配置和唯一结论；
- 不为“显得专业”而增加用户未提供的事实；
- 信息不足时给出 2-4 个能改变诊断分支的追问；
- 仍需给出一两个首轮检查动作，不能只说“去维修店”。

### 10.4 `reject`

适用：非汽车目标问题、答案无法安全改写、输入为空或内容严重损坏。记录失败原因，不进入 final。

### 10.5 统一回答顺序

```text
当前能确定什么 / 不能确定什么
→ 优先检查步骤
→ 每一步的观察点和下一分支
→ 类似案例说明（只有 analogy 才写）
→ 需要补充的信息
→ 必要安全提示
```

Judgement 与 generation 合并在一次调用中；不再单独运行 A/B/C Judge。

## 11. Phase 5：Literal Validator

实现为普通 Python 正则，不调用第二个事实验证模型。

检查：

- DTC；
- 数值+单位；
- 年款、里程；
- 工具型号；
- 零件号/控制器代号。

允许集合：

```text
allowed_literals = literals(user_query)
                 ∪ literals(selected_evidence)
```

AMCK `base_answer` 中的具体 literal 不自动加入允许集合。

如果最终答案出现 unsupported literal：

1. 调用同一个改写 Prompt 重试一次，要求删除这些 literal；
2. 仍然失败则写入 `data/work/rejected/unsupported_literal.jsonl`；
3. 不进入最终 SFT。

Literal Validator 只负责防止具体字符串凭空出现，不扩展为逐句 claim 验证系统。

## 12. Phase 6：三条 SFT 路线

### S1：AMCK + Evidence

来自 `matched` 和 `analogy`，这是项目最核心的数据。

### S2：AMCK + No Evidence

来自 `no_evidence`，专门训练模型在信息不足或缺少依据时保持克制。

### S3：Evidence→QA

用于未被 AMCK 命中的优质 D2/D3/D4 内容：

- D2：选择自包含、汽车相关的问答，改写为中文友好回答；
- D3：选择有检查过程和排除结果的案例；
- D4：每个故障案例最多生成 1 条 QA；
- 问题和答案都只能使用当前 Evidence 的事实；
- 同样运行 Literal Validator。

专业 SFT 的 4,302 条数据作为 `S4 legacy_train_only`：过滤非汽车内容并抽检后加入 train，不参与 validation/test，也不作为本项目效果的主要证明。

### Pilot 规模

第一轮：

- 1,000 条 AMCK 改写；
- 300-500 条 Evidence→QA；
- 专业 SFT 先抽检 100 条，不急于全量加入。

通过后：

- 扩到 5,000-10,000 条 AMCK；
- 扩到 1,000-3,000 条 Evidence→QA；
- 再决定专业 SFT 的加入量。

不规定 matched/analogy/no_evidence 的固定比例，以实际检索结果为准，但报告必须统计三种 route 的数量。

## 13. Phase 7：增量 PT 数据

实现 `08_build_pt.py`，PT 文件必须交付，但是否实际训练由后续对照决定。

来源：

- D2：过滤后的高质量英文维修回答；可选翻译一小部分中文，不做全量翻译；
- D3：过滤后的汽车故障原 `input`；
- D4：完整故障案例和技术文章；
- AMCK 不作为 PT 主体；
- 模型生成的 SFT 不回灌 PT。

输出格式：

```json
{"text":"一篇完整或按诊断阶段切分的汽车技术文本"}
```

PT 与 SFT 使用相同 source split。D4 同一文档的多个片段不能跨 split。

训练顺序：先完成 `Base + SFT`；有余力再运行 `Base + PT + SFT`。PT 没有提升时保留数据文件，但取消训练步骤。

## 14. Phase 8：验证与导出

实现 `07_validate_export.py`。

### 14.1 自动检查

- JSONL 全部可解析；
- 问题、答案、角色非空；
- `sample_id/query_id/route/split` 完整；
- 精确重复为 0；
- train/validation/test 无相同规范化问题；
- unsupported literal 为 0；
- token 长度不超过训练设置；
- train 和 validation 导出到不同目录。

### 14.2 人工抽检

个人项目控制在以下范围：

- 检索结果：100 条 AMCK 的 Top-3；
- 改写结果：matched 80、analogy 60、no_evidence 60，共 200 条；
- Evidence→QA：每个来源 30 条；
- 全部 test：200 条；
- 所有制动、高压、燃油、举升等高风险样本。

评分只用 4 项，每项 0/1：

1. 是否使用了相关证据或正确拒绝了无关证据；
2. 是否给出了有顺序的排查步骤；
3. 是否避免把类似案例当成当前结论；
4. 是否没有无来源的具体 literal。

改写 Pilot 总通过率达到 90%，且高风险严重错误为 0，才扩大生成。

### 14.3 MedicalGPT 导出

SFT：

```json
{"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}
```

PT：

```json
{"text":"..."}
```

目录：

```text
data/final/v1/sft/train/data.jsonl
data/final/v1/sft/validation/data.jsonl
data/final/v1/sft/test/data.jsonl
data/final/v1/pt/train/data.jsonl
data/final/v1/pt/validation/data.jsonl
```

## 15. Windows 与 Colab 分工

### Windows 本地

- 数据清点；
- JSON/JSONL/Word 提取；
- 关键词过滤和简单标签；
- Literal Validator；
- 人工抽检；
- final 合并和格式校验。

### Colab

- 多语言 embedding；
- Top-K 批量检索；
- LLM 改写和 Evidence→QA；
- tokenizer 长度统计；
- MedicalGPT LoRA PT/SFT；
- Base/SFT/PT+SFT 对照。

Colab 每 100 条生成一个批次文件，完成后再写回 Drive。embedding 和模型输出均使用 ID 对齐，不能依赖文件行顺序猜测对应关系。

## 16. 推荐执行进度

### 第 1-3 天：本地准备

- 清点和读取四类数据；
- 提取 D4；
- 过滤 D2/D3；
- 生成简单标签；
- 先完成 source split。

### 第 4-5 天：Evidence 与检索

- 构造 Evidence Corpus；
- Colab 生成 embedding；
- 运行 1,000 条 AMCK 检索；
- 人工检查 100 条 Top-3 并调整阈值。

### 第 6-8 天：改写 Pilot

- 生成 1,000 条 AMCK 改写；
- 生成 300-500 条 Evidence→QA；
- 运行 Literal Validator；
- 人工抽检 200 条改写结果。

### 第 9-10 天：导出与 SFT

- 生成 PT/SFT final；
- 运行 MedicalGPT LoRA SFT；
- 用固定 200 条 test 比较 Base 与 SFT；
- 决定是否扩到 5,000-10,000 条和是否做 PT 训练。

## 17. 首版完成定义

- [ ] D1-D4 均已转换为统一记录；
- [ ] D2/D3 非汽车内容和上下文残缺内容已过滤；
- [ ] D4 Word 文档已提取；
- [ ] Evidence Corpus 已包含 D2/D3/D4；
- [ ] AMCK 与 Evidence 已完成跨语言向量匹配；
- [ ] matched、analogy、no_evidence 三条路线均有数据；
- [ ] AMCK 的匹配判断和回答改写由一次调用完成；
- [ ] 所有生成 SFT 通过 Literal Validator；
- [ ] 未被 AMCK 使用的 Evidence 已通过 Evidence→QA 利用；
- [ ] PT 和 SFT 均已导出为 MedicalGPT 格式；
- [ ] Pilot 人工通过率达到 90%；
- [ ] Colab 已完成 Base 与 SFT 的固定测试集对比；
- [ ] 只有 Pilot 达标后才进行扩量。

这套流程保留了 `task.md` 中“AMCK 为基底、外部案例负责增强、无匹配时保守优化”的核心，同时把匹配判断和答案生成合并为一次调用，使其仍然适合个人项目实施。
