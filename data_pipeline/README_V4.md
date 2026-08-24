# V4 数据工程脚本使用说明

这组脚本实现 `automotive_medicalgpt_data_task_plan_v4_core.md`：以 AMCK 为回答基底，从 D2/D3/D4 检索维修 Evidence，一次模型调用完成证据选择、路线判断和改写；未使用的 Evidence 可另行生成中文 QA；最终分别导出 MedicalGPT 的 SFT 与 PT JSONL。

## 1. 环境和约束

- Windows 本地：数据清点、Word 提取、规则规范化与过滤，以及 API 生成、校验和导出。
- Colab GPU：03 用小型 instruct 模型复核少量规则拒绝项并构建 Evidence，04 用 `BAAI/bge-m3` embedding 与批量检索；可选 tokenizer 长度校验。
- Windows 本地：同步回 04 结果后调用 API 完成 05/06 生成，不要求本地 GPU。
- Python 3.8+ 推荐；纯预处理脚本兼容项目当前的 Python 3.7。
- 原始数据只读，所有中间结果写到 `data/work`，正式结果写到 `data/final/v1`。
- 所有抽样默认使用固定 `seed=42`；需要另一组可复现实验时传入 `--seed`。
- `.docx` 直接解析；D4 中唯一的旧 `.doc` 可用脚本 01 的 `--convert-legacy-doc` 通过 Word/LibreOffice 转换。转换件只写入 `data/work/converted`，不修改 raw。
- `hash` 检索和 `mock` 生成只用于离线联调。`07_validate_export.py` 默认排除所有 `is_mock=true` 的记录。

`data/work` 按阶段和统一运行名组织：

```text
work/
├─ converted/                 # 01
├─ inventory/                 # 01
├─ normalized/                # 02 规则保留基线，不由 03 回写
├─ semantic/                  # 03 模型决定、召回记录、有效 D1 与抽检样本
├─ evidence/                  # 03 规则保留 + 语义召回后的 Evidence
├─ reports/                   # 02/03
├─ rejected/
│  ├─ normalize/              # 02 规则拒绝
│  └─ <run-name>/             # 05 生成拒绝
├─ embeddings/<run-name>/     # 04 Colab 向量缓存
├─ retrieval/<run-name>/      # 04 检索结果
├─ review/<run-name>/         # 04 抽检样本
└─ generated/<run-name>/      # 05/06 生成结果
```

同一轮 04、05、06 使用同一个 `run-name`，不要再混用 `pilot_v2`、`rewrite_pilot_v3` 等不同名称。

## 2. Windows 本地准备

在项目根目录运行：

```powershell
python -m pip install -r data_pipeline/requirements-local.txt
python data_pipeline/scripts/01_inventory_extract.py
python data_pipeline/scripts/02_normalize_filter.py
```

本机装有 Microsoft Word 或 LibreOffice 时，如需纳入旧 `.doc`：

```powershell
python data_pipeline/scripts/01_inventory_extract.py --convert-legacy-doc
```

首次建议先限制读取量检查格式：

```powershell
python data_pipeline/scripts/02_normalize_filter.py --amck-limit 1000 --d2-limit 1000 --d3-limit 500 --professional-limit 100
```

注意：再次不带 limit 运行会原子覆盖对应中间文件，适合从 Pilot 切换到正式全量准备。

### 2.1 规范化规则和抽检字段

脚本 02 使用可解释规则，不依赖分类模型。规范化记录中保留以下辅助字段：

- D1 `intent`：区分纯价格/选购、维修问题以及“维修问题 + 价格”的混合意图；混合意图保留并写入软 flag。
- D1 `vehicle_context_status`：对 instruction 声明车型与 query 明确自述车型做一致性检查；明确冲突进入 `rejected/normalize/d1_amck.jsonl`，比较其他车型等不确定情形只标记 review。
- D2 `has_*`：仅表示正文提到了图片、链接、评论或其他回答；只有确实依赖缺失上下文时才产生阻断性的 `needs_*`。
- D2 `risk_bypass`：只匹配绕过安全联锁、限速、气囊等明确危险组合；普通 `bypass valve` 只记录 `has_bypass_term`。
- D3 `domain_status`：`automotive`、`non_automotive` 或 `uncertain`；不再把未命中汽车词表直接等同于非汽车。
- D3/D4 `diagnosis_signals`：记录诊断标题、编号步骤、检查动作、检查发现、修复结果和测量信号数量。
- D4 `document_type`：`case_evidence`、`procedure_evidence`、`technical_pt` 或 `maintenance_qa`。

规则回归测试：

```powershell
python -m unittest discover -s data_pipeline/tests -v
```

全量统计位于 `data/work/reports/normalize_report.json`；抽检时应同时查看 normalized 和 rejected，不能只看保留数量。

### 2.2 一次性收敛 Git 托管范围

旧版本曾把 retrieval、review、smoke 等运行产物纳入 Git。更新 `.gitignore` 后，需要执行一次仅清理 Git 索引的迁移；`--cached` 不会删除本机文件：

```powershell
git rm -r --cached data_pipeline/data/work
git rm --cached data_pipeline/test.json
git add .gitignore data_pipeline/README_V4.md data_pipeline/configs data_pipeline/requirements-colab.txt data_pipeline/requirements-local.txt data_pipeline/scripts data_pipeline/tests data_pipeline/.env.example
git add data_pipeline/data/work/normalized/d1_amck.jsonl data_pipeline/data/work/normalized/d2_mechanics.jsonl data_pipeline/data/work/normalized/d3_faults.jsonl data_pipeline/data/work/normalized/d4_documents.jsonl
git add data_pipeline/data/work/rejected/normalize/d1_amck.jsonl data_pipeline/data/work/rejected/normalize/d3_faults.jsonl data_pipeline/data/work/rejected/normalize/d4_documents.jsonl data_pipeline/data/work/reports/normalize_report.json
git status --short
```

确认状态中旧 pilot/smoke 为删除、上述 8 个交接文件为新增或修改后再提交和推送。以后本机重跑 02，只需重新 `git add` 这 8 个交接文件及本次代码变更；Colab 生成的 `semantic`、`evidence`、`embeddings`、`retrieval`、`review` 不纳入 Git。

当前 `normalized/d1_amck.jsonl` 已接近 100 MB。推送前应检查其大小；如果所用远端拒绝该文件，就为它启用 Git LFS，或改用 Drive 作为数据交接，不要继续把更大的普通 Git blob 写入历史。

## 3. 离线端到端冒烟测试

以下命令不需要下载模型或配置 API：

```powershell
python data_pipeline/scripts/01_inventory_extract.py --config data_pipeline/configs/pipeline.smoke.yaml
python data_pipeline/scripts/02_normalize_filter.py --config data_pipeline/configs/pipeline.smoke.yaml --amck-limit 200 --d2-limit 200 --d3-limit 100 --professional-limit 20 --d4-limit 20
python data_pipeline/scripts/03_build_evidence.py --config data_pipeline/configs/pipeline.smoke.yaml
python data_pipeline/scripts/04_embed_retrieve.py --config data_pipeline/configs/pipeline.smoke.yaml --backend hash --run-name smoke --query-limit 10
python data_pipeline/scripts/05_rewrite_amck.py --config data_pipeline/configs/pipeline.smoke.yaml --provider mock --run-name smoke --limit 10
python data_pipeline/scripts/06_generate_evidence_qa.py --config data_pipeline/configs/pipeline.smoke.yaml --provider mock --run-name smoke --limit 6
python data_pipeline/scripts/07_validate_export.py --config data_pipeline/configs/pipeline.smoke.yaml --run-name smoke --include-mock --no-legacy
python data_pipeline/scripts/08_build_pt.py --config data_pipeline/configs/pipeline.smoke.yaml --limit-per-source 20
```

该配置把产物隔离到 `data_pipeline/.tmp/smoke_verify`，不再污染正式 `data/work`。`--include-mock` 只用于验证导出器；正式导出时不要使用。

## 4. Colab 语义召回与正式检索

本地完成 01-02 后，将代码以及 03 所需的最小交接文件提交并推送：

```text
data_pipeline/data/work/normalized/d1_amck.jsonl
data_pipeline/data/work/normalized/d2_mechanics.jsonl
data_pipeline/data/work/normalized/d3_faults.jsonl
data_pipeline/data/work/normalized/d4_documents.jsonl
data_pipeline/data/work/rejected/normalize/d1_amck.jsonl
data_pipeline/data/work/rejected/normalize/d3_faults.jsonl
data_pipeline/data/work/rejected/normalize/d4_documents.jsonl
data_pipeline/data/work/reports/normalize_report.json
```

`.gitignore` 已忽略其余 work 产物。D2 的拒绝项以及 D4 提取中间件不参与本轮模型召回，因而不需要上传。Colab 中 clone/pull 仓库后切换到 GPU 运行时，在项目根目录安装依赖。

先做每个数据源最多 100 个候选的语义 Pilot：

```bash
pip install -r data_pipeline/requirements-colab.txt
python data_pipeline/scripts/03_build_evidence.py --semantic-limit 100
```

03 不会重新分类 normalized 全集：D1 只复核 rejected 中的 `price_only`，D3 只复核 `no_diagnosis_process`，D4 只复核 `technical_article`/`maintenance_qa`。模型必须给出能够在原文中逐字验证的引用；只有结构合法、无危险操作且 `confidence=high` 的 `recall` 才会进入有效数据。重点审核：

```text
data_pipeline/data/work/semantic/semantic_recall_report.json
data_pipeline/data/work/semantic/review_samples.jsonl
data_pipeline/data/work/semantic/d1_decisions.jsonl
data_pipeline/data/work/semantic/d3_decisions.jsonl
data_pipeline/data/work/semantic/d4_decisions.jsonl
data_pipeline/data/work/semantic/recalled_d1.jsonl
data_pipeline/data/work/semantic/recalled_d3.jsonl
data_pipeline/data/work/semantic/recalled_d4.jsonl
```

默认 `review_limit=300`，所以这次最多 207 条候选的 Pilot 会全部进入 `review_samples.jsonl`；`*_decisions.jsonl` 同时是可断点复用的完整决定缓存。

Pilot 只用于判断模型和 Prompt 是否合适，不能直接作为全量检索输入。审核通过后，在同一 Colab 工作区全量重跑 03；已经生成且输入指纹一致的 Pilot 决定会自动复用，然后执行 04：

```bash
python data_pipeline/scripts/03_build_evidence.py
python data_pipeline/scripts/04_embed_retrieve.py \
  --run-name pilot_v4 \
  --splits train
```

03 全量产出 `semantic/effective_d1_amck.jsonl`，它由规则保留的 D1 加模型召回的 D1 组成；04 会优先读取该文件。D3/D4 的语义召回项会在 03 内部并入 Evidence。`normalized/` 和 `rejected/normalize/` 始终保留为可复查的规则基线。

04 分别向量化纯 `query` 和 `instruction + query`，以 0.75/0.25 融合相似度；从 Top-50
候选池中进行轻量 system/DTC/powertrain 软评分、去重和系统多样性选择，最终向 05 提供 Top-5。
system 和 powertrain 不再硬排除候选，不使用不断扩张的故障关键词检索规则。

运行完成后，从 Colab 手动同步回本机：

```text
data_pipeline/data/work/semantic/
data_pipeline/data/work/evidence/
data_pipeline/data/work/retrieval/pilot_v4/
data_pipeline/data/work/review/pilot_v4/
```

`embeddings/pilot_v4` 留在 Colab/Drive 用于断点复用，无需同步回本机。如果要在两次 Colab 会话之间保留 03 的模型缓存，应额外把 `work/semantic` 复制到 Drive；否则新会话会重新推理，但不会改变 02 基线。先检查
`review/pilot_v4/retrieval_samples.jsonl` 和 `retrieval/pilot_v4/retrieval_report.json`，
确认检索质量后再运行 05。

## 5. LLM 改写与 Evidence→QA

脚本调用 OpenAI-compatible 的 Chat Completions 接口。默认自动读取
`data_pipeline/.env`，该文件已被 Git 忽略。DeepSeek V4 Flash 配置示例：

```dotenv
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
LLM_API_KEY=在这里填写你的密钥
```

不要提交真实 `.env`；可复制仓库中的 `.env.example` 作为模板。04 结果同步回本机后，
使用同一个运行名执行 100 条 train Pilot：

```powershell
python data_pipeline/scripts/05_rewrite_amck.py --run-name pilot_v4 --splits train --limit 100
```

04/05 Pilot 审核通过后，不要用全量结果覆盖 `pilot_v4`。由于 03 已经全量完成，回到 Colab 时只需带上相同的 `work/semantic` 与 `work/evidence`，新建正式运行名并完成全量检索：

```bash
python data_pipeline/scripts/04_embed_retrieve.py \
  --run-name full_v1 \
  --query-limit 0
```

将 `retrieval/full_v1` 和 `review/full_v1` 同步回本机，再执行完整 AMCK 改写和首批 500 条 Evidence→QA：

```powershell
python data_pipeline/scripts/05_rewrite_amck.py --run-name full_v1
python data_pipeline/scripts/06_generate_evidence_qa.py --run-name full_v1 --limit 500
```

首批 QA 审核通过后，去掉 `--limit 500` 继续全量运行；相同运行名会跳过已经成功生成的记录。

05 脚本默认显示 `tqdm` 进度条，包括待改写总数、当前请求/重试状态、完成速度、预计剩余时间以及接受/拒绝数；如需关闭可添加 `--no-progress`。
生成脚本每处理 10 条原子保存一次，重复运行同一 `run-name` 会跳过已成功记录，进度条总数只统计本次实际需要重新请求的样本。失败或 literal 校验不通过的记录位于 `data/work/rejected/<run-name>`。
Embedding 和生成记录都带输入指纹；模型、Prompt 或输入内容变化时会自动重算对应缓存。正式实验应给每组配置单独的 `run-name`。

## 6. 正式校验与导出

```powershell
python data_pipeline/scripts/07_validate_export.py --run-name full_v1
python data_pipeline/scripts/08_build_pt.py
```

如果要用训练模型的 tokenizer 做精确长度检查，可在 Colab 运行：

```bash
python data_pipeline/scripts/07_validate_export.py \
  --run-name pilot_v4 \
  --tokenizer /content/你的基础模型目录 \
  --max-tokens 4096
```

默认行为：

- 不导出 mock 记录；
- 专业派生 SFT 只稳定抽取 100 条并且仅进入 train；可用 `--no-legacy` 排除，或用 `--legacy-limit 0` 纳入全部合格记录；
- validation/test 各最多 200 条；
- 相同规范化问题不会跨 train/validation/test；
- D3 只进入 train；D2/D4 的 test 记录不进入 PT；
- MedicalGPT 数据目录只含训练所需字段，完整元数据保存在 `data/work/export`。

最终文件：

```text
data/final/v1/sft/train/data.jsonl
data/final/v1/sft/validation/data.jsonl
data/final/v1/sft/test/data.jsonl
data/final/v1/pt/train/data.jsonl
data/final/v1/pt/validation/data.jsonl
```

人工抽检清单位于 `data/work/review/<run-name>/review_samples.jsonl`，包含各路线/来源的稳定样本、全部导出 test 以及检测到的高风险样本。扩量前应达到 V4 计划中的 Pilot 通过标准。

## 7. 接入当前项目中的 MedicalGPT

先跑 `Base + SFT`。进入 `MedicalGPT` 目录后，以仓库自带的 `scripts/run_sft.sh` 为模板，只需将两个数据目录分别指向：

```text
../data_pipeline/data/final/v1/sft/train
../data_pipeline/data/final/v1/sft/validation
```

其中 test 目录不要传给训练脚本，用于训练完成后的固定评测。单张 Colab GPU 将 `torchrun --nproc_per_node 2` 改为直接执行：

```bash
python training/supervised_finetuning.py \
  --model_name_or_path /content/你的基础模型 \
  --train_file_dir ../data_pipeline/data/final/v1/sft/train \
  --validation_file_dir ../data_pipeline/data/final/v1/sft/validation \
  --do_train --do_eval --use_peft True \
  --max_train_samples -1 --max_eval_samples -1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --model_max_length 4096 \
  --num_train_epochs 1 \
  --output_dir outputs-automobile-sft
```

参数仍应根据所选底模和 Colab 显存调整。只有 `Base + SFT` 的固定 test 结果稳定后，才复制 `scripts/run_pt.sh` 做 `Base + PT + SFT` 对照；PT 的 train/validation 目录为 `../data_pipeline/data/final/v1/pt/train` 和 `../data_pipeline/data/final/v1/pt/validation`。PT LoRA 完成后需先按 MedicalGPT 文档合并 adapter，再把合并模型作为 SFT 的 `model_name_or_path`。
