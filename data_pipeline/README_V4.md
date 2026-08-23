# V4 数据工程脚本使用说明

这组脚本实现 `automotive_medicalgpt_data_task_plan_v4_core.md`：以 AMCK 为回答基底，从 D2/D3/D4 检索维修 Evidence，一次模型调用完成证据选择、路线判断和改写；未使用的 Evidence 可另行生成中文 QA；最终分别导出 MedicalGPT 的 SFT 与 PT JSONL。

## 1. 环境和约束

- Windows 本地：数据清点、Word 提取、规范化、过滤、PT 构建、校验与导出。
- Colab：正式的 `BAAI/bge-m3` embedding、批量检索、LLM 生成和可选 tokenizer 长度校验。
- Python 3.8+ 推荐；纯预处理脚本兼容项目当前的 Python 3.7。
- 原始数据只读，所有中间结果写到 `data/work`，正式结果写到 `data/final/v1`。
- 所有抽样默认使用固定 `seed=42`；需要另一组可复现实验时传入 `--seed`。
- `.docx` 直接解析；D4 中唯一的旧 `.doc` 可用脚本 01 的 `--convert-legacy-doc` 通过 Word/LibreOffice 转换。转换件只写入 `data/work/converted`，不修改 raw。
- `hash` 检索和 `mock` 生成只用于离线联调。`07_validate_export.py` 默认排除所有 `is_mock=true` 的记录。

## 2. Windows 本地准备

在项目根目录运行：

```powershell
python -m pip install -r data_pipeline/requirements-local.txt
python data_pipeline/scripts/01_inventory_extract.py
python data_pipeline/scripts/02_normalize_filter.py
python data_pipeline/scripts/03_build_evidence.py
```

本机装有 Microsoft Word 或 LibreOffice 时，如需纳入旧 `.doc`：

```powershell
python data_pipeline/scripts/01_inventory_extract.py --convert-legacy-doc
```

首次建议先限制读取量检查格式：

```powershell
python data_pipeline/scripts/02_normalize_filter.py --amck-limit 1000 --d2-limit 1000 --d3-limit 500 --professional-limit 100
python data_pipeline/scripts/03_build_evidence.py
```

注意：再次不带 limit 运行会原子覆盖对应中间文件，适合从 Pilot 切换到正式全量准备。

### 2.1 规范化规则和抽检字段

脚本 02 使用可解释规则，不依赖分类模型。规范化记录中保留以下辅助字段：

- D1 `intent`：区分纯价格/选购、维修问题以及“维修问题 + 价格”的混合意图；混合意图保留并写入软 flag。
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

该配置把产物隔离到 `data/work/smoke_verify` 和 `data/final/smoke_verify`。`--include-mock` 只用于验证导出器；正式导出时不要使用。

## 4. Colab 正式检索

将项目目录挂载到 Google Drive，在项目根目录执行：

```bash
pip install -r data_pipeline/requirements-colab.txt
python data_pipeline/scripts/04_embed_retrieve.py \
  --backend sentence-transformers \
  --model BAAI/bge-m3 \
  --run-name pilot_v2 \
  --embedding-cache-name pilot_v1 \
  --query-limit 1000
```

`--embedding-cache-name pilot_v1` 会尝试复用上一轮 BGE-M3 向量，同时把新结果写到
`retrieval/pilot_v2` 和 `review/pilot_v2`；如果旧缓存不存在，脚本会自动重新计算。

正式配置默认先屏蔽明确冲突的动力类型，再从 Top-50 候选池中进行 system 软评分、DTC
加分和候选去重，并以原始相似度 `0.55` 清理低分尾部。人工查看
`data/work/review/pilot_v2/retrieval_samples.jsonl` 中的 100 条 Top-3，并结合
`retrieval/pilot_v2/retrieval_report.json` 的分数区间、无候选原因和去重统计决定是否继续调整。
可用 `--min-score` 或 `--candidate-pool-k` 临时覆盖正式配置；阈值只负责清理低分候选，
最终是否采用 Evidence 仍由 05 的 route 判断。

## 5. LLM 改写与 Evidence→QA

脚本调用 OpenAI-compatible 的 `/v1/chat/completions` 接口。PowerShell 示例：

```powershell
$env:LLM_BASE_URL = "https://你的服务地址/v1"
$env:LLM_MODEL = "你的模型名"
$env:LLM_API_KEY = "你的密钥"

python data_pipeline/scripts/05_rewrite_amck.py --provider api --run-name pilot_v2 --limit 1000
python data_pipeline/scripts/06_generate_evidence_qa.py --provider api --run-name pilot_v2 --limit 500
```

生成脚本每 100 条原子保存一次，重复运行同一 `run-name` 会跳过已成功记录。失败或 literal 校验不通过的记录位于 `data/work/rejected/<run-name>`。
Embedding 和生成记录都带输入指纹；模型、Prompt 或输入内容变化时会自动重算对应缓存。正式实验应给每组配置单独的 `run-name`。

## 6. 正式校验与导出

```powershell
python data_pipeline/scripts/07_validate_export.py --run-name pilot_v1
python data_pipeline/scripts/08_build_pt.py
```

如果要用训练模型的 tokenizer 做精确长度检查，可在 Colab 运行：

```bash
python data_pipeline/scripts/07_validate_export.py \
  --run-name pilot_v1 \
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
