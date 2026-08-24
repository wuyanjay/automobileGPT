#!/usr/bin/env python3
"""Grounded small-instruct-model recall for rejected D1/D3/D4 records."""

from __future__ import print_function

import json
from collections import Counter

from tqdm import tqdm

from _common import (
    ensure_dir,
    normalize_document,
    normalize_inline,
    parse_model_json,
    read_jsonl,
    stable_hash,
    write_json,
    write_jsonl,
)


SYSTEM_PROMPT = """你是汽车数据工程中的语义召回器。输入记录已经被保守规则拒绝，你的任务只是判断它是否属于少量值得召回的误杀样本。

总原则：
1. 宁可继续拒绝，也不要为了提高召回率而放宽标准。
2. 只能依据输入原文判断，不补充车型知识，不评价技术结论是否正确。
3. supporting_quotes、action_quotes、finding_quotes、verification_quotes 中的每个字符串都必须逐字摘自输入，不得改写。
4. 只输出一个JSON对象，不要输出Markdown或解释性前后缀。
5. confidence只能是high、medium、low。只有证据完整且没有歧义时才能使用high。
"""


D1_PROMPT = """任务：复核一条被判为price_only的汽车问答问题。

标签定义：
- price_only：核心诉求只是价格、报价、工时费或更换费用。仅仅提到部件损坏、事故外观、已经更换某件，或问“换X多少钱”，仍是price_only。
- mixed_repair_price：除价格外，用户还明确要求诊断原因、判断已有维修结论是否正确、提供检查/排查/处理方案，或评估故障是否影响安全。这个维修诉求必须能够脱离价格问题独立成立。

只有明确属于mixed_repair_price时decision才可为recall；其余为keep_rejected；有歧义为review。
supporting_quotes必须摘录能够证明“独立维修诉求”的原文，不能只摘录价格或损坏部件名称。

输出字段：
{"decision":"recall|keep_rejected|review","label":"mixed_repair_price|price_only","confidence":"high|medium|low","supporting_quotes":[],"action_quotes":[],"finding_quotes":[],"verification_quotes":[],"unsafe_operation":false,"reason":"不超过60字"}
"""


EVIDENCE_PROMPT = """任务：判断一段汽车技术文本是否被规则误判，能否召回为维修Evidence。

标签定义：
- diagnostic_case：描述实际故障，并包含检查/测试动作，以及检查发现、范围收窄、原因判断或维修后验证。仅有现象和最终原因不够。
- diagnostic_procedure：提供可执行的分步或条件式诊断流程，写明检查对象以及正常/异常时的判断或下一分支。纯拆装、保养、原理或原因清单不算。
- technical_only：技术原理、参数介绍、保养知识、原因罗列或只有维修结论，没有足够诊断链。
- incomplete：文本截断、关键步骤缺失或依赖未提供的图片/外链。
- other：非汽车维修内容。

decision规则：
- 只有明确的diagnostic_case或diagnostic_procedure才能为recall。
- 涉及绕过安全联锁、气囊/安全带/限速器，危险道路复现，高压或燃油系统非专业改线、短接、并接、拆装时，unsafe_operation=true且不得recall。
- 不能因为出现“诊断、检查、故障原因、解决措施”等标题就直接召回。

引用要求：
- action_quotes：实际检查、检测、测量、试车或诊断动作。
- finding_quotes：动作得到的正常/异常结果、数据变化或排除信息。
- verification_quotes：维修后的恢复、试车、复现或回访结果；没有可以为空。
- diagnostic_case至少需要action_quotes，以及finding_quotes或verification_quotes。
- diagnostic_procedure至少需要action_quotes和finding_quotes。

输出字段：
{"decision":"recall|keep_rejected|review","label":"diagnostic_case|diagnostic_procedure|technical_only|incomplete|other","confidence":"high|medium|low","supporting_quotes":[],"action_quotes":[],"finding_quotes":[],"verification_quotes":[],"unsafe_operation":false,"reason":"不超过60字"}
"""


VALID_DECISIONS = {"recall", "keep_rejected", "review"}
VALID_CONFIDENCE = {"high", "medium", "low"}
VALID_LABELS = {
    "d1": {"mixed_repair_price", "price_only"},
    "d3": {"diagnostic_case", "diagnostic_procedure", "technical_only", "incomplete", "other"},
    "d4": {"diagnostic_case", "diagnostic_procedure", "technical_only", "incomplete", "other"},
}
QUOTE_FIELDS = (
    "supporting_quotes", "action_quotes", "finding_quotes", "verification_quotes",
)


def semantic_record_id(dataset, record):
    if dataset == "d1":
        return record["query_id"]
    if dataset == "d3":
        return record["evidence_source_id"]
    return record["document_id"]


def semantic_record_text(dataset, record):
    if dataset == "d1":
        return normalize_document(
            "instruction: {}\nquery: {}".format(record.get("instruction", ""), record.get("query", ""))
        )
    if dataset == "d4":
        return normalize_document("title: {}\ntext: {}".format(
            record.get("title", ""), record.get("text", "")
        ))
    return normalize_document(record.get("text", ""))


def is_semantic_candidate(dataset, record):
    flags = set(record.get("flags", []))
    if dataset == "d1":
        hard_blockers = {"empty", "duplicate_query", "instruction_query_vehicle_conflict", "off_topic"}
        return record.get("intent") == "price_only" and not flags.intersection(hard_blockers)
    if dataset == "d3":
        return (
            "no_diagnosis_process" in flags
            and record.get("domain_status") == "automotive"
            and not flags.intersection({"empty", "off_topic", "uncertain_domain"})
        )
    if dataset == "d4":
        return (
            bool(flags.intersection({"technical_article", "maintenance_qa"}))
            and not flags.intersection({"empty", "needs_image", "safety_review"})
        )
    return False


def build_prompt(dataset, record):
    task_prompt = D1_PROMPT if dataset == "d1" else EVIDENCE_PROMPT
    payload = {
        "dataset": dataset,
        "record_id": semantic_record_id(dataset, record),
        "text": semantic_record_text(dataset, record),
    }
    return task_prompt + "\n\n输入记录：\n" + json.dumps(payload, ensure_ascii=False)


def quote_supported(quote, source_text):
    quote = normalize_inline(quote)
    source_text = normalize_inline(source_text)
    return len(quote) >= 4 and quote in source_text


def validate_decision(dataset, record, response):
    errors = []
    if not isinstance(response, dict):
        return ["response_not_object"]
    if response.get("decision") not in VALID_DECISIONS:
        errors.append("invalid_decision")
    if response.get("label") not in VALID_LABELS[dataset]:
        errors.append("invalid_label")
    if response.get("confidence") not in VALID_CONFIDENCE:
        errors.append("invalid_confidence")
    if not isinstance(response.get("unsafe_operation", False), bool):
        errors.append("invalid_unsafe_operation")

    source_text = semantic_record_text(dataset, record)
    for field in QUOTE_FIELDS:
        values = response.get(field, [])
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            errors.append("invalid_{}".format(field))
            continue
        if any(not quote_supported(value, source_text) for value in values):
            errors.append("unsupported_{}".format(field))

    if response.get("decision") == "recall":
        if response.get("confidence") != "high":
            errors.append("recall_requires_high_confidence")
        if response.get("unsafe_operation"):
            errors.append("unsafe_recall")
        if dataset == "d1":
            if response.get("label") != "mixed_repair_price":
                errors.append("invalid_d1_recall_label")
            if not response.get("supporting_quotes"):
                errors.append("d1_recall_missing_support")
        else:
            label = response.get("label")
            if label not in {"diagnostic_case", "diagnostic_procedure"}:
                errors.append("invalid_evidence_recall_label")
            if not response.get("action_quotes"):
                errors.append("evidence_recall_missing_action")
            if label == "diagnostic_case" and not (
                response.get("finding_quotes") or response.get("verification_quotes")
            ):
                errors.append("case_recall_missing_result")
            if label == "diagnostic_procedure" and not response.get("finding_quotes"):
                errors.append("procedure_recall_missing_branch")
    return sorted(set(errors))


def decision_is_accepted(decision_record):
    return bool(
        decision_record.get("decision") == "recall"
        and decision_record.get("confidence") == "high"
        and not decision_record.get("validation_errors")
    )


def recalled_record(dataset, record, decision):
    result = dict(record)
    flags = set(result.get("flags", []))
    if dataset == "d1":
        flags.discard("non_repair_intent")
        result["intent"] = "mixed_repair_price"
        flags.add("semantic_recalled_d1")
    else:
        if dataset == "d3":
            flags.discard("no_diagnosis_process")
        else:
            flags.discard("technical_article")
            flags.discard("maintenance_qa")
            result["document_type"] = (
                "case_evidence" if decision.get("label") == "diagnostic_case"
                else "procedure_evidence"
            )
        flags.add("semantic_recalled_evidence")
        result["eligible_evidence"] = True
    result["flags"] = sorted(flags)
    result["semantic_recall"] = {
        "model": decision.get("model"),
        "prompt_version": decision.get("prompt_version"),
        "label": decision.get("label"),
        "reason": decision.get("reason", ""),
    }
    return result


def merge_effective_d1(base_records, recalled_records):
    merged = {record["query_id"]: record for record in base_records}
    for record in recalled_records:
        merged[record["query_id"]] = record
    return sorted(merged.values(), key=lambda record: (
        record.get("source_index", 10 ** 12), record["query_id"]
    ))


class TransformersSemanticClassifier(object):
    """Lazy local Hugging Face classifier used only by the Colab stage."""

    def __init__(self, settings):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "Semantic recall requires the Colab dependencies. "
                "Install data_pipeline/requirements-colab.txt."
            ) from exc

        self.torch = torch
        self.max_input_tokens = int(settings.get("max_input_tokens", 8192))
        self.max_new_tokens = int(settings.get("max_new_tokens", 384))
        model_name = settings["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model_kwargs = {"device_map": "auto", "torch_dtype": "auto"}
        if settings.get("load_in_4bit", True):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "4-bit semantic recall requires a CUDA runtime. Run stage 03 in Colab GPU mode."
                )
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()

    def generate(self, prompts):
        conversations = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        inputs = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[1]
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.batch_decode(
            generated[:, input_length:], skip_special_tokens=True
        )


def decision_fingerprint(dataset, record, model_name, prompt_version):
    return stable_hash("\0".join((
        dataset, semantic_record_id(dataset, record), semantic_record_text(dataset, record),
        model_name, prompt_version,
    )), length=64)


def parse_decision(dataset, record, raw_output, model_name, prompt_version, fingerprint):
    try:
        response = parse_model_json(raw_output)
        errors = validate_decision(dataset, record, response)
    except Exception as exc:
        response = {
            "decision": "review",
            "label": "price_only" if dataset == "d1" else "incomplete",
            "confidence": "low",
            "supporting_quotes": [],
            "action_quotes": [],
            "finding_quotes": [],
            "verification_quotes": [],
            "unsafe_operation": False,
            "reason": "模型输出无法解析",
        }
        errors = ["invalid_json:{}".format(type(exc).__name__)]
    result = {
        "dataset": dataset,
        "record_id": semantic_record_id(dataset, record),
        "fingerprint": fingerprint,
        "model": model_name,
        "prompt_version": prompt_version,
        "decision": response.get("decision"),
        "label": response.get("label"),
        "confidence": response.get("confidence"),
        "supporting_quotes": response.get("supporting_quotes", []),
        "action_quotes": response.get("action_quotes", []),
        "finding_quotes": response.get("finding_quotes", []),
        "verification_quotes": response.get("verification_quotes", []),
        "unsafe_operation": response.get("unsafe_operation", False),
        "reason": normalize_inline(response.get("reason", ""))[:200],
        "validation_errors": errors,
    }
    result["accepted"] = decision_is_accepted(result)
    return result


def load_decision_cache(path):
    return {
        record.get("record_id"): record
        for record in read_jsonl(path)
        if record.get("record_id")
    }


def run_semantic_recall(config, work_dir, per_dataset_limit=0, show_progress=True, classifier=None):
    """Recall only rejected D1/D3/D4 records and write traceable semantic artifacts."""
    settings = config.get("semantic_recall", {})
    enabled = bool(settings.get("enabled", False))
    semantic_dir = ensure_dir(work_dir / "semantic")
    normalized_dir = work_dir / "normalized"
    rejected_dir = work_dir / "rejected" / "normalize"
    model_name = settings.get("model", "Qwen/Qwen3-4B-Instruct-2507")
    prompt_version = settings.get("prompt_version", "v1")
    batch_size = max(1, int(settings.get("batch_size", 4)))
    save_every = max(batch_size, int(settings.get("save_every", 20)))
    seed = int(config.get("seed", 42))
    decisions_by_dataset = {}
    originals_by_dataset = {}
    report = {
        "enabled": enabled,
        "model": model_name,
        "prompt_version": prompt_version,
        "per_dataset_limit": int(per_dataset_limit or 0),
        "datasets": {},
    }
    review_records = []

    for dataset, filename in (
        ("d1", "d1_amck.jsonl"),
        ("d3", "d3_faults.jsonl"),
        ("d4", "d4_documents.jsonl"),
    ):
        rejected_records = list(read_jsonl(rejected_dir / filename))
        candidates = [record for record in rejected_records if is_semantic_candidate(dataset, record)]
        candidates.sort(key=lambda record: stable_hash(
            "{}:{}:{}".format(seed, dataset, semantic_record_id(dataset, record))
        ))
        selected = candidates[:per_dataset_limit] if per_dataset_limit and per_dataset_limit > 0 else candidates
        originals_by_dataset[dataset] = {
            semantic_record_id(dataset, record): record for record in selected
        }
        cache_path = semantic_dir / ("{}_decisions.jsonl".format(dataset))
        # A disabled semantic stage must be a pure rule-only pass. In particular,
        # smoke tests must not silently reuse decisions left by an earlier model run.
        cache = load_decision_cache(cache_path) if enabled else {}
        current = {}
        pending = []
        for record in selected:
            record_id = semantic_record_id(dataset, record)
            fingerprint = decision_fingerprint(
                dataset, record, model_name, prompt_version
            )
            cached = cache.get(record_id)
            if cached and cached.get("fingerprint") == fingerprint:
                current[record_id] = cached
            else:
                pending.append((record, fingerprint))

        if enabled and pending and classifier is None:
            classifier = TransformersSemanticClassifier(settings)
        newly_processed = 0
        if enabled and pending:
            iterator = range(0, len(pending), batch_size)
            iterator = tqdm(
                iterator,
                total=(len(pending) + batch_size - 1) // batch_size,
                desc="03 semantic {}".format(dataset),
                disable=not show_progress,
            )
            since_save = 0
            for start in iterator:
                batch = pending[start:start + batch_size]
                prompts = [build_prompt(dataset, record) for record, _ in batch]
                try:
                    outputs = classifier.generate(prompts)
                except RuntimeError:
                    if len(batch) == 1:
                        raise
                    outputs = []
                    for prompt in prompts:
                        outputs.extend(classifier.generate([prompt]))
                if len(outputs) != len(batch):
                    raise RuntimeError("Semantic classifier returned an unexpected batch size")
                for (record, fingerprint), raw_output in zip(batch, outputs):
                    decision = parse_decision(
                        dataset, record, raw_output, model_name, prompt_version, fingerprint
                    )
                    record_id = semantic_record_id(dataset, record)
                    current[record_id] = decision
                    cache[record_id] = decision
                    newly_processed += 1
                    since_save += 1
                if since_save >= save_every:
                    write_jsonl(cache_path, sorted(cache.values(), key=lambda item: item["record_id"]))
                    since_save = 0
            write_jsonl(cache_path, sorted(cache.values(), key=lambda item: item["record_id"]))

        decisions = [
            current[semantic_record_id(dataset, record)]
            for record in selected
            if semantic_record_id(dataset, record) in current
        ]
        decisions_by_dataset[dataset] = decisions
        accepted = sum(decision_is_accepted(record) for record in decisions)
        invalid = sum(bool(record.get("validation_errors")) for record in decisions)
        report["datasets"][dataset] = {
            "rejected_records": len(rejected_records),
            "recall_candidates": len(candidates),
            "selected": len(selected),
            "cached": len(decisions) - newly_processed,
            "processed": newly_processed,
            "deferred": len(candidates) - len(selected),
            "accepted": accepted,
            "review": sum(record.get("decision") == "review" for record in decisions),
            "invalid": invalid,
            "labels": dict(Counter(record.get("label") for record in decisions)),
        }
        for decision in decisions:
            original = originals_by_dataset[dataset][decision["record_id"]]
            review_records.append(dict(
                decision,
                source_text=semantic_record_text(dataset, original),
            ))

    recalled = {}
    for dataset in ("d1", "d3", "d4"):
        source_map = originals_by_dataset[dataset]
        recalled[dataset] = [
            recalled_record(dataset, source_map[decision["record_id"]], decision)
            for decision in decisions_by_dataset[dataset]
            if decision_is_accepted(decision)
        ]
        write_jsonl(
            semantic_dir / ("recalled_{}.jsonl".format(dataset)), recalled[dataset]
        )

    base_d1 = list(read_jsonl(normalized_dir / "d1_amck.jsonl"))
    effective_d1 = merge_effective_d1(base_d1, recalled["d1"])
    write_jsonl(semantic_dir / "effective_d1_amck.jsonl", effective_d1)
    report["base_d1_records"] = len(base_d1)
    report["effective_d1_records"] = len(effective_d1)
    report["recalled_d1_records"] = len(recalled["d1"])
    report["recalled_d3_records"] = len(recalled["d3"])
    report["recalled_d4_records"] = len(recalled["d4"])

    review_records.sort(key=lambda record: stable_hash(
        "{}:{}:{}".format(seed, record["dataset"], record["record_id"])
    ))
    review_limit = int(settings.get("review_limit", 150))
    if review_limit > 0:
        review_records = review_records[:review_limit]
    write_jsonl(semantic_dir / "review_samples.jsonl", review_records)
    write_json(semantic_dir / "semantic_recall_report.json", report)
    return {
        "effective_d1": effective_d1,
        "recalled_d3": recalled["d3"],
        "recalled_d4": recalled["d4"],
        "report": report,
    }
