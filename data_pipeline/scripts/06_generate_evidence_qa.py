#!/usr/bin/env python3
"""Generate Chinese Evidence-to-QA SFT samples from Evidence not used by AMCK rewrites."""

from __future__ import print_function

import argparse
import json
import os
import re
import time
from collections import defaultdict

from _common import (
    api_chat,
    load_config,
    load_env_file,
    normalize_document,
    parse_model_json,
    pipeline_path,
    read_jsonl,
    retry_call,
    stable_hash,
    unsupported_literals,
    write_json,
    write_jsonl,
)


SYSTEM_PROMPT = """你是汽车维修训练数据编辑器。请把单条真实维修 Evidence 转成一条中文用户问题和中文助手回答。

只输出 JSON 对象，字段为 usable、question、answer、reason。
规则：
1. 只能使用 Evidence 中已有的车型、现象、检查、原因和处理结果；
2. 问题应像真实车主/维修人员提问，不能泄露 Evidence 中尚未给用户的最终答案；
3. 回答按检查顺序组织，说明检查动作、观察点和下一步；
4. 如果 Evidence 只支持类似案例结论，写“该来源案例最终发现”，不能伪装成对所有车辆的通用结论；
5. 不新增数值、DTC、年款、工具型号、零件号；
6. 英文 Evidence 需要准确转成中文，并保留原有不确定程度；
7. 内容非汽车、严重缺上下文或存在明显危险建议时 usable=false。
"""


def balanced_records(records, limit, seed=42):
    groups = defaultdict(list)
    seen_d4_documents = set()
    for record in records:
        # D4 may be split into multiple chunks, but V4 allows at most one QA
        # from the same source document in a run.
        if record["source"] == "d4":
            document_id = record.get("source_record_id") or record.get("source_ref")
            if document_id in seen_d4_documents:
                continue
            seen_d4_documents.add(document_id)
        groups[record["source"]].append(record)
    for values in groups.values():
        values.sort(key=lambda record: stable_hash("{}:{}".format(seed, record["evidence_id"])))
    selected = []
    sources = sorted(groups)
    while sources and (not limit or len(selected) < limit):
        next_sources = []
        for source in sources:
            if groups[source] and (not limit or len(selected) < limit):
                selected.append(groups[source].pop(0))
            if groups[source]:
                next_sources.append(source)
        sources = next_sources
    return selected


def build_prompt(evidence, max_chars, unsupported=None):
    prompt = """Evidence ID: {evidence_id}
Source: {source}
Language: {language}

Evidence:
{text}

请输出 JSON。""".format(
        evidence_id=evidence["evidence_id"],
        source=evidence["source"],
        language=evidence["language"],
        text=evidence["text"][:max_chars],
    )
    if unsupported:
        prompt += "\n上一次输出新增了无来源 literal：{}。请删除后重新输出。".format("、".join(unsupported))
    return prompt


def normalize_response(value):
    usable = bool(value.get("usable", True))
    question = normalize_document(value.get("question"))
    answer = normalize_document(value.get("answer"))
    reason = str(value.get("reason", ""))[:500]
    if usable and (len(question) < 10 or len(answer) < 40):
        raise ValueError("generated question/answer is too short")
    if usable and len(re.findall(r"[\u4e00-\u9fff]", question + answer)) < 10:
        raise ValueError("generated QA is not Chinese")
    return usable, question, answer, reason


def mock_response(evidence):
    return {
        "usable": True,
        "question": "这条维修记录反映了什么问题，应该怎样按顺序排查？",
        "answer": (
            "应先核对故障现象和出现条件，再按照记录中的检查顺序确认相关部件、线路和测量结果。"
            "每完成一步都要记录观察结果，并根据结果决定下一项检查；在原因得到验证前，不应直接更换部件。"
        ),
        "reason": "mock smoke response",
    }


def generation_input_hash(evidence, max_chars, provider, model):
    payload = {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": build_prompt(evidence, max_chars),
        "provider": provider,
        "model": model if provider == "api" else "mock",
    }
    return stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True), length=32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--provider", choices=("api", "mock"), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--env-file", default=None, help="Defaults to data_pipeline/.env")
    parser.add_argument("--splits", default="train,validation,test")
    parser.add_argument("--limit", type=int, default=500, help="Maximum new samples per split")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    load_env_file(args.env_file)
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    work_dir = pipeline_path(config, "work")
    generation = config["generation"]
    provider = args.provider or generation.get("provider", "api")
    model = args.model or os.environ.get(generation.get("model_env", "LLM_MODEL")) or os.environ.get("OPENAI_MODEL")
    base_url = args.base_url or os.environ.get(generation.get("base_url_env", "LLM_BASE_URL")) or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com"
    api_key_env = args.api_key_env or generation.get("api_key_env", "LLM_API_KEY")
    api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
    if provider == "api" and not model:
        raise RuntimeError("Missing model. Pass --model or set LLM_MODEL/OPENAI_MODEL.")
    attempts = int(generation.get("attempts", 3))
    save_every = int(generation.get("save_every", 100))
    progress_every = int(generation.get("progress_every", 10))
    max_chars = int(config["filters"].get("max_evidence_chars", 6000))
    generated_dir = work_dir / "generated" / args.run_name
    rejected_dir = work_dir / "rejected" / args.run_name
    report = {"provider": provider, "model": model, "splits": {}}

    for split in [value.strip() for value in args.splits.split(",") if value.strip()]:
        evidence_records = list(read_jsonl(work_dir / "evidence" / ("evidence_{}.jsonl".format(split))))
        rewrite_records = list(read_jsonl(generated_dir / ("rewrite_{}.jsonl".format(split))))
        used_ids = {
            evidence_id
            for record in rewrite_records if record.get("passed")
            for evidence_id in record.get("selected_evidence_ids", [])
        }
        output_path = generated_dir / ("evidence_qa_{}.jsonl".format(split))
        raw_existing = [] if args.no_resume else list(read_jsonl(output_path))
        current_ids = {record["evidence_id"] for record in evidence_records}
        existing = [
            record for record in raw_existing
            if record.get("evidence_id") in current_ids and record.get("evidence_id") not in used_ids
        ]
        existing_by_evidence = {record["evidence_id"]: record for record in existing}
        candidates = [
            record for record in evidence_records
            if record["evidence_id"] not in used_ids and (
                record["evidence_id"] not in existing_by_evidence
                or existing_by_evidence[record["evidence_id"]].get("input_hash")
                != generation_input_hash(record, max_chars, provider, model)
            )
        ]
        candidates = balanced_records(candidates, args.limit, seed=seed)
        output = list(existing)
        rejected = []
        print(
            "{}: starting {} Evidence records; progress/checkpoint every {}".format(
                split, len(candidates), save_every
            ),
            flush=True,
        )
        for index, evidence in enumerate(candidates, 1):
            prompt = build_prompt(evidence, max_chars)
            input_hash = generation_input_hash(evidence, max_chars, provider, model)
            if evidence["evidence_id"] in existing_by_evidence:
                output = [
                    record for record in output
                    if record.get("evidence_id") != evidence["evidence_id"]
                ]
                existing_by_evidence.pop(evidence["evidence_id"], None)
            try:
                if provider == "mock":
                    parsed = mock_response(evidence)
                else:
                    content = retry_call(
                        lambda: api_chat(
                            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                            model, base_url, api_key,
                            temperature=float(generation.get("temperature", 0.1)),
                        ),
                        attempts=attempts,
                    )
                    parsed = parse_model_json(content)
                usable, question, answer, reason = normalize_response(parsed)
                unsupported = unsupported_literals(question + "\n" + answer, [evidence["text"]])
                if usable and unsupported and provider != "mock":
                    retry_prompt = build_prompt(evidence, max_chars, unsupported=unsupported)
                    retry_content = retry_call(
                        lambda: api_chat(
                            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": retry_prompt}],
                            model, base_url, api_key, temperature=0.0,
                        ),
                        attempts=attempts,
                    )
                    usable, question, answer, reason = normalize_response(parse_model_json(retry_content))
                    unsupported = unsupported_literals(question + "\n" + answer, [evidence["text"]])
                record = {
                    "sample_id": "evidence_qa_{}".format(evidence["evidence_id"]),
                    "evidence_id": evidence["evidence_id"],
                    "source": evidence["source"],
                    "route": "evidence_to_qa",
                    "question": question,
                    "answer": answer,
                    "reason": reason,
                    "unsupported_literals": unsupported,
                    "passed": bool(usable and not unsupported),
                    "split": split,
                    "is_mock": provider == "mock",
                    "model": model if provider == "api" else "mock",
                    "input_hash": input_hash,
                }
                if record["passed"]:
                    output.append(record)
                    existing_by_evidence[evidence["evidence_id"]] = record
                else:
                    rejected.append(record)
            except Exception as exc:
                rejected.append({
                    "evidence_id": evidence["evidence_id"],
                    "split": split,
                    "error": "generation_error",
                    "detail": str(exc)[:1000],
                })
                if args.fail_fast:
                    raise
            finally:
                if index % save_every == 0:
                    write_jsonl(output_path, output)
                    write_jsonl(rejected_dir / ("evidence_qa_{}.jsonl".format(split)), rejected)
                if index % progress_every == 0:
                    print(
                        "{}: processed={}/{}, output={}, rejected={}".format(
                            split, index, len(candidates), len(output), len(rejected)
                        ),
                        flush=True,
                    )
                if args.sleep:
                    time.sleep(args.sleep)
        write_jsonl(output_path, output)
        write_jsonl(rejected_dir / ("evidence_qa_{}.jsonl".format(split)), rejected)
        report["splits"][split] = {
            "existing": len(existing),
            "candidates": len(candidates),
            "output": len(output),
            "rejected": len(rejected),
            "used_by_rewrite": len(used_ids),
        }
        print("{}: candidates={}, output={}, rejected={}".format(
            split, len(candidates), len(output), len(rejected)
        ))
    report["seed"] = seed
    write_json(generated_dir / "evidence_qa_report.json", report)


if __name__ == "__main__":
    main()
