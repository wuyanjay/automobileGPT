#!/usr/bin/env python3
"""Use one LLM call to select a route and rewrite each retrieved AMCK answer."""

from __future__ import print_function

import argparse
import json
import os
import time
from collections import Counter

from _common import (
    api_chat,
    load_by_id,
    load_config,
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


SYSTEM_PROMPT = """你是汽车维修训练数据编辑器。你不是在远程确诊车辆，而是在根据用户输入、AMCK 草稿和候选维修证据生成一条可靠的中文 SFT 答案。

必须只输出一个 JSON 对象，字段为 route、selected_evidence_ids、answer、reason。
route 只能是 matched、analogy、no_evidence、reject。

规则：
1. matched：证据与当前问题的系统、主要现象和工况较一致。可使用证据中的检查动作和诊断顺序，但用户没有确认的最终原因不能写成当前车辆结论。
2. analogy：只是同系统或相似症状。必须写成“同类系统/类似案例可作为排查参考”，只能迁移检查方法，不能迁移其他车型的最终故障、DTC、数值、工具型号。
3. no_evidence：候选均不相关或不足。保留 AMCK 中通用合理的部分，删除无来源具体数值、故障码、零件号、价格、车型配置和唯一结论。信息不足时提出 2-4 个能改变诊断分支的问题，同时给出一两个首轮检查动作。
4. reject：非目标问题、输入损坏或无法安全改写。
5. 回答顺序：当前能/不能确定什么；优先检查步骤；每步观察点和分支；必要的补充信息；安全提示。
6. 不要只说“去 4S 店”，不要盲目建议换件，不要编造证据中没有的具体事实。
7. selected_evidence_ids 只能从候选 ID 中选择；no_evidence/reject 必须为空数组。
"""


def build_user_prompt(query, candidates, evidence_by_id, max_base_chars, max_evidence_chars, unsupported=None):
    candidate_blocks = []
    for candidate in candidates:
        evidence = evidence_by_id.get(candidate["evidence_id"])
        if not evidence:
            continue
        candidate_blocks.append(
            "[{} | source={} | score={}]\n{}".format(
                evidence["evidence_id"], evidence["source"], candidate.get("adjusted_score", candidate.get("score")),
                evidence["text"][:max_evidence_chars],
            )
        )
    prompt = """AMCK instruction:
{instruction}

用户问题:
{query}

AMCK 原答案（只是待修草稿，不是事实来源）:
{base_answer}

候选 Evidence:
{evidence}

请判断路线并输出最终 JSON。reason 只写一句内部说明。""".format(
        instruction=query["instruction"],
        query=query["query"],
        base_answer=query["base_answer"][:max_base_chars],
        evidence="\n\n".join(candidate_blocks) if candidate_blocks else "无候选证据",
    )
    if unsupported:
        prompt += (
            "\n\n上一次答案出现以下无来源具体 literal：{}。"
            "请删除或改成不含具体值的表述，并重新输出完整 JSON。".format("、".join(unsupported))
        )
    return prompt


def mock_response(query):
    return {
        "route": "no_evidence",
        "selected_evidence_ids": [],
        "answer": (
            "根据目前提供的信息，还不能确定唯一故障原因。建议先记录故障出现的工况，"
            "读取并保存车辆当前的故障信息，再从与现象直接相关的线路、连接和基础状态开始检查。"
            "在得到测量结果前不建议继续盲目更换部件。请补充故障是持续还是偶发、冷热车是否有差异、"
            "已经做过哪些测量，以及是否有明确的故障提示。"
        ),
        "reason": "mock smoke response",
    }


def normalize_response(response, candidate_ids):
    route = str(response.get("route", "")).strip()
    if route not in ("matched", "analogy", "no_evidence", "reject"):
        raise ValueError("invalid route: {}".format(route))
    selected = response.get("selected_evidence_ids") or []
    if isinstance(selected, str):
        selected = [selected]
    selected = [value for value in selected if value in candidate_ids]
    if route in ("no_evidence", "reject"):
        selected = []
    if route in ("matched", "analogy") and not selected:
        route = "no_evidence"
    answer = normalize_document(response.get("answer"))
    if route != "reject" and len(answer) < 40:
        raise ValueError("answer is empty or too short")
    return {
        "route": route,
        "selected_evidence_ids": selected,
        "answer": answer,
        "reason": str(response.get("reason", ""))[:500],
    }


def generation_input_hash(user_prompt, provider, model):
    payload = {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
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
    parser.add_argument("--splits", default="train,validation,test")
    parser.add_argument("--limit", type=int, default=0, help="Maximum new records per split")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()
    config = load_config(args.config)
    work_dir = pipeline_path(config, "work")
    generation = config["generation"]
    filters = config["filters"]
    provider = args.provider or generation.get("provider", "api")
    model = args.model or os.environ.get(generation.get("model_env", "LLM_MODEL")) or os.environ.get("OPENAI_MODEL")
    base_url = args.base_url or os.environ.get(generation.get("base_url_env", "LLM_BASE_URL")) or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com"
    api_key_env = args.api_key_env or generation.get("api_key_env", "LLM_API_KEY")
    api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
    if provider == "api" and not model:
        raise RuntimeError("Missing model. Pass --model or set LLM_MODEL/OPENAI_MODEL.")
    selected_splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    query_by_id = load_by_id(work_dir / "normalized" / "d1_amck.jsonl", "query_id")
    generated_dir = work_dir / "generated" / args.run_name
    rejected_dir = work_dir / "rejected" / args.run_name
    save_every = int(generation.get("save_every", 100))
    attempts = int(generation.get("attempts", 3))
    report = {"provider": provider, "model": model, "splits": {}}

    for split in selected_splits:
        evidence_by_id = load_by_id(work_dir / "evidence" / ("evidence_{}.jsonl".format(split)), "evidence_id")
        retrieval_path = work_dir / "retrieval" / args.run_name / ("retrieval_{}.jsonl".format(split))
        retrieval_records = list(read_jsonl(retrieval_path))
        output_path = generated_dir / ("rewrite_{}.jsonl".format(split))
        existing = [] if args.no_resume else list(read_jsonl(output_path))
        existing_by_query = {record["query_id"]: record for record in existing}
        output = list(existing)
        rejected = []
        new_count = 0
        route_counts = Counter(record.get("route") for record in existing)
        for retrieval in retrieval_records:
            query_id = retrieval["query_id"]
            query = query_by_id.get(query_id)
            if not query:
                rejected.append({"query_id": query_id, "error": "query_not_found"})
                continue
            if args.limit and new_count >= args.limit:
                break
            candidates = retrieval.get("candidates", [])
            candidate_ids = {item["evidence_id"] for item in candidates}
            user_prompt = build_user_prompt(
                query, candidates, evidence_by_id,
                filters.get("max_base_answer_chars", 5000),
                filters.get("max_evidence_chars", 6000),
            )
            input_hash = generation_input_hash(user_prompt, provider, model)
            previous = existing_by_query.get(query_id)
            if previous and previous.get("input_hash") == input_hash:
                continue
            if previous:
                output = [record for record in output if record.get("query_id") != query_id]
                route_counts[previous.get("route")] -= 1
                existing_by_query.pop(query_id, None)
            try:
                if provider == "mock":
                    parsed = mock_response(query)
                else:
                    content = retry_call(
                        lambda: api_chat(
                            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
                            model, base_url, api_key,
                            temperature=float(generation.get("temperature", 0.1)),
                        ),
                        attempts=attempts,
                    )
                    parsed = parse_model_json(content)
                normalized = normalize_response(parsed, candidate_ids)
                selected_texts = [
                    evidence_by_id[evidence_id]["text"]
                    for evidence_id in normalized["selected_evidence_ids"]
                    if evidence_id in evidence_by_id
                ]
                allowed_texts = [query["instruction"], query["query"]] + selected_texts
                unsupported = unsupported_literals(normalized["answer"], allowed_texts)
                if unsupported and provider != "mock":
                    retry_prompt = build_user_prompt(
                        query, candidates, evidence_by_id,
                        filters.get("max_base_answer_chars", 5000),
                        filters.get("max_evidence_chars", 6000),
                        unsupported=unsupported,
                    )
                    retry_content = retry_call(
                        lambda: api_chat(
                            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": retry_prompt}],
                            model, base_url, api_key,
                            temperature=0.0,
                        ),
                        attempts=attempts,
                    )
                    normalized = normalize_response(parse_model_json(retry_content), candidate_ids)
                    selected_texts = [
                        evidence_by_id[evidence_id]["text"]
                        for evidence_id in normalized["selected_evidence_ids"]
                        if evidence_id in evidence_by_id
                    ]
                    allowed_texts = [query["instruction"], query["query"]] + selected_texts
                    unsupported = unsupported_literals(normalized["answer"], allowed_texts)
                passed = normalized["route"] != "reject" and not unsupported
                record = {
                    "sample_id": "rewrite_{}".format(query_id),
                    "query_id": query_id,
                    "route": normalized["route"],
                    "selected_evidence_ids": normalized["selected_evidence_ids"],
                    "question": normalize_document(query["instruction"] + "\n" + query["query"]),
                    "answer": normalized["answer"],
                    "reason": normalized["reason"],
                    "unsupported_literals": unsupported,
                    "passed": passed,
                    "split": split,
                    "is_mock": provider == "mock",
                    "model": model if provider == "api" else "mock",
                    "input_hash": input_hash,
                }
                if passed:
                    output.append(record)
                    existing_by_query[query_id] = record
                    route_counts[record["route"]] += 1
                else:
                    rejected.append(record)
                new_count += 1
                if new_count % save_every == 0:
                    write_jsonl(output_path, output)
                    write_jsonl(rejected_dir / ("rewrite_{}.jsonl".format(split)), rejected)
                if args.sleep:
                    time.sleep(args.sleep)
            except Exception as exc:
                rejected.append({
                    "query_id": query_id,
                    "split": split,
                    "error": "generation_error",
                    "detail": str(exc)[:1000],
                })
                if args.fail_fast:
                    raise
        write_jsonl(output_path, output)
        write_jsonl(rejected_dir / ("rewrite_{}.jsonl".format(split)), rejected)
        report["splits"][split] = {
            "output": len(output),
            "new": new_count,
            "rejected": len(rejected),
            "routes": dict(route_counts),
        }
        print("{}: output={}, new={}, rejected={}".format(split, len(output), new_count, len(rejected)))
    write_json(generated_dir / "rewrite_report.json", report)


if __name__ == "__main__":
    main()
