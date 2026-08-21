#!/usr/bin/env python3
"""Normalize/filter D1-D4 and assign stable source splits."""

from __future__ import print_function

import argparse
import json
from collections import Counter

from _common import (
    classify_powertrain,
    classify_system,
    context_flags,
    extract_literals,
    has_diagnosis_process,
    is_automotive,
    is_fault_case,
    load_config,
    normalize_document,
    normalize_inline,
    off_topic_reasons,
    pipeline_path,
    read_jsonl,
    risk_flags,
    source_path,
    stable_id,
    stable_split,
    summarize_flags,
    write_json,
    write_jsonl,
)


def limited(records, limit):
    if not limit or limit <= 0:
        return records
    return records[:limit]


def normalize_amck(config, limit=0):
    path = source_path(config, "amck")
    raw = json.loads(path.read_text(encoding="utf-8"))
    output = []
    rejected = []
    seen_queries = set()
    train_pct = config["splits"]["d1_train"]
    val_pct = config["splits"]["d1_validation"]
    for index, item in enumerate(raw):
        instruction = normalize_inline(item.get("instruction"))
        query = normalize_inline(item.get("input"))
        answer = normalize_document(item.get("output"))
        query_key = normalize_inline(instruction + " " + query).lower()
        flags = []
        if not query or not answer:
            flags.append("empty")
        if query_key in seen_queries:
            flags.append("duplicate_query")
        combined = instruction + " " + query
        if not is_automotive(combined, source_hint="d1"):
            flags.append("off_topic")
        if any(word in query for word in ("值得购买", "值得拥有", "能买吗", "报价", "多少钱")):
            flags.append("non_repair_intent")
        record = {
            "query_id": stable_id("d1", query_key),
            "source_index": index,
            "instruction": instruction,
            "query": query,
            "base_answer": answer,
            "powertrain": classify_powertrain(combined),
            "system": classify_system(combined),
            "literals": extract_literals(combined),
            "flags": flags,
            "split": stable_split(query_key, train_pct, val_pct),
        }
        if flags:
            rejected.append(record)
            continue
        seen_queries.add(query_key)
        output.append(record)
        if limit and len(output) >= limit:
            break
    return output, rejected, {"raw": len(raw), "kept": len(output), "rejected": len(rejected)}


def normalize_d2(config, limit=0):
    path = source_path(config, "d2")
    output = []
    rejected = []
    train_pct = config["splits"]["d2_train"]
    val_pct = config["splits"]["d2_validation"]
    include_unsafe = config["filters"].get("include_unsafe_evidence", False)
    include_context = config["filters"].get("include_context_dependent_d2", False)
    for index, item in enumerate(read_jsonl(path)):
        question = normalize_inline(item.get("input"))
        answer = normalize_document(item.get("output"))
        combined = question + "\n" + answer
        flags = []
        if not question or not answer:
            flags.append("empty")
        reasons = off_topic_reasons(combined)
        if reasons or not is_automotive(combined, source_hint="d2"):
            flags.append("off_topic")
        flags.extend(context_flags(answer))
        flags.extend(risk_flags(combined))
        context_problem = any(flag.startswith("needs_") for flag in flags)
        # Broad risk flags (brakes, lifting, fuel, airbags, high voltage) are
        # review labels, not automatic evidence rejection. Only an explicit
        # safety-bypass signal is rejected by rule; the remaining high-risk
        # samples are retained and forced into the export review list.
        unsafe_problem = "risk_bypass" in flags
        eligible = (
            "empty" not in flags and "off_topic" not in flags
            and (include_context or not context_problem)
            and (include_unsafe or not unsafe_problem)
        )
        question_key = normalize_inline(question).lower()
        record = {
            "evidence_source_id": stable_id("d2", question + "\0" + answer),
            "source_index": index,
            "question": question,
            "answer": answer,
            "language": "en",
            "metadata": item.get("metadata", {}),
            "powertrain": classify_powertrain(combined),
            "system": classify_system(combined),
            "literals": extract_literals(combined),
            "flags": sorted(set(flags)),
            "eligible_evidence": eligible,
            "eligible_pt": eligible,
            "split": stable_split(question_key, train_pct, val_pct),
        }
        output.append(record)
        if not eligible:
            rejected.append(record)
        if limit and len(output) >= limit:
            break
    return output, rejected


def normalize_d3(config, limit=0):
    path = source_path(config, "d3")
    output = []
    rejected = []
    for index, item in enumerate(read_jsonl(path)):
        text = normalize_document(item.get("input"))
        flags = []
        automotive = is_automotive(text, source_hint="d3")
        diagnosis = has_diagnosis_process(text)
        if not text:
            flags.append("empty")
        if not automotive:
            flags.append("off_topic")
        if automotive and not diagnosis:
            flags.append("no_diagnosis_process")
        record = {
            "evidence_source_id": stable_id("d3", text),
            "source_index": index,
            "text": text,
            "weak_labels": normalize_document(item.get("output")),
            "language": "zh",
            "powertrain": classify_powertrain(text),
            "system": classify_system(text),
            "literals": extract_literals(text),
            "flags": flags,
            "eligible_evidence": bool(text and automotive and diagnosis),
            "eligible_pt": bool(text and automotive),
            "split": "train",
        }
        output.append(record)
        if not record["eligible_evidence"]:
            rejected.append(record)
        if limit and len(output) >= limit:
            break
    return output, rejected


def normalize_professional(config, limit=0):
    path = source_path(config, "professional_sft")
    output = []
    rejected = []
    for index, item in enumerate(read_jsonl(path)):
        conversations = item.get("conversations") or []
        question = ""
        answer = ""
        if len(conversations) >= 2:
            question = normalize_inline(conversations[0].get("value"))
            answer = normalize_document(conversations[1].get("value"))
        combined = question + "\n" + answer
        flags = []
        if not question or not answer:
            flags.append("empty")
        if not is_automotive(combined, source_hint="professional"):
            flags.append("off_topic")
        record = {
            "sample_id": stable_id("legacy", question + "\0" + answer),
            "source_index": index,
            "question": question,
            "answer": answer,
            "flags": flags,
            "eligible_sft": not flags,
            "split": "train",
        }
        output.append(record)
        if flags:
            rejected.append(record)
        if limit and len(output) >= limit:
            break
    return output, rejected


def normalize_d4(config, limit=0):
    work_dir = pipeline_path(config, "work")
    path = work_dir / "normalized" / "d4_extracted.jsonl"
    output = []
    rejected = []
    train_pct = config["splits"]["d4_train"]
    val_pct = config["splits"]["d4_validation"]
    for item in read_jsonl(path):
        title = normalize_inline(item.get("title"))
        text = normalize_document(item.get("text"))
        fault_case = is_fault_case(title, text)
        flags = []
        if not text:
            flags.append("empty")
        if item.get("media_count", 0) and len(text) < 300:
            flags.append("needs_image")
        if not fault_case:
            flags.append("technical_article")
        record = dict(item)
        record.update({
            "title": title,
            "text": text,
            "language": "zh",
            "powertrain": classify_powertrain(title + " " + text),
            "system": classify_system(title + " " + text),
            "literals": extract_literals(text),
            "flags": flags,
            "eligible_evidence": bool(text and fault_case and "needs_image" not in flags),
            "eligible_pt": bool(text and "needs_image" not in flags),
            "split": stable_split(item["document_id"], train_pct, val_pct),
        })
        output.append(record)
        if not record["eligible_evidence"]:
            rejected.append(record)
        if limit and len(output) >= limit:
            break
    return output, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--amck-limit", type=int, default=0)
    parser.add_argument("--d2-limit", type=int, default=0)
    parser.add_argument("--d3-limit", type=int, default=0)
    parser.add_argument("--professional-limit", type=int, default=0)
    parser.add_argument("--d4-limit", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    work_dir = pipeline_path(config, "work")
    normalized_dir = work_dir / "normalized"
    rejected_dir = work_dir / "rejected"

    d1, d1_rejected, d1_stats = normalize_amck(config, args.amck_limit)
    d2, d2_rejected = normalize_d2(config, args.d2_limit)
    d3, d3_rejected = normalize_d3(config, args.d3_limit)
    professional, professional_rejected = normalize_professional(config, args.professional_limit)
    d4, d4_rejected = normalize_d4(config, args.d4_limit)

    datasets = {
        "d1_amck": d1,
        "d2_mechanics": d2,
        "d3_faults": d3,
        "professional_sft": professional,
        "d4_documents": d4,
    }
    rejected_sets = {
        "d1_amck": d1_rejected,
        "d2_mechanics": d2_rejected,
        "d3_faults": d3_rejected,
        "professional_sft": professional_rejected,
        "d4_documents": d4_rejected,
    }
    for name, records in datasets.items():
        write_jsonl(normalized_dir / (name + ".jsonl"), records)
    for name, records in rejected_sets.items():
        write_jsonl(rejected_dir / ("normalize_" + name + ".jsonl"), records)

    report = {
        "d1": d1_stats,
        "datasets": {
            name: {
                "records": len(records),
                "flags": summarize_flags(records),
                "split_counts": dict(Counter(record.get("split") for record in records)),
            }
            for name, records in datasets.items()
        },
        "rejected_counts": {name: len(records) for name, records in rejected_sets.items()},
    }
    write_json(work_dir / "reports" / "normalize_report.json", report)
    for name, records in datasets.items():
        print("{}: {}".format(name, len(records)))


if __name__ == "__main__":
    main()
