#!/usr/bin/env python3
"""Validate generated SFT records and export MedicalGPT ShareGPT JSONL."""

from __future__ import print_function

import argparse
from collections import Counter, defaultdict

from _common import (
    load_config,
    normalize_document,
    normalize_inline,
    pipeline_path,
    read_jsonl,
    risk_flags,
    stable_hash,
    write_json,
    write_jsonl,
)


SPLITS = ("train", "validation", "test")
ROUTES = ("matched", "analogy", "no_evidence", "evidence_to_qa", "legacy_train_only")


def load_token_counter(model_name):
    if not model_name:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise RuntimeError("--tokenizer requires transformers; install requirements-colab.txt")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def count(question, answer):
        text = "用户：{}\n助手：{}".format(question, answer)
        return len(tokenizer.encode(text, add_special_tokens=True))

    return count


def generated_candidates(work_dir, run_name, include_mock):
    generated_dir = work_dir / "generated" / run_name
    for split in SPLITS:
        for kind in ("rewrite", "evidence_qa"):
            path = generated_dir / ("{}_{}.jsonl".format(kind, split))
            for record in read_jsonl(path):
                if not record.get("passed"):
                    continue
                if record.get("is_mock") and not include_mock:
                    continue
                yield {
                    "sample_id": record.get("sample_id"),
                    "query_id": record.get("query_id"),
                    "source_record_id": record.get("evidence_id"),
                    "source": "amck" if kind == "rewrite" else record.get("source", "unknown"),
                    "route": record.get("route"),
                    "question": normalize_document(record.get("question")),
                    "answer": normalize_document(record.get("answer")),
                    "unsupported_literals": record.get("unsupported_literals") or [],
                    "selected_evidence_ids": record.get("selected_evidence_ids") or [],
                    "split": split,
                    "is_mock": bool(record.get("is_mock")),
                    "model": record.get("model"),
                }


def legacy_candidates(work_dir, limit, seed):
    records = [
        record for record in read_jsonl(work_dir / "normalized" / "professional_sft.jsonl")
        if record.get("eligible_sft") and record.get("split") == "train"
    ]
    records.sort(key=lambda record: stable_hash("{}:{}".format(seed, record["sample_id"])))
    if limit and limit > 0:
        records = records[:limit]
    for record in records:
        yield {
            "sample_id": record["sample_id"],
            "query_id": None,
            "source_record_id": None,
            "source": "professional_sft",
            "route": "legacy_train_only",
            "question": normalize_document(record.get("question")),
            "answer": normalize_document(record.get("answer")),
            "unsupported_literals": [],
            "selected_evidence_ids": [],
            "split": "train",
            "is_mock": False,
            "model": "source_dataset",
        }


def validate_record(record, max_chars, max_tokens, token_counter):
    errors = []
    if not record.get("sample_id"):
        errors.append("missing_sample_id")
    if record.get("split") not in SPLITS:
        errors.append("invalid_split")
    if not record.get("route"):
        errors.append("missing_route")
    elif record.get("route") not in ROUTES:
        errors.append("invalid_route")
    if record.get("route") in ("matched", "analogy", "no_evidence") and not record.get("query_id"):
        errors.append("missing_query_id")
    if record.get("route") == "evidence_to_qa" and not record.get("source_record_id"):
        errors.append("missing_evidence_id")
    if not record.get("question"):
        errors.append("empty_question")
    if not record.get("answer"):
        errors.append("empty_answer")
    if record.get("unsupported_literals"):
        errors.append("unsupported_literal")
    char_count = len(record.get("question", "")) + len(record.get("answer", ""))
    if max_chars and char_count > max_chars:
        errors.append("too_many_chars")
    token_count = None
    if token_counter and not errors:
        token_count = token_counter(record["question"], record["answer"])
        if max_tokens and token_count > max_tokens:
            errors.append("too_many_tokens")
    return errors, char_count, token_count


def normalized_question_key(record):
    return normalize_inline(record.get("question")).lower()


def deduplicate_and_cap(by_split, validation_limit, test_limit, seed):
    # First deduplicate inside each split deterministically.
    inside_counts = Counter()
    for split in SPLITS:
        chosen = {}
        for record in sorted(
            by_split[split],
            key=lambda value: stable_hash("{}:{}".format(seed, value["sample_id"])),
        ):
            key = normalized_question_key(record)
            if key in chosen:
                inside_counts[split] += 1
                continue
            chosen[key] = record
        by_split[split] = list(chosen.values())

    if validation_limit and validation_limit > 0:
        by_split["validation"] = by_split["validation"][:validation_limit]
    if test_limit and test_limit > 0:
        by_split["test"] = by_split["test"][:test_limit]

    # Held-out splits win over train when the same normalized question appears
    # in multiple splits. This prevents train/test leakage.
    kept = {split: [] for split in SPLITS}
    reserved = set()
    cross_counts = Counter()
    for split in ("test", "validation", "train"):
        for record in by_split[split]:
            key = normalized_question_key(record)
            if key in reserved:
                cross_counts[split] += 1
                continue
            reserved.add(key)
            kept[split].append(record)
    return kept, dict(inside_counts), dict(cross_counts)


def load_evidence_risks(work_dir):
    result = {}
    for split in SPLITS:
        for record in read_jsonl(work_dir / "evidence" / ("evidence_{}.jsonl".format(split))):
            result[record["evidence_id"]] = risk_flags(record.get("text", ""))
    return result


def review_rows(records, per_group, evidence_risks, seed):
    groups = defaultdict(list)
    high_risk = []
    all_test = []
    for record in records:
        groups["route:" + record["route"]].append(record)
        if record["route"] == "evidence_to_qa":
            groups["source:" + record["source"]].append(record)
        flags = risk_flags(record["question"] + "\n" + record["answer"])
        evidence_ids = list(record.get("selected_evidence_ids") or [])
        if record.get("source_record_id"):
            evidence_ids.append(record["source_record_id"])
        for evidence_id in evidence_ids:
            flags.extend(evidence_risks.get(evidence_id, []))
        flags = sorted(set(flags))
        if flags:
            item = dict(record)
            item["review_reason"] = "high_risk:" + ",".join(flags)
            high_risk.append(item)
        if record["split"] == "test":
            item = dict(record)
            item["review_reason"] = "test"
            all_test.append(item)

    selected = {}
    for group_name, values in groups.items():
        values.sort(key=lambda value: stable_hash("{}:{}".format(seed, value["sample_id"])))
        for record in values[:per_group]:
            item = dict(record)
            item["review_reason"] = group_name
            selected.setdefault(record["sample_id"], item)
    for record in all_test + high_risk:
        selected[record["sample_id"]] = record
    return sorted(selected.values(), key=lambda value: (value["split"], value["sample_id"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--include-mock", action="store_true", help="Smoke test only; never use mock data for training")
    parser.add_argument("--no-legacy", action="store_true")
    parser.add_argument("--legacy-limit", type=int, default=None, help="0 means all eligible legacy records")
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--tokenizer", default=None, help="Optional Hugging Face tokenizer name/path")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    work_dir = pipeline_path(config, "work")
    final_dir = pipeline_path(config, "final")
    export_cfg = config["export"]
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    max_chars = int(export_cfg.get("max_conversation_chars", 8000))
    max_tokens = args.max_tokens if args.max_tokens is not None else int(export_cfg.get("max_tokens", 4096))
    validation_limit = args.validation_limit if args.validation_limit is not None else export_cfg.get("validation_limit", 200)
    test_limit = args.test_limit if args.test_limit is not None else export_cfg.get("test_limit", 200)
    legacy_limit = args.legacy_limit if args.legacy_limit is not None else export_cfg.get("legacy_sft_limit", 100)
    token_counter = load_token_counter(args.tokenizer)

    candidates = list(generated_candidates(work_dir, args.run_name, args.include_mock))
    include_legacy = bool(export_cfg.get("include_legacy_sft", True)) and not args.no_legacy
    if include_legacy:
        candidates.extend(legacy_candidates(work_dir, legacy_limit, seed))

    valid = {split: [] for split in SPLITS}
    rejected = []
    for record in candidates:
        errors, char_count, token_count = validate_record(record, max_chars, max_tokens, token_counter)
        record["char_count"] = char_count
        if token_count is not None:
            record["token_count"] = token_count
        if errors:
            rejected.append({"record": record, "errors": errors})
        else:
            valid[record["split"]].append(record)

    kept, inside_dupes, cross_dupes = deduplicate_and_cap(valid, validation_limit, test_limit, seed)
    export_dir = work_dir / "export" / args.run_name
    exported_records = []
    for split in SPLITS:
        records = kept[split]
        exported_records.extend(records)
        conversations = [
            {
                "conversations": [
                    {"from": "human", "value": record["question"]},
                    {"from": "gpt", "value": record["answer"]},
                ]
            }
            for record in records
        ]
        write_jsonl(final_dir / "sft" / split / "data.jsonl", conversations)
        # Metadata is intentionally outside final/sft: MedicalGPT recursively
        # reads JSONL files from its dataset directory.
        write_jsonl(export_dir / ("manifest_{}.jsonl".format(split)), records)

    write_jsonl(work_dir / "rejected" / args.run_name / "export_rejected.jsonl", rejected)
    reviews = review_rows(
        exported_records,
        int(export_cfg.get("review_samples_per_route", 30)),
        load_evidence_risks(work_dir),
        seed,
    )
    write_jsonl(work_dir / "review" / args.run_name / "review_samples.jsonl", reviews)

    split_counts = {split: len(kept[split]) for split in SPLITS}
    route_counts = Counter(record["route"] for record in exported_records)
    source_counts = Counter(record["source"] for record in exported_records)
    report = {
        "run_name": args.run_name,
        "include_mock": args.include_mock,
        "include_legacy": include_legacy,
        "legacy_limit": legacy_limit,
        "tokenizer": args.tokenizer,
        "seed": seed,
        "input_candidates": len(candidates),
        "rejected": len(rejected),
        "duplicates_inside_split": inside_dupes,
        "duplicates_across_splits": cross_dupes,
        "split_counts": split_counts,
        "route_counts": dict(route_counts),
        "source_counts": dict(source_counts),
        "review_rows": len(reviews),
    }
    write_json(final_dir / "sft_report.json", report)
    print("SFT exported: train={train}, validation={validation}, test={test}".format(**split_counts))
    print("Rejected: {}; review rows: {}".format(len(rejected), len(reviews)))


if __name__ == "__main__":
    main()
