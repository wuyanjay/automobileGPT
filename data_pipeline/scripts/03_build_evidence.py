#!/usr/bin/env python3
"""Recall rule-rejected records, then build the split-aware Evidence Corpus."""

from __future__ import print_function

import argparse
from collections import Counter

from _common import (
    chunk_text,
    extract_literals,
    load_config,
    normalize_document,
    pipeline_path,
    read_jsonl,
    stable_id,
    write_json,
    write_jsonl,
)
from _semantic_recall import run_semantic_recall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--semantic-limit",
        type=int,
        default=0,
        help="Pilot limit per rejected dataset; 0 processes every recall candidate",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    work_dir = pipeline_path(config, "work")
    normalized_dir = work_dir / "normalized"
    evidence_dir = work_dir / "evidence"
    min_chars = config["filters"]["min_evidence_chars"]
    max_chars = config["filters"]["max_evidence_chars"]
    semantic_result = run_semantic_recall(
        config,
        work_dir,
        per_dataset_limit=args.semantic_limit,
        show_progress=True,
    )
    semantic_report = semantic_result["report"]
    for dataset in ("d1", "d3", "d4"):
        dataset_report = semantic_report["datasets"][dataset]
        print(
            "semantic {}: candidates={}, selected={}, processed={}, cached={}, recalled={}".format(
                dataset,
                dataset_report["recall_candidates"],
                dataset_report["selected"],
                dataset_report["processed"],
                dataset_report["cached"],
                dataset_report["accepted"],
            )
        )
    by_split = {"train": [], "validation": [], "test": []}
    seen = set()
    stats = Counter()

    def add_record(source, source_id, text, language, powertrain, system, source_ref, split):
        text = normalize_document(text)
        chunks = chunk_text(text, max_chars=max_chars, min_chars=min_chars)
        for chunk_index, chunk in enumerate(chunks):
            key = normalize_document(chunk).lower()
            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)
            evidence_id = stable_id("evidence", source + "\0" + source_id + "\0" + str(chunk_index))
            by_split[split].append({
                "evidence_id": evidence_id,
                "source_record_id": source_id,
                "source": source,
                "text": chunk,
                "language": language,
                "powertrain": powertrain,
                "system": system,
                "literals": extract_literals(chunk),
                "source_ref": source_ref,
                "split": split,
            })
            stats[source] += 1

    for record in read_jsonl(normalized_dir / "d2_mechanics.jsonl"):
        if not record.get("eligible_evidence"):
            continue
        add_record(
            "d2", record["evidence_source_id"],
            "Question: {}\n\nAnswer: {}".format(record["question"], record["answer"]),
            "en", record["powertrain"], record["system"],
            "mechanics_english_sft.jsonl:{}".format(record["source_index"]), record["split"],
        )

    d3_records = list(read_jsonl(normalized_dir / "d3_faults.jsonl"))
    d3_records.extend(semantic_result["recalled_d3"])
    for record in d3_records:
        if not record.get("eligible_evidence"):
            continue
        add_record(
            "d3", record["evidence_source_id"], record["text"], "zh",
            record["powertrain"], record["system"],
            "spo_0.json:{}".format(record["source_index"]), "train",
        )

    d4_records = list(read_jsonl(normalized_dir / "d4_documents.jsonl"))
    d4_records.extend(semantic_result["recalled_d4"])
    for record in d4_records:
        if not record.get("eligible_evidence"):
            continue
        add_record(
            "d4", record["document_id"], record["text"], "zh",
            record["powertrain"], record["system"], record["source_file"], record["split"],
        )

    for split, records in by_split.items():
        write_jsonl(evidence_dir / ("evidence_{}.jsonl".format(split)), records)
    report = {
        "split_counts": {split: len(records) for split, records in by_split.items()},
        "source_counts": dict(stats),
        "semantic_recall": semantic_report,
    }
    write_json(evidence_dir / "evidence_stats.json", report)
    for split, records in by_split.items():
        print("evidence {}: {}".format(split, len(records)))


if __name__ == "__main__":
    main()
