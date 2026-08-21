#!/usr/bin/env python3
"""Build split-aware MedicalGPT PT JSONL from normalized D2/D3/D4 text."""

from __future__ import print_function

import argparse
from collections import Counter

from _common import (
    chunk_text,
    load_config,
    normalize_document,
    pipeline_path,
    read_jsonl,
    risk_flags,
    stable_hash,
    write_json,
    write_jsonl,
)


def add_chunks(target, source, source_id, split, text, min_chars, max_chars):
    for index, chunk in enumerate(chunk_text(text, max_chars=max_chars, min_chars=min_chars)):
        target[split].append({
            "pt_id": "pt_{}_{}_{}".format(source, stable_hash(source_id), index),
            "source": source,
            "source_record_id": source_id,
            "split": split,
            "text": normalize_document(chunk),
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--exclude-d2", action="store_true")
    parser.add_argument("--exclude-d3", action="store_true")
    parser.add_argument("--exclude-d4", action="store_true")
    parser.add_argument("--limit-per-source", type=int, default=0, help="Smoke/pilot limit before chunking; 0 means all")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    work_dir = pipeline_path(config, "work")
    final_dir = pipeline_path(config, "final")
    normalized_dir = work_dir / "normalized"
    min_chars = int(config["filters"].get("min_evidence_chars", 80))
    max_chars = int(config["filters"].get("max_evidence_chars", 6000))
    by_split = {"train": [], "validation": []}

    def selected(records):
        values = list(records)
        values.sort(key=lambda record: stable_hash("{}:{}".format(
            seed,
            record.get("evidence_source_id") or record.get("document_id") or record.get("source_index"),
        )))
        if args.limit_per_source and args.limit_per_source > 0:
            values = values[:args.limit_per_source]
        return values

    if not args.exclude_d2:
        records = selected(
            record for record in read_jsonl(normalized_dir / "d2_mechanics.jsonl")
            if record.get("eligible_pt") and record.get("split") in by_split
        )
        for record in records:
            text = "Question: {}\n\nAnswer: {}".format(record["question"], record["answer"])
            add_chunks(by_split, "d2", record["evidence_source_id"], record["split"], text, min_chars, max_chars)

    if not args.exclude_d3:
        records = selected(
            record for record in read_jsonl(normalized_dir / "d3_faults.jsonl")
            if record.get("eligible_pt")
        )
        for record in records:
            # V4 keeps D3 train-only because no reliable parent linkage exists
            # for the derived professional SFT dataset.
            add_chunks(by_split, "d3", record["evidence_source_id"], "train", record["text"], min_chars, max_chars)

    if not args.exclude_d4:
        records = selected(
            record for record in read_jsonl(normalized_dir / "d4_documents.jsonl")
            if record.get("eligible_pt") and record.get("split") in by_split
        )
        for record in records:
            title = normalize_document(record.get("title", ""))
            body = normalize_document(record["text"])
            text = body if title and body.startswith(title) else "{}\n\n{}".format(title, body)
            add_chunks(by_split, "d4", record["document_id"], record["split"], text, min_chars, max_chars)

    # Deduplicate with validation priority. Test source records are deliberately
    # excluded from PT so that the SFT test set remains held out.
    seen = set()
    duplicate_counts = Counter()
    kept = {"train": [], "validation": []}
    for split in ("validation", "train"):
        for record in sorted(
            by_split[split],
            key=lambda value: stable_hash("{}:{}".format(seed, value["pt_id"])),
        ):
            key = normalize_document(record["text"]).lower()
            if key in seen:
                duplicate_counts[split] += 1
                continue
            seen.add(key)
            kept[split].append(record)

    manifest_dir = work_dir / "export" / "pt"
    for split in ("train", "validation"):
        write_jsonl(
            final_dir / "pt" / split / "data.jsonl",
            [{"text": record["text"]} for record in kept[split]],
        )
        write_jsonl(manifest_dir / ("manifest_{}.jsonl".format(split)), kept[split])

    source_counts = Counter(record["source"] for split in kept.values() for record in split)
    high_risk = []
    for records in kept.values():
        for record in records:
            flags = risk_flags(record["text"])
            if flags:
                item = dict(record)
                item["risk_flags"] = flags
                high_risk.append(item)
    write_jsonl(work_dir / "review" / "pt_high_risk.jsonl", high_risk)
    report = {
        "split_counts": {split: len(records) for split, records in kept.items()},
        "source_counts": dict(source_counts),
        "duplicate_counts": dict(duplicate_counts),
        "test_records_excluded": True,
        "generated_sft_included": False,
        "high_risk_review_rows": len(high_risk),
        "seed": seed,
    }
    write_json(final_dir / "pt_report.json", report)
    print("PT exported: train={}, validation={}".format(len(kept["train"]), len(kept["validation"])))
    print("Source counts: {}".format(dict(source_counts)))


if __name__ == "__main__":
    main()
