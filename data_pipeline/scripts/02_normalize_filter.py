#!/usr/bin/env python3
"""Normalize/filter D1-D4 and assign stable source splits."""

from __future__ import print_function

import argparse
import json
import re
from collections import Counter

from _common import (
    classify_automotive_domain,
    classify_d4_document,
    classify_powertrain,
    classify_repair_intent,
    classify_system,
    context_flags,
    diagnosis_process_signals,
    extract_literals,
    has_diagnosis_process,
    is_automotive,
    load_config,
    normalize_document,
    normalize_inline,
    off_topic_reasons,
    pipeline_path,
    read_jsonl,
    risk_flags,
    requires_safety_review,
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


VEHICLE_INSTRUCTION_RE = re.compile(
    r"汽车品牌和车型信息\s*[:：]\s*(.+?)(?:(?:,|，)\s*请回答|(?:,|，)|请回答|$)"
)
VEHICLE_OWNER_RE = re.compile(
    r"(?:我(?:的|有(?:一辆|一台|一款)?|开的是)|本人(?:的)?|这辆|这台|车型(?:是|为)|车(?:型)?是).{0,8}$"
)
VEHICLE_COMPARISON_RE = re.compile(r"(?:对比|相比|比较|区别|哪个好|哪款好|还是选|和.{0,12}比)")
LEADING_VEHICLE_PREFIX_RE = re.compile(r"^(?:\d{2,4}年?(?:款)?)?$")


def normalize_vehicle_name(value):
    """Normalize a brand/model name for conservative cross-field matching."""
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", normalize_inline(value).lower())


def parse_instruction_vehicle(instruction):
    """Extract the declared brand/model from the standard AMCK instruction."""
    match = VEHICLE_INSTRUCTION_RE.search(normalize_inline(instruction))
    if not match:
        return None
    parts = match.group(1).strip().split()
    if len(parts) < 2:
        return None
    brand = parts[0]
    model = " ".join(parts[1:])
    return {
        "brand": brand,
        "model": model,
        "brand_key": normalize_vehicle_name(brand),
        "model_key": normalize_vehicle_name(model),
    }


def build_vehicle_catalog(items):
    """Build model aliases from D1 itself instead of maintaining a hand-written car list."""
    alias_map = {}
    full_aliases = set()
    pairs = {}
    for item in items:
        vehicle = parse_instruction_vehicle(item.get("instruction"))
        if not vehicle or not vehicle["model_key"]:
            continue
        pair_key = vehicle["brand_key"] + "\0" + vehicle["model_key"]
        pairs[pair_key] = vehicle
        full_alias = vehicle["brand_key"] + vehicle["model_key"]
        aliases = {full_alias}
        full_aliases.add(full_alias)
        # Two-character Chinese names such as "雅阁" are useful; pure numeric
        # or short alphanumeric names such as "350"/"X1" require the brand.
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", vehicle["model_key"])
        has_ascii = bool(re.search(r"[a-z0-9]", vehicle["model_key"]))
        # Two-character names collide heavily with ordinary repair prose:
        # "里程", "发现", "风尚" and "皮卡" all occur as model names in D1.
        # Keep them only when paired with a brand. Mixed names such as 帝豪EC7
        # remain useful because the alphanumeric suffix makes them distinctive.
        if len(chinese_chars) >= 3 or (len(chinese_chars) >= 2 and has_ascii):
            aliases.add(vehicle["model_key"])
        for alias in aliases:
            if alias:
                alias_map.setdefault(alias, set()).add(pair_key)

    # Bare model aliases shared by multiple brands are review hints, not blockers.
    brand_keys = {item["brand_key"] for item in pairs.values() if item["brand_key"]}
    unambiguous = {
        alias: next(iter(pair_keys))
        for alias, pair_keys in alias_map.items()
        if len(pair_keys) == 1 and (alias in full_aliases or alias not in brand_keys)
    }
    alias_pattern = None
    if unambiguous:
        alias_pattern = re.compile(
            "|".join(re.escape(alias) for alias in sorted(unambiguous, key=len, reverse=True))
        )
    return {
        "pairs": pairs,
        "aliases": unambiguous,
        "full_aliases": full_aliases,
        "pattern": alias_pattern,
    }


def same_vehicle_brand(left, right):
    """Allow common source variants such as '传祺'/'广汽传祺'."""
    if not left or not right:
        return False
    left_base = re.sub(r"(?:汽车|轿车)$", "", left)
    right_base = re.sub(r"(?:汽车|轿车)$", "", right)
    return left_base == right_base or (
        min(len(left_base), len(right_base)) >= 2
        and (left_base in right_base or right_base in left_base)
    )


def same_vehicle_model(left, right):
    """Treat brand-prefixed and trim-suffixed spellings as the same model."""
    if not left or not right:
        return False
    return left == right or (
        min(len(left), len(right)) >= 2 and (left in right or right in left)
    )


def vehicle_context_check(instruction, query, catalog):
    """Return a conservative status for instruction/query vehicle consistency."""
    declared = parse_instruction_vehicle(instruction)
    if not declared or not catalog.get("pattern"):
        return "unknown", declared, []
    if declared["brand_key"] in {"其他", "其他品牌", "未知", "未知品牌"}:
        return "unknown", declared, []

    query_key = normalize_vehicle_name(query)
    mentions = []
    occupied = []
    for match in catalog["pattern"].finditer(query_key):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        pair = catalog["pairs"][catalog["aliases"][match.group(0)]]
        mentions.append({
            "brand": pair["brand"],
            "model": pair["model"],
            "brand_key": pair["brand_key"],
            "model_key": pair["model_key"],
            "start": match.start(),
            "explicit_brand": match.group(0) in catalog.get("full_aliases", set()),
        })
        occupied.append((match.start(), match.end()))

    if not mentions:
        return "no_explicit_vehicle", declared, []
    # Same-brand mentions commonly differ only by translated name, generation,
    # trim or the "进口" suffix. They are not safe hard-conflict signals.
    if any(
        same_vehicle_brand(item["brand_key"], declared["brand_key"])
        or same_vehicle_model(item["model_key"], declared["model_key"])
        for item in mentions
    ):
        return "consistent", declared, mentions

    different = [
        item for item in mentions
        if item["brand_key"] != declared["brand_key"]
        or item["model_key"] != declared["model_key"]
    ]
    if VEHICLE_COMPARISON_RE.search(query_key):
        return "review", declared, mentions
    for item in different:
        prefix = query_key[max(0, item["start"] - 20):item["start"]]
        if (
            VEHICLE_OWNER_RE.search(prefix)
            or (
                item.get("explicit_brand")
                and LEADING_VEHICLE_PREFIX_RE.match(prefix)
            )
        ):
            return "conflict", declared, mentions
    if any(item.get("explicit_brand") for item in different):
        return "review", declared, mentions
    return "no_explicit_vehicle", declared, mentions


def normalize_amck(config, limit=0):
    path = source_path(config, "amck")
    raw = json.loads(path.read_text(encoding="utf-8"))
    vehicle_catalog = build_vehicle_catalog(raw)
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
        domain_status = classify_automotive_domain(combined, source_hint="d1")
        if domain_status != "automotive":
            flags.append("off_topic")
        intent = classify_repair_intent(query)
        if intent in ("purchase_only", "price_only"):
            flags.append("non_repair_intent")
        elif intent in ("mixed_repair_price", "mixed_repair_purchase"):
            flags.append(intent)
        vehicle_status, declared_vehicle, query_vehicle_mentions = vehicle_context_check(
            instruction, query, vehicle_catalog,
        )
        if vehicle_status == "conflict":
            flags.append("instruction_query_vehicle_conflict")
        elif vehicle_status == "review":
            flags.append("vehicle_context_mismatch_review")
        record = {
            "query_id": stable_id("d1", query_key),
            "source_index": index,
            "instruction": instruction,
            "query": query,
            "base_answer": answer,
            "domain_status": domain_status,
            "intent": intent,
            "vehicle_context_status": vehicle_status,
            "instruction_vehicle": (
                {"brand": declared_vehicle["brand"], "model": declared_vehicle["model"]}
                if declared_vehicle else None
            ),
            "query_vehicle_mentions": [
                {"brand": item["brand"], "model": item["model"]}
                for item in query_vehicle_mentions
            ],
            "powertrain": classify_powertrain(combined),
            "system": classify_system(combined),
            "literals": extract_literals(combined),
            "flags": flags,
            "split": stable_split(query_key, train_pct, val_pct),
        }
        blocking_flags = {
            "empty", "duplicate_query", "off_topic", "non_repair_intent",
            "instruction_query_vehicle_conflict",
        }
        if any(flag in blocking_flags for flag in flags):
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
        # Determine the topic from the question. Examples and analogies in an
        # otherwise useful answer must not turn an automotive question into an
        # off-topic record (for example "spider gear" or battery "cycling").
        reasons = off_topic_reasons(question)
        domain_status = classify_automotive_domain(question, source_hint="d2")
        if reasons or domain_status == "non_automotive":
            flags.append("off_topic")
        flags.extend(context_flags(answer, question))
        flags.extend(risk_flags(answer, question))
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
            "domain_status": domain_status,
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
        weak_labels = normalize_document(item.get("output"))
        flags = []
        domain_status = classify_automotive_domain(text, source_hint="d3", auxiliary_text=weak_labels)
        automotive = domain_status == "automotive"
        diagnosis_signals = diagnosis_process_signals(text)
        diagnosis = has_diagnosis_process(text)
        if not text:
            flags.append("empty")
        if domain_status == "non_automotive":
            flags.append("off_topic")
        elif domain_status == "uncertain":
            flags.append("uncertain_domain")
        if automotive and not diagnosis:
            flags.append("no_diagnosis_process")
        record = {
            "evidence_source_id": stable_id("d3", text),
            "source_index": index,
            "text": text,
            "weak_labels": weak_labels,
            "language": "zh",
            "domain_status": domain_status,
            "diagnosis_signals": diagnosis_signals,
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
        document_type = classify_d4_document(title, text)
        diagnosis_signals = diagnosis_process_signals(text)
        safety_review = requires_safety_review(title + " " + text)
        flags = []
        if not text:
            flags.append("empty")
        if item.get("media_count", 0) and len(text) < 300:
            flags.append("needs_image")
        if document_type == "technical_pt":
            flags.append("technical_article")
        elif document_type == "maintenance_qa":
            flags.append("maintenance_qa")
        flags.extend(risk_flags(title + " " + text))
        if safety_review:
            flags.append("safety_review")
        evidence_type = document_type in ("case_evidence", "procedure_evidence")
        record = dict(item)
        record.update({
            "title": title,
            "text": text,
            "language": "zh",
            "document_type": document_type,
            "diagnosis_signals": diagnosis_signals,
            "powertrain": classify_powertrain(title + " " + text),
            "system": classify_system(title + " " + text),
            "literals": extract_literals(text),
            "flags": sorted(set(flags)),
            "eligible_evidence": bool(
                text and evidence_type and "needs_image" not in flags and not safety_review
            ),
            "eligible_pt": bool(text and "needs_image" not in flags and not safety_review),
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
    rejected_dir = work_dir / "rejected" / "normalize"

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
        write_jsonl(rejected_dir / (name + ".jsonl"), records)

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
        "classification_counts": {
            "d1_intent": dict(Counter(record.get("intent") for record in d1 + d1_rejected)),
            "d1_vehicle_context": dict(Counter(
                record.get("vehicle_context_status") for record in d1 + d1_rejected
            )),
            "d2_domain": dict(Counter(record.get("domain_status") for record in d2)),
            "d3_domain": dict(Counter(record.get("domain_status") for record in d3)),
            "d4_document_type": dict(Counter(record.get("document_type") for record in d4)),
        },
    }
    write_json(work_dir / "reports" / "normalize_report.json", report)
    for name, records in datasets.items():
        print("{}: {}".format(name, len(records)))


if __name__ == "__main__":
    main()
