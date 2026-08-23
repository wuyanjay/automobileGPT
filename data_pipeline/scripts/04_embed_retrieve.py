#!/usr/bin/env python3
"""Embed AMCK queries/Evidence and produce filtered Top-K retrieval results."""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from _common import (
    ensure_dir,
    load_config,
    normalize_inline,
    normalize_literal,
    pipeline_path,
    read_jsonl,
    stable_hash,
    write_json,
    write_jsonl,
)


def hash_embeddings(texts, dim=512):
    """Dependency-free smoke backend; not suitable for final cross-language retrieval."""
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        normalized = normalize_inline(text).lower()
        tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", normalized)
        tokens.extend(normalized[i:i + 3] for i in range(max(0, len(normalized) - 2)))
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            matrix[row, index] += sign
        norm = float(np.linalg.norm(matrix[row]))
        if norm:
            matrix[row] /= norm
    return matrix


def model_embeddings(texts, model_name, batch_size):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError(
            "sentence-transformers is required for final embeddings. "
            "Install requirements-colab.txt or use --backend hash for a smoke test."
        )
    model = SentenceTransformer(model_name)
    values = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return values.astype(np.float32)


def embedding_fingerprint(ids, texts, backend, model_name):
    digest = hashlib.sha256()
    digest.update((backend + "\0" + model_name).encode("utf-8"))
    for record_id, value in zip(ids, texts):
        digest.update(b"\0")
        digest.update(str(record_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value).encode("utf-8"))
    return digest.hexdigest()


def load_or_embed(path, ids_path, meta_path, ids, texts, backend, model_name, batch_size):
    fingerprint = embedding_fingerprint(ids, texts, backend, model_name)
    if path.exists() and ids_path.exists() and meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            cached_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            if metadata.get("fingerprint") == fingerprint and cached_ids == ids:
                matrix = np.load(str(path))
                if matrix.shape[0] == len(ids):
                    return matrix, True
        except Exception:
            pass
    if not texts:
        dimensions = 512 if backend == "hash" else 0
        matrix = np.zeros((0, dimensions), dtype=np.float32)
    elif backend == "hash":
        matrix = hash_embeddings(texts)
    else:
        matrix = model_embeddings(texts, model_name, batch_size)
    np.save(str(path), matrix)
    write_json(ids_path, ids)
    write_json(meta_path, {
        "fingerprint": fingerprint,
        "backend": backend,
        "model": model_name,
        "records": len(ids),
        "dimensions": int(matrix.shape[1]) if len(matrix.shape) > 1 else 0,
    })
    return matrix, False


def dtc_set(record):
    result = set()
    for value in record.get("literals", []):
        normalized = normalize_literal(value)
        if re.match(r"^[PCBU][0O0-9A-F][0-9A-F]{3,5}$", normalized):
            result.add(normalized)
    return result


def powertrain_compatible(query, evidence):
    query_powertrain = query.get("powertrain", "unknown")
    evidence_powertrain = evidence.get("powertrain", "unknown")
    return not (
        query_powertrain != "unknown" and evidence_powertrain != "unknown"
        and query_powertrain != evidence_powertrain
    )


def system_adjustment(query, evidence, match_bonus, mismatch_penalty):
    query_system = query.get("system", "other")
    evidence_system = evidence.get("system", "other")
    if query_system != "other" and query_system == evidence_system:
        return float(match_bonus), "exact"
    if query_system == "other" or evidence_system == "other":
        return 0.0, "other"
    return -float(mismatch_penalty), "mismatch"


def confidence_label(raw_score):
    if raw_score is None:
        return "none"
    if raw_score >= 0.65:
        return "high"
    if raw_score >= 0.60:
        return "medium"
    return "low"


def score_summary(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
        }

    def percentile(fraction):
        position = (len(values) - 1) * fraction
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return values[lower]
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    return {
        "min": round(values[0], 6),
        "p25": round(percentile(0.25), 6),
        "median": round(percentile(0.50), 6),
        "p75": round(percentile(0.75), 6),
        "max": round(values[-1], 6),
        "mean": round(sum(values) / float(len(values)), 6),
    }


def normalized_evidence_key(record):
    return normalize_inline(record.get("text", "")).lower()


def near_duplicate(candidate_index, selected_indices, evidence_matrix, threshold):
    if threshold is None or threshold <= 0:
        return False
    vector = evidence_matrix[candidate_index]
    for selected_index in selected_indices:
        if float(np.dot(vector, evidence_matrix[selected_index])) >= threshold:
            return True
    return False


def retrieve(
    query_records,
    evidence_records,
    query_matrix,
    evidence_matrix,
    candidate_pool_k,
    final_top_k,
    min_score,
    system_match_bonus=0.02,
    system_mismatch_penalty=0.03,
    dtc_boost=0.05,
    dedup_similarity=0.98,
):
    """Retrieve diverse candidates and return results plus aggregate diagnostics."""
    results = []
    stats = Counter()
    stats["queries"] = len(query_records)
    if not evidence_records:
        stats["queries_without_candidates"] = len(query_records)
        stats["no_candidate_no_evidence"] = len(query_records)
        return [
            {
                "query_id": query["query_id"],
                "split": query["split"],
                "retrieval_status": "no_candidate",
                "no_candidate_reason": "no_evidence",
                "confidence": "none",
                "candidates": [],
            }
            for query in query_records
        ], dict(stats)

    compatible_by_powertrain = {}
    for powertrain in {record.get("powertrain", "unknown") for record in query_records}:
        compatible_by_powertrain[powertrain] = np.asarray([
            index
            for index, evidence in enumerate(evidence_records)
            if powertrain_compatible({"powertrain": powertrain}, evidence)
        ], dtype=np.int64)
    evidence_dtcs = [dtc_set(record) for record in evidence_records]
    evidence_keys = [normalized_evidence_key(record) for record in evidence_records]

    for start in range(0, len(query_records), 128):
        block = query_matrix[start:start + 128]
        score_block = np.matmul(block, evidence_matrix.T)
        for offset, scores in enumerate(score_block):
            query = query_records[start + offset]
            query_powertrain = query.get("powertrain", "unknown")
            compatible_indices = compatible_by_powertrain[query_powertrain]
            stats["powertrain_conflicts_masked"] += len(evidence_records) - len(compatible_indices)
            if not len(compatible_indices):
                results.append({
                    "query_id": query["query_id"],
                    "split": query["split"],
                    "retrieval_status": "no_candidate",
                    "no_candidate_reason": "powertrain_conflict",
                    "confidence": "none",
                    "candidates": [],
                })
                stats["queries_without_candidates"] += 1
                stats["no_candidate_powertrain_conflict"] += 1
                continue

            compatible_scores = scores[compatible_indices]
            k = min(candidate_pool_k, len(compatible_indices))
            if k == len(compatible_indices):
                pool_positions = np.argsort(-compatible_scores)[:k]
            else:
                pool_positions = np.argpartition(-compatible_scores, k - 1)[:k]
                pool_positions = pool_positions[np.argsort(-compatible_scores[pool_positions])]
            raw_indices = compatible_indices[pool_positions]
            stats["candidate_pool_records"] += len(raw_indices)
            query_dtcs = dtc_set(query)
            if query_dtcs:
                stats["queries_with_dtc"] += 1
            candidates = []
            for index in raw_indices:
                evidence = evidence_records[int(index)]
                raw_score = float(scores[int(index)])
                if min_score is not None and raw_score < min_score:
                    stats["below_min_score_candidates"] += 1
                    continue
                system_delta, system_relation = system_adjustment(
                    query, evidence, system_match_bonus, system_mismatch_penalty,
                )
                stats["system_{}_candidates".format(system_relation)] += 1
                dtc_delta = (
                    float(dtc_boost)
                    if query_dtcs and query_dtcs.intersection(evidence_dtcs[int(index)])
                    else 0.0
                )
                if dtc_delta:
                    stats["dtc_boosted_candidates"] += 1
                adjusted = raw_score + system_delta + dtc_delta
                candidates.append({
                    "evidence_id": evidence["evidence_id"],
                    "score": round(raw_score, 6),
                    "raw_score": round(raw_score, 6),
                    "system_adjustment": round(system_delta, 6),
                    "dtc_adjustment": round(dtc_delta, 6),
                    "adjusted_score": round(adjusted, 6),
                    "source": evidence["source"],
                    "powertrain": evidence.get("powertrain", "unknown"),
                    "system": evidence.get("system", "other"),
                    "_index": int(index),
                })
            candidates.sort(key=lambda item: item["adjusted_score"], reverse=True)

            selected = []
            selected_indices = []
            seen_source_records = set()
            seen_texts = set()
            for candidate in candidates:
                index = candidate["_index"]
                evidence = evidence_records[index]
                source_record_id = evidence.get("source_record_id")
                text_key = evidence_keys[index]
                if source_record_id and source_record_id in seen_source_records:
                    stats["deduplicated_source_record"] += 1
                    continue
                if text_key and text_key in seen_texts:
                    stats["deduplicated_exact_text"] += 1
                    continue
                if near_duplicate(index, selected_indices, evidence_matrix, dedup_similarity):
                    stats["deduplicated_near_text"] += 1
                    continue
                candidate.pop("_index", None)
                selected.append(candidate)
                selected_indices.append(index)
                if source_record_id:
                    seen_source_records.add(source_record_id)
                if text_key:
                    seen_texts.add(text_key)
                if len(selected) >= final_top_k:
                    break

            if selected:
                status = "retrieved"
                reason = None
                confidence = confidence_label(selected[0]["raw_score"])
                stats["queries_with_candidates"] += 1
                stats["final_candidates"] += len(selected)
            else:
                status = "no_candidate"
                reason = "below_min_score" if candidates == [] and min_score is not None else "after_dedup"
                confidence = "none"
                stats["queries_without_candidates"] += 1
                stats["no_candidate_{}".format(reason)] += 1
            results.append({
                "query_id": query["query_id"],
                "split": query["split"],
                "retrieval_status": status,
                "no_candidate_reason": reason,
                "confidence": confidence,
                "candidates": selected,
            })
    return results, dict(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", choices=("sentence-transformers", "hash"), default="sentence-transformers")
    parser.add_argument("--model", default=None)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--candidate-pool-k", type=int, default=None)
    parser.add_argument(
        "--embedding-cache-name",
        default=None,
        help="Reuse embeddings from another run name while writing separate retrieval outputs",
    )
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--splits", default="train,validation,test")
    parser.add_argument("--review-limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    work_dir = pipeline_path(config, "work")
    embedding_cfg = config["embedding"]
    model_name = args.model or embedding_cfg["model"]
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    query_limit = args.query_limit
    if query_limit is None:
        query_limit = embedding_cfg.get("pilot_query_limit", 0)
    min_score = args.min_score if args.min_score is not None else embedding_cfg.get("min_score")
    candidate_pool_k = (
        args.candidate_pool_k
        if args.candidate_pool_k is not None
        else embedding_cfg.get("candidate_pool_k", embedding_cfg.get("raw_top_k", 50))
    )
    final_top_k = embedding_cfg.get("final_top_k", 3)
    batch_size = embedding_cfg.get("batch_size", 16)
    system_match_bonus = float(embedding_cfg.get("system_match_bonus", 0.02))
    system_mismatch_penalty = float(embedding_cfg.get("system_mismatch_penalty", 0.03))
    dtc_boost = float(embedding_cfg.get("dtc_boost", 0.05))
    dedup_similarity = embedding_cfg.get("dedup_similarity", 0.98)
    if dedup_similarity is not None:
        dedup_similarity = float(dedup_similarity)
    if candidate_pool_k <= 0:
        raise ValueError("candidate_pool_k must be positive")
    if final_top_k <= 0:
        raise ValueError("final_top_k must be positive")
    if dedup_similarity is not None and not 0.0 <= dedup_similarity <= 1.0:
        raise ValueError("dedup_similarity must be between 0 and 1")
    selected_splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    embedding_cache_name = args.embedding_cache_name or args.run_name
    embedding_dir = ensure_dir(work_dir / "embeddings" / embedding_cache_name)
    retrieval_dir = ensure_dir(work_dir / "retrieval" / args.run_name)

    all_queries = list(read_jsonl(work_dir / "normalized" / "d1_amck.jsonl"))
    run_report = {
        "backend": args.backend,
        "model": model_name,
        "embedding_cache_name": embedding_cache_name,
        "settings": {
            "candidate_pool_k": candidate_pool_k,
            "final_top_k": final_top_k,
            "min_score": min_score,
            "system_match_bonus": system_match_bonus,
            "system_mismatch_penalty": system_mismatch_penalty,
            "dtc_boost": dtc_boost,
            "dedup_similarity": dedup_similarity,
        },
        "splits": {},
    }
    review_records = []
    for split in selected_splits:
        queries = [record for record in all_queries if record.get("split") == split]
        queries.sort(key=lambda record: stable_hash("{}:{}".format(seed, record["query_id"])))
        if query_limit and query_limit > 0:
            queries = queries[:query_limit]
        evidence = list(read_jsonl(work_dir / "evidence" / ("evidence_{}.jsonl".format(split))))
        query_texts = [record["instruction"] + "\n" + record["query"] for record in queries]
        evidence_texts = [record["text"] for record in evidence]
        query_ids = [record["query_id"] for record in queries]
        evidence_ids = [record["evidence_id"] for record in evidence]
        query_matrix, query_cached = load_or_embed(
            embedding_dir / ("queries_{}.npy".format(split)),
            embedding_dir / ("query_ids_{}.json".format(split)),
            embedding_dir / ("query_meta_{}.json".format(split)),
            query_ids, query_texts, args.backend, model_name, batch_size,
        )
        evidence_matrix, evidence_cached = load_or_embed(
            embedding_dir / ("evidence_{}.npy".format(split)),
            embedding_dir / ("evidence_ids_{}.json".format(split)),
            embedding_dir / ("evidence_meta_{}.json".format(split)),
            evidence_ids, evidence_texts, args.backend, model_name, batch_size,
        )
        results, retrieval_stats = retrieve(
            queries, evidence, query_matrix, evidence_matrix,
            candidate_pool_k, final_top_k, min_score,
            system_match_bonus=system_match_bonus,
            system_mismatch_penalty=system_mismatch_penalty,
            dtc_boost=dtc_boost,
            dedup_similarity=dedup_similarity,
        )
        write_jsonl(retrieval_dir / ("retrieval_{}.jsonl".format(split)), results)
        query_by_id = {record["query_id"]: record for record in queries}
        evidence_by_id = {record["evidence_id"]: record for record in evidence}
        for result in results:
            query = query_by_id[result["query_id"]]
            review_records.append({
                "query_id": query["query_id"],
                "split": split,
                "query": query["instruction"] + "\n" + query["query"],
                "powertrain": query.get("powertrain"),
                "system": query.get("system"),
                "candidates": [dict(candidate, text=evidence_by_id[candidate["evidence_id"]]["text"])
                               for candidate in result["candidates"]],
            })
        with_candidates = sum(bool(record["candidates"]) for record in results)
        all_candidates = [candidate for record in results for candidate in record["candidates"]]
        top_candidates = [record["candidates"][0] for record in results if record["candidates"]]
        candidate_counts = Counter(len(record["candidates"]) for record in results)
        confidence_counts = Counter(record["confidence"] for record in results)
        no_candidate_reasons = Counter(
            record["no_candidate_reason"]
            for record in results
            if record.get("no_candidate_reason")
        )
        candidate_sources = Counter(candidate["source"] for candidate in all_candidates)
        top1_sources = Counter(candidate["source"] for candidate in top_candidates)
        run_report["splits"][split] = {
            "queries": len(queries),
            "evidence": len(evidence),
            "queries_with_candidates": with_candidates,
            "mean_candidate_count": (
                sum(len(record["candidates"]) for record in results) / float(len(results))
                if results else 0.0
            ),
            "query_embeddings_cached": query_cached,
            "evidence_embeddings_cached": evidence_cached,
            "candidate_count_distribution": {
                str(count): candidate_counts[count] for count in sorted(candidate_counts)
            },
            "confidence_counts": dict(sorted(confidence_counts.items())),
            "no_candidate_reasons": dict(sorted(no_candidate_reasons.items())),
            "top1_raw_score": score_summary(candidate["raw_score"] for candidate in top_candidates),
            "all_candidate_raw_score": score_summary(
                candidate["raw_score"] for candidate in all_candidates
            ),
            "candidate_sources": dict(sorted(candidate_sources.items())),
            "top1_sources": dict(sorted(top1_sources.items())),
            "retrieval_diagnostics": retrieval_stats,
        }
        print("{}: queries={}, evidence={}, with_candidates={}".format(
            split, len(queries), len(evidence), with_candidates
        ))
    review_records.sort(key=lambda record: stable_hash("{}:{}".format(seed, record["query_id"])))
    if args.review_limit and args.review_limit > 0:
        review_records = review_records[:args.review_limit]
    write_jsonl(work_dir / "review" / args.run_name / "retrieval_samples.jsonl", review_records)
    run_report["review_records"] = len(review_records)
    run_report["seed"] = seed
    write_json(retrieval_dir / "retrieval_report.json", run_report)


if __name__ == "__main__":
    main()
