#!/usr/bin/env python3
"""Embed AMCK queries/Evidence and produce filtered Top-K retrieval results."""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import re
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


def compatible(query, evidence):
    query_powertrain = query.get("powertrain", "unknown")
    evidence_powertrain = evidence.get("powertrain", "unknown")
    if (
        query_powertrain != "unknown" and evidence_powertrain != "unknown"
        and query_powertrain != evidence_powertrain
    ):
        return False
    query_system = query.get("system", "other")
    evidence_system = evidence.get("system", "other")
    if query_system != "other" and evidence_system != "other" and query_system != evidence_system:
        return False
    return True


def retrieve(query_records, evidence_records, query_matrix, evidence_matrix, raw_top_k, final_top_k, min_score):
    results = []
    if not evidence_records:
        return [
            {"query_id": query["query_id"], "split": query["split"], "candidates": []}
            for query in query_records
        ]
    for start in range(0, len(query_records), 128):
        block = query_matrix[start:start + 128]
        score_block = np.matmul(block, evidence_matrix.T)
        for offset, scores in enumerate(score_block):
            query = query_records[start + offset]
            k = min(raw_top_k, len(evidence_records))
            if k == len(evidence_records):
                raw_indices = np.argsort(-scores)[:k]
            else:
                raw_indices = np.argpartition(-scores, k - 1)[:k]
                raw_indices = raw_indices[np.argsort(-scores[raw_indices])]
            query_dtcs = dtc_set(query)
            candidates = []
            for index in raw_indices:
                evidence = evidence_records[int(index)]
                if not compatible(query, evidence):
                    continue
                raw_score = float(scores[int(index)])
                boost = 0.05 if query_dtcs and query_dtcs.intersection(dtc_set(evidence)) else 0.0
                adjusted = raw_score + boost
                if min_score is not None and adjusted < min_score:
                    continue
                candidates.append({
                    "evidence_id": evidence["evidence_id"],
                    "score": round(raw_score, 6),
                    "adjusted_score": round(adjusted, 6),
                    "source": evidence["source"],
                    "powertrain": evidence.get("powertrain", "unknown"),
                    "system": evidence.get("system", "other"),
                })
            candidates.sort(key=lambda item: item["adjusted_score"], reverse=True)
            results.append({
                "query_id": query["query_id"],
                "split": query["split"],
                "candidates": candidates[:final_top_k],
            })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--backend", choices=("sentence-transformers", "hash"), default="sentence-transformers")
    parser.add_argument("--model", default=None)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=None)
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
    raw_top_k = embedding_cfg.get("raw_top_k", 10)
    final_top_k = embedding_cfg.get("final_top_k", 3)
    batch_size = embedding_cfg.get("batch_size", 16)
    selected_splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    embedding_dir = ensure_dir(work_dir / "embeddings" / args.run_name)
    retrieval_dir = ensure_dir(work_dir / "retrieval" / args.run_name)

    all_queries = list(read_jsonl(work_dir / "normalized" / "d1_amck.jsonl"))
    run_report = {"backend": args.backend, "model": model_name, "splits": {}}
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
        results = retrieve(
            queries, evidence, query_matrix, evidence_matrix,
            raw_top_k, final_top_k, min_score,
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
