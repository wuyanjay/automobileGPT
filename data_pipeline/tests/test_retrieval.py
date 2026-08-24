#!/usr/bin/env python3
"""Regression tests for the lightweight dense retrieval stage."""

from __future__ import print_function

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location("embed_retrieve", SCRIPT_DIR / "04_embed_retrieve.py")
RETRIEVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RETRIEVAL)


def query(**updates):
    record = {
        "query_id": "q1",
        "split": "train",
        "powertrain": "unknown",
        "system": "other",
        "literals": [],
    }
    record.update(updates)
    return record


def evidence(number, **updates):
    record = {
        "evidence_id": "e{}".format(number),
        "source_record_id": "source{}".format(number),
        "source": "d3",
        "text": "evidence text {}".format(number),
        "powertrain": "unknown",
        "system": "other",
        "literals": [],
    }
    record.update(updates)
    return record


class RetrievalTests(unittest.TestCase):
    def run_retrieve(
        self,
        query_record,
        evidence_records,
        query_vector,
        evidence_vectors,
        **updates
    ):
        options = {
            "candidate_pool_k": 50,
            "final_top_k": 3,
            "min_score": None,
            "system_match_bonus": 0.02,
            "system_mismatch_penalty": 0.03,
            "dtc_boost": 0.05,
            "dedup_similarity": 0.98,
        }
        options.update(updates)
        return RETRIEVAL.retrieve(
            [query_record],
            evidence_records,
            np.asarray([query_vector], dtype=np.float32),
            np.asarray(evidence_vectors, dtype=np.float32),
            **options
        )

    def test_powertrain_conflict_is_soft_penalty_not_hard_mask(self):
        results, stats = self.run_retrieve(
            query(powertrain="bev"),
            [
                evidence(1, powertrain="ice"),
                evidence(2, powertrain="bev"),
            ],
            [1.0, 0.0],
            [
                [1.0, 0.0],
                [0.8, 0.6],
            ],
            candidate_pool_k=2,
            final_top_k=2,
            dedup_similarity=None,
        )

        self.assertEqual(results[0]["candidates"][0]["evidence_id"], "e1")
        self.assertEqual(results[0]["candidates"][0]["powertrain_adjustment"], -0.01)
        self.assertEqual(stats["powertrain_mismatch_candidates"], 1)

    def test_system_mismatch_is_penalized_instead_of_deleted(self):
        results, stats = self.run_retrieve(
            query(system="engine"),
            [
                evidence(1, system="transmission"),
                evidence(2, system="engine"),
            ],
            [1.0, 0.0],
            [
                [0.92, 0.391918],
                [0.85, 0.526783],
            ],
            final_top_k=2,
            dedup_similarity=None,
        )

        self.assertEqual([item["evidence_id"] for item in results[0]["candidates"]], ["e1", "e2"])
        self.assertEqual(results[0]["candidates"][0]["system_adjustment"], -0.03)
        self.assertEqual(results[0]["candidates"][1]["system_adjustment"], 0.02)
        self.assertEqual(stats["system_mismatch_candidates"], 1)
        self.assertEqual(stats["system_exact_candidates"], 1)

    def test_min_score_uses_raw_score_before_metadata_adjustment(self):
        results, stats = self.run_retrieve(
            query(system="engine"),
            [evidence(1, system="engine")],
            [1.0, 0.0],
            [[0.54, 0.841665]],
            min_score=0.55,
        )

        self.assertEqual(results[0]["candidates"], [])
        self.assertEqual(results[0]["no_candidate_reason"], "below_min_score")
        self.assertEqual(stats["below_min_score_candidates"], 1)

    def test_matching_dtc_can_reorder_candidates(self):
        results, stats = self.run_retrieve(
            query(literals=["P0300"]),
            [
                evidence(1, literals=[]),
                evidence(2, literals=["P0300"]),
            ],
            [1.0, 0.0],
            [
                [0.62, 0.784602],
                [0.60, 0.8],
            ],
            final_top_k=2,
        )

        self.assertEqual(results[0]["candidates"][0]["evidence_id"], "e2")
        self.assertEqual(results[0]["candidates"][0]["dtc_adjustment"], 0.05)
        self.assertEqual(stats["dtc_boosted_candidates"], 1)

    def test_chunks_from_same_source_record_are_deduplicated(self):
        results, stats = self.run_retrieve(
            query(),
            [
                evidence(1, source_record_id="shared", text="first chunk"),
                evidence(2, source_record_id="shared", text="second chunk"),
                evidence(3, source_record_id="another", text="another case"),
            ],
            [1.0, 0.0, 0.0],
            [
                [0.90, 0.43589, 0.0],
                [0.89, 0.455961, 0.0],
                [0.88, 0.0, 0.474974],
            ],
            final_top_k=2,
            dedup_similarity=None,
        )

        self.assertEqual([item["evidence_id"] for item in results[0]["candidates"]], ["e1", "e3"])
        self.assertEqual(stats["deduplicated_source_record"], 1)

    def test_near_duplicate_embeddings_do_not_fill_top_three(self):
        results, stats = self.run_retrieve(
            query(),
            [
                evidence(1, text="case version one"),
                evidence(2, text="case version two"),
                evidence(3, text="different diagnostic case"),
            ],
            [1.0, 0.0, 0.0],
            [
                [1.0, 0.0, 0.0],
                [0.99995, 0.01, 0.0],
                [0.0, 1.0, 0.0],
            ],
            final_top_k=2,
            dedup_similarity=0.98,
        )

        self.assertEqual([item["evidence_id"] for item in results[0]["candidates"]], ["e1", "e3"])
        self.assertEqual(stats["deduplicated_near_text"], 1)

    def test_symptom_and_context_views_are_fused(self):
        results, _ = self.run_retrieve(
            query(),
            [evidence(1), evidence(2)],
            [1.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            context_query_matrix=np.asarray([[0.0, 1.0]], dtype=np.float32),
            symptom_weight=0.75,
            final_top_k=2,
            dedup_similarity=None,
        )

        candidates = results[0]["candidates"]
        self.assertEqual([item["evidence_id"] for item in candidates], ["e1", "e2"])
        self.assertEqual(candidates[0]["symptom_score"], 1.0)
        self.assertEqual(candidates[0]["context_score"], 0.0)

    def test_close_alternate_system_is_kept_for_diversity(self):
        results, stats = self.run_retrieve(
            query(system="engine"),
            [
                evidence(1, system="engine"),
                evidence(2, system="engine"),
                evidence(3, system="engine"),
                evidence(4, system="transmission"),
            ],
            [1.0, 0.0, 0.0, 0.0],
            [
                [0.90, 0.0, 0.0, 0.0],
                [0.89, 0.0, 0.0, 0.0],
                [0.88, 0.0, 0.0, 0.0],
                [0.85, 0.0, 0.0, 0.0],
            ],
            final_top_k=3,
            system_match_bonus=0.0,
            system_mismatch_penalty=0.0,
            diversity_score_margin=0.08,
            dedup_similarity=None,
        )

        self.assertEqual(
            [item["evidence_id"] for item in results[0]["candidates"]],
            ["e1", "e2", "e4"],
        )
        self.assertEqual(stats["system_diversity_candidates_added"], 1)


if __name__ == "__main__":
    unittest.main()
