#!/usr/bin/env python3
"""Regression tests for grounded semantic recall."""

from __future__ import print_function

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from _semantic_recall import (  # noqa: E402
    D1_PROMPT,
    EVIDENCE_PROMPT,
    decision_is_accepted,
    is_semantic_candidate,
    merge_effective_d1,
    parse_decision,
    recalled_record,
    run_semantic_recall,
)
from _common import read_jsonl, write_jsonl  # noqa: E402


def parse(dataset, record, payload):
    import json

    return parse_decision(dataset, record, json.dumps(payload, ensure_ascii=False), "test-model", "v1", "fp")


class CandidateTests(unittest.TestCase):
    def test_d1_reviews_only_price_rejects(self):
        record = {
            "query_id": "d1_1",
            "instruction": "请回答汽车问题",
            "query": "空调不制冷是什么原因，换压缩机多少钱",
            "intent": "price_only",
            "flags": ["non_repair_intent"],
        }
        self.assertTrue(is_semantic_candidate("d1", record))
        record["intent"] = "purchase_only"
        self.assertFalse(is_semantic_candidate("d1", record))

    def test_d1_hard_reject_is_not_recalled(self):
        record = {
            "query_id": "d1_2",
            "instruction": "请回答汽车问题",
            "query": "换保险杠多少钱",
            "intent": "price_only",
            "flags": ["duplicate_query"],
        }
        self.assertFalse(is_semantic_candidate("d1", record))

    def test_d3_and_d4_candidate_boundaries(self):
        d3 = {
            "evidence_source_id": "d3_1",
            "text": "检查线路后发现插头松动，固定后试车正常。",
            "domain_status": "automotive",
            "flags": ["no_diagnosis_process"],
        }
        d4 = {
            "document_id": "d4_1",
            "title": "无法充电检修",
            "text": "测量端子电压，低于标准时检查线束。",
            "flags": ["technical_article"],
        }
        self.assertTrue(is_semantic_candidate("d3", d3))
        self.assertTrue(is_semantic_candidate("d4", d4))
        d4["flags"].append("needs_image")
        self.assertFalse(is_semantic_candidate("d4", d4))


class PromptContractTests(unittest.TestCase):
    def test_d1_prompt_covers_missed_mixed_intents(self):
        self.assertIn("外倾角怎么调", D1_PROMPT)
        self.assertIn("二保该换什么", D1_PROMPT)
        self.assertIn("灯罩破了能只换灯罩吗", D1_PROMPT)

    def test_evidence_prompt_does_not_trust_article_style(self):
        self.assertIn("标题、文体和来源不能决定分类", EVIDENCE_PROMPT)
        self.assertIn("技术研究文章", EVIDENCE_PROMPT)

    def test_evidence_prompt_separates_hypothesis_and_observation(self):
        self.assertIn("假设性原理或应急操作", EVIDENCE_PROMPT)
        self.assertIn("不能只是原始故障现象", EVIDENCE_PROMPT)

    def test_evidence_prompt_blocks_dangerous_road_reproduction(self):
        self.assertIn("上路加速至140km/h", EVIDENCE_PROMPT)
        self.assertIn("unsafe_operation=true", EVIDENCE_PROMPT)

    def test_uncertain_evidence_must_be_reviewed(self):
        self.assertIn("不要用keep_rejected加low", EVIDENCE_PROMPT)


class ValidationTests(unittest.TestCase):
    def test_d1_high_confidence_grounded_recall_is_accepted(self):
        record = {
            "query_id": "d1_3",
            "instruction": "请回答汽车问题",
            "query": "空调不制冷是什么原因，换压缩机多少钱",
        }
        decision = parse("d1", record, {
            "decision": "recall",
            "label": "mixed_repair_price",
            "confidence": "high",
            "supporting_quotes": ["空调不制冷是什么原因"],
            "action_quotes": [],
            "finding_quotes": [],
            "verification_quotes": [],
            "unsafe_operation": False,
            "reason": "包含独立诊断诉求",
        })
        self.assertTrue(decision_is_accepted(decision))

    def test_invented_quote_blocks_recall(self):
        record = {
            "query_id": "d1_4",
            "instruction": "请回答汽车问题",
            "query": "换压缩机多少钱",
        }
        decision = parse("d1", record, {
            "decision": "recall",
            "label": "mixed_repair_price",
            "confidence": "high",
            "supporting_quotes": ["请诊断空调为什么不制冷"],
            "action_quotes": [],
            "finding_quotes": [],
            "verification_quotes": [],
            "unsafe_operation": False,
            "reason": "声称存在诊断诉求",
        })
        self.assertIn("unsupported_supporting_quotes", decision["validation_errors"])
        self.assertFalse(decision_is_accepted(decision))

    def test_d3_requires_action_and_result_quotes(self):
        record = {
            "evidence_source_id": "d3_2",
            "text": "先测量蓄电池电压，发现只有10V。更换蓄电池后试车正常。",
        }
        decision = parse("d3", record, {
            "decision": "recall",
            "label": "diagnostic_case",
            "confidence": "high",
            "supporting_quotes": [],
            "action_quotes": ["测量蓄电池电压"],
            "finding_quotes": ["发现只有10V"],
            "verification_quotes": ["更换蓄电池后试车正常"],
            "unsafe_operation": False,
            "reason": "诊断链完整",
        })
        self.assertTrue(decision_is_accepted(decision))

    def test_unsafe_evidence_cannot_be_recalled(self):
        record = {
            "document_id": "d4_2",
            "title": "高压互锁测试",
            "text": "短接高压互锁后上电测试，发现车辆可以启动。",
        }
        decision = parse("d4", record, {
            "decision": "recall",
            "label": "diagnostic_case",
            "confidence": "high",
            "supporting_quotes": [],
            "action_quotes": ["短接高压互锁后上电测试"],
            "finding_quotes": ["发现车辆可以启动"],
            "verification_quotes": [],
            "unsafe_operation": True,
            "reason": "存在危险短接",
        })
        self.assertIn("unsafe_recall", decision["validation_errors"])
        self.assertFalse(decision_is_accepted(decision))


class MergeTests(unittest.TestCase):
    def test_recalled_record_removes_rule_reject_flag(self):
        source = {
            "evidence_source_id": "d3_3",
            "flags": ["no_diagnosis_process"],
            "eligible_evidence": False,
        }
        result = recalled_record("d3", source, {
            "model": "test-model",
            "prompt_version": "v1",
            "label": "diagnostic_case",
            "reason": "诊断链完整",
        })
        self.assertTrue(result["eligible_evidence"])
        self.assertNotIn("no_diagnosis_process", result["flags"])
        self.assertIn("semantic_recalled_evidence", result["flags"])

    def test_effective_d1_merge_is_idempotent(self):
        base = [{"query_id": "d1_a", "source_index": 1}]
        recalled = [
            {"query_id": "d1_b", "source_index": 2},
            {"query_id": "d1_b", "source_index": 2, "intent": "mixed_repair_price"},
        ]
        merged = merge_effective_d1(base, recalled)
        self.assertEqual([record["query_id"] for record in merged], ["d1_a", "d1_b"])

    def test_disabled_stage_does_not_reuse_model_cache(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            work_dir = Path(tmp_name)
            d1 = {
                "query_id": "d1_cached",
                "source_index": 1,
                "instruction": "请回答汽车问题",
                "query": "空调不制冷是什么原因，换压缩机多少钱",
                "intent": "price_only",
                "flags": ["non_repair_intent"],
            }
            write_jsonl(work_dir / "normalized" / "d1_amck.jsonl", [])
            write_jsonl(work_dir / "rejected" / "normalize" / "d1_amck.jsonl", [d1])
            write_jsonl(work_dir / "rejected" / "normalize" / "d3_faults.jsonl", [])
            write_jsonl(work_dir / "rejected" / "normalize" / "d4_documents.jsonl", [])
            write_jsonl(work_dir / "semantic" / "d1_decisions.jsonl", [{
                "record_id": "d1_cached",
                "fingerprint": "stale",
                "decision": "recall",
                "confidence": "high",
                "validation_errors": [],
            }])
            result = run_semantic_recall(
                {"seed": 42, "semantic_recall": {"enabled": False}},
                work_dir,
                show_progress=False,
            )
            self.assertEqual(result["report"]["recalled_d1_records"], 0)
            self.assertEqual(list(read_jsonl(work_dir / "semantic" / "recalled_d1.jsonl")), [])


if __name__ == "__main__":
    unittest.main()
