#!/usr/bin/env python3
"""Regression tests for route and answer policy in AMCK rewriting."""

from __future__ import print_function

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location("rewrite_amck", SCRIPT_DIR / "05_rewrite_amck.py")
REWRITE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REWRITE)


def response(route="analogy", answer="目前不能确定，可先检查相关连接是否松动。", reason="内部判断", **extra):
    value = {
        "route": route,
        "selected_evidence_ids": ["e1"] if route in ("matched", "analogy") else [],
        "answer": answer,
        "reason": reason,
    }
    value.update(extra)
    return value


class RewritePolicyTests(unittest.TestCase):
    def test_pending_count_excludes_matching_cache_and_honors_limit(self):
        queries = {
            "q1": {"instruction": "汽车问题", "query": "异响", "base_answer": "待改写答案"},
            "q2": {"instruction": "汽车问题", "query": "抖动", "base_answer": "待改写答案"},
        }
        retrievals = [
            {"query_id": "q1", "candidates": []},
            {"query_id": "q2", "candidates": []},
        ]
        q1_prompt = REWRITE.build_user_prompt(queries["q1"], [], {}, 5000, 6000)
        q1_hash = REWRITE.generation_input_hash(q1_prompt, "api", "test-model")
        existing = {"q1": {"query_id": "q1", "input_hash": q1_hash}}

        count = REWRITE.count_pending_rewrites(
            retrievals, queries, {}, existing, "api", "test-model", 5000, 6000, 0,
        )
        self.assertEqual(1, count)

        count_without_cache = REWRITE.count_pending_rewrites(
            retrievals, queries, {}, {}, "api", "test-model", 5000, 6000, 1,
        )
        self.assertEqual(1, count_without_cache)

    def test_retry_prompt_explains_policy_failure_in_chinese(self):
        query = {"instruction": "汽车问题", "query": "应该加什么油", "base_answer": "待改写答案"}
        prompt = REWRITE.build_user_prompt(
            query, [], {}, 5000, 6000,
            policy_violations=["ambiguous_fluid_not_clarified"],
        )
        self.assertIn("请先区分汽油/柴油与发动机机油", prompt)

    def test_pipeline_meta_language_is_rejected(self):
        value = response(answer="同类系统/类似案例可作为排查参考，但不能直接套用其他车型结论。")
        self.assertIn(
            "answer_contains_pipeline_meta",
            REWRITE.response_policy_violations(value),
        )

    def test_user_answer_cannot_mention_internal_evidence_shortage(self):
        value = response(answer="现有证据不足，建议补充车辆年款后再判断。")
        self.assertIn(
            "answer_contains_pipeline_meta",
            REWRITE.response_policy_violations(value),
        )

    def test_no_evidence_user_procedure_is_rejected(self):
        value = response(
            route="no_evidence",
            answer="先断开蓄电池负极，再拆下高压油管并完成泄压。",
        )
        self.assertIn(
            "no_evidence_contains_procedure",
            REWRITE.response_policy_violations(value),
        )

    def test_no_evidence_professional_referral_is_allowed(self):
        value = response(
            route="no_evidence",
            answer="目前不能确定原因。建议由专业维修人员使用诊断仪读取故障信息。",
        )
        self.assertNotIn(
            "no_evidence_contains_procedure",
            REWRITE.response_policy_violations(value),
        )

    def test_no_evidence_negated_procedure_is_allowed(self):
        value = response(
            route="no_evidence",
            answer="目前无法确定结构，因此不要拆卸任何部件，也请勿拔下线束插头。",
        )
        self.assertNotIn(
            "no_evidence_contains_procedure",
            REWRITE.response_policy_violations(value),
        )

    def test_no_evidence_procedure_question_is_allowed(self):
        value = response(
            route="no_evidence",
            answer="目前无法确定状态。请先确认是否已经拆下车轮，以及之前是否断开过蓄电池。",
        )
        self.assertNotIn(
            "no_evidence_contains_procedure",
            REWRITE.response_policy_violations(value),
        )

    def test_no_evidence_procedure_purpose_question_is_allowed(self):
        value = response(
            route="no_evidence",
            answer="目前无法确定具体方法。请先说明拆卸进气管是为了清洗、更换还是维修。",
        )
        self.assertNotIn(
            "no_evidence_contains_procedure",
            REWRITE.response_policy_violations(value),
        )

    def test_dipstick_observation_is_allowed(self):
        value = response(
            route="no_evidence",
            answer="目前不能确定加注量。可在车辆冷却后拔出机油尺，观察液位所在位置。",
        )
        self.assertNotIn(
            "no_evidence_contains_procedure",
            REWRITE.response_policy_violations(value),
        )

    def test_no_evidence_specific_cause_is_rejected(self):
        value = response(
            route="no_evidence",
            answer="目前不能确定。常见原因包括喷油器堵塞和高压油泵故障。",
        )
        self.assertIn(
            "no_evidence_contains_specific_cause",
            REWRITE.response_policy_violations(value),
        )

    def test_no_evidence_unsourced_general_claim_is_rejected(self):
        value = response(
            route="no_evidence",
            answer="目前无法确认配件来源。原厂件通常做工更细致，可以比较包装和重量。",
        )
        self.assertIn(
            "no_evidence_contains_unsourced_claim",
            REWRITE.response_policy_violations(value),
        )

    def test_matched_reason_cannot_describe_analogy(self):
        value = response(route="matched", reason="仅为相似症状，需要迁移检查方法。")
        self.assertIn(
            "matched_reason_describes_analogy",
            REWRITE.response_policy_violations(value),
        )

    def test_no_evidence_reason_cannot_claim_analogy(self):
        value = response(route="no_evidence", reason="采用类比路线迁移检查方法。")
        self.assertIn(
            "no_evidence_reason_describes_analogy",
            REWRITE.response_policy_violations(value),
        )

    def test_no_evidence_reason_cannot_claim_reused_evidence(self):
        value = response(
            route="no_evidence",
            reason="证据6与系统一致且检查原则可复用，故选为analogy。",
        )
        self.assertIn(
            "no_evidence_reason_describes_analogy",
            REWRITE.response_policy_violations(value),
        )

    def test_no_evidence_reason_can_deny_analogy(self):
        value = response(route="no_evidence", reason="候选均不相关，无法迁移检查方法。")
        self.assertNotIn(
            "no_evidence_reason_describes_analogy",
            REWRITE.response_policy_violations(value),
        )

    def test_matched_reason_can_warn_against_copying_conclusion(self):
        value = response(route="matched", reason="检查流程与现象匹配，但不能直接套用最终结论。")
        self.assertNotIn(
            "matched_reason_describes_analogy",
            REWRITE.response_policy_violations(value),
        )

    def test_invalid_evidence_id_is_not_silently_demoted(self):
        raw = {
            "route": "analogy",
            "selected_evidence_ids": ["证据6"],
            "answer": "目前不能确定具体原因，可以先记录故障出现的完整工况和仪表提示，再交由具备资质的专业人员检查相关系统。",
            "reason": "采用证据6的检查原则。",
        }
        normalized = REWRITE.normalize_response(raw, {"evidence_real_id"})
        self.assertEqual("analogy", normalized["route"])
        self.assertEqual(["证据6"], normalized["invalid_selected_evidence_ids"])
        violations = REWRITE.response_policy_violations(normalized)
        self.assertIn("invalid_selected_evidence_ids", violations)
        self.assertIn("route_missing_selected_evidence", violations)

    def test_no_evidence_cannot_return_selected_ids(self):
        value = response(
            route="no_evidence",
            unexpected_selected_evidence_ids=["e1"],
        )
        self.assertIn(
            "route_forbids_selected_evidence",
            REWRITE.response_policy_violations(value),
        )

    def test_ambiguous_oil_question_must_clarify_fluid_type(self):
        value = response(
            route="analogy",
            answer="如果是指发动机机油，请以用户手册要求为准，按规定规格购买。",
        )
        violations = REWRITE.response_policy_violations(value, "这款车应该加什么油")
        self.assertIn("ambiguous_fluid_not_clarified", violations)

    def test_ambiguous_oil_question_can_ask_fuel_or_engine_oil(self):
        value = response(
            route="no_evidence",
            answer="请先确认您问的是汽油标号还是发动机机油规格，两者需要的信息和答案不同。",
        )
        violations = REWRITE.response_policy_violations(value, "这款车应该加什么油")
        self.assertNotIn("ambiguous_fluid_not_clarified", violations)

    def test_unsafe_exhaust_smell_is_rejected(self):
        value = response(
            route="no_evidence",
            answer="目前不能确定原因。可以靠近排气管闻一下尾气属于酸味还是氨味。",
        )
        self.assertIn(
            "answer_contains_unsafe_user_action",
            REWRITE.response_policy_violations(value),
        )

    def test_unsafe_current_measurement_is_rejected_even_when_matched(self):
        value = response(
            route="matched",
            answer="把万用表调到电流档，断开蓄电池接线并将表笔串联到回路中测量。",
        )
        self.assertIn(
            "answer_contains_unsafe_user_action",
            REWRITE.response_policy_violations(value),
        )

    def test_professional_current_measurement_is_allowed(self):
        value = response(
            route="matched",
            answer="建议由专业维修人员将万用表串联到回路中测量车辆静态电流。",
        )
        self.assertNotIn(
            "answer_contains_unsafe_user_action",
            REWRITE.response_policy_violations(value),
        )

    def test_guarded_low_speed_observation_is_allowed(self):
        value = response(
            route="no_evidence",
            answer="可在空旷停车场低速通过减速带，观察辅助驾驶是否再次异常。",
        )
        self.assertNotIn(
            "answer_suggests_high_risk_reproduction",
            REWRITE.response_policy_violations(value, "车辆出现LCC画龙和幽灵刹车"),
        )

    def test_unguarded_high_risk_reproduction_is_rejected(self):
        value = response(
            route="no_evidence",
            answer="可以驾驶车辆通过减速带，观察辅助驾驶是否再次异常。",
        )
        self.assertIn(
            "answer_suggests_high_risk_reproduction",
            REWRITE.response_policy_violations(value, "车辆出现LCC画龙和幽灵刹车"),
        )

    def test_dangerous_reproduction_is_rejected_even_with_location_hint(self):
        value = response(
            route="no_evidence",
            answer="可以在空旷停车场反复急刹，观察制动故障是否复现。",
        )
        self.assertIn(
            "answer_suggests_high_risk_reproduction",
            REWRITE.response_policy_violations(value, "车辆制动异常"),
        )

    def test_prompt_reads_untrusted_draft_after_candidates(self):
        query = {"instruction": "汽车问题", "query": "冷车异响", "base_answer": "草稿内容"}
        evidence_by_id = {"e1": {"evidence_id": "e1", "source": "d3", "text": "检查记录"}}
        prompt = REWRITE.build_user_prompt(
            query, [{"evidence_id": "e1", "score": 0.7}], evidence_by_id, 5000, 6000,
        )
        self.assertLess(prompt.index("候选 Evidence"), prompt.index("<untrusted_draft>"))
        self.assertIn("信息强度守恒", REWRITE.SYSTEM_PROMPT)

    def test_overlong_answer_is_rejected(self):
        value = response(answer="检查相关连接。" * 100)
        self.assertIn("answer_too_long", REWRITE.response_policy_violations(value))


if __name__ == "__main__":
    unittest.main()
