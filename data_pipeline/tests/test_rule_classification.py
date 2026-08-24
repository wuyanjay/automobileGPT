#!/usr/bin/env python3
"""Regression tests for the lightweight rule classifiers."""

from __future__ import print_function

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    classify_automotive_domain,
    classify_d4_document,
    classify_repair_intent,
    context_flags,
    diagnosis_process_signals,
    has_diagnosis_process,
    off_topic_reasons,
    requires_safety_review,
    risk_flags,
)

NORMALIZE_SPEC = importlib.util.spec_from_file_location(
    "normalize_filter", SCRIPT_DIR / "02_normalize_filter.py"
)
NORMALIZE = importlib.util.module_from_spec(NORMALIZE_SPEC)
NORMALIZE_SPEC.loader.exec_module(NORMALIZE)


class VehicleContextTests(unittest.TestCase):
    def setUp(self):
        self.catalog = NORMALIZE.build_vehicle_catalog([
            {"instruction": "以下为汽车品牌和车型信息:本田 雅阁,请回答相关问题:"},
            {"instruction": "以下为汽车品牌和车型信息:本田 里程,请回答相关问题:"},
            {"instruction": "以下为汽车品牌和车型信息:大众 桑塔纳,请回答相关问题:"},
            {"instruction": "以下为汽车品牌和车型信息:别克 昂科威,请回答相关问题:"},
        ])

    def test_explicit_owned_vehicle_conflict_is_blocking(self):
        status, declared, mentions = NORMALIZE.vehicle_context_check(
            "以下为汽车品牌和车型信息:本田 雅阁,请回答相关问题:",
            "我有一款08年的桑塔纳，没有风且风量无法调节。",
            self.catalog,
        )
        self.assertEqual(status, "conflict")
        self.assertEqual(declared["model"], "雅阁")
        self.assertEqual(mentions[0]["model"], "桑塔纳")

    def test_same_explicit_model_is_consistent(self):
        status, _, _ = NORMALIZE.vehicle_context_check(
            "以下为汽车品牌和车型信息:别克 昂科威,请回答相关问题:",
            "昂科威低速行驶时转速突然升高。",
            self.catalog,
        )
        self.assertEqual(status, "consistent")

    def test_comparison_with_another_model_is_review_only(self):
        status, _, _ = NORMALIZE.vehicle_context_check(
            "以下为汽车品牌和车型信息:本田 雅阁,请回答相关问题:",
            "桑塔纳与这款车相比，空调结构一样吗？",
            self.catalog,
        )
        self.assertEqual(status, "review")

    def test_implicit_vehicle_reference_is_not_blocked(self):
        status, _, mentions = NORMALIZE.vehicle_context_check(
            "以下为汽车品牌和车型信息:别克 昂科威,请回答相关问题:",
            "我的车低速行驶时转速突然升高。",
            self.catalog,
        )
        self.assertEqual(status, "no_explicit_vehicle")
        self.assertEqual(mentions, [])

    def test_two_character_model_word_in_repair_prose_is_not_a_vehicle(self):
        status, _, mentions = NORMALIZE.vehicle_context_check(
            "以下为汽车品牌和车型信息:别克 昂科威,请回答相关问题:",
            "我的车续航里程显示不准，跑一段路也没有变化。",
            self.catalog,
        )
        self.assertEqual(status, "no_explicit_vehicle")
        self.assertEqual(mentions, [])


class RepairIntentTests(unittest.TestCase):
    def test_price_only(self):
        self.assertEqual(classify_repair_intent("换一个保险杠多少钱"), "price_only")

    def test_purchase_only(self):
        self.assertEqual(classify_repair_intent("这款二手车值得购买吗"), "purchase_only")

    def test_mixed_price_keeps_repair_problem(self):
        self.assertEqual(
            classify_repair_intent("空调不制冷是什么原因，换压缩机多少钱"),
            "mixed_repair_price",
        )

    def test_mixed_price_keeps_diagnostic_wording(self):
        self.assertEqual(
            classify_repair_intent("ABS线被剪短，怎么维修？维修费用大概多少"),
            "mixed_repair_price",
        )

    def test_mixed_price_keeps_abnormal_battery_symptoms(self):
        self.assertEqual(
            classify_repair_intent("动力电池掉电异常，充电功率低，换电多少钱"),
            "mixed_repair_price",
        )

    def test_mixed_price_keeps_maintenance_interval_question(self):
        self.assertEqual(
            classify_repair_intent("刹车片多少公里可以换，大概多少钱"),
            "mixed_repair_price",
        )

    def test_mixed_price_keeps_maintenance_effect_question(self):
        self.assertEqual(
            classify_repair_intent("空气滤芯一直没换，对发动机影响大吗，清洗节气门多少钱"),
            "mixed_repair_price",
        )

    def test_mixed_price_keeps_repair_necessity_question(self):
        self.assertEqual(
            classify_repair_intent("碰撞后需要换车门吗，换门多少钱"),
            "mixed_repair_price",
        )

    def test_mixed_price_keeps_diagnostic_guess(self):
        self.assertEqual(
            classify_repair_intent("漏机油了，是不是油底壳坏了，换一个多少钱"),
            "mixed_repair_price",
        )

    def test_mixed_price_keeps_procedure_challenge(self):
        self.assertEqual(
            classify_repair_intent("换分泵真的需要放完刹车油吗，报价多少"),
            "mixed_repair_price",
        )

    def test_repair_only(self):
        self.assertEqual(classify_repair_intent("发动机异响应该怎么排查"), "repair_only")


class DomainTests(unittest.TestCase):
    def test_battery_cycling_is_not_bicycle(self):
        self.assertEqual(off_topic_reasons("The battery keeps cycling and loses charge"), [])

    def test_spider_gear_is_automotive_term(self):
        self.assertEqual(off_topic_reasons("Inspect the differential spider gears"), [])

    def test_spider_vehicle_model_is_not_an_insect(self):
        self.assertEqual(off_topic_reasons("Alfa Romeo Spider stored for 12 years"), [])

    def test_spider_spring_is_not_an_insect(self):
        self.assertEqual(off_topic_reasons("Can I drive with a broken spider spring in my clutch?"), [])

    def test_spider_pest_control_is_off_topic(self):
        self.assertIn("spider", off_topic_reasons("How do I get rid of spiders inside my car?"))

    def test_vehicle_key_fob_pcb_is_automotive(self):
        self.assertEqual(off_topic_reasons("My car keyless remote fob only works outside the PCB housing"), [])

    def test_bicycle_question_is_off_topic(self):
        self.assertIn("bicycle", off_topic_reasons("How do I repair a bicycle chain?"))

    def test_expanded_chinese_component_vocabulary(self):
        self.assertEqual(
            classify_automotive_domain("中控锁达到45km/h后车门无法自动落锁", source_hint="d3"),
            "automotive",
        )

    def test_weak_labels_can_confirm_domain(self):
        self.assertEqual(
            classify_automotive_domain("起步时偶尔发响", source_hint="d3", auxiliary_text="离合器_异响"),
            "automotive",
        )

    def test_power_grid_text_is_non_automotive(self):
        self.assertEqual(
            classify_automotive_domain("值班调控检查换流站直流母线", source_hint="d3"),
            "non_automotive",
        )


class ContextTests(unittest.TestCase):
    def test_figure_out_is_not_an_image_reference(self):
        flags = context_flags("You can figure out the value by measuring voltage at the connector.")
        self.assertNotIn("has_image_reference", flags)

    def test_out_of_the_picture_is_not_an_image_reference(self):
        flags = context_flags("Once the engine is running, the starter circuit is out of the picture.")
        self.assertNotIn("has_image_reference", flags)

    def test_supplementary_url_is_soft_reference(self):
        answer = (
            "First check battery voltage at the posts. Then inspect both terminals for corrosion and "
            "measure voltage drop while cranking. Replace the battery only if it fails the load test. "
            "A supplementary wiring diagram is available at https://example.com/diagram"
        )
        flags = context_flags(answer, "Why does the car intermittently have no power?")
        self.assertIn("has_external_link", flags)
        self.assertNotIn("needs_external_link", flags)
        self.assertIn("has_image_reference", flags)
        self.assertNotIn("needs_image", flags)

    def test_link_only_answer_is_hard_dependency(self):
        flags = context_flags("See https://example.com/manual for the complete procedure.")
        self.assertIn("needs_external_link", flags)

    def test_external_site_required_for_result_is_hard_dependency(self):
        answer = (
            "Use the video to retrieve the serial number, then go to the manufacturer website "
            "https://example.com/code to obtain the unlock code."
        )
        self.assertIn("needs_external_link", context_flags(answer))

    def test_concise_direct_answer_with_supplementary_link_is_soft(self):
        answer = (
            "No, the worn steering-rack bushings can be bought and replaced separately. "
            "Parts are listed at https://example.com/rack-bushings"
        )
        flags = context_flags(answer, "Must I replace the whole steering rack?")
        self.assertIn("has_external_link", flags)
        self.assertNotIn("needs_external_link", flags)

    def test_video_required_for_location_is_hard_dependency(self):
        answer = (
            "I am only answering where the sensor is located. This video explains the procedure, "
            "including the location: https://example.com/video"
        )
        self.assertIn("needs_external_link", context_flags(answer, "Where is the sensor?"))

    def test_identification_question_needs_image(self):
        flags = context_flags("It is the purge valve.", "What is this part in the picture?")
        self.assertIn("needs_image", flags)

    def test_photo_included_identification_needs_image(self):
        flags = context_flags(
            "That is the crankcase ventilation hose.",
            "What is the part called that connects the engine to the intake? Photo included",
        )
        self.assertIn("needs_image", flags)

    def test_image_based_damage_judgment_needs_image(self):
        flags = context_flags(
            "The image does not show much detail, but from what I can see it is only scratched.",
            "Is this tyre damage safe to drive on?",
        )
        self.assertIn("needs_image", flags)

    def test_implicit_object_identification_needs_image(self):
        flags = context_flags(
            "The picture is unclear, but that looks like a sway bar end-link.",
            "This came off in my car.",
        )
        self.assertIn("needs_image", flags)

    def test_demonstrative_symptom_is_not_object_identification(self):
        answer = (
            "A worn idler or tensioner pulley can squeak. Remove the belt and spin each pulley by hand; "
            "replace the one that feels rough or has play. Pictures of both pulley types are included for reference."
        )
        flags = context_flags(answer, "What is this squeak in my engine, and how can I fix it?")
        self.assertIn("has_image_reference", flags)
        self.assertNotIn("needs_image", flags)

    def test_plural_picture_basis_needs_image(self):
        flags = context_flags(
            "Based on the pictures you provided, the lower control arm is broken.",
            "Is the steering damage serious?",
        )
        self.assertIn("needs_image", flags)

    def test_long_self_contained_answer_with_illustrative_image_is_soft(self):
        answer = (
            "Inspect the hose inside and outside for cracks. Push it fully past the raised ridge, then "
            "place the clamp behind the ridge with rubber extending beyond both sides. Tighten it evenly "
            "and pressure-test the joint for leaks. The first picture shows an incorrect position and the "
            "second picture shows the corrected position."
        )
        flags = context_flags(answer, "How should a hose clamp be installed?")
        self.assertIn("has_image_reference", flags)
        self.assertNotIn("needs_image", flags)

    def test_self_contained_comment_reference_is_soft(self):
        answer = (
            "As noted in the comments, check voltage directly at the battery posts. If voltage remains "
            "above 12.4 V, inspect the terminal connection and perform a voltage-drop test while cranking."
        )
        flags = context_flags(answer)
        self.assertIn("has_comment_reference", flags)
        self.assertNotIn("needs_comment", flags)


class RiskTests(unittest.TestCase):
    def test_bypass_valve_is_not_safety_bypass(self):
        flags = risk_flags("Inspect the oil filter bypass valve for sticking.")
        self.assertIn("has_bypass_term", flags)
        self.assertNotIn("risk_bypass", flags)

    def test_parking_brake_interlock_bypass_is_risky(self):
        flags = risk_flags("How can I bypass the parking brake video interlock?")
        self.assertIn("risk_bypass", flags)

    def test_short_bursts_near_parking_brake_are_not_bypass(self):
        flags = risk_flags(
            "Put the parking brake on and rev the engine in short bursts to inspect the mounts.",
            "How do I test an engine mount?",
        )
        self.assertNotIn("risk_bypass", flags)

    def test_diagnostic_immobilizer_mention_is_not_bypass(self):
        flags = risk_flags(
            "The immobilizer is supposed to be hard to bypass. Check its warning light and read DTCs.",
            "Why does the engine start and stop after two seconds?",
        )
        self.assertNotIn("risk_bypass", flags)

    def test_global_procedure_word_does_not_make_warning_a_bypass(self):
        flags = risk_flags(
            "Check the owner manual for a recovery procedure. The immobilizer is supposed to be hard to bypass.",
            "Why does the engine start and stop after two seconds?",
        )
        self.assertNotIn("risk_bypass", flags)

    def test_question_bypass_intent_is_still_blocked_when_answer_warns(self):
        flags = risk_flags(
            "Disabling an airbag is dangerous and is not recommended.",
            "How do I disable the passenger airbag?",
        )
        self.assertIn("risk_bypass", flags)


class DiagnosisTests(unittest.TestCase):
    D3_EXAMPLE = (
        "故障原因发动机连杆轴瓦异响诊断排除1、从发动机加速时的异响特征可初步确定为连杆轴瓦异响;"
        "2、放出机油检查,从机油带有铝屑可判断有连杆轴瓦磨损;3、拆开油底壳检查连杆瓦,发现工作面磨损;"
        "4、用测隙条检查轴瓦间隙是否在标准范围内,换掉后装机路试,上述异响消失。"
    )
    D4_EXAMPLE = (
        "慢充系统指示灯都不亮的检修方法。当指示灯均不亮时可按照下述步骤检修:"
        "(1)测量充电枪N脚导通,阻值应小于0.5Ω,否则更换充电线。"
        "(2)测量L脚导通,阻值应小于0.5Ω,否则更换充电线。"
        "(3)检查插接件是否烧蚀,继续测量线束电阻。"
    )

    def test_d3_user_example_has_process(self):
        self.assertTrue(has_diagnosis_process(self.D3_EXAMPLE))
        signals = diagnosis_process_signals(self.D3_EXAMPLE)
        self.assertGreaterEqual(signals["step_count"], 4)
        self.assertGreaterEqual(signals["outcome_count"], 1)

    def test_d3_outcome_word_xiaochu_is_recognized(self):
        text = (
            "故障原因简要分析，先固定可能共振的部件后试车，故障依旧。"
            "检查发现汽油管与水管碰撞，分别固定后试车，异响消除。"
        )
        self.assertTrue(has_diagnosis_process(text))
        self.assertGreaterEqual(diagnosis_process_signals(text)["outcome_count"], 1)

    def test_short_d3_analysis_with_verified_outcome_is_process(self):
        text = (
            "故障分析检查车门线路，发现地板灯线路破损搭铁。"
            "故障排除处理破损线路后，仪表指示正常，故障排除。"
        )
        self.assertTrue(has_diagnosis_process(text))

    def test_outcome_heading_alone_is_not_verified_outcome(self):
        signals = diagnosis_process_signals("故障分析可能是继电器。故障排除建议检查继电器。")
        self.assertEqual(signals["outcome_count"], 0)

    def test_d4_user_example_is_procedure(self):
        self.assertTrue(has_diagnosis_process(self.D4_EXAMPLE))
        self.assertEqual(
            classify_d4_document("北汽新能源EV200慢充系统指示灯都不亮的检修方法", self.D4_EXAMPLE),
            "procedure_evidence",
        )

    def test_symptom_only_is_not_process(self):
        self.assertFalse(has_diagnosis_process("车辆行驶时发动机异响，怠速时声音减轻，可能是轴瓦磨损。"))

    def test_heading_with_one_suggested_check_is_not_full_process(self):
        self.assertFalse(has_diagnosis_process("诊断排除重点检查油泵继电器，可判断发动机可能断油。"))

    def test_concrete_case_is_case_evidence(self):
        text = "一辆轿车无法启动。维修过程:1、检查蓄电池电压;2、测量起动机端子发现无电压。更换继电器后试车正常。"
        self.assertEqual(classify_d4_document("轿车无法启动故障", text), "case_evidence")

    def test_high_voltage_modification_requires_review(self):
        self.assertTrue(requires_safety_review("动力电池高压互锁线路需要短接后继续测试"))

    def test_shielded_wire_and_jumper_measurement_are_not_unsafe_modification(self):
        self.assertFalse(requires_safety_review("高压车型使用跨接线测量旋变绕组，并检查屏蔽线是否断路"))


if __name__ == "__main__":
    unittest.main()
