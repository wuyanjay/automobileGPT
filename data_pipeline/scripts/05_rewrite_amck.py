#!/usr/bin/env python3
"""Use one LLM call to select a route and rewrite each retrieved AMCK answer."""

from __future__ import print_function

import argparse
import json
import os
import re
import time
from collections import Counter

from tqdm import tqdm

from _common import (
    api_chat,
    load_by_id,
    load_config,
    load_env_file,
    normalize_document,
    parse_model_json,
    pipeline_path,
    read_jsonl,
    retry_call,
    stable_hash,
    unsupported_literals,
    write_json,
    write_jsonl,
)


SYSTEM_PROMPT = """你是汽车维修训练数据编辑器。你不是在远程确诊车辆，而是在根据用户输入、候选维修资料和一份不可信草稿生成可靠、克制、自然的中文 SFT 答案。

必须只输出一个 JSON 对象，字段为 route、selected_evidence_ids、answer、reason。
route 只能是 matched、analogy、no_evidence、reject。

输出前请在内部完成以下检查，但不要输出检查过程：先独立理解用户问题；再判断候选是否真的覆盖当前对象、现象和工况；仅用用户输入与所选候选起草；最后逐句确认技术事实、因果关系和操作建议都有来源，且语气没有被加强。

规则：
1. matched：只有候选证据与当前问题的系统、主要现象、触发工况都较一致时才可使用。可采用所选证据中的检查动作和诊断顺序，但不得把来源车辆的最终原因写成当前车辆结论。
2. analogy：候选与当前问题至少属于同一系统，且某个检查原则确实可以复用，但车型、现象或工况有明显差异。只能采用所选证据支持的检查原则；不得迁移最终故障、DTC、数值、工具型号和车型专用操作。
3. no_evidence：候选均不相关或不足。AMCK 原答案也不是事实来源。回答只能说明目前不能确定，提出 2-4 个能改变判断分支的问题，并给出最多 2 个无需拆卸、无需专用工具、无需危险试车的安全观察动作。不得自行补充具体故障原因、车型专用重启/刷写方法、维修参数、拆装步骤或保养结论。
4. reject：非目标问题、输入损坏、缺少图片等关键上下文，或无法安全改写。
5. selected_evidence_ids 是 answer 中技术事实和操作建议的唯一资料来源，只能逐字复制候选中形如 evidence_xxx 的完整 ID，禁止填写“证据1”“候选2”等序号。不得使用未选候选中的专有内容；no_evidence/reject 必须为空数组。
6. answer 面向普通用户，禁止出现“候选证据”“根据证据”“证据显示”“现有证据不足”“检索结果”“同类系统”“类似案例”“迁移”“不能直接套用”等内部元话术。analogy 也应自然表达为“目前不能确定，可先检查……”。资料是否充分只能写在 reason 中。
7. 遵守“信息强度守恒”：候选中的“可能、怀疑、有时、个别案例、初步判断”不得改成“通常、主要原因、已经证明、明显下降、必然导致”。单个维修案例不能推导常见原因，多条候选不能组合出原文没有的新因果链，检查某项也不能扩写成该项就是当前故障原因。
8. AMCK 原答案是不可信草稿，只能用来确认是否漏答了用户的问题，不能提供技术事实。先完成基于用户输入和候选的答案，再检查草稿；不得沿用草稿中未被所选候选支持的机理、结论、数值或车型知识。
9. 当 answer 包含用户可执行的观察、启动、低速验证或简单检查时，应在相关句中说明适用环境、操作边界或停止条件。草稿或当前正文已经有正确且相关的安全提示时，应自然保留或合并，不得重复追加；最终最多保留一处完整安全说明。纯知识问答不要机械添加固定警告。
10. 允许在安全空旷场地进行短时间、低速、非破坏性的现象确认，但不得要求高速、急刹、失控边缘或持续保持机构极限位置来复现故障。涉及高压电池、高压燃油、制动失效、转向失控、安全气囊、车辆下方作业时，不得让普通用户通过一句安全提示自行拆卸、拔插、短接、泄压、刷写或复现危险故障。
11. 不得建议普通用户闻尾气、触碰运行中的发动机部件、堵塞真空管、串联万用表测大电流、向气缸注油或拆检发动机。必要检测应明确交给具备资质的人员。
12. 油液牌号、零件互换、保养周期、维修价格、质保和车型功能都属于车型专用事实；只有同车型同配置的直接资料才能给出明确结论。候选没有直接比较用户所说的两个对象时，不得根据名称、标号或模型常识自行推导差异。尤其不得只根据 5W、10W、30、40 等标号推导当前发动机的噪声、磨损、流动速度或适配结论。
13. 不要只说“去 4S 店”，不要盲目建议换件。不要编造候选中没有的事实、因果关系、车型配置、价格、保险理赔、保修政策、数值或操作方法。
14. answer 建议 150-450 个中文字符，最多 4 个检查步骤。遇到“加什么油”“这个位置”“图中/以下异常”等歧义或缺失上下文时，必须先澄清所指对象；不得让候选替用户补全问题。缺失图片且无法作答时选择 reject。
"""


ANSWER_MAX_CHARS = 500
ANSWER_META_RE = re.compile(
    r"证据|候选(?:Evidence)?|检索结果|"
    r"同类系统|类似案例|参考案例|不能直接套用|不可直接套用|"
    r"迁移(?:检查|方法|结论)|作为排查参考"
)
MATCHED_REASON_ANALOGY_RE = re.compile(
    r"(?:仅|只是).{0,12}(?:类似|相似)|(?:采用|属于|应走|按).{0,8}(?:类比|analogy)|"
    r"(?:车型|主要现象|触发工况).{0,12}(?:明显)?(?:不同|不一致|不匹配)",
    re.I,
)
NO_EVIDENCE_REASON_ANALOGY_RE = re.compile(
    r"(?:采用|选择|判定为|按).{0,8}(?:类比|analogy)|可(?:以)?迁移|"
    r"(?:能够|适合|尝试)迁移.{0,8}(?:检查|方法|思路)|"
    r"故选(?:择)?为?.{0,4}(?:类比|analogy)|可(?:以)?复用|"
    r"(?:采用|借用).{0,20}(?:证据|检查|方法|原则|思路)",
    re.I,
)
ANALOGY_REASON_MATCHED_RE = re.compile(
    r"(?:系统|主要现象|触发工况).{0,20}(?:完全|高度).{0,5}(?:一致|匹配)|"
    r"(?:完全|高度).{0,8}(?:一致|匹配).{0,12}(?:直接采用|matched)",
    re.I,
)
NO_EVIDENCE_PROCEDURE_RE = re.compile(
    r"拆(?:下|卸|开)|拔(?:下|出|掉)|短接|跨接|泄压|打磨|恢复出厂|"
    r"刷写|重装系统|松开.{0,12}螺栓|断开.{0,12}(?:蓄电池|电瓶|高压|线束|插头)|"
    r"用(?:万用表|试灯|缸压表|压力表|诊断仪)"
)
PROFESSIONAL_RE = re.compile(
    r"专业|技师|维修人员|维修店|维修点|服务中心|授权|具备资质"
)
ACTION_GUARD_RE = re.compile(
    r"(?:(?:不要|切勿|严禁|不得|避免|不建议|禁止|无需|不必|不可|不能|请勿|"
    r"不应|无法(?:提供|确定|给出)?).{0,24}|[不未])$"
)
QUESTION_GUARD_RE = re.compile(
    r"(?:是否|有没有|有无|是否已经|当前是否|之前是否|是否已).{0,20}$"
)
ACTION_RISK_DESCRIPTION_RE = re.compile(
    r".{0,12}(?:涉及|存在|可能造成).{0,16}(?:风险|危险|受伤|损坏)"
)
ACTION_QUESTION_DESCRIPTION_RE = re.compile(
    r".{0,18}(?:是为了|原因).{0,32}(?:还是|是否)"
)
SAFE_OBSERVATION_RE = re.compile(
    r"拔出.{0,6}机油尺|抽出.{0,6}机油尺"
)
DANGEROUS_USER_ACTION_RE = re.compile(
    r"(?:靠近.{0,12}(?:排气管|尾气).{0,8}闻|闻(?:一下)?(?:尾气|排气|排气管气味))|"
    r"(?:(?:堵住|夹住|捏住).{0,12}(?:真空管|进气管|管路))|"
    r"(?:(?:串联|电流档).{0,30}(?:万用表|表笔|回路))|"
    r"(?:断开.{0,12}(?:电瓶|蓄电池).{0,40}(?:串联|电流档|表笔))|"
    r"(?:向(?:气缸|缸内).{0,12}(?:加|注入).{0,8}机油)|"
    r"(?:拆检.{0,8}(?:缸盖|气门|高压系统|制动系统|转向系统))"
)
HIGH_RISK_DRIVING_CONTEXT_RE = re.compile(
    r"LCC|辅助驾驶|智能驾驶|智驾|幽灵刹车|ABS|制动|刹车|转向失灵|方向失控",
    re.I,
)
ROAD_REPRO_RE = re.compile(
    r"(?:试车|路试|低速行驶|驾驶|驶过|通过).{0,20}(?:减速带|坑洼|湿滑|急刹|复现|测试)|"
    r"(?:录像|拍摄).{0,20}(?:行驶|故障发生|路况)"
)
SAFETY_BOUNDARY_RE = re.compile(
    r"安全|空旷|停车场|不影响交通|低速|短时间|不要长时间|停止|立即停止|"
    r"异常加剧|警告灯|具备资质|专业人员"
)
INHERENTLY_DANGEROUS_REPRO_RE = re.compile(
    r"高速|急刹|紧急制动|失控|甩尾|打滑|持续.{0,8}(?:打死|极限)|"
    r"反复.{0,8}(?:急刹|紧急制动)"
)
AMBIGUOUS_FLUID_QUERY_RE = re.compile(
    r"(?:加|用|换).{0,8}(?:什么|哪种|哪一类)油|应该加什么油"
)
EXPLICIT_FLUID_RE = re.compile(
    r"机油|汽油|柴油|燃油|变速箱油|齿轮油|刹车油|制动液|冷却液|防冻液|助力油"
)
NO_EVIDENCE_CAUSE_RE = re.compile(
    r"常见(?:原因|方向)|通常(?:指向|由于|与|是|指|表示|代表)|多半(?:是|来自)|"
    r"可能(?:是|由|涉及|来自|与).{0,50}(?:故障|损坏|磨损|堵塞|异常|不良|问题|"
    r"有关|相关|导致|引起)|(?:原因|故障点)(?:包括|有|为)"
)
NO_EVIDENCE_UNSOURCED_CLAIM_RE = re.compile(
    r"(?:通常|一般|往往|多数情况下|普遍).{0,50}(?:是|会|可|能|指|表示|代表|"
    r"做工|意味着|说明|需要|取决于)"
)
POLICY_RETRY_MESSAGES = {
    "answer_too_long": "答案超过500字，请压缩到150-450字",
    "answer_contains_pipeline_meta": "答案含数据工程元话术，请改成自然的用户回答",
    "invalid_selected_evidence_ids": "证据ID无效，请只复制候选中完整的 evidence_xxx ID",
    "route_forbids_selected_evidence": "no_evidence/reject 路线的证据ID必须为空",
    "route_missing_selected_evidence": "matched/analogy 必须至少选择一个有效候选证据ID",
    "matched_reason_describes_analogy": "reason 描述的是类比关系，不能标为 matched",
    "analogy_reason_describes_match": "reason 描述的是直接匹配关系，请重新判断 route",
    "no_evidence_reason_describes_analogy": "no_evidence 不得声称采用或复用候选证据",
    "ambiguous_fluid_not_clarified": "用户没有说明油液类型，请先区分汽油/柴油与发动机机油等含义",
    "answer_contains_unsafe_user_action": "删除面向普通用户的危险操作，改由具备资质的人员检查",
    "answer_suggests_high_risk_reproduction": "不要让用户在道路上主动复现制动、转向或辅助驾驶异常",
    "no_evidence_contains_specific_cause": "没有证据时不得列举或暗示具体故障原因",
    "no_evidence_contains_unsourced_claim": "没有证据时不得补充通常/一般等无来源技术结论",
    "no_evidence_contains_procedure": "没有证据时只能给安全观察动作，不能给拆装或仪器检测步骤",
}


def action_is_guarded(sentence, match):
    """Whether an action is negated, merely queried, professional, or explicitly safe."""
    local_context = sentence[max(0, match.start() - 24):min(len(sentence), match.end() + 16)]
    action_context = sentence[match.start():min(len(sentence), match.end() + 12)]
    if PROFESSIONAL_RE.search(local_context) or SAFE_OBSERVATION_RE.search(action_context):
        return True
    prefix = sentence[max(0, match.start() - 24):match.start()]
    suffix = sentence[match.end():min(len(sentence), match.end() + 40)]
    return bool(
        ACTION_GUARD_RE.search(prefix)
        or QUESTION_GUARD_RE.search(prefix)
        or ACTION_RISK_DESCRIPTION_RE.match(suffix)
        or ACTION_QUESTION_DESCRIPTION_RE.match(suffix)
    )


def reason_describes_positive_analogy(reason):
    """Ignore phrases such as '无法复用'; only flag affirmative evidence reuse."""
    for match in NO_EVIDENCE_REASON_ANALOGY_RE.finditer(reason):
        prefix = reason[max(0, match.start() - 10):match.start()]
        if re.search(r"(?:无|不|未|缺乏|不能|不可|无法).{0,8}$", prefix):
            continue
        return True
    return False


def response_policy_violations(response, query_text=""):
    """Return deterministic policy failures not covered by literal checks."""
    route = response["route"]
    answer = response["answer"]
    reason = response.get("reason", "")
    violations = []
    if len(answer) > ANSWER_MAX_CHARS:
        violations.append("answer_too_long")
    if ANSWER_META_RE.search(answer):
        violations.append("answer_contains_pipeline_meta")
    if response.get("invalid_selected_evidence_ids"):
        violations.append("invalid_selected_evidence_ids")
    if response.get("unexpected_selected_evidence_ids"):
        violations.append("route_forbids_selected_evidence")
    if route in ("matched", "analogy") and not response.get("selected_evidence_ids"):
        violations.append("route_missing_selected_evidence")
    if route == "matched" and MATCHED_REASON_ANALOGY_RE.search(reason):
        violations.append("matched_reason_describes_analogy")
    if route == "analogy" and ANALOGY_REASON_MATCHED_RE.search(reason):
        violations.append("analogy_reason_describes_match")
    if route == "no_evidence" and reason_describes_positive_analogy(reason):
        violations.append("no_evidence_reason_describes_analogy")
    if (
        AMBIGUOUS_FLUID_QUERY_RE.search(query_text)
        and not EXPLICIT_FLUID_RE.search(query_text)
        and not ("机油" in answer and re.search(r"汽油|柴油|燃油", answer))
    ):
        violations.append("ambiguous_fluid_not_clarified")
    for sentence in re.split(r"[。！？!?；;\n]", answer):
        for match in DANGEROUS_USER_ACTION_RE.finditer(sentence):
            if not action_is_guarded(sentence, match):
                violations.append("answer_contains_unsafe_user_action")
                break
        if "answer_contains_unsafe_user_action" in violations:
            break
    if HIGH_RISK_DRIVING_CONTEXT_RE.search(query_text):
        if INHERENTLY_DANGEROUS_REPRO_RE.search(answer):
            violations.append("answer_suggests_high_risk_reproduction")
        elif ROAD_REPRO_RE.search(answer) and not SAFETY_BOUNDARY_RE.search(answer):
            violations.append("answer_suggests_high_risk_reproduction")
    if route == "no_evidence":
        if NO_EVIDENCE_CAUSE_RE.search(answer):
            violations.append("no_evidence_contains_specific_cause")
        if NO_EVIDENCE_UNSOURCED_CLAIM_RE.search(answer):
            violations.append("no_evidence_contains_unsourced_claim")
        for sentence in re.split(r"[。！？!?；;\n]", answer):
            for match in NO_EVIDENCE_PROCEDURE_RE.finditer(sentence):
                if not action_is_guarded(sentence, match):
                    violations.append("no_evidence_contains_procedure")
                    break
            if "no_evidence_contains_procedure" in violations:
                break
    return list(dict.fromkeys(violations))


def build_user_prompt(
    query, candidates, evidence_by_id, max_base_chars, max_evidence_chars,
    unsupported=None, policy_violations=None,
):
    candidate_blocks = []
    for candidate in candidates:
        evidence = evidence_by_id.get(candidate["evidence_id"])
        if not evidence:
            continue
        candidate_blocks.append(
            "[{} | source={} | score={}]\n{}".format(
                evidence["evidence_id"], evidence["source"], candidate.get("adjusted_score", candidate.get("score")),
                evidence["text"][:max_evidence_chars],
            )
        )
    prompt = """AMCK instruction:
{instruction}

用户问题:
{query}

候选 Evidence:
{evidence}

AMCK 原答案（不可信草稿，最后阅读；只能检查是否漏答，不是事实来源）:
<untrusted_draft>
{base_answer}
</untrusted_draft>

请判断路线并输出最终 JSON。reason 只写一句内部说明。""".format(
        instruction=query["instruction"],
        query=query["query"],
        base_answer=query["base_answer"][:max_base_chars],
        evidence="\n\n".join(candidate_blocks) if candidate_blocks else "无候选证据",
    )
    if unsupported:
        prompt += (
            "\n\n上一次答案出现以下无来源具体 literal：{}。"
            "请删除或改成不含具体值的表述，并重新输出完整 JSON。".format("、".join(unsupported))
        )
    if policy_violations:
        policy_feedback = [
            POLICY_RETRY_MESSAGES.get(value, value) for value in policy_violations
        ]
        prompt += (
            "\n\n上一次输出未通过以下策略校验：{}。"
            "请严格按系统规则修正 route、证据边界、风险操作或答案长度，并重新输出完整 JSON。".format(
                "；".join(policy_feedback)
            )
        )
    return prompt


def mock_response(query):
    return {
        "route": "no_evidence",
        "selected_evidence_ids": [],
        "answer": (
            "根据目前提供的信息，还不能确定唯一故障原因。建议先记录故障出现的工况，"
            "读取并保存车辆当前的故障信息，再从与现象直接相关的线路、连接和基础状态开始检查。"
            "在得到测量结果前不建议继续盲目更换部件。请补充故障是持续还是偶发、冷热车是否有差异、"
            "已经做过哪些测量，以及是否有明确的故障提示。"
        ),
        "reason": "mock smoke response",
    }


def normalize_response(response, candidate_ids):
    route = str(response.get("route", "")).strip()
    if route not in ("matched", "analogy", "no_evidence", "reject"):
        raise ValueError("invalid route: {}".format(route))
    raw_selected = response.get("selected_evidence_ids") or []
    if isinstance(raw_selected, str):
        raw_selected = [raw_selected]
    raw_selected = [str(value).strip() for value in raw_selected if str(value).strip()]
    invalid_selected = [value for value in raw_selected if value not in candidate_ids]
    selected = [value for value in raw_selected if value in candidate_ids]
    unexpected_selected = []
    if route in ("no_evidence", "reject"):
        unexpected_selected = list(raw_selected)
        selected = []
    answer = normalize_document(response.get("answer"))
    if route != "reject" and len(answer) < 40:
        raise ValueError("answer is empty or too short")
    return {
        "route": route,
        "selected_evidence_ids": selected,
        "answer": answer,
        "reason": str(response.get("reason", ""))[:500],
        "invalid_selected_evidence_ids": invalid_selected,
        "unexpected_selected_evidence_ids": unexpected_selected,
    }


def generation_input_hash(user_prompt, provider, model):
    payload = {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "provider": provider,
        "model": model if provider == "api" else "mock",
    }
    return stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True), length=32)


def count_pending_rewrites(
    retrieval_records, query_by_id, evidence_by_id, existing_by_query,
    provider, model, max_base_chars, max_evidence_chars, limit,
):
    """Count uncached records so tqdm reports the actual rewrite workload."""
    pending = 0
    for retrieval in retrieval_records:
        query = query_by_id.get(retrieval["query_id"])
        if not query:
            continue
        candidates = retrieval.get("candidates", [])
        user_prompt = build_user_prompt(
            query, candidates, evidence_by_id, max_base_chars, max_evidence_chars,
        )
        input_hash = generation_input_hash(user_prompt, provider, model)
        previous = existing_by_query.get(retrieval["query_id"])
        if previous and previous.get("input_hash") == input_hash:
            continue
        pending += 1
        if limit and pending >= limit:
            break
    return pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-name", default="default")
    parser.add_argument(
        "--retrieval-run-name", default=None,
        help="Retrieval input name; defaults to --run-name",
    )
    parser.add_argument("--provider", choices=("api", "mock"), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--env-file", default=None, help="Defaults to data_pipeline/.env")
    parser.add_argument("--splits", default="train,validation,test")
    parser.add_argument("--limit", type=int, default=0, help="Maximum new records per split")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()
    load_env_file(args.env_file)
    config = load_config(args.config)
    work_dir = pipeline_path(config, "work")
    generation = config["generation"]
    filters = config["filters"]
    provider = args.provider or generation.get("provider", "api")
    model = args.model or os.environ.get(generation.get("model_env", "LLM_MODEL")) or os.environ.get("OPENAI_MODEL")
    base_url = args.base_url or os.environ.get(generation.get("base_url_env", "LLM_BASE_URL")) or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com"
    api_key_env = args.api_key_env or generation.get("api_key_env", "LLM_API_KEY")
    api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
    if provider == "api" and not model:
        raise RuntimeError("Missing model. Pass --model or set LLM_MODEL/OPENAI_MODEL.")
    selected_splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    retrieval_run_name = args.retrieval_run_name or args.run_name
    query_by_id = load_by_id(work_dir / "normalized" / "d1_amck.jsonl", "query_id")
    generated_dir = work_dir / "generated" / args.run_name
    rejected_dir = work_dir / "rejected" / args.run_name
    save_every = int(generation.get("save_every", 100))
    attempts = int(generation.get("attempts", 3))
    max_base_chars = filters.get("max_base_answer_chars", 5000)
    max_evidence_chars = filters.get("max_evidence_chars", 6000)
    report = {
        "provider": provider,
        "model": model,
        "retrieval_run_name": retrieval_run_name,
        "output_run_name": args.run_name,
        "splits": {},
    }

    for split in selected_splits:
        evidence_by_id = load_by_id(work_dir / "evidence" / ("evidence_{}.jsonl".format(split)), "evidence_id")
        retrieval_path = (
            work_dir / "retrieval" / retrieval_run_name / ("retrieval_{}.jsonl".format(split))
        )
        retrieval_records = list(read_jsonl(retrieval_path))
        output_path = generated_dir / ("rewrite_{}.jsonl".format(split))
        existing = [] if args.no_resume else list(read_jsonl(output_path))
        existing_by_query = {record["query_id"]: record for record in existing}
        output = list(existing)
        rejected = []
        new_count = 0
        attempted_count = 0
        accepted_count = 0
        route_counts = Counter(record.get("route") for record in existing)
        target = count_pending_rewrites(
            retrieval_records, query_by_id, evidence_by_id, existing_by_query,
            provider, model, max_base_chars, max_evidence_chars, args.limit,
        )
        progress = tqdm(
            total=target,
            desc="05 rewrite {}".format(split),
            unit="sample",
            dynamic_ncols=True,
            ascii=os.name == "nt",
            disable=args.no_progress,
        )
        for retrieval in retrieval_records:
            if attempted_count >= target:
                break
            query_id = retrieval["query_id"]
            query = query_by_id.get(query_id)
            if not query:
                rejected.append({"query_id": query_id, "error": "query_not_found"})
                continue
            candidates = retrieval.get("candidates", [])
            candidate_ids = {item["evidence_id"] for item in candidates}
            user_prompt = build_user_prompt(
                query, candidates, evidence_by_id,
                max_base_chars, max_evidence_chars,
            )
            input_hash = generation_input_hash(user_prompt, provider, model)
            previous = existing_by_query.get(query_id)
            if previous and previous.get("input_hash") == input_hash:
                continue
            if previous:
                output = [record for record in output if record.get("query_id") != query_id]
                route_counts[previous.get("route")] -= 1
                existing_by_query.pop(query_id, None)
            try:
                progress.set_postfix(status="requesting", query=query_id[-8:], refresh=True)
                if provider == "mock":
                    parsed = mock_response(query)
                else:
                    content = retry_call(
                        lambda: api_chat(
                            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
                            model, base_url, api_key,
                            temperature=float(generation.get("temperature", 0.1)),
                        ),
                        attempts=attempts,
                    )
                    parsed = parse_model_json(content)
                normalized = normalize_response(parsed, candidate_ids)
                selected_texts = [
                    evidence_by_id[evidence_id]["text"]
                    for evidence_id in normalized["selected_evidence_ids"]
                    if evidence_id in evidence_by_id
                ]
                allowed_texts = [query["instruction"], query["query"]] + selected_texts
                unsupported = unsupported_literals(normalized["answer"], allowed_texts)
                query_text = query["instruction"] + "\n" + query["query"]
                policy_violations = response_policy_violations(normalized, query_text)
                if (unsupported or policy_violations) and provider != "mock":
                    progress.set_postfix(status="retrying", query=query_id[-8:], refresh=True)
                    retry_prompt = build_user_prompt(
                        query, candidates, evidence_by_id,
                        max_base_chars, max_evidence_chars,
                        unsupported=unsupported,
                        policy_violations=policy_violations,
                    )
                    retry_content = retry_call(
                        lambda: api_chat(
                            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": retry_prompt}],
                            model, base_url, api_key,
                            temperature=0.0,
                        ),
                        attempts=attempts,
                    )
                    normalized = normalize_response(parse_model_json(retry_content), candidate_ids)
                    selected_texts = [
                        evidence_by_id[evidence_id]["text"]
                        for evidence_id in normalized["selected_evidence_ids"]
                        if evidence_id in evidence_by_id
                    ]
                    allowed_texts = [query["instruction"], query["query"]] + selected_texts
                    unsupported = unsupported_literals(normalized["answer"], allowed_texts)
                    policy_violations = response_policy_violations(normalized, query_text)
                passed = (
                    normalized["route"] != "reject"
                    and not unsupported
                    and not policy_violations
                )
                record = {
                    "sample_id": "rewrite_{}".format(query_id),
                    "query_id": query_id,
                    "route": normalized["route"],
                    "selected_evidence_ids": normalized["selected_evidence_ids"],
                    "invalid_selected_evidence_ids": normalized["invalid_selected_evidence_ids"],
                    "unexpected_selected_evidence_ids": normalized["unexpected_selected_evidence_ids"],
                    "question": normalize_document(query["instruction"] + "\n" + query["query"]),
                    "answer": normalized["answer"],
                    "reason": normalized["reason"],
                    "unsupported_literals": unsupported,
                    "policy_violations": policy_violations,
                    "passed": passed,
                    "split": split,
                    "is_mock": provider == "mock",
                    "model": model if provider == "api" else "mock",
                    "input_hash": input_hash,
                }
                if passed:
                    output.append(record)
                    existing_by_query[query_id] = record
                    route_counts[record["route"]] += 1
                    accepted_count += 1
                else:
                    rejected.append(record)
                new_count += 1
            except Exception as exc:
                rejected.append({
                    "query_id": query_id,
                    "split": split,
                    "error": "generation_error",
                    "detail": str(exc)[:1000],
                })
                if args.fail_fast:
                    raise
            finally:
                attempted_count += 1
                if attempted_count % save_every == 0:
                    write_jsonl(output_path, output)
                    write_jsonl(rejected_dir / ("rewrite_{}.jsonl".format(split)), rejected)
                progress.update(1)
                progress.set_postfix(
                    status="done", accepted=accepted_count, rejected=len(rejected),
                    refresh=False,
                )
                if args.sleep:
                    time.sleep(args.sleep)
        progress.close()
        write_jsonl(output_path, output)
        write_jsonl(rejected_dir / ("rewrite_{}.jsonl".format(split)), rejected)
        report["splits"][split] = {
            "output": len(output),
            "new": new_count,
            "attempted": attempted_count,
            "rejected": len(rejected),
            "routes": dict(route_counts),
        }
        print("{}: output={}, new={}, rejected={}".format(split, len(output), new_count, len(rejected)))
    write_json(generated_dir / "rewrite_report.json", report)


if __name__ == "__main__":
    main()
