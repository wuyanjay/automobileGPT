#!/usr/bin/env python3
"""Shared helpers for the lightweight V4 automotive data pipeline.

The module intentionally depends only on the Python standard library.  GPU/model
dependencies are imported lazily by the scripts that need them.
"""

from __future__ import print_function

import hashlib
import json
import os
import re
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent


def load_env_file(env_path=None, override=False):
    """Load simple KEY=VALUE settings from data_pipeline/.env.

    Existing process environment variables win by default, so Colab secrets or
    explicitly configured shell variables can still override the local file.
    The parser intentionally uses only the Python standard library.
    """
    path = Path(env_path) if env_path else PIPELINE_ROOT / ".env"
    if not path.is_absolute():
        candidate = Path.cwd() / path
        path = candidate if candidate.exists() else PIPELINE_ROOT / path
    if not path.exists():
        return None

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError("Invalid .env line {} in {}".format(line_number, path))
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            raise ValueError("Invalid .env key on line {} in {}".format(line_number, path))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if override or key not in os.environ:
            os.environ[key] = value
    return path.resolve()


def load_config(config_path=None):
    path = Path(config_path) if config_path else PIPELINE_ROOT / "configs" / "pipeline.yaml"
    if not path.is_absolute():
        candidate = Path.cwd() / path
        path = candidate if candidate.exists() else PIPELINE_ROOT / path
    text = path.read_text(encoding="utf-8")
    try:
        config = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError:
            raise RuntimeError(
                "pipeline.yaml is not JSON-compatible YAML and PyYAML is not installed"
            )
        config = yaml.safe_load(text)
    config["_config_path"] = str(path.resolve())
    return config


def pipeline_path(config, *parts):
    """Resolve paths relative to data_pipeline, never to the shell cwd."""
    if not parts:
        return PIPELINE_ROOT
    first = str(parts[0])
    configured = config.get("paths", {}).get(first)
    if configured is not None:
        base = Path(configured)
        rest = parts[1:]
    else:
        base = Path(first)
        rest = parts[1:]
    if not base.is_absolute():
        base = PIPELINE_ROOT / base
    return base.joinpath(*[str(p) for p in rest])


def source_path(config, name):
    path = Path(config["sources"][name])
    return path if path.is_absolute() else PIPELINE_ROOT / path


def ensure_parent(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=False)


def write_json(path, value):
    path = ensure_parent(path)
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, records):
    path = ensure_parent(path)
    lines = [json_dumps(record) for record in records]
    text = "\n".join(lines)
    if text:
        text += "\n"
    _atomic_write_text(path, text)


def _atomic_write_text(path, text):
    path = Path(path)
    ensure_parent(path)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        Path(tmp_name).replace(path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def read_jsonl(path, strict=True):
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception as exc:
                if strict:
                    raise ValueError("Invalid JSONL at {}:{}: {}".format(path, line_number, exc))


def count_jsonl(path):
    return sum(1 for _ in read_jsonl(path))


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(text, length=16):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:length]


def stable_id(prefix, text):
    return "{}_{}".format(prefix, stable_hash(text))


def normalize_inline(text):
    text = unicodedata.normalize("NFKC", str(text or ""))
    return re.sub(r"\s+", " ", text).strip()


def normalize_document(text):
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    compact = []
    previous_blank = False
    for line in lines:
        is_blank = not line
        if is_blank and previous_blank:
            continue
        compact.append(line)
        previous_blank = is_blank
    return "\n".join(compact).strip()


def stable_split(key, train_pct=90, validation_pct=5):
    bucket = int(hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < train_pct:
        return "train"
    if bucket < train_pct + validation_pct:
        return "validation"
    return "test"


POWERTRAIN_PATTERNS = [
    ("phev", ["phev", "插电混动", "插电式混合动力", "plug-in hybrid"]),
    ("bev", ["bev", "纯电", "电动车", "electric vehicle", "ev battery", "动力电池", "充电桩"]),
    ("hev", ["hev", "混合动力", "混动车", "hybrid", "ima"]),
    ("ice", ["燃油车", "汽油机", "柴油机", "gasoline", "diesel", "spark plug", "火花塞"]),
]


SYSTEM_PATTERNS = {
    "charging": ["充电", "充电桩", "charger", "charging", "on-board charger", "obc"],
    "battery": ["动力电池", "高压电池", "蓄电池", "battery", "bms", "soc", "cell voltage"],
    "brake": ["制动", "刹车", "abs", "brake", "caliper", "rotor"],
    "steering": ["转向", "方向机", "助力泵", "eps", "steering", "tie rod"],
    "transmission": ["变速箱", "变速器", "离合器", "档位", "换挡", "transmission", "gearbox", "clutch"],
    "hvac": ["空调", "制冷", "暖风", "压缩机", "冷凝器", "hvac", "a/c", "air conditioning"],
    "engine": ["发动机", "引擎", "点火", "喷油", "怠速", "机油", "engine", "misfire", "injector", "oil pressure"],
    "electrical": ["can总线", "can通信", "线束", "继电器", "保险丝", "短路", "断路", "wiring", "relay", "fuse", "electrical"],
    "chassis": ["悬挂", "减震", "车轮", "轮胎", "轴承", "底盘", "suspension", "wheel bearing", "tire", "tyre"],
    "body": ["车门", "车窗", "天窗", "雨刮", "仪表", "车灯", "door", "window", "sunroof", "wiper", "headlight"],
}


def classify_powertrain(text):
    lowered = normalize_inline(text).lower()
    for label, patterns in POWERTRAIN_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return label
    return "unknown"


def classify_system(text):
    lowered = normalize_inline(text).lower()
    scores = {}
    for label, patterns in SYSTEM_PATTERNS.items():
        scores[label] = sum(lowered.count(pattern) for pattern in patterns)
    best = max(scores, key=scores.get) if scores else "other"
    return best if scores.get(best, 0) > 0 else "other"


OFF_TOPIC_PATTERNS = [
    "printed circuit board", "pcb", "solder pad", "arduino", "raspberry pi",
    "bicycle", "mountain bike", "pedal bike",
    "garage bill", "legal advice", "insurance claim",
    "insect", "bug remains", "car wash", "detailing wax",
    "直流母线", "交流滤波器", "值班调控", "换流站", "抽油机", "油井",
    "接户线", "配电线路", "电网调度", "电厂", "变电站", "绕组接地", "备用排污泵",
]

AUTOMOTIVE_PATTERNS = [
    "汽车", "车辆", "轿车", "客车", "车主", "行驶", "发动机", "变速箱", "变速器",
    "方向机", "制动", "刹车", "空调", "轮胎", "蓄电池", "动力电池", "故障码", "诊断仪",
    "车门", "车窗", "后备箱", "中控锁", "门锁", "大灯", "前照灯", "雨刮", "雨刷",
    "凸轮轴", "曲轴", "连杆", "轴瓦", "正时", "离合器", "离合踏板", "助力泵",
    "档位", "挡位", "挂挡", "换挡", "脱挡", "底盘", "悬架", "减震器", "车身控制",
    "点火开关", "起动机", "发电机", "火花塞", "喷油器", "节气门", "仪表灯", "里程表",
    "ecu", "bcm", "obd", "can总线", "can通信", "abs", "vin",
    "car", "vehicle", "truck", "engine", "transmission", "gearbox", "brake", "steering",
    "battery", "clutch", "wheel", "tire", "tyre", "radiator", "alternator", "starter", "motorcycle",
    "camshaft", "crankshaft", "timing belt", "timing chain", "headlight", "wiper", "door lock",
]


def _contains_term(text, pattern):
    """Match English tokens/phrases on boundaries and Chinese terms by substring."""
    pattern = str(pattern).lower()
    if re.match(r"^[a-z0-9 ]+$", pattern):
        expression = r"(?<![a-z0-9]){}(?![a-z0-9])".format(re.escape(pattern))
        return re.search(expression, text) is not None
    return pattern in text


def matched_terms(text, patterns):
    lowered = normalize_inline(text).lower()
    return [pattern for pattern in patterns if _contains_term(lowered, pattern)]


def off_topic_reasons(text):
    lowered = normalize_inline(text).lower()
    reasons = matched_terms(lowered, OFF_TOPIC_PATTERNS)
    # A PCB inside a key fob/ECU is still an automotive repair topic. Keep the
    # generic PCB exclusion only when the question has no automotive anchor.
    if "pcb" in reasons and automotive_reasons(lowered):
        reasons = [reason for reason in reasons if reason != "pcb"]
    # "Spider" is also a vehicle model and a clutch/differential component.
    # Only treat it as pest control when the surrounding wording says so.
    pest_spider = re.search(r"\bspiders?\b", lowered) and re.search(
        r"(?i)\b(?:get\s+rid\s+of|remove|kill|poison|fog|infest(?:ed|ation)?|nest(?:ing)?)\b|"
        r"\b(?:inside|within)\s+(?:my\s+|the\s+)?(?:car|vehicle)\b",
        lowered,
    )
    if pest_spider:
        reasons.append("spider")
    return sorted(set(reasons))


def automotive_reasons(text):
    return matched_terms(text, AUTOMOTIVE_PATTERNS)


def classify_automotive_domain(text, source_hint=None, auxiliary_text=""):
    """Return automotive/non_automotive/uncertain without treating unknown as off-topic."""
    primary = normalize_inline(text).lower()
    auxiliary = normalize_inline(auxiliary_text).lower()
    negative = off_topic_reasons(primary)
    positive = automotive_reasons(primary)
    auxiliary_positive = automotive_reasons(auxiliary)
    if negative:
        return "non_automotive"
    if positive or auxiliary_positive:
        return "automotive"
    # Mechanics Stack Exchange is mostly automotive. A question with no
    # explicit non-automotive signal stays in scope instead of being rejected
    # because the answer happened not to repeat a component name.
    if source_hint == "d2":
        return "automotive"
    return "uncertain"


def is_automotive(text, source_hint=None, auxiliary_text=""):
    return classify_automotive_domain(text, source_hint, auxiliary_text) == "automotive"


URL_RE = re.compile(r"(?i)(?:https?://|www\.)\S+")
IMAGE_REFERENCE_RE = re.compile(
    r"(?i)\b(?:images?|pictures?|photos?|diagrams?)\b|\bthe\s+figure\b|\bfigure\s+\d+\b|"
    r"图中|下图|上图|图片|照片|示意图"
)
COMMENT_REFERENCE_RE = re.compile(r"(?i)\bcomments?\b|评论中|评论区")
OTHER_ANSWER_REFERENCE_RE = re.compile(
    r"(?i)\b(?:other|another|accepted|previous)\s+answer\b|其他回答|采纳的回答"
)
SUBSTANTIVE_ANSWER_RE = re.compile(
    r"(?i)\b(?:check|test|inspect|measure|replace|remove|install|because|therefore|voltage|resistance|"
    r"diagnos|fault|engine|battery|brake)\b|检查|检测|测量|更换|拆卸|安装|因为|因此|故障|电压|电阻"
)


def strip_urls(text):
    return normalize_inline(URL_RE.sub(" ", str(text or "")))


def answer_is_substantive(text):
    cleaned = strip_urls(text)
    return len(cleaned) >= 180 or (len(cleaned) >= 90 and SUBSTANTIVE_ANSWER_RE.search(cleaned))


def context_flags(answer, question=""):
    """Separate harmless references from records that truly need missing context."""
    answer = normalize_document(answer)
    question = normalize_inline(question)
    combined = normalize_inline(question + " " + answer)
    flags = []
    substantive = answer_is_substantive(answer)

    image_context = re.sub(
        r"(?i)\bout\s+of\s+the\s+picture\b|\bif\s+you\s+can\s+picture\b|\bpicture\s+(?:this|a)\b",
        " ",
        combined,
    )
    if IMAGE_REFERENCE_RE.search(image_context):
        flags.append("has_image_reference")
        question_has_image = IMAGE_REFERENCE_RE.search(question) is not None
        broad_identification_question = re.search(
            r"(?i)\b(?:what|which)\s+(?:is|are)\s+(?:the\s+)?(?:name\s+of\s+)?(?:this|these)\b|"
            r"\b(?:is|are|can)\b.{0,35}\b(?:this|these)\b|"
            r"\bidentify\b.{0,40}\b(?:part|component|item)\b|"
            r"这是什么|哪个部件|识别.{0,12}(?:部件|零件)",
            question,
        )
        implicit_attachment_identification = re.search(
            r"(?i)\b(?:this|these)\b.{0,45}\b(?:came|fell|broke)\s+off\b|"
            r"\b(?:what|which)\b.{0,35}\b(?:part|component|connector|item|piece|bolt|nut|hose|"
            r"wire|box|container|device)\b.{0,25}\b(?:this|these)\b|"
            r"\bwhat\s+(?:is|are)\s+(?:this|these)\s+(?:part|component|connector|item|piece|box|"
            r"container|device)\b",
            question,
        )
        identification_question = bool(
            implicit_attachment_identification or (question_has_image and broad_identification_question)
        )
        question_image_dependency = question_has_image and re.search(
            r"(?i)\b(?:what|which|identify|called|name|part|component|connector|colour|color|damage|safe|"
            r"where|this|these)\b|什么|哪个|识别|部件|零件|损伤|安全吗|位置",
            question,
        )
        dependent_answer = re.search(
            r"(?i)\b(?:the|this)\s+(?:arrow|circled|pictured)\b|\b(?:shown|pictured)\s+(?:above|below)\b|"
            r"图中(?:箭头|圈出|所示)|箭头所示|圈出的",
            answer,
        )
        strong_image_dependency = re.search(
            r"(?i)\b(?:from|based\s+on|judging\s+from|according\s+to)\s+(?:the\s+|this\s+|your\s+)?"
            r"(?:images?|pictures?|photos?)\b|\blooking\s+at\b.{0,30}\b(?:image|picture|photo)\b|"
            r"\b(?:image|picture)\s+(?:doesn't|does\s+not)\s+show\b|\bfrom\s+what\s+i\s+can\s+see\b|"
            r"\bpicture\s*\d+\b|\bmarked\s+(?:with|as)\b|"
            r"\b(?:combining|using)\b.{0,30}\b(?:image|picture|photo)\b|"
            r"\b(?:image|picture|photo)\b.{0,80}\b(?:that|it|this)\s+looks?\s+like\b|"
            r"根据(?:图|图片)|从图中|图中可见",
            answer,
        )
        illustrative_image_claim = re.search(
            r"(?i)\b(?:image|picture)\s+(?:shows|contains|is|describes|depicts)\b|"
            r"\bin\s+(?:the|this|your|these|first|second|third)\s+(?:image|picture|photo)\b|"
            r"\b(?:image|picture|photo)\s+(?:above|below)\b|\bshown\s+(?:above|below)\b",
            answer,
        )
        if (
            identification_question or question_image_dependency or strong_image_dependency
            or ((dependent_answer or illustrative_image_claim) and not substantive)
        ):
            flags.append("needs_image")

    if URL_RE.search(answer):
        flags.append("has_external_link")
        cleaned = strip_urls(answer)
        directs_to_link = re.search(
            r"(?i)\b(?:see|read|follow|visit|refer to|check)\b.{0,50}\b(?:link|url|website|page)\b|"
            r"\b(?:here(?:'s|\s+is)|this)\b.{0,35}\b(?:video|website|pdf|link|page|article|post|thread|manual|instructions?)\b|"
            r"\b(?:video|website|page|article|post|thread|manual)\b.{0,35}\b(?:shows|tells|lists|explains|describes)\b|"
            r"\bgo\s+to\b.{0,45}\b(?:website|page|link|url)\b|\bif\s+it\s+(?:sounds|looks)\s+like\s+this\b|"
            r"详见链接|参见链接|访问链接",
            answer,
        )
        standalone_actions = len(re.findall(
            r"(?i)\b(?:check|test|inspect|measure|replace|remove|install|diagnos)\w*\b|"
            r"检查|检测|测量|更换|拆卸|安装|诊断",
            cleaned,
        ))
        link_fulfills_request = re.search(
            r"(?i)\b(?:video|website|page|manual|article|post|thread)\b.{0,70}"
            r"\b(?:complete\s+procedure|whole\s+procedure|including\s+the\s+location|"
            r"where\s+it\s+is|obtain\s+the|retrieve\s+the|gives?\s+the\s+answer)\b|"
            r"\bonly\s+answering\s+where\b.{0,120}\bvideo\b",
            answer,
        )
        if (
            len(cleaned) < 60 or link_fulfills_request
            or (directs_to_link and not substantive and len(cleaned) < 300 and standalone_actions < 2)
        ):
            flags.append("needs_external_link")

    if COMMENT_REFERENCE_RE.search(answer):
        flags.append("has_comment_reference")
        depends_on_comment = re.search(
            r"(?i)\b(?:see|read|check|refer to)\b.{0,40}\bcomments?\b|"
            r"\bfollowing\s+(?:the\s+)?link\s+in\s+(?:this|the)\s+comment\b|"
            r"as\s+(?:described|explained)\s+in\s+(?:the\s+)?comments?|详见评论|参见评论",
            answer,
        )
        if depends_on_comment and not substantive:
            flags.append("needs_comment")

    if OTHER_ANSWER_REFERENCE_RE.search(answer):
        flags.append("has_other_answer_reference")
        depends_on_answer = re.search(
            r"(?i)\b(?:see|read|refer to)\b.{0,50}\b(?:other|another|accepted|previous)\s+answer\b|"
            r"答案同(?:上|其他回答)|详见其他回答",
            answer,
        )
        if depends_on_answer and not substantive:
            flags.append("needs_other_answer")

    return sorted(set(flags))


RISK_PATTERNS = {
    "risk_brake": ["brake", "制动", "刹车"],
    "risk_lifting": ["jack stand", "jack up", "举升", "千斤顶"],
    "risk_fuel": ["fuel leak", "gasoline leak", "燃油泄漏", "汽油泄漏"],
    "risk_airbag": ["airbag", "安全气囊"],
    "risk_high_voltage": ["high voltage", "高压电", "高压系统", "动力电池"],
}

BYPASS_ACTION_RE = re.compile(r"(?i)\b(?:bypass|disable|defeat|override|jump)\b|绕过|禁用|屏蔽|短接|拆除")
BYPASS_SAFETY_OBJECT_RE = re.compile(
    r"(?i)\b(?:safety\s+(?:system|switch|interlock)|interlock|speed\s+limiter|governor|airbag|"
    r"emissions?\s+(?:system|control)|parking\s+brake|seat\s*belt|video\s+lock|immobilizer|"
    r"high[- ]voltage\s+interlock)\b|安全装置|安全联锁|速度限制|限速器|驻车制动|安全气囊|"
    r"排放控制|高压互锁"
)
BYPASS_INSTRUCTION_RE = re.compile(
    r"(?i)\b(?:how\s+(?:to|can|do)|you\s+can|you\s+should|try\s+to|need\s+to|must|as\s+follows|"
    r"instructions?|procedure|simply)\b|如何|怎么|可以|可将|需要|步骤"
)


def _has_bypass_pair(text, max_distance=55):
    actions = list(BYPASS_ACTION_RE.finditer(text))
    objects = list(BYPASS_SAFETY_OBJECT_RE.finditer(text))
    for action in actions:
        for obj in objects:
            if abs(action.start() - obj.start()) > max_distance:
                continue
            middle = text[min(action.end(), obj.end()):max(action.start(), obj.start())]
            if not re.search(r"[.!?;。！？；\n]", middle):
                return True
    return False


def _has_instructional_bypass_pair(text, max_distance=55):
    actions = list(BYPASS_ACTION_RE.finditer(text))
    objects = list(BYPASS_SAFETY_OBJECT_RE.finditer(text))
    for action in actions:
        for obj in objects:
            if abs(action.start() - obj.start()) > max_distance:
                continue
            pair_start = min(action.start(), obj.start())
            pair_end = max(action.end(), obj.end())
            middle = text[min(action.end(), obj.end()):max(action.start(), obj.start())]
            if re.search(r"[.!?;。！？；\n]", middle):
                continue
            prefix = text[:pair_start]
            suffix = text[pair_end:]
            previous_boundary = max(
                [prefix.rfind(mark) for mark in ".!?;。！？；\n"] + [-1]
            )
            next_offsets = [suffix.find(mark) for mark in ".!?;。！？；\n" if suffix.find(mark) >= 0]
            next_boundary = pair_end + (min(next_offsets) if next_offsets else len(suffix))
            local_context = text[previous_boundary + 1:next_boundary]
            if BYPASS_INSTRUCTION_RE.search(local_context):
                return True
    return False


def risk_flags(text, question=""):
    answer = normalize_document(text).lower()
    question = normalize_inline(question).lower()
    lowered = normalize_inline(question + " " + answer).lower()
    flags = []
    for flag, patterns in RISK_PATTERNS.items():
        if any(_contains_term(lowered, pattern) for pattern in patterns):
            flags.append(flag)
    bypass_text = re.sub(r"屏蔽(?:线|层|电缆)", "", lowered)
    if re.search(r"(?i)\bbypass\b|绕过|屏蔽|短接", bypass_text):
        flags.append("has_bypass_term")
    question_pair = _has_bypass_pair(question) if question else False
    # Blocking is based on explicit bypass intent in the question, or an
    # affirmative bypass procedure in the answer. Mere warnings, negations,
    # and diagnostic mentions stay available with soft risk labels.
    if question_pair or _has_instructional_bypass_pair(answer):
        flags.append("risk_bypass")
    return sorted(set(flags))


PRICE_INTENT_RE = re.compile(r"多少钱|报价|费用|价格|价钱|收费|工时费|预算|大概(?:要|得)多少")
PURCHASE_INTENT_RE = re.compile(r"值得购买|值得拥有|能买吗|值得买吗|值不值得买|买哪款|选购|购车推荐|落地价")
REPAIR_PROBLEM_RE = re.compile(
    r"故障|异响|抖动|顿挫|失灵|报警|故障灯|故障码|报码|无法|不能|不启动|不着车|"
    r"不制冷|不制热|不工作|漏油|漏液|漏水|漏电|亏电|熄火|冒烟|高温|过热|烧机油|"
    r"磨损|松动|断裂|卡滞|怎么修|怎么检查|怎么检测|怎么排查|什么原因|原因是什么|怎么办|"
    r"是否正常|保养周期|多久保养|多久更换|异常|漏风|塌陷|怎么回事|咋回事|什么毛病|"
    r"哪里的问题|是什么问题|是不是.{0,8}问题|是否有.{0,8}问题|有没有.{0,8}问题|"
    r"是不是.{0,12}(?:坏了|烂了|故障|损坏)|"
    r"严重问题|安全问题|怎么维修|如何维修|怎么修复|如何修复|能不能修|能否修|能修好|"
    r"好修吗|可以维修吗|是否该换|是否需要(?:换|更换)|需不需要(?:换|更换)|要不要(?:换|更换)|"
    r"需要(?:换|更换).{0,5}吗|"
    r"怎么样确定是否|如何确定是否|如何检查|有什么方法|其他方法|有必要(?:换|更换|修|吗)|"
    r"保养(?:哪些|什么)(?:东西|项目)?|(?:保养|滤芯|节气门).{0,25}(?:影响|周期|项目)|"
    r"多少公里.{0,6}(?:换|更换|保养)|啸叫|噪声|"
    r"(?:换|更换).{0,25}(?:要|需要).{0,12}吗|"
    r"掉电.{0,8}(?:快|异常|离谱)|充电功率.{0,8}(?:低|上不去)|充不满"
)


def classify_repair_intent(text):
    text = normalize_inline(text)
    has_price = PRICE_INTENT_RE.search(text) is not None
    has_purchase = PURCHASE_INTENT_RE.search(text) is not None
    has_repair_problem = REPAIR_PROBLEM_RE.search(text) is not None
    if has_purchase and has_repair_problem:
        return "mixed_repair_purchase"
    if has_price and has_repair_problem:
        return "mixed_repair_price"
    if has_purchase:
        return "purchase_only"
    if has_price:
        return "price_only"
    if has_repair_problem:
        return "repair_only"
    return "other"


DIAGNOSIS_HEADING_RE = re.compile(
    r"(?i)故障诊断|诊断过程|诊断排除|诊断与排除|分析诊断|故障分析及排除|故障分析与排除|"
    r"维修过程|维修方案|处理方法|解决措施|故障排除|检修方法|检修步骤|排除方法|"
    r"diagnos(?:is|tic)|troubleshoot(?:ing)?|diagnostic procedure"
)
DIAGNOSIS_ACTION_RE = re.compile(
    r"(?i)检查|检测|测量|读取|拆检|拆开|拆下|拆卸|松下|调整|清洗|紧固|修复|更换|换掉|"
    r"连接|断开|拔掉|插入|插上|观察|启动|处理|清理|初始化|激活|清除|设定|复位|编程|匹配|"
    r"试车|路试|试验|验证|排查|check(?:ed|ing)?|test(?:ed|ing)?|inspect(?:ed|ing)?|"
    r"measure(?:d|ment|ing)?|read\s+(?:the\s+)?(?:code|dtc)|remove(?:d)?|replace(?:d)?|road\s+test"
)
DIAGNOSIS_FINDING_RE = re.compile(
    r"(?i)发现|显示|测得|读得|表明|确认|判断|可确定|可判断|说明|查明|"
    r"found|showed|indicated|confirmed|determined"
)
DIAGNOSIS_OUTCOME_RE = re.compile(
    r"(?i)恢复正常|故障(?:已|被|得到|彻底)?排除(?:[。.!！]|$)|问题解决|"
    r"(?:故障|问题|症状|异响)(?:被)?消除|"
    r"故障解除|症状解除|异响消失|故障消失|症状消失|不再出现|不再发生|未再出现|未再发生|"
    r"不再异响|不再响|"
    r"正常工作|路试正常|试车正常|运行正常|resolved|fixed|returned\s+to\s+normal|no\s+longer"
)
DIAGNOSIS_MEASUREMENT_RE = re.compile(
    r"(?i)测量|测得|测隙|电压|电阻|电流|压力|间隙|故障码|数据流|"
    r"measure(?:d|ment|ing)?|voltage|resistance|current|pressure|clearance|dtc|fault\s+code"
)
NUMBERED_STEP_RE = re.compile(
    r"(?m)(?:^|\n)\s*(?:\(?\d{1,2}\)?[、.．)]|[（(]\d{1,2}[）)]|步骤[一二三四五六七八九十\d]+)|"
    r"(?<![\d.])(?:[（(]\d{1,2}[）)]|\d{1,2}[、．])(?!\d)"
)


def diagnosis_process_signals(text):
    text = normalize_document(text)
    return {
        "heading_count": len(DIAGNOSIS_HEADING_RE.findall(text)),
        "step_count": len(NUMBERED_STEP_RE.findall(text)),
        "action_count": len(DIAGNOSIS_ACTION_RE.findall(text)),
        "finding_count": len(DIAGNOSIS_FINDING_RE.findall(text)),
        "outcome_count": len(DIAGNOSIS_OUTCOME_RE.findall(text)),
        "measurement_count": len(DIAGNOSIS_MEASUREMENT_RE.findall(text)),
    }


def has_diagnosis_process(text):
    signals = diagnosis_process_signals(text)
    heading = signals["heading_count"] >= 1
    steps = signals["step_count"] >= 2
    actions = signals["action_count"]
    findings = signals["finding_count"]
    outcomes = signals["outcome_count"]
    measurements = signals["measurement_count"]
    if heading and actions >= 1 and steps:
        return True
    if heading and actions >= 1 and findings >= 1 and outcomes >= 1:
        return True
    if heading and actions >= 2 and (findings >= 1 or outcomes >= 1):
        return True
    if steps and actions >= 2 and (findings >= 1 or outcomes >= 1 or measurements >= 1):
        return True
    if actions >= 2 and findings >= 1 and outcomes >= 1:
        return True
    return actions >= 4 and findings >= 1 and measurements >= 2


D4_CASE_RE = re.compile(r"一辆|一台|该车|本车|进厂|进店|送修|车主反映|故障现象|行驶里程|维修案例|经检查")
D4_MAINTENANCE_RE = re.compile(r"保养|维护|库存电池|存放|定期充电")
UNSAFE_MODIFICATION_RE = re.compile(
    r"(?:短接|屏蔽|绕过|拆除|禁用).{0,45}(?:高压互锁|安全联锁)|"
    r"(?:高压互锁|安全联锁).{0,45}(?:短接|屏蔽|绕过|拆除|禁用)|"
    r"(?:改线|改接|强制接通).{0,60}(?:充电|高压|动力电池)|"
    r"(?:充电|高压|动力电池).{0,60}(?:改线|改接|强制接通)|"
    r"(?i:(?:bypass|disable|defeat).{0,50}high[- ]voltage\s+interlock|"
    r"high[- ]voltage\s+interlock.{0,50}(?:bypass|disable|defeat))"
)


def classify_d4_document(title, text):
    combined = normalize_inline(title + " " + text)
    if has_diagnosis_process(text):
        if D4_CASE_RE.search(combined):
            return "case_evidence"
        return "procedure_evidence"
    if D4_MAINTENANCE_RE.search(normalize_inline(title)):
        return "maintenance_qa"
    return "technical_pt"


def requires_safety_review(text):
    text = normalize_inline(text)
    return bool(UNSAFE_MODIFICATION_RE.search(text))


def is_fault_case(title, text):
    return classify_d4_document(title, text) in ("case_evidence", "procedure_evidence")


DTC_RE = re.compile(r"(?i)\b[PCBU][0O0-9A-F][0-9A-F]{3,5}\b")
NUMBER_UNIT_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])\d+(?:\.\d+)?\s*(?:mV|V|mA|A|Ω|ohms?|bar|kPa|MPa|psi|rpm|r/min|"
    r"km/h|km|公里|mm|cm|m|L|mL|升|毫升|Nm|N·m|%|℃|°C|度|年|个月|月|周|天|"
    r"小时|分钟|秒|万元|元)(?![A-Za-z])"
)
YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?:年|款)?(?!\d)")
SHORT_YEAR_RE = re.compile(r"(?<!\d)\d{2}(?:年|款)(?!\d)")
CODE_RE = re.compile(r"(?i)\b(?:[A-Z]{1,5}[-/]?\d{2,}[A-Z0-9/-]*|\d+[A-Z]{1,5}\d+[A-Z0-9/-]*)\b")
TEN_THOUSAND_DISTANCE_RE = re.compile(r"(?i)(?<![\d.])\d+(?:\.\d+)?\s*万\s*(?:km|公里)")
OIL_GRADE_RE = re.compile(r"(?i)(?<![A-Z0-9])\d{1,2}W[- ]?\d{2}(?![A-Z0-9])")
FUEL_GRADE_RE = re.compile(r"(?<!\d)(?:89|9\d|10[0-2])号(?:汽油)?")


def normalize_literal(value):
    normalized = re.sub(r"\s+", "", str(value)).upper()
    year_match = re.fullmatch(r"(\d{2}|\d{4})(?:年|款)", normalized)
    if year_match:
        digits = year_match.group(1)
        if len(digits) == 2:
            short_year = int(digits)
            if short_year <= 29:
                digits = str(2000 + short_year)
            elif short_year >= 80:
                digits = str(1900 + short_year)
            else:
                return normalized
        return "MODEL_YEAR:{}".format(digits)
    return normalized


def extract_literals(text):
    text = str(text or "")
    values = []
    for regex in (
        DTC_RE, NUMBER_UNIT_RE, YEAR_RE, SHORT_YEAR_RE, CODE_RE,
        TEN_THOUSAND_DISTANCE_RE, OIL_GRADE_RE, FUEL_GRADE_RE,
    ):
        values.extend(match.group(0) for match in regex.finditer(text))
    deduped = []
    seen = set()
    for value in values:
        normalized = normalize_literal(value)
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(value)
    return deduped


def unsupported_literals(answer, allowed_texts):
    allowed = set()
    for text in allowed_texts:
        allowed.update(normalize_literal(value) for value in extract_literals(text))
    unsupported = []
    for value in extract_literals(answer):
        if normalize_literal(value) not in allowed:
            unsupported.append(value)
    return unsupported


def extract_docx_text(path):
    """Extract paragraphs from a docx without requiring python-docx."""
    word_ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(str(path)) as archive:
        xml = archive.read("word/document.xml")
        media_count = sum(
            1 for name in archive.namelist()
            if name.startswith("word/media/") and not name.endswith("/")
        )
    root = ElementTree.fromstring(xml)
    paragraphs = []
    for paragraph in root.iter(word_ns + "p"):
        text = "".join(node.text or "" for node in paragraph.iter(word_ns + "t"))
        text = normalize_inline(text)
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs), media_count


def chunk_text(text, max_chars=6000, min_chars=200):
    text = normalize_document(text)
    if len(text) <= max_chars:
        return [text] if len(text) >= min_chars else []
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = []
    current_len = 0
    for paragraph in paragraphs:
        if current and current_len + len(paragraph) + 1 > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        if len(paragraph) > max_chars:
            for start in range(0, len(paragraph), max_chars):
                piece = paragraph[start:start + max_chars]
                if len(piece) >= min_chars:
                    chunks.append(piece)
            continue
        current.append(paragraph)
        current_len += len(paragraph) + 1
    if current:
        chunks.append("\n".join(current))
    return [chunk for chunk in chunks if len(chunk) >= min_chars]


def parse_model_json(text):
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError("Model output does not contain a JSON object")


def chat_completions_endpoint(base_url):
    """Resolve an OpenAI-compatible base URL to its Chat Completions endpoint."""
    endpoint = str(base_url or "").strip().rstrip("/")
    if not endpoint:
        raise RuntimeError("Missing LLM base URL.")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    if endpoint.endswith("/v1"):
        return endpoint + "/chat/completions"

    parsed = urllib.parse.urlsplit(endpoint)
    # DeepSeek's official OpenAI-format base URL currently omits /v1.
    if parsed.netloc.lower() == "api.deepseek.com" and not parsed.path.rstrip("/"):
        return endpoint + "/chat/completions"
    return endpoint + "/v1/chat/completions"


def api_chat(messages, model, base_url, api_key, temperature=0.1, timeout=120):
    if not api_key:
        raise RuntimeError("Missing API key. Fill LLM_API_KEY in data_pipeline/.env.")
    endpoint = chat_completions_endpoint(base_url)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": "Bearer {}".format(api_key),
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("LLM API HTTP {}: {}".format(exc.code, body[:1000]))
    return result["choices"][0]["message"]["content"]


def retry_call(function, attempts=3, base_delay=2.0):
    last_error = None
    for attempt in range(attempts):
        try:
            return function()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(base_delay * (2 ** attempt))
    raise last_error


def load_by_id(path, id_field):
    return {record[id_field]: record for record in read_jsonl(path)}


def summarize_flags(records):
    counter = Counter()
    for record in records:
        for flag in record.get("flags", []):
            counter[flag] += 1
    return dict(counter)
