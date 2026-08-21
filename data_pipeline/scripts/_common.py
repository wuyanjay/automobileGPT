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
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent


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
    "bicycle", "cycling", "mountain bike", "pedal bike",
    "mot test", "garage bill", "legal advice", "insurance claim",
    "spider", "insect", "bug remains", "car wash", "detailing wax",
    "直流母线", "交流滤波器", "值班调控", "换流站", "抽油机", "油井",
    "电厂", "变电站", "绕组接地", "备用排污泵",
]

AUTOMOTIVE_PATTERNS = [
    "汽车", "车辆", "轿车", "客车", "车主", "行驶", "发动机", "变速箱", "变速器",
    "方向机", "制动", "刹车", "空调", "轮胎", "蓄电池", "动力电池", "故障码", "诊断仪",
    "car", "vehicle", "truck", "engine", "transmission", "gearbox", "brake", "steering",
    "battery", "clutch", "wheel", "tire", "tyre", "radiator", "alternator", "starter", "motorcycle",
]


def off_topic_reasons(text):
    lowered = normalize_inline(text).lower()
    return [pattern for pattern in OFF_TOPIC_PATTERNS if pattern in lowered]


def is_automotive(text, source_hint=None):
    lowered = normalize_inline(text).lower()
    if off_topic_reasons(lowered):
        return False
    if any(pattern in lowered for pattern in AUTOMOTIVE_PATTERNS):
        return True
    # Mechanics Stack Exchange is mostly automotive; ambiguous records stay in
    # the pool and can be reviewed rather than being silently discarded.
    return source_hint == "d2"


CONTEXT_PATTERNS = {
    "needs_image": ["image", "picture", "photo", "diagram", "figure", "图中", "下图", "图片"],
    "needs_comment": ["comment", "评论中", "as mentioned in the comment"],
    "needs_other_answer": ["other answer", "another answer", "accepted answer", "previous answer"],
    "needs_external_link": ["http://", "https://", "www."],
}


def context_flags(text):
    lowered = normalize_inline(text).lower()
    flags = []
    for flag, patterns in CONTEXT_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            flags.append(flag)
    return flags


RISK_PATTERNS = {
    "risk_brake": ["brake", "制动", "刹车"],
    "risk_lifting": ["jack stand", "jack up", "举升", "千斤顶"],
    "risk_fuel": ["fuel leak", "gasoline leak", "燃油泄漏", "汽油泄漏"],
    "risk_airbag": ["airbag", "安全气囊"],
    "risk_high_voltage": ["high voltage", "高压电", "高压系统", "动力电池"],
    "risk_bypass": ["bypass", "disable safety", "defeat", "短接安全", "屏蔽安全"],
}


def risk_flags(text):
    lowered = normalize_inline(text).lower()
    flags = []
    for flag, patterns in RISK_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            flags.append(flag)
    return flags


DIAGNOSIS_PATTERNS = [
    "故障诊断", "诊断过程", "维修过程", "处理方法", "故障排除", "原因分析", "检查发现",
    "测量", "读取故障码", "更换后", "排查", "diagnos", "test", "inspect", "measure",
    "replaced", "found", "check", "troubleshoot",
]


def has_diagnosis_process(text):
    lowered = normalize_inline(text).lower()
    hits = sum(1 for pattern in DIAGNOSIS_PATTERNS if pattern in lowered)
    return hits >= 2


FAULT_CASE_PATTERNS = [
    "故障", "无法", "报警", "灯亮", "不工作", "异响", "检修", "诊断", "排除", "维修过程",
]


def is_fault_case(title, text):
    combined = normalize_inline(title + " " + text)
    return any(pattern in combined for pattern in FAULT_CASE_PATTERNS) and has_diagnosis_process(text)


DTC_RE = re.compile(r"(?i)\b[PCBU][0O0-9A-F][0-9A-F]{3,5}\b")
NUMBER_UNIT_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])\d+(?:\.\d+)?\s*(?:mV|V|mA|A|Ω|ohms?|bar|kPa|MPa|psi|rpm|r/min|"
    r"km/h|km|公里|mm|cm|m|L|mL|升|毫升|Nm|N·m|%|℃|°C|度)(?![A-Za-z])"
)
YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?:年|款)?(?!\d)")
CODE_RE = re.compile(r"(?i)\b(?:[A-Z]{1,5}[-/]?\d{2,}[A-Z0-9/-]*|\d+[A-Z]{1,5}\d+[A-Z0-9/-]*)\b")


def normalize_literal(value):
    return re.sub(r"\s+", "", str(value)).upper()


def extract_literals(text):
    text = str(text or "")
    values = []
    for regex in (DTC_RE, NUMBER_UNIT_RE, YEAR_RE, CODE_RE):
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


def api_chat(messages, model, base_url, api_key, temperature=0.1, timeout=120):
    if not api_key:
        raise RuntimeError("Missing API key. Set LLM_API_KEY or OPENAI_API_KEY.")
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        if not endpoint.endswith("/v1"):
            endpoint += "/v1"
        endpoint += "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
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
