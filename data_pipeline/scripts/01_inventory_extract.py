#!/usr/bin/env python3
"""Inventory raw files and extract readable D4 docx documents."""

from __future__ import print_function

import argparse
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from _common import (
    PIPELINE_ROOT,
    extract_docx_text,
    load_config,
    normalize_document,
    pipeline_path,
    sha256_file,
    stable_id,
    write_json,
    write_jsonl,
)


def inspect_json_like(path):
    result = {"records": 0, "parse_errors": 0, "format": None}
    if path.suffix.lower() == ".jsonl":
        result["format"] = "jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                    result["records"] += 1
                except Exception:
                    result["parse_errors"] += 1
        return result
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        result["format"] = "json_array" if isinstance(data, list) else "json_object"
        result["records"] = len(data) if isinstance(data, list) else 1
        return result
    except Exception:
        result["format"] = "jsonl_with_json_extension"
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                    result["records"] += 1
                except Exception:
                    result["parse_errors"] += 1
        return result


def convert_legacy_doc(source, target):
    """Convert one .doc to a work-directory .docx using LibreOffice or Word."""
    target.parent.mkdir(parents=True, exist_ok=True)
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    errors = []
    if soffice:
        try:
            result = subprocess.run(
                [soffice, "--headless", "--convert-to", "docx", "--outdir", str(target.parent), str(source)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
            )
            generated = target.parent / (source.stem + ".docx")
            if result.returncode == 0 and generated.exists():
                if generated != target:
                    generated.replace(target)
                return target, "libreoffice"
            errors.append("LibreOffice exit {}".format(result.returncode))
        except Exception as exc:
            errors.append("LibreOffice: {}".format(exc))

    if os.name == "nt":
        source_literal = str(source.resolve()).replace("'", "''")
        target_literal = str(target.resolve()).replace("'", "''")
        command = (
            "$source='{}';$target='{}';"
            "$word=New-Object -ComObject Word.Application;"
            "$word.Visible=$false;$word.DisplayAlerts=0;"
            "try{{$doc=$word.Documents.Open($source);$doc.SaveAs2($target,16);$doc.Close()}}"
            "finally{{$word.Quit()}}"
        ).format(source_literal, target_literal)
        try:
            if target.exists():
                target.unlink()
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
            )
            if result.returncode == 0 and target.exists():
                return target, "microsoft_word"
            errors.append("Microsoft Word exit {}".format(result.returncode))
        except Exception as exc:
            errors.append("Microsoft Word: {}".format(exc))
    raise RuntimeError("; ".join(errors) or "No LibreOffice or Microsoft Word converter available")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--convert-legacy-doc", action="store_true",
        help="Convert D4 .doc files into data/work via LibreOffice or Microsoft Word",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    raw_dir = pipeline_path(config, "raw")
    work_dir = pipeline_path(config, "work")

    inventory = []
    extension_counts = Counter()
    for path in sorted(item for item in raw_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(PIPELINE_ROOT).as_posix()
        entry = {
            "path": relative,
            "bytes": path.stat().st_size,
            "extension": path.suffix.lower(),
            "sha256": sha256_file(path),
            "temporary": path.name.startswith("~$"),
        }
        extension_counts[entry["extension"] or "<none>"] += 1
        if path.suffix.lower() in (".json", ".jsonl"):
            entry.update(inspect_json_like(path))
        inventory.append(entry)

    d4_dir = PIPELINE_ROOT / config["sources"]["d4_dir"]
    extracted = []
    extraction_errors = []
    legacy_doc = []
    converted_doc = []
    documents = []
    for path in sorted(item for item in d4_dir.iterdir() if item.is_file()):
        if path.name.startswith("~$"):
            continue
        if path.suffix.lower() == ".doc":
            if args.convert_legacy_doc:
                target = work_dir / "converted" / "d4" / (path.stem + ".docx")
                try:
                    converted_path, converter = convert_legacy_doc(path, target)
                    documents.append((converted_path, path))
                    converted_doc.append({"source": path.name, "converter": converter})
                except Exception as exc:
                    legacy_doc.append(path.name)
                    extraction_errors.append({"file": path.name, "stage": "conversion", "error": str(exc)})
            else:
                legacy_doc.append(path.name)
            continue
        if path.suffix.lower() != ".docx":
            continue
        documents.append((path, path))

    for path, source in documents:
        try:
            text, media_count = extract_docx_text(path)
            text = normalize_document(text)
            if not text:
                raise ValueError("empty extracted text")
            title = next((line.strip() for line in text.split("\n") if line.strip()), source.stem)
            extracted.append({
                "document_id": stable_id("d4", source.name),
                "title": title,
                "text": text,
                "source_file": source.relative_to(PIPELINE_ROOT).as_posix(),
                "source_sha256": sha256_file(source),
                "media_count": media_count,
                "char_count": len(text),
            })
        except Exception as exc:
            extraction_errors.append({"file": source.name, "stage": "extraction", "error": str(exc)})

    report = {
        "raw_root": raw_dir.relative_to(PIPELINE_ROOT).as_posix(),
        "file_count": len(inventory),
        "extension_counts": dict(extension_counts),
        "files": inventory,
        "d4": {
            "extracted_docx": len(extracted),
            "converted_legacy_doc": converted_doc,
            "legacy_doc_needs_conversion": legacy_doc,
            "errors": extraction_errors,
        },
    }
    write_json(work_dir / "inventory" / "inventory.json", report)
    write_jsonl(work_dir / "normalized" / "d4_extracted.jsonl", extracted)
    print("Inventory files: {}".format(len(inventory)))
    print("D4 extracted docx: {}".format(len(extracted)))
    if converted_doc:
        print("D4 converted legacy .doc: {}".format(len(converted_doc)))
    if legacy_doc:
        print("D4 .doc files requiring manual conversion: {}".format(", ".join(legacy_doc)))
    if extraction_errors:
        print("D4 extraction errors: {}".format(len(extraction_errors)))


if __name__ == "__main__":
    main()
