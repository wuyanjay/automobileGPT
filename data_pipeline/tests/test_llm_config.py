#!/usr/bin/env python3
"""Tests for local LLM configuration and endpoint resolution."""

from __future__ import print_function

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from _common import (  # noqa: E402
    chat_completions_endpoint,
    extract_literals,
    load_env_file,
    unsupported_literals,
)


class EnvFileTests(unittest.TestCase):
    def test_loads_values_without_overwriting_existing_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "LLM_BASE_URL=https://api.deepseek.com\n"
                "LLM_MODEL=deepseek-v4-flash\n"
                "LLM_API_KEY=file-secret\n",
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"LLM_API_KEY": "process-secret"}, clear=True):
                loaded = load_env_file(env_path)
                self.assertEqual(loaded, env_path.resolve())
                self.assertEqual(os.environ["LLM_BASE_URL"], "https://api.deepseek.com")
                self.assertEqual(os.environ["LLM_MODEL"], "deepseek-v4-flash")
                self.assertEqual(os.environ["LLM_API_KEY"], "process-secret")


class ChatEndpointTests(unittest.TestCase):
    def test_deepseek_official_base_url_omits_v1(self):
        self.assertEqual(
            chat_completions_endpoint("https://api.deepseek.com"),
            "https://api.deepseek.com/chat/completions",
        )

    def test_generic_openai_base_url_adds_v1(self):
        self.assertEqual(
            chat_completions_endpoint("https://api.openai.com"),
            "https://api.openai.com/v1/chat/completions",
        )

    def test_full_endpoint_is_preserved(self):
        endpoint = "https://example.com/v1/chat/completions"
        self.assertEqual(chat_completions_endpoint(endpoint), endpoint)


class LiteralTests(unittest.TestCase):
    def test_extracts_warranty_distance_oil_and_fuel_grades(self):
        values = extract_literals("质保3年或10万公里，使用5W-30机油和95号汽油")
        normalized = {value.replace(" ", "").upper() for value in values}
        self.assertIn("3年", normalized)
        self.assertIn("10万公里", normalized)
        self.assertIn("5W-30", normalized)
        self.assertIn("95号汽油", normalized)

    def test_new_model_specific_literals_are_rejected(self):
        answer = "建议使用5W-30，质保通常为3年或10万公里。"
        self.assertEqual(
            unsupported_literals(answer, ["请按车辆手册操作"]),
            ["3年", "10万公里", "5W-30"],
        )

    def test_short_and_full_model_years_are_equivalent(self):
        self.assertEqual(
            [],
            unsupported_literals("这是2007年的车型。", ["车辆为07款车型"]),
        )


if __name__ == "__main__":
    unittest.main()
