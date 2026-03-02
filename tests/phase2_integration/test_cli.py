"""Phase 2 — CLI smoke tests.

Verifies that the ``bio-sentinel`` CLI can be invoked programmatically
and produces the expected output files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bio_sentinel.cli import main


class TestCLICompare:
    def test_compare_mock_model(self, tmp_path):
        """The compare command should produce a valid JSON report."""
        out = tmp_path / "report.json"
        main([
            "compare",
            "--models", "mock",
            "--output", str(out),
            "--severity", "0.5",
            "--threshold", "0.40",
        ])
        assert out.exists()
        data = json.loads(out.read_text())
        assert "models" in data
        assert len(data["models"]) == 1
        # baseline + 4 distortions
        assert len(data["models"][0]["conditions"]) == 5

    def test_compare_multiple_mocks(self, tmp_path):
        """Multiple model keys should produce multiple model entries."""
        out = tmp_path / "multi.json"
        main([
            "compare",
            "--models", "mock,mock",
            "--output", str(out),
        ])
        data = json.loads(out.read_text())
        assert len(data["models"]) == 2


class TestCLIListModels:
    def test_list_models(self, capsys):
        """The list-models command should print available keys."""
        main(["list-models"])
        captured = capsys.readouterr()
        assert "mock" in captured.out.lower()
        assert "mdv5" in captured.out.lower()
