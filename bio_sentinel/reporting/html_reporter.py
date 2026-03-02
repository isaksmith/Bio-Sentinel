"""pytest-html configuration hooks for Bio-Sentinel reports."""

from __future__ import annotations

import pytest


def pytest_html_report_title(report):
    """Set a custom title for the HTML report."""
    report.title = "Bio-Sentinel QA Report"


def pytest_html_results_summary(prefix, summary, postfix):
    """Add a project description line to the report summary."""
    prefix.extend(
        [
            "<p>Automated Validation Framework for Conservation AI</p>",
            "<p>Tests cover: unit checks, regression, robustness, "
            "and edge-case integration.</p>",
        ]
    )
