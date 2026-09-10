import json
from pathlib import Path

import pytest

from scripts.populate_report_fields import populate_report_fields


def test_populate_report_fields(tmp_path: Path) -> None:
    """Populate all report fields from build metadata."""

    build_info = {
        "repository": "jay-a/granular-rves",
        "source_commit": "abc1234",
        "generated": "2026-09-10T12:45:00Z",
        "author": "Jay M. Appleton",
        "status": "Early development",
        "tests_passed": 7,
        "tests_total": 7,
        "build": "passed",
    }

    build_info_path = tmp_path / "build_info.json"
    report_template = tmp_path / "main.tex"
    output = tmp_path / "_build" / "main.tex"

    build_info_path.write_text(
        json.dumps(build_info),
        encoding="utf-8",
    )

    template = r"""
\title{Granular RVEs}

Repository: __REPOSITORY__

Source commit: __SOURCE_COMMIT__

Generated: __GENERATED__

Author: __AUTHOR__

Status: __STATUS__

Tests: __TESTS_PASSED__ / __TESTS_TOTAL__

Build: __BUILD_STATUS__
"""

    report_template.write_text(template, encoding="utf-8")

    populate_report_fields(
        build_info_path=build_info_path,
        report_template=report_template,
        output=output,
    )

    rendered = output.read_text(encoding="utf-8")

    assert "jay-a/granular-rves" in rendered
    assert "abc1234" in rendered
    assert "2026-09-10T12:45:00Z" in rendered
    assert "Jay M. Appleton" in rendered
    assert "Early development" in rendered
    assert "7 / 7" in rendered
    assert "passed" in rendered

    assert "__REPOSITORY__" not in rendered
    assert "__SOURCE_COMMIT__" not in rendered
    assert "__GENERATED__" not in rendered
    assert "__AUTHOR__" not in rendered
    assert "__STATUS__" not in rendered
    assert "__TESTS_PASSED__" not in rendered
    assert "__TESTS_TOTAL__" not in rendered
    assert "__BUILD_STATUS__" not in rendered


def test_populate_report_fields_does_not_modify_template(
    tmp_path: Path,
) -> None:
    """The source LaTeX template remains unchanged."""

    build_info = {
        "repository": "jay-a/granular-rves",
        "source_commit": "abc1234",
        "generated": "2026-09-10T12:45:00Z",
        "author": "Jay M. Appleton",
        "status": "Early development",
        "tests_passed": 7,
        "tests_total": 7,
        "build": "passed",
    }

    build_info_path = tmp_path / "build_info.json"
    report_template = tmp_path / "main.tex"
    output = tmp_path / "_build" / "main.tex"

    build_info_path.write_text(
        json.dumps(build_info),
        encoding="utf-8",
    )

    template = "Source commit: __SOURCE_COMMIT__\n"
    report_template.write_text(template, encoding="utf-8")

    original_template = report_template.read_text(encoding="utf-8")

    populate_report_fields(
        build_info_path=build_info_path,
        report_template=report_template,
        output=output,
    )

    assert report_template.read_text(encoding="utf-8") == original_template

