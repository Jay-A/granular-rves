import json
from pathlib import Path

import pytest

from scripts.generate_build_info import generate_build_info


def test_generate_build_info(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate build metadata from pytest JUnit XML."""

    monkeypatch.setattr(
        "scripts.generate_build_info.get_source_commit",
        lambda: "abc1234",
    )

    pytest_xml = tmp_path / "pytest.xml"
    pytest_xml.write_text(
        """\
<testsuites>
    <testsuite name="tests" tests="7" failures="0" errors="0" skipped="0">
        <testcase classname="tests" name="test_one" />
        <testcase classname="tests" name="test_two" />
        <testcase classname="tests" name="test_three" />
        <testcase classname="tests" name="test_four" />
        <testcase classname="tests" name="test_five" />
        <testcase classname="tests" name="test_six" />
        <testcase classname="tests" name="test_seven" />
    </testsuite>
</testsuites>
""",
        encoding="utf-8",
    )

    output = tmp_path / "build_info.json"

    generate_build_info(
        output=output,
        pytest_xml=pytest_xml,
    )

    assert output.exists()

    build_info = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert build_info["repository"] == "jay-a/granular-rves"
    assert build_info["source_commit"] == "abc1234"
    assert build_info["author"] == "Jay M. Appleton"
    assert build_info["status"] == "Early development"
    assert build_info["tests_passed"] == 7
    assert build_info["tests_total"] == 7
    assert build_info["build"] == "passed"
    assert build_info["generated"]


def test_generate_build_info_records_failed_and_skipped_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record test counts reported by pytest."""

    monkeypatch.setattr(
        "scripts.generate_build_info.get_source_commit",
        lambda: "abc1234",
    )

    pytest_xml = tmp_path / "pytest.xml"
    pytest_xml.write_text(
        """\
<testsuites>
    <testsuite name="tests" tests="8" failures="1" errors="1" skipped="2">
        <testcase classname="tests" name="passed_one" />
        <testcase classname="tests" name="passed_two" />
        <testcase classname="tests" name="passed_three" />
        <testcase classname="tests" name="passed_four" />
        <testcase classname="tests" name="failed">
            <failure />
        </testcase>
        <testcase classname="tests" name="errored">
            <error />
        </testcase>
        <testcase classname="tests" name="skipped_one">
            <skipped />
        </testcase>
        <testcase classname="tests" name="skipped_two">
            <skipped />
        </testcase>
    </testsuite>
</testsuites>
""",
        encoding="utf-8",
    )

    output = tmp_path / "build_info.json"

    generate_build_info(
        output=output,
        pytest_xml=pytest_xml,
    )

    build_info = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert build_info["tests_passed"] == 4
    assert build_info["tests_total"] == 8


def test_generate_build_info_rejects_missing_pytest_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise an error when the pytest JUnit XML report is missing."""

    monkeypatch.setattr(
        "scripts.generate_build_info.get_source_commit",
        lambda: "abc1234",
    )

    pytest_xml = tmp_path / "missing.xml"
    output = tmp_path / "build_info.json"

    with pytest.raises(
        FileNotFoundError,
        match="JUnit XML report not found",
    ):
        generate_build_info(
            output=output,
            pytest_xml=pytest_xml,
        )
