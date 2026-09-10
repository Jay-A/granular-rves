from pathlib import Path

import pytest

from scripts.pytest_metadata import read_pytest_metadata


def test_read_pytest_metadata(tmp_path: Path) -> None:
    """Read passed and total test counts from JUnit XML."""

    junitxml = tmp_path / "pytest.xml"

    junitxml.write_text(
        """\
<testsuites>
    <testsuite name="tests" tests="5" failures="0" errors="0" skipped="0">
        <testcase classname="tests" name="test_one" />
        <testcase classname="tests" name="test_two" />
        <testcase classname="tests" name="test_three" />
        <testcase classname="tests" name="test_four" />
        <testcase classname="tests" name="test_five" />
    </testsuite>
</testsuites>
""",
        encoding="utf-8",
    )

    metadata = read_pytest_metadata(junitxml)

    assert metadata == {
        "tests_passed": 5,
        "tests_total": 5,
    }


def test_read_pytest_metadata_excludes_failures_and_skips(
    tmp_path: Path,
) -> None:
    """Exclude failed, errored, and skipped tests from passed count."""

    junitxml = tmp_path / "pytest.xml"

    junitxml.write_text(
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

    metadata = read_pytest_metadata(junitxml)

    assert metadata == {
        "tests_passed": 4,
        "tests_total": 8,
    }


def test_read_pytest_metadata_missing_file(
    tmp_path: Path,
) -> None:
    """Raise an error when the JUnit XML report is missing."""

    junitxml = tmp_path / "missing.xml"

    with pytest.raises(
        FileNotFoundError,
        match="JUnit XML report not found",
    ):
        read_pytest_metadata(junitxml)
