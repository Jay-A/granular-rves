import json
from pathlib import Path

from scripts.generate_build_info import generate_build_info


def test_generate_build_info(tmp_path: Path) -> None:
    """Generate build metadata from committed test status."""
    status_json = tmp_path / "status.json"
    status_json.write_text(
        """
        {
          "tests": {
            "summary": {
              "status": "passed",
              "total": 7,
              "passed": 7,
              "failed": 0,
              "skipped": 0
            },
            "pytest_tests": []
          }
        }
        """,
        encoding="utf-8",
    )

    output = tmp_path / "build_info.json"

    generate_build_info(
        output=output,
        status_json=status_json,
        source_commit="abc1234",
        generated="2026-09-17T00:30:00Z",
    )

    assert output.exists()

    build_info = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert build_info["repository"] == "jay-a/granular-rves"
    assert build_info["source_commit"] == "abc1234"
    assert build_info["generated"] == "2026-09-17T00:30:00Z"
    assert build_info["author"] == "Jay M. Appleton"
    assert build_info["status"] == "Early development"
    assert build_info["tests_passed"] == 7
    assert build_info["tests_total"] == 7
    assert build_info["build"] == "passed"


def test_generate_build_info_records_failed_and_skipped_tests(
    tmp_path: Path,
) -> None:
    """Record test counts from committed test status."""
    status_json = tmp_path / "status.json"
    status_json.write_text(
        """
        {
          "tests": {
            "summary": {
              "status": "failed",
              "total": 8,
              "passed": 4,
              "failed": 2,
              "skipped": 2
            },
            "pytest_tests": []
          }
        }
        """,
        encoding="utf-8",
    )

    output = tmp_path / "build_info.json"

    generate_build_info(
        output=output,
        status_json=status_json,
        source_commit="abc1234",
        generated="2026-09-17T00:30:00Z",
    )

    build_info = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert build_info["tests_passed"] == 4
    assert build_info["tests_total"] == 8


def test_generate_build_info_rejects_missing_status_file(
    tmp_path: Path,
) -> None:
    """Raise an error when the test status JSON is missing."""
    status_json = tmp_path / "missing.json"
    output = tmp_path / "build_info.json"

    try:
        generate_build_info(
            output=output,
            status_json=status_json,
            source_commit="abc1234",
            generated="2026-09-17T00:30:00Z",
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError(
            "generate_build_info() did not raise FileNotFoundError"
        )

