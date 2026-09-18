#!/usr/bin/env python3

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
STATUS_FILE = ROOT / ".github" / "status.json"
PYTEST_XML = ROOT / ".github" / ".pytest-results.xml"


def main() -> int:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--junitxml",
            str(PYTEST_XML),
        ],
        cwd=ROOT,
    )

    if result.returncode != 0:
        print("Tests failed; status.json was not updated.")
        return result.returncode


    tree = ET.parse(PYTEST_XML)
    testsuites = tree.getroot()

    testsuite = testsuites.find("testsuite")
    if testsuite is None:
        raise RuntimeError("Could not find pytest testsuite in JUnit XML.")

    total = int(testsuite.attrib.get("tests", 0))
    failures = int(testsuite.attrib.get("failures", 0))
    errors = int(testsuite.attrib.get("errors", 0))
    skipped = int(testsuite.attrib.get("skipped", 0))
    passed = total - failures - errors - skipped

    test_results = []

    for testcase in testsuite.findall("testcase"):
        name = testcase.attrib.get("name", "")
        classname = testcase.attrib.get("classname", "")

        if testcase.find("failure") is not None:
            status = "failed"
        elif testcase.find("error") is not None:
            status = "error"
        elif testcase.find("skipped") is not None:
            status = "skipped"
        else:
            status = "passed"

        test_results.append(
            {
                "name": name,
                "file": classname,
                "status": status,
            }
        )

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()

    status = {
        "commit": commit,
        "tests": {
            "summary": {
                "status": "passed",
                "total": total,
                "passed": passed,
                "failed": failures + errors,
                "skipped": skipped,
            },
            "pytest_tests": test_results,
        },
    }

    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)

    STATUS_FILE.write_text(
        json.dumps(status, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Tests passed: {passed}/{total}")
    print(f"Updated {STATUS_FILE}")

    PYTEST_XML.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


