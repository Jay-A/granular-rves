from __future__ import annotations

import json
from pathlib import Path


REPORT_TEMPLATE = Path("docs/report/main.tex")
BUILD_INFO = Path("build_info.json")
OUTPUT = Path("_build/report/main.tex")


def populate_report_fields(
    build_info_path: Path,
    report_template: Path,
    output: Path,
) -> None:
    """Populate report metadata fields from ``build_info.json``.

    Reads build metadata from the specified JSON file, substitutes the
    corresponding ``__MACRO__`` fields in the LaTeX report template, and
    writes the populated report to the specified output path.

    The source template is never modified.

    Parameters
    ----------
    build_info_path
        Path to the JSON file containing build metadata.
    report_template
        Path to the LaTeX report template.
    output
        Path where the populated LaTeX file will be written.

    Raises
    ------
    KeyError
        If a required field is missing from ``build_info.json``.
    """
    with build_info_path.open("r", encoding="utf-8") as f:
        build_info = json.load(f)

    template = report_template.read_text(encoding="utf-8")

    replacements = {
        "__REPOSITORY__": build_info["repository"],
        "__SOURCE_COMMIT__": build_info["source_commit"],
        "__GENERATED__": build_info["generated"],
        "__AUTHOR__": build_info["author"],
        "__STATUS__": build_info["status"],
        "__TESTS_PASSED__": str(build_info["tests_passed"]),
        "__TESTS_TOTAL__": str(build_info["tests_total"]),
        "__BUILD_STATUS__": build_info["build"],
    }

    rendered = template

    for token, value in replacements.items():
        rendered = rendered.replace(token, value)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


def main() -> None:
    """Run report-field population."""
    populate_report_fields(
        build_info_path=BUILD_INFO,
        report_template=REPORT_TEMPLATE,
        output=OUTPUT,
    )


if __name__ == "__main__":
    main()

