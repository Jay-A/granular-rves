from __future__ import annotations

import argparse
import json
from pathlib import Path

BUILD_INFO = Path("build_info.json")
REPOSITORY = "jay-a/granular-rves"
AUTHOR = "Jay M. Appleton"
STATUS = "Early development"


def generate_build_info(
    output: Path,
    status_json: Path,
    source_commit: str,
    generated: str,
    build_status: str = "passed",
) -> None:
    """Generate repository build metadata.

    Parameters
    ----------
    output
        Path where ``build_info.json`` will be written.
    status_json
        Path to the committed test status JSON file.
    source_commit
        Git commit hash for the source being built.
    generated
        UTC timestamp for when the build metadata was generated.
    build_status
        Status of the documentation and report build.

    Raises
    ------
    FileNotFoundError
        If the status JSON file does not exist.
    KeyError
        If the expected test summary fields are missing.
    """
    status = json.loads(status_json.read_text(encoding="utf-8"))
    summary = status["tests"]["summary"]

    build_info = {
        "repository": REPOSITORY,
        "source_commit": source_commit,
        "generated": generated,
        "author": AUTHOR,
        "status": STATUS,
        "tests_passed": summary["passed"],
        "tests_total": summary["total"],
        "build": build_status,
    }

    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(build_info, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate build metadata for the project report."
    )

    parser.add_argument(
        "--status-json",
        type=Path,
        required=True,
        help="Path to the committed test status JSON file.",
    )

    parser.add_argument(
        "--commit",
        required=True,
        help="Git commit hash for the source being built.",
    )

    parser.add_argument(
        "--timestamp",
        required=True,
        help="UTC timestamp for when the build metadata was generated.",
    )

    parser.add_argument(
        "--build-status",
        default="passed",
        help="Status of the documentation/report build.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=BUILD_INFO,
        help="Path for the generated build metadata JSON.",
    )

    return parser.parse_args()


def main() -> None:
    """Generate build metadata from committed test results."""
    args = parse_args()

    generate_build_info(
        output=args.output,
        status_json=args.status_json,
        source_commit=args.commit,
        generated=args.timestamp,
        build_status=args.build_status,
    )


if __name__ == "__main__":
    main()

