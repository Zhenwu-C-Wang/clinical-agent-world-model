from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


REPORT_COMMANDS = {
    "reports/baseline_results.md": [
        "scripts/run_baselines.py",
        "--count",
        "1000",
        "--seed",
        "42",
    ],
    "reports/world_model_eval.md": [
        "scripts/train_world_model.py",
        "--input",
        "data/synthetic_trajectories.jsonl",
        "--seed",
        "42",
    ],
    "reports/planner_results.md": [
        "scripts/run_planner.py",
        "--training-data",
        "data/synthetic_trajectories.jsonl",
        "--count",
        "1000",
        "--seed",
        "42",
        "--horizon",
        "3",
    ],
    "reports/stress_results.md": [
        "scripts/run_stress_eval.py",
        "--training-data",
        "data/synthetic_trajectories.jsonl",
        "--count",
        "500",
        "--seed",
        "99",
        "--horizon",
        "3",
    ],
}


def _run_report(script_args: list[str], output_path: Path) -> None:
    subprocess.run(
        [sys.executable, *script_args, "--output", str(output_path)],
        cwd=REPO_ROOT,
        check=True,
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def check_reports_fresh(keep_generated: bool = False) -> int:
    stale_reports: list[tuple[str, Path]] = []

    with tempfile.TemporaryDirectory(prefix="clinical-world-reports-") as tmp:
        tmp_root = Path(tmp)
        for report_path, script_args in REPORT_COMMANDS.items():
            generated_path = tmp_root / report_path
            generated_path.parent.mkdir(parents=True, exist_ok=True)
            _run_report(script_args, generated_path)

            checked_in_path = REPO_ROOT / report_path
            if _read_text(generated_path) != _read_text(checked_in_path):
                stale_reports.append((report_path, generated_path))

        if stale_reports:
            print("stale reports detected:", file=sys.stderr)
            for report_path, generated_path in stale_reports:
                print(
                    f"- {report_path} differs from regenerated output at {generated_path}",
                    file=sys.stderr,
                )
            if keep_generated:
                print(f"generated reports kept under {tmp_root}", file=sys.stderr)
                input("Press Enter to remove generated reports and exit...")
            return 1

    print(f"fresh=true checked_reports={len(REPORT_COMMANDS)}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate checked-in reports and fail if they are stale."
    )
    parser.add_argument(
        "--keep-generated",
        action="store_true",
        help="Keep the temporary directory open on failure for manual inspection.",
    )
    args = parser.parse_args()
    raise SystemExit(check_reports_fresh(keep_generated=args.keep_generated))


if __name__ == "__main__":
    main()

