from __future__ import annotations

import argparse
from pathlib import Path

from clinical_world_model.metrics import evaluate_baselines, render_baseline_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline policies on synthetic workflow scenarios."
    )
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--output", default="reports/baseline_results.md")
    args = parser.parse_args()

    metrics, trajectories_by_policy = evaluate_baselines(
        count=args.count,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    report = render_baseline_report(
        metrics=metrics,
        trajectories_by_policy=trajectories_by_policy,
        count=args.count,
        seed=args.seed,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote baseline report to {output_path}")


if __name__ == "__main__":
    main()

