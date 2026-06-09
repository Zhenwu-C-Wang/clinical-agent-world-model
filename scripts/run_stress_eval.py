from __future__ import annotations

import argparse
from pathlib import Path

from clinical_world_model.stress import evaluate_stress, render_stress_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate policies under a harder synthetic stress distribution."
    )
    parser.add_argument("--training-data", default="data/synthetic_trajectories.jsonl")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--output", default="reports/stress_results.md")
    args = parser.parse_args()

    metrics, trajectories_by_policy, summary = evaluate_stress(
        training_path=args.training_data,
        count=args.count,
        seed=args.seed,
        horizon=args.horizon,
        max_steps=args.max_steps,
    )
    report = render_stress_report(
        metrics=metrics,
        trajectories_by_policy=trajectories_by_policy,
        summary=summary,
        seed=args.seed,
        horizon=args.horizon,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote stress evaluation report to {output_path}")


if __name__ == "__main__":
    main()

