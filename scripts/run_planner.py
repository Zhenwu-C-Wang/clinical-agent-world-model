from __future__ import annotations

import argparse
from pathlib import Path

from clinical_world_model.planner import (
    evaluate_planner_comparison,
    render_planner_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a world-model lookahead planner on synthetic workflows."
    )
    parser.add_argument("--training-data", default="data/synthetic_trajectories.jsonl")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--output", default="reports/planner_results.md")
    args = parser.parse_args()

    metrics, trajectories_by_policy = evaluate_planner_comparison(
        training_path=args.training_data,
        count=args.count,
        seed=args.seed,
        horizon=args.horizon,
        max_steps=args.max_steps,
    )
    report = render_planner_report(
        metrics=metrics,
        trajectories_by_policy=trajectories_by_policy,
        count=args.count,
        seed=args.seed,
        horizon=args.horizon,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote planner report to {output_path}")


if __name__ == "__main__":
    main()

