from __future__ import annotations

import argparse
from pathlib import Path

from clinical_world_model.world_model import (
    render_world_model_report,
    train_and_evaluate_from_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a lightweight synthetic workflow world model."
    )
    parser.add_argument("--input", default="data/synthetic_trajectories.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--output", default="reports/world_model_eval.md")
    args = parser.parse_args()

    evaluation = train_and_evaluate_from_jsonl(
        path=args.input,
        seed=args.seed,
        test_size=args.test_size,
    )
    report = render_world_model_report(evaluation)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote world model evaluation to {output_path}")


if __name__ == "__main__":
    main()

