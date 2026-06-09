from __future__ import annotations

import argparse

from clinical_world_model.generation import (
    generate_trajectories,
    write_trajectories_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic hospital workflow trajectories."
    )
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--output", default="data/synthetic_trajectories.jsonl")
    args = parser.parse_args()

    trajectories = generate_trajectories(
        count=args.count,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    write_trajectories_jsonl(args.output, trajectories)
    print(f"Wrote {len(trajectories)} trajectories to {args.output}")


if __name__ == "__main__":
    main()

