# Data

This directory is for generated synthetic artifacts.

- `synthetic_trajectories.jsonl` is produced by `python scripts/generate_trajectories.py --count 1000 --seed 42 --output data/synthetic_trajectories.jsonl`.
- Records are synthetic workflow trajectories only.
- No PHI content is used, requested, stored, or generated.
- Some records include boolean PHI-risk flags such as `contains_phi`; these are synthetic labels, not PHI.
- The data is not medical advice and is not suitable for clinical use.
