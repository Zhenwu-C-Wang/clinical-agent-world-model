# Changelog

## Unreleased

- Made world-model calibration reporting stable across scikit-learn wheels by using decision-level probability bins.
- Added `Makefile` release commands for smoke checks, report regeneration, report freshness checks, and release verification.
- Added `scripts/check_reports_fresh.py` to regenerate checked-in reports and fail when committed reports drift from reproducible outputs.
- Added report freshness checking to CI.

## v0.1.2

- Made editable/dev installation reproducible with `python -m pip install -e ".[dev]"`.
- Lazy-loaded heavy top-level exports so `import clinical_world_model` no longer imports the world-model stack eagerly; local smoke time dropped from about 60 seconds to about 0.02 seconds.
- Added packaging smoke coverage for installed imports from outside the repository, without relying on `PYTHONPATH`.
- Expanded CI smoke coverage to run trajectory generation, world-model training, and planner scripts after editable installation.
- Added a top-level 5-minute reproduction path to the README.
