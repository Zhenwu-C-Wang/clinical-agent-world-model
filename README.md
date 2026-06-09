# Clinical Agent World Model

[![CI](https://github.com/Zhenwu-C-Wang/clinical-agent-world-model/actions/workflows/ci.yml/badge.svg)](https://github.com/Zhenwu-C-Wang/clinical-agent-world-model/actions/workflows/ci.yml)

Synthetic hospital workflow world model for tool-using clinical AI agents, with state-transition prediction, safety-risk forecasting, audit-completeness tracking, delay estimation, and lookahead planning.

## Problem

Clinical AI agents that use tools can fail through workflow mistakes: missing context, unsupported evidence, unsafe chart writeback, missing clinician review, overbroad PHI exposure, or scope escalation. This project builds a synthetic environment where those workflow risks can be measured before moving to learned world models and planning.

## Safety Scope

- Synthetic data only.
- No PHI is used, requested, stored, or generated.
- Not medical advice.
- Not a clinical decision support system.
- Not intended for diagnosis, treatment, or real clinical deployment.

## Approach

The project models a synthetic hospital workflow as:

- `ClinicalState`: current workflow stage, context availability, note/order draft status, review status, audit events, task progress, and elapsed delay.
- `AgentAction`: tool-using agent action such as context retrieval, note generation, order drafting, writeback, or escalation.
- `Transition`: `state + action -> outcome + next_state`.
- `Outcome`: progress delta, delay, audit event, review burden, and safety violations.
- `SafetyViolation`: synthetic risk labels for unsafe writeback, PHI leakage, hallucinated evidence, missing clinician review, and scope escalation.

## Quickstart

Requires Python >=3.10.

```bash
python -m pip install -e ".[dev]"
pytest
```

## Generate Synthetic Trajectories

```bash
python scripts/generate_trajectories.py --count 1000 --seed 42 --output data/synthetic_trajectories.jsonl
```

The generated JSONL contains synthetic workflow trajectories only. It does not contain PHI content; some records include boolean PHI-risk flags for safety evaluation. It is not medical advice.

## Run Baseline Evaluation

```bash
python scripts/run_baselines.py --count 1000 --seed 42 --output reports/baseline_results.md
```

The baseline report compares direct, safety-review, and conservative human-review policies on task success, unsafe action rate, false decline rate, audit completeness, expected delay, and clinician review burden.

## Current Status: M2

- Python package under `src/clinical_world_model`.
- Core schemas implemented in `src/clinical_world_model/schemas.py`.
- Minimal rule-based simulator implemented in `src/clinical_world_model/simulator.py`.
- Five handwritten synthetic workflow examples are in `examples/handwritten_episodes.jsonl`.
- Three generation policies are implemented in `src/clinical_world_model/policies.py`.
- Synthetic trajectory generation is implemented in `scripts/generate_trajectories.py`.
- Baseline metrics and reporting are implemented in `src/clinical_world_model/metrics.py`.
- Baseline results are in `reports/baseline_results.md`.
- The checked-in M1 dataset is `data/synthetic_trajectories.jsonl`.
- CI runs `pytest`.
- Project roadmap is in `reports/roadmap.md`.

## M0 Schema

- `ClinicalState`
- `AgentAction`
- `Transition`
- `Outcome`
- `SafetyViolation`

## Roadmap

- M0: project skeleton, schema, examples, minimal simulator, pytest, CI.
- M1: rule-based hospital workflow simulator and 1,000+ synthetic trajectories.
- M2: baseline policies and metrics.
- M3: learned transition/risk world model.
- M4: 3-step lookahead planner.
- M5: portfolio release v0.1 with final report and release packaging.

## Limitations

The simulator is intentionally simplified. It is useful for agent-safety engineering practice, evaluation design, and world-model prototyping, but it does not represent real clinical operations or real patient care.
