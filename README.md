# Clinical Agent World Model

[![CI](https://github.com/Zhenwu-C-Wang/clinical-agent-world-model/actions/workflows/ci.yml/badge.svg)](https://github.com/Zhenwu-C-Wang/clinical-agent-world-model/actions/workflows/ci.yml)

Synthetic hospital workflow world model for tool-using clinical AI agents, with state-transition prediction, safety-risk forecasting, audit-completeness tracking, delay estimation, and lookahead planning.

## Key Results

- Generated `1,000` synthetic hospital workflow trajectories with no PHI content.
- Covered six workflow actions: context retrieval, note generation, note revision, order drafting, draft writeback, and human review escalation.
- Covered five safety risks: unsafe writeback, PHI leakage, hallucinated evidence, missing clinician review, and scope escalation.
- Trained a lightweight random-forest world model to predict next workflow state, safety risk, delay, and audit completeness.
- Reduced unsafe action rate from `0.505` with direct action to `0.000` with 3-step world-model lookahead while improving task success from `0.822` to `1.000` on synthetic scenarios.

## Read the Technical Blog

- [Building a Synthetic World Model for Clinical AI Agents](reports/blog_world_model.md)
- [Portfolio report](reports/portfolio.md)

## Problem

Clinical AI agents that use tools can fail through workflow mistakes: missing context, unsupported evidence, unsafe chart writeback, missing clinician review, overbroad PHI exposure, or scope escalation. This project builds a synthetic environment where those workflow risks can be measured before moving to learned world models and planning.

## Safety Scope

- Synthetic data only.
- No PHI content is used, requested, stored, or generated.
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

## Train World Model

```bash
python scripts/train_world_model.py --input data/synthetic_trajectories.jsonl --seed 42 --output reports/world_model_eval.md
```

The M3 world model trains lightweight random forests to predict next workflow state, safety violation risk, expected action delay, and audit completeness from synthetic `state + action` features.

## Run Lookahead Planner

```bash
python scripts/run_planner.py --training-data data/synthetic_trajectories.jsonl --count 1000 --seed 42 --horizon 3 --output reports/planner_results.md
```

The M4 planner uses the learned world model to score candidate actions with a 3-step lookahead objective that rewards task progress and penalizes safety risk, delay, and clinician review burden.

## Run Stress Evaluation

```bash
python scripts/run_stress_eval.py --training-data data/synthetic_trajectories.jsonl --count 500 --seed 99 --horizon 3 --output reports/stress_results.md
```

The stress evaluation over-samples high-acuity cases, sensitive context, order support, and unsupported autonomous-ordering requests to probe robustness beyond the default synthetic distribution.

## Current Status: v0.1.1

- Python package under `src/clinical_world_model`.
- Core schemas implemented in `src/clinical_world_model/schemas.py`.
- Minimal rule-based simulator implemented in `src/clinical_world_model/simulator.py`.
- Five handwritten synthetic workflow examples are in `examples/handwritten_episodes.jsonl`.
- Three generation policies are implemented in `src/clinical_world_model/policies.py`.
- Synthetic trajectory generation is implemented in `scripts/generate_trajectories.py`.
- Baseline metrics and reporting are implemented in `src/clinical_world_model/metrics.py`.
- Baseline results are in `reports/baseline_results.md`.
- Learned world-model training and evaluation are implemented in `src/clinical_world_model/world_model.py`.
- World-model evaluation is in `reports/world_model_eval.md`.
- 3-step world-model lookahead planning is implemented in `src/clinical_world_model/planner.py`.
- Planner comparison results are in `reports/planner_results.md`.
- Stress evaluation is implemented in `src/clinical_world_model/stress.py`.
- Stress results are in `reports/stress_results.md`.
- Portfolio report is in `reports/portfolio.md`.
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
- M5: portfolio release v0.1/v0.1.1 with final report and release packaging.

## Limitations

The simulator is intentionally simplified. It is useful for agent-safety engineering practice, evaluation design, and world-model prototyping, but it does not represent real clinical operations or real patient care. This is synthetic only, uses no PHI content, is not medical advice, and is not a clinical decision support system.
