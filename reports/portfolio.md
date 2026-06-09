# Portfolio Report: Clinical Agent World Model v0.1

## One-Line Summary

Built a synthetic hospital workflow world model for tool-using clinical AI agents that predicts workflow state, safety risk, audit completeness, and delay after agent actions, then uses 3-step lookahead planning to choose safer actions.

## Safety Scope

- Synthetic data only.
- No PHI content is used, requested, stored, or generated.
- Some records include synthetic PHI-risk flags for evaluation; these are labels, not PHI.
- Not medical advice.
- Not a clinical decision support system.
- Not intended for diagnosis, treatment, or real clinical deployment.

## Problem

Tool-using clinical agents can fail through workflow errors even when individual text outputs look plausible. Common failures include missing context, unsupported evidence, unsafe chart writeback, missing clinician review, overbroad PHI exposure, and scope escalation. This project creates a synthetic environment where those failures can be simulated, measured, modeled, and reduced through planning.

## Approach

The project implements a synthetic hospital workflow environment with typed schemas:

- `ClinicalState`: workflow stage, context availability, draft status, review status, audit events, task progress, and elapsed delay.
- `AgentAction`: context retrieval, note generation, note revision, order drafting, writeback, review escalation, or decline.
- `Transition`: `state + action -> outcome + next_state`.
- `Outcome`: progress delta, delay, audit event, review burden, and safety violations.
- `SafetyViolation`: unsafe writeback, PHI leakage, hallucinated evidence, missing clinician review, and scope escalation.

The environment generates trajectories, evaluates baseline policies, trains a lightweight learned world model, and applies 3-step lookahead planning.

## Artifacts

- Synthetic trajectories: `data/synthetic_trajectories.jsonl`
- Baseline report: `reports/baseline_results.md`
- World model evaluation: `reports/world_model_eval.md`
- Planner comparison: `reports/planner_results.md`
- Roadmap: `reports/roadmap.md`

## Milestone Summary

| milestone | result |
| --- | --- |
| M0 | Project skeleton, schemas, examples, simulator, tests, and CI |
| M1 | Rule-based simulator and 1,000 synthetic trajectories |
| M2 | Direct, safety-review, and conservative human-review baselines |
| M3 | Learned transition/risk/delay/audit world model |
| M4 | 3-step world-model lookahead planner |
| M5 | Portfolio packaging and v0.1 release |

## Key Results

Baseline policy comparison on 1,000 synthetic scenarios:

| policy | task_success_rate | unsafe_action_rate | false_decline_rate | audit_completeness | expected_delay_min | clinician_review_burden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct_policy | 0.822 | 0.505 | 0.000 | 0.602 | 22.6 | 0.00 |
| safety_review_policy | 0.926 | 0.000 | 0.090 | 0.926 | 40.7 | 0.75 |
| conservative_human_review_policy | 1.000 | 0.000 | 0.000 | 1.000 | 57.4 | 1.18 |

World-model lookahead comparison on 1,000 synthetic scenarios:

| policy | task_success_rate | unsafe_action_rate | false_decline_rate | audit_completeness | expected_delay_min | clinician_review_burden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct_policy | 0.822 | 0.505 | 0.000 | 0.602 | 22.6 | 0.00 |
| safety_review_policy | 0.926 | 0.000 | 0.090 | 0.926 | 40.7 | 0.75 |
| world_model_lookahead_policy | 1.000 | 0.000 | 0.000 | 0.928 | 42.3 | 0.82 |

The learned world model achieved synthetic simulator evaluation metrics:

| target | result |
| --- | ---: |
| next workflow state accuracy | 1.000 |
| next workflow state macro F1 | 1.000 |
| safety violation F1 | 1.000 |
| safety violation Brier score | 0.005 |
| delay MAE | 0.00 min |
| audit completeness MAE | 0.000 |

These high scores are expected because the current environment is deterministic and rule-generated.

## Technical Highlights

- Designed typed clinical workflow schemas and JSONL trajectory serialization.
- Implemented a reproducible rule-based simulator for synthetic workflow state transitions.
- Built baseline policy evaluation with safety, audit, delay, decline, and review-burden metrics.
- Trained a lightweight learned world model over `state + action` features.
- Implemented 3-step lookahead action selection with a utility objective:

```text
utility = task_progress
        - safety_risk_penalty
        - delay_penalty
        - clinician_burden_penalty
```

## Reproduce

```bash
python -m pip install -e ".[dev]"
pytest
python scripts/generate_trajectories.py --count 1000 --seed 42 --output data/synthetic_trajectories.jsonl
python scripts/run_baselines.py --count 1000 --seed 42 --output reports/baseline_results.md
python scripts/train_world_model.py --input data/synthetic_trajectories.jsonl --seed 42 --output reports/world_model_eval.md
python scripts/run_planner.py --training-data data/synthetic_trajectories.jsonl --count 1000 --seed 42 --horizon 3 --output reports/planner_results.md
```

## Resume Bullets

- Built a synthetic clinical-agent workflow simulator with typed state/action/outcome schemas, generating 1,000 no-PHI-content trajectories across six workflow actions and five safety-risk classes.
- Trained a lightweight world model to predict next workflow state, safety risk, delay, and audit completeness from `state + action` features, with automated evaluation reports and CI-backed tests.
- Implemented a 3-step lookahead planner that reduced synthetic unsafe action rate from 50.5% to 0.0% while improving task success from 82.2% to 100.0%.

## Limitations

- The simulator is synthetic and simplified.
- The learned model currently learns deterministic simulator dynamics, not real hospital behavior.
- The project does not use real patient data and should not be used for clinical care.
- The planner objective is intentionally simple and should be stress-tested with harder synthetic distributions before any broader claims.
