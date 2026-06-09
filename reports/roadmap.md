# Roadmap

## M0: Project Skeleton & Schema

Definition of done:

- README states project goal, synthetic-only scope, no PHI, and not-medical-advice limitations.
- Core schemas exist: `ClinicalState`, `AgentAction`, `Transition`, `Outcome`, and `SafetyViolation`.
- Minimal simulator can execute one action from one state and produce a transition.
- Five handwritten synthetic workflow examples exist.
- `pytest` passes locally and in GitHub Actions.

## M1: Rule-Based Hospital Workflow Simulator

Definition of done:

- Simulator generates reproducible synthetic trajectories.
- Covers at least six action types: retrieve EMR context, generate note, revise note, draft order, writeback draft, and escalate to human review.
- Covers at least five risk types: unsafe writeback, PHI leakage, hallucinated evidence, missing clinician review, and scope escalation.
- Generates at least 1,000 trajectories to `data/synthetic_trajectories.jsonl`.

## M2: Baseline Policies & Metrics

Definition of done:

- Implements direct agent, rule-based safety, and human-review-heavy policies.
- Reports task success rate, unsafe action rate, false decline rate, audit completeness, expected delay, and clinician review burden.
- Produces `reports/baseline_results.md` with a policy comparison table.

## M3: Learned World Model

Definition of done:

- Trains a lightweight model for `state + action -> next outcome`.
- Predicts next workflow state, safety violation probability, expected delay, and audit completeness.
- Reports accuracy/F1, calibration, and confusion matrix in `reports/world_model_eval.md`.

## M4: Lookahead Planner

Definition of done:

- Implements a 3-step lookahead planner.
- Compares direct, rule-based safety, and world-model lookahead policies.
- Produces `reports/planner_results.md`.

## M5: Portfolio Release v0.1

Definition of done:

- Produces `reports/portfolio.md`.
- README includes problem, approach, quickstart, key results, and limitations.
- Release is tagged `v0.1`.
- GitHub topics include `world-models`, `clinical-ai`, `llm-agents`, `agent-safety`, `evals`, `synthetic-data`, and `python`.

## M6: Synthetic Stress Evaluation

Definition of done:

- Adds a harder synthetic stress distribution.
- Evaluates direct, rule-based safety, and world-model lookahead policies under stress.
- Produces `reports/stress_results.md`.
- Confirms the planner reduces unsafe actions while preserving supported workflow completion under stress.
