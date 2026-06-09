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

```bash
python -m pip install -e ".[dev]"
pytest
```

## Current Status: M0

- Python package under `src/clinical_world_model`.
- Core schemas implemented in `src/clinical_world_model/schemas.py`.
- Minimal rule-based simulator implemented in `src/clinical_world_model/simulator.py`.
- Five handwritten synthetic workflow examples are in `examples/handwritten_episodes.jsonl`.
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
