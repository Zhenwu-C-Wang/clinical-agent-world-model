# Planner Results

This report compares direct action, rule-based safety, and learned world-model lookahead policies on the same synthetic scenarios.
The lookahead policy uses a lightweight learned world model to score candidate actions before executing them in the simulator.

Scenario count per policy: `1000`.
Scenario seed: `42`.
Lookahead horizon: `3` steps.

## Planner Objective

```text
utility = task_progress
        - safety_risk_penalty
        - delay_penalty
        - clinician_burden_penalty
```

## Policy Comparison

| policy | n | task_success_rate | unsafe_action_rate | false_decline_rate | audit_completeness | expected_delay_min | clinician_review_burden |
| --- | --- | --- | --- | --- | --- | --- | --- |
| direct_policy | 1000 | 0.822 | 0.505 | 0.000 | 0.602 | 22.6 | 0.00 |
| safety_review_policy | 1000 | 0.926 | 0.000 | 0.090 | 0.926 | 40.7 | 0.75 |
| world_model_lookahead_policy | 1000 | 1.000 | 0.000 | 0.000 | 0.928 | 42.3 | 0.82 |

## Safety Violation Counts

| policy | unsafe_writeback | phi_leakage | hallucinated_evidence | missing_clinician_review | scope_escalation |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct_policy | 1000 | 111 | 425 | 1000 | 1146 |
| safety_review_policy | 0 | 0 | 0 | 0 | 0 |
| world_model_lookahead_policy | 0 | 0 | 0 | 0 | 0 |

## Interpretation

- `direct_policy` is fast but exposes unsafe writeback and scope escalation.
- `safety_review_policy` uses fixed rules and accepts conservative false declines.
- `world_model_lookahead_policy` uses the learned world model to avoid high-risk actions while preserving task success on supported workflows.

## Limitations

- Planning is evaluated only in this synthetic simulator.
- The learned world model is trained on rule-generated data, so strong planner results do not imply clinical readiness.
- This is not medical advice and is not a clinical decision support system.
