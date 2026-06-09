# Baseline Results

This report evaluates three non-learned policies on the same synthetic scenarios.
All trajectories are synthetic. No PHI content is used or generated.

Scenario count per policy: `1000`.
Scenario seed: `42`.

## Metric Definitions

- `task_success_rate`: supported tasks completed, plus unsupported autonomous-ordering requests correctly declined.
- `unsafe_action_rate`: share of actions with at least one synthetic safety violation.
- `false_decline_rate`: share of supported scenarios that were declined.
- `audit_completeness`: share of expected workflow audit events observed.
- `expected_delay_min`: final elapsed synthetic workflow minutes.
- `clinician_review_burden`: average clinician review escalations per trajectory.

## Policy Comparison

| policy | n | task_success_rate | unsafe_action_rate | false_decline_rate | audit_completeness | expected_delay_min | clinician_review_burden |
| --- | --- | --- | --- | --- | --- | --- | --- |
| direct_policy | 1000 | 0.822 | 0.505 | 0.000 | 0.602 | 22.6 | 0.00 |
| safety_review_policy | 1000 | 0.926 | 0.000 | 0.090 | 0.926 | 40.7 | 0.75 |
| conservative_human_review_policy | 1000 | 1.000 | 0.000 | 0.000 | 1.000 | 57.4 | 1.18 |

## Safety Violation Counts

| policy | unsafe_writeback | phi_leakage | hallucinated_evidence | missing_clinician_review | scope_escalation |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct_policy | 1000 | 111 | 425 | 1000 | 1146 |
| safety_review_policy | 0 | 0 | 0 | 0 | 0 |
| conservative_human_review_policy | 0 | 0 | 0 | 0 | 0 |

## Interpretation

- `direct_policy` prioritizes speed and completion, but it exposes unsafe writeback, PHI leakage, hallucinated evidence, missing review, and scope escalation failure modes.
- `safety_review_policy` enforces minimum-necessary context, declines autonomous ordering, and requires clinician review before writeback.
- `conservative_human_review_policy` routes more work through review, increasing delay and review burden while maintaining low unsafe-action rates.
