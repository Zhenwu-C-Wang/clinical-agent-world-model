# Stress Evaluation Results

This report evaluates policies on a harder synthetic scenario distribution than the default trajectory generator.
The stress distribution over-samples high-acuity cases, sensitive context, order support, and unsupported autonomous ordering requests.

Scenario count per policy: `500`.
Scenario seed: `99`.
Lookahead horizon: `3` steps.

## Stress Distribution

| property | rate |
| --- | ---: |
| high acuity | 0.610 |
| sensitive context | 0.696 |
| order required | 0.964 |
| autonomous ordering request | 0.252 |
| order-support request | 0.576 |

## Policy Comparison

| policy | n | task_success_rate | unsafe_action_rate | false_decline_rate | audit_completeness | expected_delay_min | clinician_review_burden |
| --- | --- | --- | --- | --- | --- | --- | --- |
| direct_policy | 500 | 0.748 | 0.568 | 0.000 | 0.563 | 25.7 | 0.00 |
| safety_review_policy | 500 | 0.632 | 0.000 | 0.492 | 0.632 | 22.1 | 0.38 |
| world_model_lookahead_policy | 500 | 1.000 | 0.000 | 0.000 | 0.885 | 38.7 | 0.75 |

## Safety Violation Counts

| policy | unsafe_writeback | phi_leakage | hallucinated_evidence | missing_clinician_review | scope_escalation |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct_policy | 500 | 116 | 245 | 500 | 964 |
| safety_review_policy | 0 | 0 | 0 | 0 | 0 |
| world_model_lookahead_policy | 0 | 0 | 0 | 0 | 0 |

## Scope Mix

| requested scope | count |
| --- | ---: |
| autonomous_ordering | 126 |
| documentation | 86 |
| order_support | 288 |

## Interpretation

- `direct_policy` remains fast but unsafe under stressed order and review conditions.
- `safety_review_policy` stays safe but declines many supported high-acuity order-support cases.
- `world_model_lookahead_policy` maintains low unsafe-action rate while recovering supported workflows that fixed safety rules decline.

## Limitations

- This is still a synthetic distribution shift, not real clinical validation.
- The stress generator intentionally over-samples hard cases, so rates should not be interpreted as real hospital prevalence.
- No PHI content is used or generated, and this is not medical advice or a clinical decision support system.
