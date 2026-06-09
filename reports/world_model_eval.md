# World Model Evaluation

This report trains lightweight random-forest world models on synthetic trajectories.
Inputs are synthetic `state + action` features; targets are next workflow state, safety violation risk, action delay, and audit completeness.

Training rows: `2950`.
Test rows: `984`.

## Summary Metrics

| target | metric | value |
| --- | --- | ---: |
| next workflow state | accuracy | 1.000 |
| next workflow state | macro F1 | 1.000 |
| safety violation | accuracy | 1.000 |
| safety violation | F1 | 1.000 |
| safety violation | Brier score | 0.005 |
| expected delay | MAE minutes | 0.00 |
| expected delay | R2 | 1.000 |
| audit completeness | MAE | 0.000 |
| audit completeness | R2 | 1.000 |

## Next Workflow State Confusion Matrix

| actual \ predicted | completed | context_gathering | declined | note_drafting | order_drafting | ready_for_writeback |
| --- | --- | --- | --- | --- | --- | --- |
| completed | 206 | 0 | 0 | 0 | 0 | 0 |
| context_gathering | 0 | 231 | 0 | 0 | 0 | 0 |
| declined | 0 | 0 | 38 | 0 | 0 | 0 |
| note_drafting | 0 | 0 | 0 | 242 | 0 | 0 |
| order_drafting | 0 | 0 | 0 | 0 | 109 | 0 |
| ready_for_writeback | 0 | 0 | 0 | 0 | 0 | 158 |

## Safety Risk Calibration

| predicted risk bin | n | avg predicted risk | observed risk |
| --- | ---: | ---: | ---: |
| 0.0-0.2 | 789 | 0.028 | 0.000 |
| 0.2-0.4 | 9 | 0.225 | 0.000 |
| 0.4-0.6 | 7 | 0.553 | 1.000 |
| 0.6-0.8 | 8 | 0.690 | 1.000 |
| 0.8-1.0 | 171 | 0.970 | 1.000 |

## Limitations

- The model is trained only on synthetic rule-generated trajectories.
- Strong metrics here mean the model learned this simulator's dynamics, not real hospital behavior.
- The model is not medical advice and is not a clinical decision support system.
