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
| safety violation | Brier score | <0.01 |
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

Decision-level bins keep the checked-in report stable across scikit-learn wheels while still showing whether low-risk and high-risk predictions are separated.

| predicted risk decision | n | observed risk |
| --- | ---: | ---: |
| predicted low risk | 798 | 0.000 |
| predicted high risk | 186 | 1.000 |

## Limitations

- The model is trained only on synthetic rule-generated trajectories.
- Strong metrics here mean the model learned this simulator's dynamics, not real hospital behavior.
- The model is not medical advice and is not a clinical decision support system.
