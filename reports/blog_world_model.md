# Building a Synthetic World Model for Clinical AI Agents

Clinical AI demos are often charmingly well behaved. The prompt is tidy, the patient is synthetic, the model answers in complete sentences, and everyone leaves the notebook feeling briefly optimistic.

Then the workflow arrives.

The workflow has missing context, audit requirements, review steps, writeback rules, scope boundaries, and the quiet but persistent expectation that nobody should accidentally turn a drafting assistant into an autonomous ordering system. In other words, the hard part is not always getting the model to say something plausible. The hard part is getting an agent to act safely inside a sequence of clinical operations.

This project, `clinical-agent-world-model`, is a small attempt to make that problem measurable.

It builds a synthetic hospital workflow world model for tool-using clinical AI agents. The world is deliberately small, structured, and synthetic. No patient data is used. No PHI content is generated. Nobody should use it for clinical care. Its job is humbler: predict what happens after an agent action, then use that prediction to choose safer actions before execution.

## The Practical Problem

In many agent demos, the unit of evaluation is an answer. In a hospital workflow, the unit of risk is often an action.

An answer can be fluent and still be operationally unsafe. For example:

- a note can be drafted without enough evidence
- context can be retrieved too broadly
- a writeback can happen before clinician review
- an order-support task can drift into unsupported autonomous ordering
- an external communication can carry information it should not carry

These are not just text-quality issues. They are state-transition issues. The question becomes:

> If the agent takes this action now, what state will the workflow be in next?

That is the reason to use a world model.

## What Counts as a World Here?

In AI, "world model" can mean many things. Sometimes it means learning physical dynamics. Sometimes it means simulating interactive environments. Here it means something much narrower and more useful for this portfolio project:

> a model of how a synthetic clinical workflow changes after an agent action.

This world does not contain real patients, real notes, or real diagnoses. It contains structured workflow states:

- whether context is available
- whether the draft exists
- whether the draft has evidence references
- whether clinician review is required or completed
- whether writeback is pending
- how much progress has been made
- how much delay has accumulated
- whether the action created a synthetic safety violation

That is not a grand theory of medicine. It is a practical model of consequences.

## The Safety Label on the Box

Before getting carried away, the project has a large label on the box:

- synthetic data only
- no PHI content
- not medical advice
- not a clinical decision support system
- not intended for diagnosis, treatment, or deployment

The synthetic environment is useful because it lets us test failure modes without touching real patient data. That is the point. If a system cannot behave in a toy workflow where every variable is typed and every case is synthetic, it has not earned the right to be vague in a real one.

## Modeling the Workflow

The simulator uses a few typed objects:

| object | meaning |
| --- | --- |
| `ClinicalState` | current workflow stage, context, draft status, review status, audit trail, progress, and delay |
| `AgentAction` | a tool-using action such as context retrieval, note generation, order drafting, writeback, review escalation, or decline |
| `Transition` | `state + action -> outcome + next_state` |
| `Outcome` | progress delta, delay, audit event, review burden, and safety violations |
| `SafetyViolation` | synthetic labels for unsafe writeback, PHI leakage, hallucinated evidence, missing review, and scope escalation |

The simulator covers six workflow actions:

| action | what it does |
| --- | --- |
| `retrieve_emr_context` | retrieves minimum-necessary synthetic context |
| `generate_note` | creates a synthetic note draft |
| `revise_note` | revises the note with evidence references |
| `draft_order` | creates an order-support draft |
| `writeback_draft` | writes a reviewed draft back into the synthetic workflow |
| `escalate_to_human_review` | routes the case to clinician review |

And five risk classes:

| risk | example |
| --- | --- |
| `unsafe_writeback` | writeback before required review |
| `phi_leakage` | overbroad context or external exposure |
| `hallucinated_evidence` | drafting without sufficient evidence references |
| `missing_clinician_review` | bypassing review controls |
| `scope_escalation` | drifting from assistive support into unsupported autonomous action |

There is no hidden clinical magic here. The system is closer to a carefully labeled workflow sandbox than a medical oracle. That is intentional.

## Three Baseline Personalities

I compared three simple policies on 1,000 synthetic scenarios.

The first is `direct_policy`. It is fast, confident, and exactly the sort of agent that makes a demo look good until someone asks about auditability.

The second is `safety_review_policy`. It follows fixed rules, requires review, and declines unsupported requests. It is safer, but sometimes too ready to say no.

The third is `conservative_human_review_policy`. It is the policy equivalent of saying, "Let's ask a clinician," which is often wise, but not free.

| policy | task success | unsafe action rate | false decline | audit completeness | delay |
| --- | ---: | ---: | ---: | ---: | ---: |
| `direct_policy` | 0.822 | 0.505 | 0.000 | 0.602 | 22.6 min |
| `safety_review_policy` | 0.926 | 0.000 | 0.090 | 0.926 | 40.7 min |
| `conservative_human_review_policy` | 1.000 | 0.000 | 0.000 | 1.000 | 57.4 min |

This is the basic tradeoff: speed, safety, auditability, and review burden rarely improve together by accident.

## Learning the Synthetic World

The world model is deliberately lightweight: random forests trained on synthetic `state + action` features.

It predicts:

- next workflow state
- probability of a safety violation
- expected delay
- audit completeness

On the current deterministic simulator, the results are almost suspiciously clean:

| target | result |
| --- | ---: |
| next workflow state accuracy | 1.000 |
| safety violation F1 | 1.000 |
| safety violation Brier score | 0.005 |
| delay MAE | 0.00 min |
| audit completeness MAE | 0.000 |

This is not a claim that clinical reality has been solved. It is a claim that the model learned the rules of the synthetic environment. That is useful for engineering, and it is also a useful warning label. A deterministic simulator can be learned; reality tends to be less cooperative.

## Planning Before Acting

Once the model can predict consequences, the agent can use it for lookahead planning.

The current planner evaluates candidate action sequences over a 3-step horizon. It scores them with a simple utility function:

```text
utility = task_progress
        - safety_risk_penalty
        - delay_penalty
        - clinician_burden_penalty
```

This is model-predictive control in a modest outfit. The agent does not need to be brilliant. It needs to ask, before acting, whether the next few steps are likely to end in progress or in a preventable workflow mess.

On 1,000 synthetic scenarios:

| policy | task success | unsafe action rate | false decline | audit completeness | delay | review burden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_policy` | 0.822 | 0.505 | 0.000 | 0.602 | 22.6 min | 0.00 |
| `safety_review_policy` | 0.926 | 0.000 | 0.090 | 0.926 | 40.7 min | 0.75 |
| `world_model_lookahead_policy` | 1.000 | 0.000 | 0.000 | 0.928 | 42.3 min | 0.82 |

The lookahead planner avoids unsafe actions like the fixed safety policy, but it also recovers supported workflows that the fixed policy declines. That is the point of modeling consequences rather than only applying static rules.

## Stress Testing the Toy Hospital

The project also includes a synthetic stress evaluation. It over-samples:

- high-acuity cases
- sensitive context
- order-support workflows
- unsupported autonomous-ordering requests

On 500 stressed scenarios:

| policy | task success | unsafe action rate | false decline | audit completeness | delay |
| --- | ---: | ---: | ---: | ---: | ---: |
| `direct_policy` | 0.748 | 0.568 | 0.000 | 0.563 | 25.7 min |
| `safety_review_policy` | 0.632 | 0.000 | 0.492 | 0.632 | 22.1 min |
| `world_model_lookahead_policy` | 1.000 | 0.000 | 0.000 | 0.885 | 38.7 min |

The stressed distribution makes the fixed safety policy's weakness visible. It remains safe, but it declines many supported high-acuity order-support cases. The lookahead planner can choose a reviewed path instead of giving up too early.

Again, this is synthetic distribution shift, not clinical validation. Still, synthetic stress tests are useful. They give the system a place to fail politely before it fails expensively.

## Reproducing the Project

```bash
python -m pip install -e ".[dev]"
pytest
python scripts/generate_trajectories.py --count 1000 --seed 42 --output data/synthetic_trajectories.jsonl
python scripts/run_baselines.py --count 1000 --seed 42 --output reports/baseline_results.md
python scripts/train_world_model.py --input data/synthetic_trajectories.jsonl --seed 42 --output reports/world_model_eval.md
python scripts/run_planner.py --training-data data/synthetic_trajectories.jsonl --count 1000 --seed 42 --horizon 3 --output reports/planner_results.md
python scripts/run_stress_eval.py --training-data data/synthetic_trajectories.jsonl --count 500 --seed 99 --horizon 3 --output reports/stress_results.md
```

The v0.1.1 release includes package discovery and CI smoke checks, so the scripts run after normal editable installation without setting `PYTHONPATH`.

## What This Project Is Not

This project does not model real clinical behavior. It does not use real patient data. It does not provide medical advice. It is not a clinical decision support system.

The current simulator is deterministic and simplified. The learned model is trained on rule-generated trajectories, so strong metrics mean it learned those rules. Future versions should add stochastic transitions, harder counterfactual episodes, richer stress distributions, and direct integration with runtime safety evaluators.

## Why I Built It

Clinical AI systems are often discussed as if prediction is the finish line. In practice, prediction is often the opening act. The harder questions arrive later:

- What action will the system take?
- What workflow state changes after that action?
- Was the evidence sufficient?
- Was review required?
- Was the action within scope?
- Can we audit what happened?

A world model is one way to make those questions explicit. It is not a substitute for clinical validation, governance, or human responsibility. It is a tool for making agent behavior more measurable before anyone pretends it is ready for the real world.

That seems like a reasonable place to start.

