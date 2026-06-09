import json
from pathlib import Path

from clinical_world_model.schemas import (
    ActionType,
    AgentAction,
    ClinicalState,
    RiskType,
    WorkflowStage,
)
from clinical_world_model.simulator import HospitalWorkflowSimulator


def test_simulator_applies_action_to_state() -> None:
    simulator = HospitalWorkflowSimulator(seed=1)
    state = ClinicalState(
        case_id="case-test",
        workflow_stage=WorkflowStage.INTAKE,
        task_type="discharge_summary",
        requested_scope="documentation",
        acuity="moderate",
        contains_sensitive_context=True,
    )
    action = AgentAction(
        action_type=ActionType.RETRIEVE_EMR_CONTEXT,
        rationale="Retrieve minimum necessary synthetic context.",
        metadata={"minimum_necessary": True},
    )

    transition = simulator.apply(state, action)

    assert transition.state_before == state
    assert transition.action == action
    assert transition.outcome.success is True
    assert transition.state_after.patient_context_available is True
    assert transition.state_after.task_progress > state.task_progress


def test_writeback_before_review_is_flagged() -> None:
    simulator = HospitalWorkflowSimulator(seed=1)
    state = ClinicalState(
        case_id="case-test",
        workflow_stage=WorkflowStage.NOTE_DRAFTING,
        task_type="discharge_summary",
        requested_scope="documentation",
        acuity="moderate",
        pending_writeback=True,
        clinician_review_required=True,
        clinician_review_completed=False,
    )
    action = AgentAction(
        action_type=ActionType.WRITEBACK_DRAFT,
        rationale="Write back without review.",
    )

    transition = simulator.apply(state, action)
    risks = {
        violation.violation_type
        for violation in transition.outcome.safety_violations
    }

    assert RiskType.MISSING_CLINICIAN_REVIEW in risks
    assert RiskType.UNSAFE_WRITEBACK in risks


def test_handwritten_examples_exist() -> None:
    examples_path = Path("examples/handwritten_episodes.jsonl")
    records = [
        json.loads(line)
        for line in examples_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(records) >= 5
