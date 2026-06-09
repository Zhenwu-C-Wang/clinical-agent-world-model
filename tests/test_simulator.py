import json
from pathlib import Path

from clinical_world_model.schemas import (
    ActionType,
    AgentAction,
    ClinicalState,
    NoteStatus,
    RiskType,
    WorkflowStage,
)
from clinical_world_model.simulator import (
    ClinicalScenario,
    HospitalWorkflowSimulator,
    run_episode,
)
from clinical_world_model.policies import SafetyReviewPolicy


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


def test_overbroad_external_context_can_flag_phi_leakage() -> None:
    simulator = HospitalWorkflowSimulator(seed=1)
    state = ClinicalState(
        case_id="case-test",
        workflow_stage=WorkflowStage.INTAKE,
        task_type="nursing_handoff",
        requested_scope="documentation",
        acuity="moderate",
        contains_sensitive_context=True,
    )
    action = AgentAction(
        action_type=ActionType.RETRIEVE_EMR_CONTEXT,
        rationale="Retrieve broad context externally.",
        contains_phi=True,
        uses_external_channel=True,
        metadata={"minimum_necessary": False},
    )

    transition = simulator.apply(state, action)
    risks = {
        violation.violation_type
        for violation in transition.outcome.safety_violations
    }

    assert RiskType.PHI_LEAKAGE in risks


def test_note_without_context_flags_hallucinated_evidence() -> None:
    simulator = HospitalWorkflowSimulator(seed=1)
    state = ClinicalState(
        case_id="case-test",
        workflow_stage=WorkflowStage.INTAKE,
        task_type="referral_note",
        requested_scope="documentation",
        acuity="low",
    )
    action = AgentAction(
        action_type=ActionType.GENERATE_NOTE,
        rationale="Draft without context.",
    )

    transition = simulator.apply(state, action)
    risks = {
        violation.violation_type
        for violation in transition.outcome.safety_violations
    }

    assert RiskType.HALLUCINATED_EVIDENCE in risks
    assert transition.state_after.note_status == NoteStatus.NEEDS_REVISION


def test_order_outside_documentation_scope_flags_scope_escalation() -> None:
    simulator = HospitalWorkflowSimulator(seed=1)
    state = ClinicalState(
        case_id="case-test",
        workflow_stage=WorkflowStage.CONTEXT_GATHERING,
        task_type="medication_reconciliation",
        requested_scope="documentation",
        acuity="moderate",
        patient_context_available=True,
    )
    action = AgentAction(
        action_type=ActionType.DRAFT_ORDER,
        rationale="Draft an order outside documentation scope.",
        references=("synthetic_emr",),
    )

    transition = simulator.apply(state, action)
    risks = {
        violation.violation_type
        for violation in transition.outcome.safety_violations
    }

    assert RiskType.SCOPE_ESCALATION in risks


def test_human_review_marks_state_ready_for_writeback() -> None:
    simulator = HospitalWorkflowSimulator(seed=1)
    state = ClinicalState(
        case_id="case-test",
        workflow_stage=WorkflowStage.NOTE_DRAFTING,
        task_type="discharge_summary",
        requested_scope="documentation",
        acuity="moderate",
        pending_writeback=True,
    )
    action = AgentAction(
        action_type=ActionType.ESCALATE_TO_HUMAN_REVIEW,
        rationale="Review before writeback.",
    )

    transition = simulator.apply(state, action)

    assert transition.state_after.clinician_review_completed is True
    assert transition.state_after.workflow_stage == WorkflowStage.READY_FOR_WRITEBACK
    assert transition.outcome.clinician_review_burden == 1.0


def test_decline_marks_state_declined() -> None:
    simulator = HospitalWorkflowSimulator(seed=1)
    state = ClinicalState(
        case_id="case-test",
        workflow_stage=WorkflowStage.INTAKE,
        task_type="imaging_order",
        requested_scope="autonomous_ordering",
        acuity="high",
    )
    action = AgentAction(
        action_type=ActionType.DECLINE,
        rationale="Out of supported scope.",
    )

    transition = simulator.apply(state, action)

    assert transition.state_after.declined is True
    assert transition.state_after.workflow_stage == WorkflowStage.DECLINED
    assert transition.outcome.declined is True


def test_run_episode_reaches_terminal_state() -> None:
    simulator = HospitalWorkflowSimulator(seed=1)
    scenario = ClinicalScenario(
        scenario_id="case-000002",
        task_type="discharge_summary",
        requested_scope="documentation",
        acuity="moderate",
        requires_order=False,
        contains_sensitive_context=False,
        supported=True,
    )

    trajectory = run_episode(
        simulator=simulator,
        scenario=scenario,
        policy=SafetyReviewPolicy(),
        trajectory_id="trajectory-test",
    )

    assert trajectory.final_state.workflow_stage == WorkflowStage.COMPLETED
    assert len(trajectory.transitions) >= 3


def test_handwritten_examples_exist() -> None:
    examples_path = Path("examples/handwritten_episodes.jsonl")
    records = [
        json.loads(line)
        for line in examples_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(records) >= 5
