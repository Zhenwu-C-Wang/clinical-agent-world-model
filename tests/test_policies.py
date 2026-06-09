from clinical_world_model.policies import (
    ConservativeHumanReviewPolicy,
    DirectPolicy,
    SafetyReviewPolicy,
)
from clinical_world_model.schemas import (
    ActionType,
    ClinicalState,
    NoteStatus,
    WorkflowStage,
)
from clinical_world_model.simulator import ClinicalScenario


def test_direct_policy_can_skip_context_for_unguarded_mode() -> None:
    scenario = ClinicalScenario(
        scenario_id="case-000000",
        task_type="referral_note",
        requested_scope="documentation",
        acuity="low",
        requires_order=False,
        contains_sensitive_context=False,
        supported=True,
    )
    state = scenario.initial_state()

    action = DirectPolicy().next_action(state, scenario, ())

    assert action.action_type == ActionType.GENERATE_NOTE
    assert action.references == ()


def test_safety_review_policy_declines_autonomous_ordering() -> None:
    scenario = ClinicalScenario(
        scenario_id="case-000001",
        task_type="imaging_order",
        requested_scope="autonomous_ordering",
        acuity="high",
        requires_order=True,
        contains_sensitive_context=False,
        supported=False,
    )
    state = scenario.initial_state()

    action = SafetyReviewPolicy().next_action(state, scenario, ())

    assert action.action_type == ActionType.DECLINE


def test_conservative_policy_escalates_pending_draft() -> None:
    scenario = ClinicalScenario(
        scenario_id="case-000002",
        task_type="discharge_summary",
        requested_scope="documentation",
        acuity="moderate",
        requires_order=False,
        contains_sensitive_context=False,
        supported=True,
    )
    state = ClinicalState(
        case_id=scenario.scenario_id,
        workflow_stage=WorkflowStage.NOTE_DRAFTING,
        task_type=scenario.task_type,
        requested_scope=scenario.requested_scope,
        acuity=scenario.acuity,
        patient_context_available=True,
        note_status=NoteStatus.DRAFT,
        pending_writeback=True,
        clinician_review_required=True,
        clinician_review_completed=False,
    )

    action = ConservativeHumanReviewPolicy().next_action(state, scenario, ())

    assert action.action_type == ActionType.ESCALATE_TO_HUMAN_REVIEW

