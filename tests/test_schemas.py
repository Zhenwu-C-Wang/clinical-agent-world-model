from clinical_world_model.schemas import (
    ActionType,
    AgentAction,
    ClinicalState,
    Outcome,
    RiskType,
    SafetyViolation,
    Severity,
    Transition,
    WorkflowStage,
)


def test_transition_round_trip() -> None:
    state = ClinicalState(
        case_id="case-test",
        workflow_stage=WorkflowStage.INTAKE,
        task_type="discharge_summary",
        requested_scope="documentation",
        acuity="moderate",
    )
    action = AgentAction(
        action_type=ActionType.GENERATE_NOTE,
        rationale="test",
        references=("synthetic_emr",),
    )
    violation = SafetyViolation(
        violation_type=RiskType.HALLUCINATED_EVIDENCE,
        severity=Severity.HIGH,
        message="test",
    )
    outcome = Outcome(
        success=True,
        task_progress_delta=0.1,
        delay_minutes=1,
        audit_event="note_draft_created",
        safety_violations=(violation,),
    )
    transition = Transition(
        state_before=state,
        action=action,
        outcome=outcome,
        state_after=state,
    )

    assert Transition.from_dict(transition.to_dict()) == transition


def test_invalid_progress_rejected() -> None:
    try:
        ClinicalState(
            case_id="case-test",
            workflow_stage=WorkflowStage.INTAKE,
            task_type="discharge_summary",
            requested_scope="documentation",
            acuity="moderate",
            task_progress=1.5,
        )
    except ValueError as error:
        assert "task_progress" in str(error)
    else:
        raise AssertionError("Expected invalid progress to be rejected")

