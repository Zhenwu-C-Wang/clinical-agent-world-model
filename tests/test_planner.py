import subprocess
import sys

from clinical_world_model.generation import (
    generate_trajectories,
    write_trajectories_jsonl,
)
from clinical_world_model.planner import (
    candidate_actions,
    evaluate_planner_comparison,
    project_next_state,
)
from clinical_world_model.schemas import (
    ActionType,
    AgentAction,
    ClinicalState,
    NoteStatus,
    WorkflowStage,
)
from clinical_world_model.simulator import ClinicalScenario
from clinical_world_model.world_model import WorldModelPrediction


def test_candidate_actions_include_review_and_unsafe_comparator() -> None:
    state = ClinicalState(
        case_id="case-test",
        workflow_stage=WorkflowStage.NOTE_DRAFTING,
        task_type="discharge_summary",
        requested_scope="documentation",
        acuity="moderate",
        patient_context_available=True,
        note_status=NoteStatus.DRAFT,
        pending_writeback=True,
        clinician_review_completed=False,
    )
    scenario = ClinicalScenario(
        scenario_id="case-test",
        task_type="discharge_summary",
        requested_scope="documentation",
        acuity="moderate",
        requires_order=False,
        contains_sensitive_context=False,
        supported=True,
    )

    actions = candidate_actions(state, scenario)
    action_types = {action.action_type for action in actions}

    assert ActionType.ESCALATE_TO_HUMAN_REVIEW in action_types
    assert ActionType.WRITEBACK_DRAFT in action_types


def test_project_next_state_applies_world_model_prediction() -> None:
    state = ClinicalState(
        case_id="case-test",
        workflow_stage=WorkflowStage.NOTE_DRAFTING,
        task_type="discharge_summary",
        requested_scope="documentation",
        acuity="moderate",
        patient_context_available=True,
        note_status=NoteStatus.DRAFT,
        pending_writeback=True,
    )
    action = AgentAction(
        action_type=ActionType.ESCALATE_TO_HUMAN_REVIEW,
        rationale="Review before writeback.",
    )
    prediction = WorldModelPrediction(
        next_workflow_stage=WorkflowStage.READY_FOR_WRITEBACK.value,
        safety_violation_probability=0.0,
        expected_delay_minutes=35.0,
        audit_completeness=0.75,
        next_task_progress=0.5,
    )

    next_state = project_next_state(state, action, prediction)

    assert next_state.workflow_stage == WorkflowStage.READY_FOR_WRITEBACK
    assert next_state.elapsed_minutes == 35
    assert next_state.task_progress == 0.5


def test_world_model_planner_reduces_unsafe_rate_and_preserves_success(tmp_path) -> None:
    training_path = tmp_path / "training.jsonl"
    write_trajectories_jsonl(
        training_path,
        generate_trajectories(count=240, seed=42),
    )

    metrics, _ = evaluate_planner_comparison(
        training_path=str(training_path),
        count=90,
        seed=42,
        horizon=3,
    )
    by_policy = {item.policy_name: item for item in metrics}

    direct = by_policy["direct_policy"]
    safety = by_policy["safety_review_policy"]
    planner = by_policy["world_model_lookahead_policy"]

    assert planner.unsafe_action_rate < direct.unsafe_action_rate
    assert planner.unsafe_action_rate <= safety.unsafe_action_rate
    assert planner.task_success_rate >= direct.task_success_rate


def test_run_planner_script_writes_report(tmp_path) -> None:
    training_path = tmp_path / "training.jsonl"
    output = tmp_path / "planner_results.md"
    write_trajectories_jsonl(
        training_path,
        generate_trajectories(count=180, seed=7),
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_planner.py",
            "--training-data",
            str(training_path),
            "--count",
            "20",
            "--seed",
            "42",
            "--horizon",
            "3",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = output.read_text(encoding="utf-8")
    assert "Wrote planner report" in result.stdout
    assert "# Planner Results" in report
    assert "world_model_lookahead_policy" in report
