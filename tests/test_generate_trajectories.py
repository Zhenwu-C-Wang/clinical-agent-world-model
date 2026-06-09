import subprocess
import sys

from clinical_world_model.generation import (
    generate_trajectories,
    read_trajectories_jsonl,
    write_trajectories_jsonl,
)
from clinical_world_model.schemas import ActionType, RiskType


def test_generate_trajectories_is_deterministic() -> None:
    first = [
        trajectory.to_dict()
        for trajectory in generate_trajectories(count=25, seed=42)
    ]
    second = [
        trajectory.to_dict()
        for trajectory in generate_trajectories(count=25, seed=42)
    ]

    assert first == second


def test_generated_trajectories_cover_m1_actions_and_risks() -> None:
    trajectories = generate_trajectories(count=300, seed=42)
    actions = {
        transition.action.action_type
        for trajectory in trajectories
        for transition in trajectory.transitions
    }
    risks = {
        violation.violation_type
        for trajectory in trajectories
        for transition in trajectory.transitions
        for violation in transition.outcome.safety_violations
    }

    assert {
        ActionType.RETRIEVE_EMR_CONTEXT,
        ActionType.GENERATE_NOTE,
        ActionType.REVISE_NOTE,
        ActionType.DRAFT_ORDER,
        ActionType.WRITEBACK_DRAFT,
        ActionType.ESCALATE_TO_HUMAN_REVIEW,
    }.issubset(actions)
    assert {
        RiskType.UNSAFE_WRITEBACK,
        RiskType.PHI_LEAKAGE,
        RiskType.HALLUCINATED_EVIDENCE,
        RiskType.MISSING_CLINICIAN_REVIEW,
        RiskType.SCOPE_ESCALATION,
    }.issubset(risks)


def test_write_and_read_trajectories_jsonl_round_trip(tmp_path) -> None:
    path = tmp_path / "synthetic_trajectories.jsonl"
    trajectories = generate_trajectories(count=10, seed=7)

    write_trajectories_jsonl(path, trajectories)
    loaded = read_trajectories_jsonl(path)

    assert [item.to_dict() for item in loaded] == [
        item.to_dict() for item in trajectories
    ]


def test_generate_trajectories_script_writes_jsonl(tmp_path) -> None:
    output = tmp_path / "trajectories.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_trajectories.py",
            "--count",
            "12",
            "--seed",
            "42",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Wrote 12 trajectories" in result.stdout
    assert len(output.read_text(encoding="utf-8").splitlines()) == 12
