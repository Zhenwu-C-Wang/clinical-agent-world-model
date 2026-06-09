import subprocess
import sys

from clinical_world_model.generation import generate_trajectories
from clinical_world_model.world_model import (
    build_dataset,
    render_world_model_report,
    train_and_evaluate_world_model,
)


def test_build_dataset_flattens_transitions() -> None:
    trajectories = generate_trajectories(count=30, seed=42)

    dataset = build_dataset(trajectories)

    assert len(dataset.features) > len(trajectories)
    assert len(dataset.features) == len(dataset.next_workflow_stage)
    assert len(dataset.features) == len(dataset.safety_violation)
    assert len(dataset.features) == len(dataset.delay_minutes)
    assert len(dataset.features) == len(dataset.audit_completeness)
    assert len(dataset.features) == len(dataset.next_task_progress)
    assert "action_type" in dataset.features[0]
    assert "workflow_stage" in dataset.features[0]


def test_world_model_evaluation_returns_required_metrics() -> None:
    trajectories = generate_trajectories(count=180, seed=42)

    evaluation = train_and_evaluate_world_model(trajectories, seed=42)

    assert evaluation.train_rows > evaluation.test_rows
    assert 0.0 <= evaluation.next_stage.accuracy <= 1.0
    assert 0.0 <= evaluation.next_stage.macro_f1 <= 1.0
    assert 0.0 <= evaluation.safety_violation.accuracy <= 1.0
    assert 0.0 <= evaluation.safety_violation.f1 <= 1.0
    assert 0.0 <= evaluation.safety_violation.brier_score <= 1.0
    assert evaluation.delay.mae >= 0.0
    assert evaluation.audit_completeness.mae >= 0.0
    assert evaluation.next_stage.accuracy >= 0.80
    assert evaluation.safety_violation.f1 >= 0.70


def test_world_model_report_contains_confusion_and_calibration() -> None:
    trajectories = generate_trajectories(count=120, seed=7)
    evaluation = train_and_evaluate_world_model(trajectories, seed=7)

    report = render_world_model_report(evaluation)

    assert "# World Model Evaluation" in report
    assert "Summary Metrics" in report
    assert "Next Workflow State Confusion Matrix" in report
    assert "Safety Risk Calibration" in report
    assert "Brier score" in report


def test_train_world_model_script_writes_report(tmp_path) -> None:
    output = tmp_path / "world_model_eval.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_world_model.py",
            "--input",
            "data/synthetic_trajectories.jsonl",
            "--seed",
            "42",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = output.read_text(encoding="utf-8")
    assert "Wrote world model evaluation" in result.stdout
    assert "# World Model Evaluation" in report
    assert "Summary Metrics" in report
