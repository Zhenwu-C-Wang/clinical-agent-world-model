import subprocess
import sys

from clinical_world_model.generation import (
    generate_trajectories,
    write_trajectories_jsonl,
)
from clinical_world_model.stress import (
    evaluate_stress,
    render_stress_report,
    sample_stress_scenarios,
    summarize_stress_scenarios,
)


def test_stress_scenarios_are_skewed_toward_harder_cases() -> None:
    scenarios = sample_stress_scenarios(count=120, seed=99)

    summary = summarize_stress_scenarios(scenarios)

    assert summary.total == 120
    assert summary.high_acuity_rate >= 0.50
    assert summary.sensitive_context_rate >= 0.60
    assert summary.order_required_rate >= 0.85
    assert summary.autonomous_ordering_rate >= 0.15
    assert summary.order_support_rate >= 0.45


def test_stress_evaluation_shows_planner_safety_and_recovery(tmp_path) -> None:
    training_path = tmp_path / "training.jsonl"
    write_trajectories_jsonl(
        training_path,
        generate_trajectories(count=160, seed=42),
    )

    metrics, trajectories_by_policy, summary = evaluate_stress(
        training_path=str(training_path),
        count=60,
        seed=99,
        horizon=3,
    )
    by_policy = {item.policy_name: item for item in metrics}

    direct = by_policy["direct_policy"]
    safety = by_policy["safety_review_policy"]
    planner = by_policy["world_model_lookahead_policy"]

    assert planner.unsafe_action_rate < direct.unsafe_action_rate
    assert planner.unsafe_action_rate <= safety.unsafe_action_rate
    assert planner.task_success_rate >= direct.task_success_rate
    assert planner.task_success_rate >= safety.task_success_rate
    assert safety.false_decline_rate > planner.false_decline_rate

    report = render_stress_report(
        metrics=metrics,
        trajectories_by_policy=trajectories_by_policy,
        summary=summary,
        seed=99,
        horizon=3,
    )
    assert "Stress Distribution" in report
    assert "world_model_lookahead_policy" in report


def test_run_stress_eval_script_writes_report(tmp_path) -> None:
    training_path = tmp_path / "training.jsonl"
    output = tmp_path / "stress_results.md"
    write_trajectories_jsonl(
        training_path,
        generate_trajectories(count=120, seed=7),
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_stress_eval.py",
            "--training-data",
            str(training_path),
            "--count",
            "20",
            "--seed",
            "99",
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
    assert "Wrote stress evaluation report" in result.stdout
    assert "# Stress Evaluation Results" in report
    assert "Stress Distribution" in report
