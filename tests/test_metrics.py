import subprocess
import sys

from clinical_world_model.metrics import (
    evaluate_baselines,
    metrics_markdown_table,
    safety_violation_counts,
)


def test_baseline_metrics_are_bounded_and_show_tradeoffs() -> None:
    metrics, _ = evaluate_baselines(count=180, seed=42)
    by_policy = {item.policy_name: item for item in metrics}

    assert set(by_policy) == {
        "direct_policy",
        "safety_review_policy",
        "conservative_human_review_policy",
    }
    for item in metrics:
        assert item.trajectories == 180
        assert 0.0 <= item.task_success_rate <= 1.0
        assert 0.0 <= item.unsafe_action_rate <= 1.0
        assert 0.0 <= item.false_decline_rate <= 1.0
        assert 0.0 <= item.audit_completeness <= 1.0
        assert item.expected_delay_minutes >= 0
        assert item.clinician_review_burden >= 0

    assert (
        by_policy["direct_policy"].unsafe_action_rate
        > by_policy["safety_review_policy"].unsafe_action_rate
    )
    assert (
        by_policy["direct_policy"].unsafe_action_rate
        > by_policy["conservative_human_review_policy"].unsafe_action_rate
    )
    assert (
        by_policy["safety_review_policy"].false_decline_rate
        > by_policy["direct_policy"].false_decline_rate
    )
    assert (
        by_policy["conservative_human_review_policy"].clinician_review_burden
        >= by_policy["safety_review_policy"].clinician_review_burden
    )


def test_violation_counts_include_all_risk_columns() -> None:
    _, trajectories_by_policy = evaluate_baselines(count=60, seed=7)

    counts = safety_violation_counts(trajectories_by_policy["direct_policy"])

    assert set(counts) == {
        "unsafe_writeback",
        "phi_leakage",
        "hallucinated_evidence",
        "missing_clinician_review",
        "scope_escalation",
    }
    assert counts["unsafe_writeback"] > 0
    assert counts["missing_clinician_review"] > 0


def test_metrics_markdown_table_has_required_columns() -> None:
    metrics, _ = evaluate_baselines(count=30, seed=3)

    table = metrics_markdown_table(metrics)

    assert "task_success_rate" in table
    assert "unsafe_action_rate" in table
    assert "false_decline_rate" in table
    assert "audit_completeness" in table
    assert "expected_delay_min" in table
    assert "clinician_review_burden" in table


def test_run_baselines_script_writes_report(tmp_path) -> None:
    output = tmp_path / "baseline_results.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_baselines.py",
            "--count",
            "40",
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
    assert "Wrote baseline report" in result.stdout
    assert "# Baseline Results" in report
    assert "Policy Comparison" in report
    assert "direct_policy" in report
    assert "safety_review_policy" in report
    assert "conservative_human_review_policy" in report
