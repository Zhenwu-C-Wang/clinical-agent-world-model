"""Synthetic-only guardrail tests.

These pin the contract that the scenario generators only ever emit structured
workflow labels drawn from a fixed vocabulary -- never free-text or
patient-identifiable content. They are intentionally the single place to update
when new (e.g. ICU) scenario labels are introduced, so that any new label is a
deliberate, reviewed change rather than an accidental data-shape drift.
"""

import re

from clinical_world_model.generation import sample_scenarios
from clinical_world_model.simulator import ClinicalScenario
from clinical_world_model.stress import sample_stress_scenarios

ALLOWED_TASK_TYPES = {
    "discharge_summary",
    "medication_reconciliation",
    "lab_followup",
    "referral_note",
    "imaging_order",
    "nursing_handoff",
}
ALLOWED_REQUESTED_SCOPES = {"order_support", "autonomous_ordering", "documentation"}
ALLOWED_ACUITIES = {"low", "moderate", "high"}
SCENARIO_ID_PATTERN = re.compile(r"^(?:case|stress-case)-\d{6}$")


def _assert_scenario_in_vocabulary(scenario: ClinicalScenario) -> None:
    assert scenario.task_type in ALLOWED_TASK_TYPES
    assert scenario.requested_scope in ALLOWED_REQUESTED_SCOPES
    assert scenario.acuity in ALLOWED_ACUITIES
    assert SCENARIO_ID_PATTERN.match(scenario.scenario_id) is not None
    # `supported` must follow deterministically from the requested scope, so an
    # unsupported autonomous-ordering request can never be silently marked safe.
    assert scenario.supported == (scenario.requested_scope != "autonomous_ordering")
    # Flag fields must stay boolean -- never free-form values that could smuggle
    # in non-synthetic content.
    assert isinstance(scenario.requires_order, bool)
    assert isinstance(scenario.contains_sensitive_context, bool)
    assert isinstance(scenario.supported, bool)


def test_default_scenarios_stay_within_synthetic_vocabulary() -> None:
    for scenario in sample_scenarios(count=500, seed=42):
        _assert_scenario_in_vocabulary(scenario)


def test_stress_scenarios_stay_within_synthetic_vocabulary() -> None:
    for scenario in sample_stress_scenarios(count=500, seed=99):
        _assert_scenario_in_vocabulary(scenario)


def test_default_distribution_exercises_full_vocabulary() -> None:
    # Guards against the vocabulary checks passing vacuously on a narrow sample:
    # every acuity tier and every requested scope should actually appear.
    scenarios = sample_scenarios(count=500, seed=42)
    assert {scenario.acuity for scenario in scenarios} == ALLOWED_ACUITIES
    assert {
        scenario.requested_scope for scenario in scenarios
    } == ALLOWED_REQUESTED_SCOPES
