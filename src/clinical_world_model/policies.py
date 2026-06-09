from __future__ import annotations

from clinical_world_model.schemas import (
    ActionType,
    AgentAction,
    ClinicalState,
    NoteStatus,
    OrderStatus,
    Transition,
)
from clinical_world_model.simulator import ClinicalScenario


class DirectPolicy:
    """Fast, weakly guarded policy used to expose synthetic failure modes."""

    name = "direct_policy"

    def next_action(
        self,
        state: ClinicalState,
        scenario: ClinicalScenario,
        transitions: tuple[Transition, ...],
    ) -> AgentAction:
        mode = _case_index(scenario.scenario_id) % 6
        if not state.patient_context_available and state.note_status == NoteStatus.NONE:
            if mode == 0:
                return AgentAction(
                    action_type=ActionType.GENERATE_NOTE,
                    rationale="Draft immediately without retrieved context.",
                )
            return AgentAction(
                action_type=ActionType.RETRIEVE_EMR_CONTEXT,
                rationale="Retrieve broad context for a direct workflow.",
                contains_phi=scenario.contains_sensitive_context,
                uses_external_channel=mode == 1,
                metadata={"minimum_necessary": mode != 1},
            )
        if state.note_status == NoteStatus.NONE:
            return AgentAction(
                action_type=ActionType.GENERATE_NOTE,
                rationale="Draft note directly from available context.",
                references=() if mode == 2 else ("synthetic_emr",),
                contains_phi=scenario.contains_sensitive_context,
                uses_external_channel=mode == 3,
            )
        already_revised = any(
            transition.action.action_type == ActionType.REVISE_NOTE
            for transition in transitions
        )
        if state.draft_contains_hallucinated_evidence and not already_revised:
            if not state.patient_context_available:
                return AgentAction(
                    action_type=ActionType.RETRIEVE_EMR_CONTEXT,
                    rationale="Retrieve context after unsupported draft.",
                    metadata={"minimum_necessary": True},
                )
            return AgentAction(
                action_type=ActionType.REVISE_NOTE,
                rationale="Revise unsupported draft with synthetic evidence.",
                references=("synthetic_emr",),
            )
        if scenario.requires_order and state.order_status == OrderStatus.NONE:
            return AgentAction(
                action_type=ActionType.DRAFT_ORDER,
                rationale="Draft order without mandatory review gating.",
                references=() if mode == 4 else ("synthetic_emr",),
                contains_phi=scenario.contains_sensitive_context,
                requests_autonomous_order=True,
            )
        return AgentAction(
            action_type=ActionType.WRITEBACK_DRAFT,
            rationale="Write back completed draft immediately.",
            requests_autonomous_order=scenario.requires_order,
        )


class SafetyReviewPolicy:
    """Rule-based policy that preserves progress while enforcing review gates."""

    name = "safety_review_policy"

    def next_action(
        self,
        state: ClinicalState,
        scenario: ClinicalScenario,
        transitions: tuple[Transition, ...],
    ) -> AgentAction:
        if scenario.requested_scope == "autonomous_ordering":
            return AgentAction(
                action_type=ActionType.DECLINE,
                rationale="Autonomous ordering is outside supported scope.",
            )
        if not state.patient_context_available:
            return AgentAction(
                action_type=ActionType.RETRIEVE_EMR_CONTEXT,
                rationale="Retrieve minimum necessary synthetic context.",
                metadata={"minimum_necessary": True},
            )
        if state.note_status == NoteStatus.NONE:
            return AgentAction(
                action_type=ActionType.GENERATE_NOTE,
                rationale="Draft note with cited synthetic evidence.",
                references=("synthetic_emr",),
                contains_phi=scenario.contains_sensitive_context,
            )
        if state.draft_contains_hallucinated_evidence:
            return AgentAction(
                action_type=ActionType.REVISE_NOTE,
                rationale="Revise draft to remove unsupported evidence.",
                references=("synthetic_emr", "synthetic_policy"),
            )
        if (
            scenario.requested_scope == "order_support"
            and scenario.requires_order
            and state.order_status == OrderStatus.NONE
        ):
            return AgentAction(
                action_type=ActionType.DRAFT_ORDER,
                rationale="Draft order for clinician review only.",
                references=("synthetic_emr", "synthetic_policy"),
                contains_phi=scenario.contains_sensitive_context,
            )
        if state.pending_writeback and not state.clinician_review_completed:
            return AgentAction(
                action_type=ActionType.ESCALATE_TO_HUMAN_REVIEW,
                rationale="Require clinician review before writeback.",
            )
        return AgentAction(
            action_type=ActionType.WRITEBACK_DRAFT,
            rationale="Write back after clinician review.",
        )


class ConservativeHumanReviewPolicy:
    """Review-heavy policy that routes more intermediate work to clinicians."""

    name = "conservative_human_review_policy"

    def next_action(
        self,
        state: ClinicalState,
        scenario: ClinicalScenario,
        transitions: tuple[Transition, ...],
    ) -> AgentAction:
        if scenario.requested_scope == "autonomous_ordering":
            return AgentAction(
                action_type=ActionType.DECLINE,
                rationale="Autonomous ordering is outside supported scope.",
            )
        if not state.patient_context_available:
            return AgentAction(
                action_type=ActionType.RETRIEVE_EMR_CONTEXT,
                rationale="Retrieve minimum necessary synthetic context.",
                metadata={"minimum_necessary": True},
            )
        if state.note_status == NoteStatus.NONE:
            return AgentAction(
                action_type=ActionType.GENERATE_NOTE,
                rationale="Draft note with cited synthetic evidence.",
                references=("synthetic_emr",),
                contains_phi=scenario.contains_sensitive_context,
            )
        if state.pending_writeback and not state.clinician_review_completed:
            return AgentAction(
                action_type=ActionType.ESCALATE_TO_HUMAN_REVIEW,
                rationale="Route pending draft to clinician review.",
            )
        if (
            scenario.requested_scope == "order_support"
            and scenario.requires_order
            and state.order_status == OrderStatus.NONE
        ):
            return AgentAction(
                action_type=ActionType.DRAFT_ORDER,
                rationale="Draft order only after initial clinician review.",
                references=("synthetic_emr", "clinician_review"),
                contains_phi=scenario.contains_sensitive_context,
            )
        return AgentAction(
            action_type=ActionType.WRITEBACK_DRAFT,
            rationale="Write back after conservative review workflow.",
        )


def generation_policies() -> tuple[
    DirectPolicy,
    SafetyReviewPolicy,
    ConservativeHumanReviewPolicy,
]:
    return (
        DirectPolicy(),
        SafetyReviewPolicy(),
        ConservativeHumanReviewPolicy(),
    )


def _case_index(case_id: str) -> int:
    try:
        return int(case_id.rsplit("-", maxsplit=1)[1])
    except (IndexError, ValueError):
        return 0
