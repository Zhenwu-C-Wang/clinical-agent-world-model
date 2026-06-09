from __future__ import annotations

from dataclasses import dataclass, replace

from clinical_world_model.generation import read_trajectories_jsonl, sample_scenarios
from clinical_world_model.metrics import (
    PolicyMetrics,
    evaluate_policy,
    metrics_markdown_table,
    safety_violation_counts,
)
from clinical_world_model.policies import DirectPolicy, SafetyReviewPolicy
from clinical_world_model.schemas import (
    ActionType,
    AgentAction,
    ClinicalState,
    NoteStatus,
    OrderStatus,
    WorkflowStage,
)
from clinical_world_model.simulator import (
    AUDIT_CLINICIAN_REVIEW_COMPLETED,
    AUDIT_DECLINED_OUT_OF_SCOPE,
    AUDIT_EMR_CONTEXT_RETRIEVED,
    AUDIT_NOTE_DRAFT_CREATED,
    AUDIT_NOTE_REVISED,
    AUDIT_ORDER_DRAFT_CREATED,
    AUDIT_OVERBROAD_CONTEXT_RETRIEVED,
    AUDIT_WRITEBACK_RECORDED,
    ClinicalScenario,
    Trajectory,
    Transition,
)
from clinical_world_model.world_model import (
    TrainedWorldModel,
    WorldModelPrediction,
    train_world_model,
)


@dataclass(frozen=True)
class PlannerObjective:
    safety_risk_weight: float = 2.5
    delay_weight: float = 0.01
    clinician_burden_weight: float = 0.25
    future_discount: float = 0.85


class WorldModelLookaheadPolicy:
    name = "world_model_lookahead_policy"

    def __init__(
        self,
        world_model: TrainedWorldModel,
        horizon: int = 3,
        objective: PlannerObjective | None = None,
    ) -> None:
        self.world_model = world_model
        self.horizon = horizon
        self.objective = objective or PlannerObjective()
        self._prediction_cache: dict[
            tuple[tuple[object, ...], tuple[object, ...]],
            WorldModelPrediction,
        ] = {}
        self._score_cache: dict[
            tuple[tuple[object, ...], tuple[object, ...], int],
            float,
        ] = {}

    def next_action(
        self,
        state: ClinicalState,
        scenario: ClinicalScenario,
        transitions: tuple[Transition, ...],
    ) -> AgentAction:
        candidates = candidate_actions(state, scenario)
        if not candidates:
            return AgentAction(
                action_type=ActionType.DECLINE,
                rationale="No valid synthetic workflow action remains.",
            )
        scored = [
            (
                self._score_action(action, state, scenario, self.horizon),
                action,
            )
            for action in candidates
        ]
        return max(scored, key=lambda item: item[0])[1]

    def _score_action(
        self,
        action: AgentAction,
        state: ClinicalState,
        scenario: ClinicalScenario,
        depth: int,
    ) -> float:
        prediction = self._predict(state, action)
        next_state = project_next_state(state, action, prediction)
        progress_gain = max(0.0, prediction.next_task_progress - state.task_progress)
        clinician_burden = (
            1.0 if action.action_type == ActionType.ESCALATE_TO_HUMAN_REVIEW else 0.0
        )
        immediate_utility = (
            progress_gain
            - self.objective.safety_risk_weight
            * prediction.safety_violation_probability
            - self.objective.delay_weight * prediction.expected_delay_minutes
            - self.objective.clinician_burden_weight * clinician_burden
        )
        if depth <= 1 or next_state.workflow_stage in {
            WorkflowStage.COMPLETED,
            WorkflowStage.DECLINED,
        }:
            return immediate_utility
        return immediate_utility + self.objective.future_discount * self._score_state(
            next_state,
            scenario,
            depth - 1,
        )

    def _score_state(
        self,
        state: ClinicalState,
        scenario: ClinicalScenario,
        depth: int,
    ) -> float:
        cache_key = (_state_key(state), _scenario_key(scenario), depth)
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]
        candidates = candidate_actions(state, scenario)
        if not candidates:
            return 0.0
        score = max(
            self._score_action(action, state, scenario, depth)
            for action in candidates
        )
        self._score_cache[cache_key] = score
        return score

    def _predict(
        self,
        state: ClinicalState,
        action: AgentAction,
    ) -> WorldModelPrediction:
        cache_key = (_state_key(state), _action_key(action))
        if cache_key not in self._prediction_cache:
            self._prediction_cache[cache_key] = self.world_model.predict(state, action)
        return self._prediction_cache[cache_key]


def candidate_actions(
    state: ClinicalState,
    scenario: ClinicalScenario,
) -> list[AgentAction]:
    if state.workflow_stage in {WorkflowStage.COMPLETED, WorkflowStage.DECLINED}:
        return []
    if scenario.requested_scope == "autonomous_ordering":
        return [
            AgentAction(
                action_type=ActionType.DECLINE,
                rationale="Autonomous ordering is outside supported scope.",
            )
        ]

    candidates: list[AgentAction] = []
    if not state.patient_context_available:
        candidates.extend(
            [
                AgentAction(
                    action_type=ActionType.RETRIEVE_EMR_CONTEXT,
                    rationale="Retrieve minimum necessary synthetic context.",
                    metadata={"minimum_necessary": True},
                ),
                AgentAction(
                    action_type=ActionType.GENERATE_NOTE,
                    rationale="Draft without context for comparison.",
                ),
            ]
        )
        return candidates

    if state.note_status == NoteStatus.NONE:
        candidates.extend(
            [
                AgentAction(
                    action_type=ActionType.GENERATE_NOTE,
                    rationale="Draft note with cited synthetic evidence.",
                    references=("synthetic_emr",),
                    contains_phi=scenario.contains_sensitive_context,
                ),
                AgentAction(
                    action_type=ActionType.GENERATE_NOTE,
                    rationale="Draft note without evidence for comparison.",
                    contains_phi=scenario.contains_sensitive_context,
                ),
            ]
        )
        return candidates

    if state.draft_contains_hallucinated_evidence:
        candidates.append(
            AgentAction(
                action_type=ActionType.REVISE_NOTE,
                rationale="Revise unsupported note with synthetic evidence.",
                references=("synthetic_emr",),
            )
        )

    if (
        scenario.requested_scope == "order_support"
        and scenario.requires_order
        and state.order_status == OrderStatus.NONE
    ):
        candidates.extend(
            [
                AgentAction(
                    action_type=ActionType.DRAFT_ORDER,
                    rationale="Draft order for clinician review.",
                    references=("synthetic_emr", "synthetic_policy"),
                    contains_phi=scenario.contains_sensitive_context,
                ),
                AgentAction(
                    action_type=ActionType.DRAFT_ORDER,
                    rationale="Draft autonomous order for comparison.",
                    references=("synthetic_emr",),
                    contains_phi=scenario.contains_sensitive_context,
                    requests_autonomous_order=True,
                ),
            ]
        )

    if state.pending_writeback and not state.clinician_review_completed:
        candidates.extend(
            [
                AgentAction(
                    action_type=ActionType.ESCALATE_TO_HUMAN_REVIEW,
                    rationale="Escalate to clinician review before writeback.",
                ),
                AgentAction(
                    action_type=ActionType.WRITEBACK_DRAFT,
                    rationale="Write back without review for comparison.",
                ),
            ]
        )
    elif state.pending_writeback:
        candidates.append(
            AgentAction(
                action_type=ActionType.WRITEBACK_DRAFT,
                rationale="Write back after clinician review.",
            )
        )

    if not candidates:
        candidates.append(
            AgentAction(
                action_type=ActionType.ESCALATE_TO_HUMAN_REVIEW,
                rationale="Escalate ambiguous workflow state.",
            )
        )
    return candidates


def project_next_state(
    state: ClinicalState,
    action: AgentAction,
    prediction: WorldModelPrediction,
) -> ClinicalState:
    audit_event = audit_event_for_action(action)
    audit_events = state.audit_events + ((audit_event,) if audit_event else ())
    next_stage = WorkflowStage(prediction.next_workflow_stage)
    elapsed_minutes = state.elapsed_minutes + max(
        0,
        int(round(prediction.expected_delay_minutes)),
    )
    updates = {
        "workflow_stage": next_stage,
        "elapsed_minutes": elapsed_minutes,
        "task_progress": max(state.task_progress, prediction.next_task_progress),
        "audit_events": audit_events,
    }

    if action.action_type == ActionType.RETRIEVE_EMR_CONTEXT:
        updates.update(
            patient_context_available=True,
            minimum_necessary_context=bool(
                action.metadata.get("minimum_necessary", True)
            ),
        )
    elif action.action_type == ActionType.GENERATE_NOTE:
        has_evidence = state.patient_context_available and bool(action.references)
        updates.update(
            note_status=NoteStatus.DRAFT if has_evidence else NoteStatus.NEEDS_REVISION,
            pending_writeback=True,
            evidence_refs=tuple(sorted(set(state.evidence_refs + action.references))),
            draft_contains_phi=state.draft_contains_phi or action.contains_phi,
            draft_contains_hallucinated_evidence=(
                state.draft_contains_hallucinated_evidence or not has_evidence
            ),
            clinician_review_completed=False,
        )
    elif action.action_type == ActionType.REVISE_NOTE:
        updates.update(
            note_status=NoteStatus.REVISED,
            evidence_refs=tuple(sorted(set(state.evidence_refs + action.references))),
            draft_contains_hallucinated_evidence=False,
            draft_contains_phi=state.draft_contains_phi or action.contains_phi,
            clinician_review_completed=False,
        )
    elif action.action_type == ActionType.DRAFT_ORDER:
        has_evidence = state.patient_context_available and bool(action.references)
        updates.update(
            order_status=OrderStatus.DRAFT,
            pending_writeback=True,
            clinician_review_required=True,
            clinician_review_completed=False,
            evidence_refs=tuple(sorted(set(state.evidence_refs + action.references))),
            draft_contains_phi=state.draft_contains_phi or action.contains_phi,
            draft_contains_hallucinated_evidence=(
                state.draft_contains_hallucinated_evidence or not has_evidence
            ),
        )
    elif action.action_type == ActionType.ESCALATE_TO_HUMAN_REVIEW:
        updates.update(
            clinician_review_required=True,
            clinician_review_completed=True,
            order_status=(
                OrderStatus.READY_FOR_REVIEW
                if state.order_status == OrderStatus.DRAFT
                else state.order_status
            ),
        )
    elif action.action_type == ActionType.WRITEBACK_DRAFT:
        updates.update(
            pending_writeback=False,
            note_status=(
                NoteStatus.WRITTEN
                if state.note_status != NoteStatus.NONE
                else state.note_status
            ),
            order_status=(
                OrderStatus.WRITTEN
                if state.order_status != OrderStatus.NONE
                else state.order_status
            ),
        )
    elif action.action_type == ActionType.DECLINE:
        updates.update(declined=True, pending_writeback=False)

    return replace(state, **updates)


def audit_event_for_action(action: AgentAction) -> str:
    if action.action_type == ActionType.RETRIEVE_EMR_CONTEXT:
        return (
            AUDIT_EMR_CONTEXT_RETRIEVED
            if action.metadata.get("minimum_necessary", True)
            else AUDIT_OVERBROAD_CONTEXT_RETRIEVED
        )
    if action.action_type == ActionType.GENERATE_NOTE:
        return AUDIT_NOTE_DRAFT_CREATED
    if action.action_type == ActionType.REVISE_NOTE:
        return AUDIT_NOTE_REVISED
    if action.action_type == ActionType.DRAFT_ORDER:
        return AUDIT_ORDER_DRAFT_CREATED
    if action.action_type == ActionType.ESCALATE_TO_HUMAN_REVIEW:
        return AUDIT_CLINICIAN_REVIEW_COMPLETED
    if action.action_type == ActionType.WRITEBACK_DRAFT:
        return AUDIT_WRITEBACK_RECORDED
    if action.action_type == ActionType.DECLINE:
        return AUDIT_DECLINED_OUT_OF_SCOPE
    return ""


def _state_key(state: ClinicalState) -> tuple[object, ...]:
    return (
        state.workflow_stage.value,
        state.task_type,
        state.requested_scope,
        state.acuity,
        state.contains_sensitive_context,
        state.patient_context_available,
        state.minimum_necessary_context,
        state.note_status.value,
        state.order_status.value,
        state.pending_writeback,
        state.clinician_review_required,
        state.clinician_review_completed,
        len(state.audit_events),
        len(state.evidence_refs),
        state.draft_contains_phi,
        state.draft_contains_hallucinated_evidence,
        round(state.task_progress, 3),
        state.elapsed_minutes,
        state.declined,
    )


def _action_key(action: AgentAction) -> tuple[object, ...]:
    return (
        action.action_type.value,
        action.contains_phi,
        action.uses_external_channel,
        action.requests_autonomous_order,
        len(action.references),
        bool(action.references),
        bool(action.metadata.get("minimum_necessary", False)),
    )


def _scenario_key(scenario: ClinicalScenario) -> tuple[object, ...]:
    return (
        scenario.task_type,
        scenario.requested_scope,
        scenario.acuity,
        scenario.requires_order,
        scenario.contains_sensitive_context,
        scenario.supported,
    )


def evaluate_planner_comparison(
    training_path: str,
    count: int = 1000,
    seed: int = 42,
    horizon: int = 3,
    max_steps: int = 8,
) -> tuple[list[PolicyMetrics], dict[str, list[Trajectory]]]:
    training_trajectories = read_trajectories_jsonl(training_path)
    world_model = train_world_model(training_trajectories, seed=seed)
    scenarios = sample_scenarios(count=count, seed=seed)
    policies = [
        DirectPolicy(),
        SafetyReviewPolicy(),
        WorldModelLookaheadPolicy(world_model=world_model, horizon=horizon),
    ]
    metrics: list[PolicyMetrics] = []
    trajectories_by_policy: dict[str, list[Trajectory]] = {}
    for policy in policies:
        policy_metrics, trajectories = evaluate_policy(
            policy=policy,
            scenarios=scenarios,
            seed=seed,
            max_steps=max_steps,
        )
        metrics.append(policy_metrics)
        trajectories_by_policy[policy.name] = trajectories
    return metrics, trajectories_by_policy


def render_planner_report(
    metrics: list[PolicyMetrics],
    trajectories_by_policy: dict[str, list[Trajectory]],
    count: int,
    seed: int,
    horizon: int,
) -> str:
    lines = [
        "# Planner Results",
        "",
        "This report compares direct action, rule-based safety, and learned world-model lookahead policies on the same synthetic scenarios.",
        "The lookahead policy uses a lightweight learned world model to score candidate actions before executing them in the simulator.",
        "",
        f"Scenario count per policy: `{count}`.",
        f"Scenario seed: `{seed}`.",
        f"Lookahead horizon: `{horizon}` steps.",
        "",
        "## Planner Objective",
        "",
        "```text",
        "utility = task_progress",
        "        - safety_risk_penalty",
        "        - delay_penalty",
        "        - clinician_burden_penalty",
        "```",
        "",
        "## Policy Comparison",
        "",
        metrics_markdown_table(metrics),
        "",
        "## Safety Violation Counts",
        "",
        "| policy | unsafe_writeback | phi_leakage | hallucinated_evidence | missing_clinician_review | scope_escalation |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy_name, trajectories in trajectories_by_policy.items():
        counts = safety_violation_counts(trajectories)
        lines.append(
            "| "
            + " | ".join(
                [
                    policy_name,
                    str(counts["unsafe_writeback"]),
                    str(counts["phi_leakage"]),
                    str(counts["hallucinated_evidence"]),
                    str(counts["missing_clinician_review"]),
                    str(counts["scope_escalation"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `direct_policy` is fast but exposes unsafe writeback and scope escalation.",
            "- `safety_review_policy` uses fixed rules and accepts conservative false declines.",
            "- `world_model_lookahead_policy` uses the learned world model to avoid high-risk actions while preserving task success on supported workflows.",
            "",
            "## Limitations",
            "",
            "- Planning is evaluated only in this synthetic simulator.",
            "- The learned world model is trained on rule-generated data, so strong planner results do not imply clinical readiness.",
            "- This is not medical advice and is not a clinical decision support system.",
            "",
        ]
    )
    return "\n".join(lines)
