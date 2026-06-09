from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StringEnum(str, Enum):
    """Enum that serializes naturally as a string value."""

    def __str__(self) -> str:
        return self.value


class WorkflowStage(StringEnum):
    INTAKE = "intake"
    CONTEXT_GATHERING = "context_gathering"
    NOTE_DRAFTING = "note_drafting"
    ORDER_DRAFTING = "order_drafting"
    PENDING_REVIEW = "pending_review"
    READY_FOR_WRITEBACK = "ready_for_writeback"
    COMPLETED = "completed"
    DECLINED = "declined"


class NoteStatus(StringEnum):
    NONE = "none"
    DRAFT = "draft"
    NEEDS_REVISION = "needs_revision"
    REVISED = "revised"
    WRITTEN = "written"


class OrderStatus(StringEnum):
    NONE = "none"
    DRAFT = "draft"
    READY_FOR_REVIEW = "ready_for_review"
    WRITTEN = "written"


class ActionType(StringEnum):
    RETRIEVE_EMR_CONTEXT = "retrieve_emr_context"
    GENERATE_NOTE = "generate_note"
    REVISE_NOTE = "revise_note"
    DRAFT_ORDER = "draft_order"
    WRITEBACK_DRAFT = "writeback_draft"
    ESCALATE_TO_HUMAN_REVIEW = "escalate_to_human_review"
    DECLINE = "decline"


class RiskType(StringEnum):
    UNSAFE_WRITEBACK = "unsafe_writeback"
    PHI_LEAKAGE = "phi_leakage"
    HALLUCINATED_EVIDENCE = "hallucinated_evidence"
    MISSING_CLINICIAN_REVIEW = "missing_clinician_review"
    SCOPE_ESCALATION = "scope_escalation"


class Severity(StringEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class SafetyViolation:
    violation_type: RiskType
    severity: Severity
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "violation_type": self.violation_type.value,
            "severity": self.severity.value,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SafetyViolation":
        return cls(
            violation_type=RiskType(data["violation_type"]),
            severity=Severity(data["severity"]),
            message=data["message"],
        )


@dataclass(frozen=True)
class ClinicalState:
    case_id: str
    workflow_stage: WorkflowStage
    task_type: str
    requested_scope: str
    acuity: str
    contains_sensitive_context: bool = False
    patient_context_available: bool = False
    minimum_necessary_context: bool = False
    note_status: NoteStatus = NoteStatus.NONE
    order_status: OrderStatus = OrderStatus.NONE
    pending_writeback: bool = False
    clinician_review_required: bool = True
    clinician_review_completed: bool = False
    audit_events: tuple[str, ...] = field(default_factory=tuple)
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)
    draft_contains_phi: bool = False
    draft_contains_hallucinated_evidence: bool = False
    task_progress: float = 0.0
    elapsed_minutes: int = 0
    declined: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.task_progress <= 1.0:
            raise ValueError("task_progress must be between 0 and 1")
        if self.elapsed_minutes < 0:
            raise ValueError("elapsed_minutes must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "workflow_stage": self.workflow_stage.value,
            "task_type": self.task_type,
            "requested_scope": self.requested_scope,
            "acuity": self.acuity,
            "contains_sensitive_context": self.contains_sensitive_context,
            "patient_context_available": self.patient_context_available,
            "minimum_necessary_context": self.minimum_necessary_context,
            "note_status": self.note_status.value,
            "order_status": self.order_status.value,
            "pending_writeback": self.pending_writeback,
            "clinician_review_required": self.clinician_review_required,
            "clinician_review_completed": self.clinician_review_completed,
            "audit_events": list(self.audit_events),
            "evidence_refs": list(self.evidence_refs),
            "draft_contains_phi": self.draft_contains_phi,
            "draft_contains_hallucinated_evidence": self.draft_contains_hallucinated_evidence,
            "task_progress": self.task_progress,
            "elapsed_minutes": self.elapsed_minutes,
            "declined": self.declined,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClinicalState":
        return cls(
            case_id=data["case_id"],
            workflow_stage=WorkflowStage(data["workflow_stage"]),
            task_type=data["task_type"],
            requested_scope=data["requested_scope"],
            acuity=data["acuity"],
            contains_sensitive_context=data.get("contains_sensitive_context", False),
            patient_context_available=data.get("patient_context_available", False),
            minimum_necessary_context=data.get("minimum_necessary_context", False),
            note_status=NoteStatus(data.get("note_status", NoteStatus.NONE.value)),
            order_status=OrderStatus(data.get("order_status", OrderStatus.NONE.value)),
            pending_writeback=data.get("pending_writeback", False),
            clinician_review_required=data.get("clinician_review_required", True),
            clinician_review_completed=data.get("clinician_review_completed", False),
            audit_events=tuple(data.get("audit_events", ())),
            evidence_refs=tuple(data.get("evidence_refs", ())),
            draft_contains_phi=data.get("draft_contains_phi", False),
            draft_contains_hallucinated_evidence=data.get(
                "draft_contains_hallucinated_evidence", False
            ),
            task_progress=float(data.get("task_progress", 0.0)),
            elapsed_minutes=int(data.get("elapsed_minutes", 0)),
            declined=data.get("declined", False),
        )


@dataclass(frozen=True)
class AgentAction:
    action_type: ActionType
    rationale: str
    references: tuple[str, ...] = field(default_factory=tuple)
    contains_phi: bool = False
    uses_external_channel: bool = False
    requests_autonomous_order: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "rationale": self.rationale,
            "references": list(self.references),
            "contains_phi": self.contains_phi,
            "uses_external_channel": self.uses_external_channel,
            "requests_autonomous_order": self.requests_autonomous_order,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentAction":
        return cls(
            action_type=ActionType(data["action_type"]),
            rationale=data.get("rationale", ""),
            references=tuple(data.get("references", ())),
            contains_phi=data.get("contains_phi", False),
            uses_external_channel=data.get("uses_external_channel", False),
            requests_autonomous_order=data.get("requests_autonomous_order", False),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class Outcome:
    success: bool
    task_progress_delta: float
    delay_minutes: int
    audit_event: str
    safety_violations: tuple[SafetyViolation, ...] = field(default_factory=tuple)
    clinician_review_burden: float = 0.0
    declined: bool = False
    message: str = ""

    def __post_init__(self) -> None:
        if self.delay_minutes < 0:
            raise ValueError("delay_minutes must be non-negative")
        if self.clinician_review_burden < 0:
            raise ValueError("clinician_review_burden must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "task_progress_delta": self.task_progress_delta,
            "delay_minutes": self.delay_minutes,
            "audit_event": self.audit_event,
            "safety_violations": [item.to_dict() for item in self.safety_violations],
            "clinician_review_burden": self.clinician_review_burden,
            "declined": self.declined,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Outcome":
        return cls(
            success=data["success"],
            task_progress_delta=float(data.get("task_progress_delta", 0.0)),
            delay_minutes=int(data.get("delay_minutes", 0)),
            audit_event=data.get("audit_event", ""),
            safety_violations=tuple(
                SafetyViolation.from_dict(item)
                for item in data.get("safety_violations", ())
            ),
            clinician_review_burden=float(data.get("clinician_review_burden", 0.0)),
            declined=data.get("declined", False),
            message=data.get("message", ""),
        )


@dataclass(frozen=True)
class Transition:
    state_before: ClinicalState
    action: AgentAction
    outcome: Outcome
    state_after: ClinicalState

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_before": self.state_before.to_dict(),
            "action": self.action.to_dict(),
            "outcome": self.outcome.to_dict(),
            "state_after": self.state_after.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transition":
        return cls(
            state_before=ClinicalState.from_dict(data["state_before"]),
            action=AgentAction.from_dict(data["action"]),
            outcome=Outcome.from_dict(data["outcome"]),
            state_after=ClinicalState.from_dict(data["state_after"]),
        )

