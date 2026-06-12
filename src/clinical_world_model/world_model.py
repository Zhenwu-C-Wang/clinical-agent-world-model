from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from clinical_world_model.generation import read_trajectories_jsonl
from clinical_world_model.schemas import AgentAction, ClinicalState
from clinical_world_model.simulator import (
    AUDIT_CLINICIAN_REVIEW_COMPLETED,
    AUDIT_DECLINED_OUT_OF_SCOPE,
    AUDIT_EMR_CONTEXT_RETRIEVED,
    AUDIT_NOTE_DRAFT_CREATED,
    AUDIT_ORDER_DRAFT_CREATED,
    AUDIT_WRITEBACK_RECORDED,
    ClinicalScenario,
    Trajectory,
)


FeatureRow = dict[str, str | int | float | bool]


@dataclass(frozen=True)
class WorldModelDataset:
    features: list[FeatureRow]
    next_workflow_stage: list[str]
    safety_violation: list[int]
    delay_minutes: list[float]
    audit_completeness: list[float]
    next_task_progress: list[float]


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    labels: list[str]
    confusion_matrix: list[list[int]]


@dataclass(frozen=True)
class BinaryRiskMetrics:
    accuracy: float
    f1: float
    brier_score: float
    calibration_bins: list[dict[str, float | int]]


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    r2: float


@dataclass(frozen=True)
class WorldModelEval:
    train_rows: int
    test_rows: int
    next_stage: ClassificationMetrics
    safety_violation: BinaryRiskMetrics
    delay: RegressionMetrics
    audit_completeness: RegressionMetrics


@dataclass(frozen=True)
class WorldModelPrediction:
    next_workflow_stage: str
    safety_violation_probability: float
    expected_delay_minutes: float
    audit_completeness: float
    next_task_progress: float


@dataclass(frozen=True)
class TrainedWorldModel:
    next_stage_model: Pipeline
    safety_model: Pipeline
    delay_model: Pipeline
    audit_model: Pipeline
    progress_model: Pipeline

    def predict(
        self,
        state: ClinicalState,
        action: AgentAction,
    ) -> WorldModelPrediction:
        features = [featurize_state_action(state, action)]
        safety_probability = _positive_class_probability(
            self.safety_model,
            features,
        )
        return WorldModelPrediction(
            next_workflow_stage=str(self.next_stage_model.predict(features)[0]),
            safety_violation_probability=safety_probability,
            expected_delay_minutes=max(0.0, float(self.delay_model.predict(features)[0])),
            audit_completeness=min(
                1.0,
                max(0.0, float(self.audit_model.predict(features)[0])),
            ),
            next_task_progress=min(
                1.0,
                max(0.0, float(self.progress_model.predict(features)[0])),
            ),
        )


def build_dataset(trajectories: list[Trajectory]) -> WorldModelDataset:
    features: list[FeatureRow] = []
    next_workflow_stage: list[str] = []
    safety_violation: list[int] = []
    delay_minutes: list[float] = []
    audit_completeness: list[float] = []
    next_task_progress: list[float] = []
    for trajectory in trajectories:
        for transition in trajectory.transitions:
            features.append(
                featurize_state_action(
                    transition.state_before,
                    transition.action,
                )
            )
            next_workflow_stage.append(transition.state_after.workflow_stage.value)
            safety_violation.append(int(bool(transition.outcome.safety_violations)))
            delay_minutes.append(float(transition.outcome.delay_minutes))
            audit_completeness.append(
                audit_completeness_for_state(
                    trajectory.scenario,
                    transition.state_after,
                )
            )
            next_task_progress.append(transition.state_after.task_progress)
    if not features:
        raise ValueError("Cannot build a world-model dataset from zero transitions")
    return WorldModelDataset(
        features=features,
        next_workflow_stage=next_workflow_stage,
        safety_violation=safety_violation,
        delay_minutes=delay_minutes,
        audit_completeness=audit_completeness,
        next_task_progress=next_task_progress,
    )


def featurize_state_action(
    state: ClinicalState,
    action: AgentAction,
) -> FeatureRow:
    return {
        "workflow_stage": state.workflow_stage.value,
        "task_type": state.task_type,
        "requested_scope": state.requested_scope,
        "acuity": state.acuity,
        "contains_sensitive_context": state.contains_sensitive_context,
        "patient_context_available": state.patient_context_available,
        "minimum_necessary_context": state.minimum_necessary_context,
        "note_status": state.note_status.value,
        "order_status": state.order_status.value,
        "pending_writeback": state.pending_writeback,
        "clinician_review_required": state.clinician_review_required,
        "clinician_review_completed": state.clinician_review_completed,
        "audit_event_count": len(state.audit_events),
        "evidence_ref_count": len(state.evidence_refs),
        "draft_contains_phi": state.draft_contains_phi,
        "draft_contains_hallucinated_evidence": (
            state.draft_contains_hallucinated_evidence
        ),
        "task_progress": state.task_progress,
        "elapsed_minutes": state.elapsed_minutes,
        "declined": state.declined,
        "action_type": action.action_type.value,
        "action_contains_phi": action.contains_phi,
        "action_uses_external_channel": action.uses_external_channel,
        "action_requests_autonomous_order": action.requests_autonomous_order,
        "action_reference_count": len(action.references),
        "action_has_references": bool(action.references),
        "action_minimum_necessary": bool(
            action.metadata.get("minimum_necessary", False)
        ),
    }


def audit_completeness_for_state(
    scenario: ClinicalScenario,
    state: ClinicalState,
) -> float:
    event_set = set(state.audit_events)
    if not scenario.supported:
        expected = {AUDIT_DECLINED_OUT_OF_SCOPE}
    else:
        expected = {
            AUDIT_EMR_CONTEXT_RETRIEVED,
            AUDIT_NOTE_DRAFT_CREATED,
            AUDIT_CLINICIAN_REVIEW_COMPLETED,
            AUDIT_WRITEBACK_RECORDED,
        }
        if scenario.requested_scope == "order_support" and scenario.requires_order:
            expected.add(AUDIT_ORDER_DRAFT_CREATED)
    return len(event_set & expected) / len(expected)


def train_and_evaluate_world_model(
    trajectories: list[Trajectory],
    seed: int = 42,
    test_size: float = 0.25,
) -> WorldModelEval:
    dataset = build_dataset(trajectories)
    indices = list(range(len(dataset.features)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=dataset.safety_violation,
    )

    train_features = [dataset.features[index] for index in train_indices]
    test_features = [dataset.features[index] for index in test_indices]

    next_stage_model = _classifier(seed)
    safety_model = _classifier(seed + 1)
    delay_model = _regressor(seed + 2)
    audit_model = _regressor(seed + 3)
    progress_model = _regressor(seed + 4)

    next_stage_model.fit(
        train_features,
        [dataset.next_workflow_stage[index] for index in train_indices],
    )
    safety_model.fit(
        train_features,
        [dataset.safety_violation[index] for index in train_indices],
    )
    delay_model.fit(
        train_features,
        [dataset.delay_minutes[index] for index in train_indices],
    )
    audit_model.fit(
        train_features,
        [dataset.audit_completeness[index] for index in train_indices],
    )
    progress_model.fit(
        train_features,
        [dataset.next_task_progress[index] for index in train_indices],
    )

    true_next_stage = [dataset.next_workflow_stage[index] for index in test_indices]
    pred_next_stage = list(next_stage_model.predict(test_features))
    labels = sorted(set(true_next_stage) | set(pred_next_stage))

    true_safety = [dataset.safety_violation[index] for index in test_indices]
    pred_safety = list(safety_model.predict(test_features))
    safety_prob = [
        float(probabilities[1])
        for probabilities in safety_model.predict_proba(test_features)
    ]

    true_delay = [dataset.delay_minutes[index] for index in test_indices]
    pred_delay = list(delay_model.predict(test_features))

    true_audit = [dataset.audit_completeness[index] for index in test_indices]
    pred_audit = list(audit_model.predict(test_features))

    return WorldModelEval(
        train_rows=len(train_indices),
        test_rows=len(test_indices),
        next_stage=ClassificationMetrics(
            accuracy=accuracy_score(true_next_stage, pred_next_stage),
            macro_f1=f1_score(true_next_stage, pred_next_stage, average="macro"),
            labels=labels,
            confusion_matrix=confusion_matrix(
                true_next_stage,
                pred_next_stage,
                labels=labels,
            ).tolist(),
        ),
        safety_violation=BinaryRiskMetrics(
            accuracy=accuracy_score(true_safety, pred_safety),
            f1=f1_score(true_safety, pred_safety, zero_division=0),
            brier_score=brier_score_loss(true_safety, safety_prob),
            calibration_bins=calibration_bins(true_safety, safety_prob, bin_count=2),
        ),
        delay=RegressionMetrics(
            mae=mean_absolute_error(true_delay, pred_delay),
            r2=r2_score(true_delay, pred_delay),
        ),
        audit_completeness=RegressionMetrics(
            mae=mean_absolute_error(true_audit, pred_audit),
            r2=r2_score(true_audit, pred_audit),
        ),
    )


def train_world_model(
    trajectories: list[Trajectory],
    seed: int = 42,
) -> TrainedWorldModel:
    dataset = build_dataset(trajectories)
    next_stage_model = _classifier(seed)
    safety_model = _classifier(seed + 1)
    delay_model = _regressor(seed + 2)
    audit_model = _regressor(seed + 3)
    progress_model = _regressor(seed + 4)

    next_stage_model.fit(dataset.features, dataset.next_workflow_stage)
    safety_model.fit(dataset.features, dataset.safety_violation)
    delay_model.fit(dataset.features, dataset.delay_minutes)
    audit_model.fit(dataset.features, dataset.audit_completeness)
    progress_model.fit(dataset.features, dataset.next_task_progress)

    return TrainedWorldModel(
        next_stage_model=next_stage_model,
        safety_model=safety_model,
        delay_model=delay_model,
        audit_model=audit_model,
        progress_model=progress_model,
    )


def train_and_evaluate_from_jsonl(
    path: str | Path,
    seed: int = 42,
    test_size: float = 0.25,
) -> WorldModelEval:
    return train_and_evaluate_world_model(
        trajectories=read_trajectories_jsonl(path),
        seed=seed,
        test_size=test_size,
    )


def calibration_bins(
    true_labels: list[int],
    probabilities: list[float],
    bin_count: int = 5,
) -> list[dict[str, float | int]]:
    bins: list[dict[str, float | int]] = []
    for bin_index in range(bin_count):
        lower = bin_index / bin_count
        upper = (bin_index + 1) / bin_count
        selected = [
            (true_label, probability)
            for true_label, probability in zip(true_labels, probabilities)
            if lower <= probability < upper
            or (bin_index == bin_count - 1 and probability == 1.0)
        ]
        if not selected:
            bins.append(
                {
                    "bin": f"{lower:.1f}-{upper:.1f}",
                    "count": 0,
                    "avg_predicted_risk": 0.0,
                    "observed_risk": 0.0,
                }
            )
            continue
        bins.append(
            {
                "bin": f"{lower:.1f}-{upper:.1f}",
                "count": len(selected),
                "avg_predicted_risk": sum(item[1] for item in selected)
                / len(selected),
                "observed_risk": sum(item[0] for item in selected) / len(selected),
            }
        )
    return bins


def render_world_model_report(evaluation: WorldModelEval) -> str:
    lines = [
        "# World Model Evaluation",
        "",
        "This report trains lightweight random-forest world models on synthetic trajectories.",
        "Inputs are synthetic `state + action` features; targets are next workflow state, safety violation risk, action delay, and audit completeness.",
        "",
        f"Training rows: `{evaluation.train_rows}`.",
        f"Test rows: `{evaluation.test_rows}`.",
        "",
        "## Summary Metrics",
        "",
        "| target | metric | value |",
        "| --- | --- | ---: |",
        f"| next workflow state | accuracy | {evaluation.next_stage.accuracy:.3f} |",
        f"| next workflow state | macro F1 | {evaluation.next_stage.macro_f1:.3f} |",
        f"| safety violation | accuracy | {evaluation.safety_violation.accuracy:.3f} |",
        f"| safety violation | F1 | {evaluation.safety_violation.f1:.3f} |",
        f"| safety violation | Brier score | {evaluation.safety_violation.brier_score:.3f} |",
        f"| expected delay | MAE minutes | {evaluation.delay.mae:.2f} |",
        f"| expected delay | R2 | {evaluation.delay.r2:.3f} |",
        f"| audit completeness | MAE | {evaluation.audit_completeness.mae:.3f} |",
        f"| audit completeness | R2 | {evaluation.audit_completeness.r2:.3f} |",
        "",
        "## Next Workflow State Confusion Matrix",
        "",
        confusion_matrix_markdown(
            evaluation.next_stage.labels,
            evaluation.next_stage.confusion_matrix,
        ),
        "",
        "## Safety Risk Calibration",
        "",
        "Decision-level probability bins keep the checked-in report stable across scikit-learn wheels while still showing whether low-risk and high-risk predictions are separated.",
        "",
        "| predicted risk bin | n | observed risk |",
        "| --- | ---: | ---: |",
    ]
    for item in evaluation.safety_violation.calibration_bins:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["bin"]),
                    str(item["count"]),
                    f"{float(item['observed_risk']):.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The model is trained only on synthetic rule-generated trajectories.",
            "- Strong metrics here mean the model learned this simulator's dynamics, not real hospital behavior.",
            "- The model is not medical advice and is not a clinical decision support system.",
            "",
        ]
    )
    return "\n".join(lines)


def confusion_matrix_markdown(labels: list[str], matrix: list[list[int]]) -> str:
    headers = ["actual \\ predicted", *labels]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for label, values in zip(labels, matrix):
        rows.append(
            "| "
            + " | ".join([label, *[str(value) for value in values]])
            + " |"
        )
    return "\n".join(rows)


def _classifier(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=False)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=10,
                    min_samples_leaf=2,
                    random_state=seed,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _regressor(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=False)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=120,
                    max_depth=10,
                    min_samples_leaf=2,
                    random_state=seed,
                ),
            ),
        ]
    )


def _positive_class_probability(model: Pipeline, features: list[FeatureRow]) -> float:
    probabilities = model.predict_proba(features)[0]
    classes = list(model.named_steps["model"].classes_)
    if 1 not in classes:
        return 0.0
    return float(probabilities[classes.index(1)])
