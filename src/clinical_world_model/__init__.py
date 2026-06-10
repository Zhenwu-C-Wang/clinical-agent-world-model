"""Synthetic hospital workflow world model package."""

from typing import Any

from clinical_world_model.schemas import (
    ActionType,
    AgentAction,
    ClinicalState,
    Outcome,
    RiskType,
    SafetyViolation,
    Transition,
)
from clinical_world_model.simulator import (
    ClinicalScenario,
    HospitalWorkflowSimulator,
    Trajectory,
    run_episode,
)
from clinical_world_model.policies import (
    ConservativeHumanReviewPolicy,
    DirectPolicy,
    SafetyReviewPolicy,
)
from clinical_world_model.metrics import PolicyMetrics, TrajectoryMetrics

_LAZY_EXPORTS = {
    "StressScenarioSummary": "clinical_world_model.stress",
    "TrainedWorldModel": "clinical_world_model.world_model",
    "WorldModelEval": "clinical_world_model.world_model",
    "WorldModelLookaheadPolicy": "clinical_world_model.planner",
    "build_dataset": "clinical_world_model.world_model",
    "train_and_evaluate_world_model": "clinical_world_model.world_model",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    "ActionType",
    "AgentAction",
    "ClinicalScenario",
    "ConservativeHumanReviewPolicy",
    "DirectPolicy",
    "ClinicalState",
    "HospitalWorkflowSimulator",
    "Outcome",
    "PolicyMetrics",
    "RiskType",
    "SafetyViolation",
    "SafetyReviewPolicy",
    "StressScenarioSummary",
    "Trajectory",
    "TrajectoryMetrics",
    "Transition",
    "TrainedWorldModel",
    "WorldModelLookaheadPolicy",
    "WorldModelEval",
    "build_dataset",
    "run_episode",
    "train_and_evaluate_world_model",
]
