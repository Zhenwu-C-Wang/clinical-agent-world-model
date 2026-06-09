from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from clinical_world_model.policies import generation_policies
from clinical_world_model.simulator import (
    ClinicalScenario,
    HospitalWorkflowSimulator,
    Trajectory,
    run_episode,
)


def sample_scenarios(count: int, seed: int) -> list[ClinicalScenario]:
    simulator = HospitalWorkflowSimulator(seed=seed)
    return [simulator.sample_scenario(index) for index in range(count)]


def generate_trajectories(
    count: int,
    seed: int,
    max_steps: int = 8,
) -> list[Trajectory]:
    scenarios = sample_scenarios(count=count, seed=seed)
    policies = generation_policies()
    trajectories: list[Trajectory] = []
    for index, scenario in enumerate(scenarios):
        policy = policies[index % len(policies)]
        simulator = HospitalWorkflowSimulator(seed=seed + index)
        trajectories.append(
            run_episode(
                simulator=simulator,
                scenario=scenario,
                policy=policy,
                trajectory_id=f"traj-{index:06d}",
                max_steps=max_steps,
            )
        )
    return trajectories


def write_trajectories_jsonl(path: str | Path, trajectories: Iterable[Trajectory]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for trajectory in trajectories:
            file.write(json.dumps(trajectory.to_dict(), sort_keys=True) + "\n")


def read_trajectories_jsonl(path: str | Path) -> list[Trajectory]:
    records: list[Trajectory] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                records.append(Trajectory.from_dict(json.loads(stripped)))
    return records

