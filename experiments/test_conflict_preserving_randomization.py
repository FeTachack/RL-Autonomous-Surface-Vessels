from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

from experiments.envs.commonocean_env import CommonOceanEnv
from experiments.scenarios.randomize_conflict_scenario import (
    ConflictPreservingConfig,
    generate_conflict_preserving_scenario,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_XML_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "scenarios"
    / "one_ego_one_traffic.xml"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "scenarios"
    / "generated"
    / "conflict_preserving"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "conflict_preserving_randomization"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "summary.json"
)

SEEDS = list(
    range(
        92000,
        92010,
    )
)

MAX_EPISODE_STEPS = 220


def to_serializable(
    value: Any,
) -> Any:
    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        dict,
    ):
        return {
            key: to_serializable(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            to_serializable(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        tuple,
    ):
        return [
            to_serializable(
                item
            )
            for item in value
        ]

    return value


def save_json(
    path: Path,
    payload: Any,
) -> None:
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            to_serializable(
                payload
            ),
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Saved JSON: {path}"
    )


def project_relative_scenario_path(
    xml_path: Path,
) -> str:
    relative_path = (
        xml_path
        .resolve()
        .relative_to(
            PROJECT_ROOT.resolve()
        )
        .as_posix()
    )

    return (
        "/"
        + relative_path
    )


def run_zero_action_episode(
    scenario_path: str,
    seed: int,
) -> dict[str, Any]:
    env = CommonOceanEnv(
        max_episode_steps=MAX_EPISODE_STEPS,
        render_mode=None,
        scenario_path=scenario_path,
        randomize_scenario=False,
    )

    observation, info = env.reset(
        seed=seed
    )

    total_reward = 0.0

    min_distance = float(
        info[
            "distance_to_traffic"
        ]
    )

    min_dcpa = float(
        info[
            "dcpa"
        ]
    )

    terminated = False
    truncated = False

    while not (
        terminated
        or truncated
    ):
        action = np.zeros(
            2,
            dtype=np.float32,
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

        total_reward += float(
            reward
        )

        min_distance = min(
            min_distance,
            float(
                info[
                    "distance_to_traffic"
                ]
            ),
        )

        min_dcpa = min(
            min_dcpa,
            float(
                info[
                    "dcpa"
                ]
            ),
        )

    result = {
        "seed": int(
            seed
        ),
        "reward": float(
            total_reward
        ),
        "steps": int(
            info[
                "step"
            ]
        ),
        "collision": bool(
            info[
                "collision"
            ]
        ),
        "goal_reached": bool(
            info[
                "goal_reached"
            ]
        ),
        "truncated": bool(
            truncated
        ),
        "min_distance": float(
            min_distance
        ),
        "min_dcpa": float(
            min_dcpa
        ),
        "final_goal_distance": float(
            info[
                "distance_to_goal"
            ]
        ),
    }

    env.close()

    return result


def main() -> None:
    print(
        "="
        * 88
    )

    print(
        "CONFLICT-PRESERVING RANDOMIZATION TEST"
    )

    print(
        "="
        * 88
    )

    config = ConflictPreservingConfig()

    results = []

    for seed in SEEDS:
        output_xml_path = (
            OUTPUT_DIR
            / f"conflict_preserving_seed_{seed}.xml"
        )

        metadata = generate_conflict_preserving_scenario(
            base_xml_path=BASE_XML_PATH,
            output_xml_path=output_xml_path,
            seed=seed,
            config=config,
        )

        scenario_path = project_relative_scenario_path(
            output_xml_path
        )

        episode_result = run_zero_action_episode(
            scenario_path=scenario_path,
            seed=seed,
        )

        params = metadata[
            "parameters"
        ]

        row = {
            "seed": int(
                seed
            ),
            "scenario_path": scenario_path,
            "metadata": metadata,
            "zero_action": episode_result,
        }

        results.append(
            row
        )

        print(
            f"seed={seed} "
            f"side={params['initial_bearing_side']:>9s} "
            f"TCPA0={params['initial_tcpa']:7.2f} "
            f"DCPA0={params['initial_dcpa']:7.2f} "
            f"d0={params['initial_distance']:7.2f} "
            f"col={episode_result['collision']} "
            f"goal={episode_result['goal_reached']} "
            f"steps={episode_result['steps']:3d} "
            f"dmin={episode_result['min_distance']:7.2f} "
            f"R={episode_result['reward']:+8.2f}"
        )

    collisions = np.asarray(
        [
            item[
                "zero_action"
            ][
                "collision"
            ]
            for item in results
        ],
        dtype=np.float64,
    )

    min_distances = np.asarray(
        [
            item[
                "zero_action"
            ][
                "min_distance"
            ]
            for item in results
        ],
        dtype=np.float64,
    )

    initial_dcpas = np.asarray(
        [
            item[
                "metadata"
            ][
                "parameters"
            ][
                "initial_dcpa"
            ]
            for item in results
        ],
        dtype=np.float64,
    )

    initial_tcpas = np.asarray(
        [
            item[
                "metadata"
            ][
                "parameters"
            ][
                "initial_tcpa"
            ]
            for item in results
        ],
        dtype=np.float64,
    )

    summary = {
        "config": to_serializable(
            config.__dict__
        ),
        "seeds": SEEDS,
        "episodes": int(
            len(
                results
            )
        ),
        "zero_action_collision_rate": float(
            np.mean(
                collisions
            )
        ),
        "zero_action_min_distance_mean": float(
            np.mean(
                min_distances
            )
        ),
        "zero_action_min_distance_min": float(
            np.min(
                min_distances
            )
        ),
        "initial_dcpa_mean": float(
            np.mean(
                initial_dcpas
            )
        ),
        "initial_dcpa_max": float(
            np.max(
                initial_dcpas
            )
        ),
        "initial_tcpa_mean": float(
            np.mean(
                initial_tcpas
            )
        ),
        "initial_tcpa_min": float(
            np.min(
                initial_tcpas
            )
        ),
        "initial_tcpa_max": float(
            np.max(
                initial_tcpas
            )
        ),
        "results": results,
    }

    save_json(
        SUMMARY_PATH,
        summary,
    )

    print()
    print(
        "="
        * 88
    )

    print(
        "SUMMARY"
    )

    print(
        "="
        * 88
    )

    print(
        f"episodes                    : {summary['episodes']}"
    )

    print(
        f"zero_action_collision_rate  : {summary['zero_action_collision_rate']:.3f}"
    )

    print(
        f"zero_action_min_distance_mean: {summary['zero_action_min_distance_mean']:.3f}"
    )

    print(
        f"zero_action_min_distance_min : {summary['zero_action_min_distance_min']:.3f}"
    )

    print(
        f"initial_dcpa_mean           : {summary['initial_dcpa_mean']:.3f}"
    )

    print(
        f"initial_dcpa_max            : {summary['initial_dcpa_max']:.3f}"
    )

    print(
        f"initial_tcpa_mean           : {summary['initial_tcpa_mean']:.3f}"
    )

    print(
        f"initial_tcpa_min            : {summary['initial_tcpa_min']:.3f}"
    )

    print(
        f"initial_tcpa_max            : {summary['initial_tcpa_max']:.3f}"
    )


if __name__ == "__main__":
    main()
