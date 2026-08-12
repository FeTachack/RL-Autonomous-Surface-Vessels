from __future__ import annotations

import numpy as np

from experiments.envs.commonocean_env import CommonOceanEnv
from experiments.scenarios.randomize_conflict_scenario import (
    ConflictPreservingConfig,
)


SEEDS = list(
    range(
        93000,
        93010,
    )
)

MAX_EPISODE_STEPS = 220


def run_zero_action(
    seed: int,
) -> dict:
    env = CommonOceanEnv(
        max_episode_steps=MAX_EPISODE_STEPS,
        render_mode=None,
        randomize_scenario=True,
        randomization_mode="conflict_preserving",
        randomization_config=ConflictPreservingConfig(),
    )

    observation, info = env.reset(
        seed=seed
    )

    metadata = dict(
        env.scenario_metadata
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

    env.close()

    return {
        "seed": int(
            seed
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
        "steps": int(
            info[
                "step"
            ]
        ),
        "reward": float(
            total_reward
        ),
        "min_distance": float(
            min_distance
        ),
        "min_dcpa": float(
            min_dcpa
        ),
        "metadata": metadata,
    }


def main() -> None:
    results = []

    print(
        "="
        * 88
    )

    print(
        "COMMONOCEAN ENV - CONFLICT PRESERVING MODE TEST"
    )

    print(
        "="
        * 88
    )

    for seed in SEEDS:
        result = run_zero_action(
            seed=seed
        )

        params = result[
            "metadata"
        ][
            "parameters"
        ]

        results.append(
            result
        )

        print(
            f"seed={seed} "
            f"mode={result['metadata']['randomization_type']} "
            f"side={params['initial_bearing_side']:>9s} "
            f"TCPA0={params['initial_tcpa']:7.2f} "
            f"DCPA0={params['initial_dcpa']:7.2f} "
            f"col={result['collision']} "
            f"goal={result['goal_reached']} "
            f"steps={result['steps']:3d} "
            f"dmin={result['min_distance']:7.2f} "
            f"R={result['reward']:+8.2f}"
        )

    collisions = np.asarray(
        [
            item[
                "collision"
            ]
            for item in results
        ],
        dtype=np.float64,
    )

    min_distances = np.asarray(
        [
            item[
                "min_distance"
            ]
            for item in results
        ],
        dtype=np.float64,
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
        f"episodes                    : {len(results)}"
    )

    print(
        f"zero_action_collision_rate  : {float(np.mean(collisions)):.3f}"
    )

    print(
        f"zero_action_min_distance_mean: {float(np.mean(min_distances)):.3f}"
    )

    print(
        f"zero_action_min_distance_min : {float(np.min(min_distances)):.3f}"
    )


if __name__ == "__main__":
    main()
