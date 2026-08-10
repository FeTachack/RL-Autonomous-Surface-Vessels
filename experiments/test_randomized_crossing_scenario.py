from __future__ import annotations

import numpy as np

from experiments.envs.commonocean_env import (
    CommonOceanEnv,
)


MAX_EPISODE_STEPS = 220

SEEDS = [
    3000,
    3001,
    3002,
    3003,
    3004,
]


def run_no_action_episode(
    seed: int,
):
    env = CommonOceanEnv(
        max_episode_steps=MAX_EPISODE_STEPS,
        render_mode=None,
        randomize_scenario=True,
    )

    observation, info = env.reset(
        seed=seed
    )

    scenario = info[
        "scenario"
    ]

    params = scenario[
        "parameters"
    ]

    print()
    print("=" * 72)
    print(f"seed = {seed}")
    print("=" * 72)

    print(
        "scenario_path      =",
        scenario[
            "scenario_path"
        ],
    )

    print(
        "traffic_x0         =",
        f"{params['traffic_x0']:.3f}",
    )

    print(
        "traffic_y0         =",
        f"{params['traffic_y0']:.3f}",
    )

    print(
        "traffic_speed      =",
        f"{params['traffic_speed']:.3f}",
    )

    print(
        "traffic_heading    =",
        f"{params['traffic_heading']:.3f}",
    )

    print(
        "ego_speed          =",
        f"{params['ego_speed']:.3f}",
    )

    terminated = False
    truncated = False

    total_reward = 0.0
    min_distance = float(
        "inf"
    )

    zero_action = np.zeros(
        2,
        dtype=np.float32,
    )

    while not (
        terminated
        or truncated
    ):
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            zero_action
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
        "final_goal_distance": float(
            info[
                "distance_to_goal"
            ]
        ),
    }

    print(
        "result             =",
        result,
    )

    env.close()

    return result


def main():
    print("=" * 72)
    print("RANDOMIZED CROSSING SCENARIO TEST")
    print("=" * 72)

    results = []

    for seed in SEEDS:
        result = run_no_action_episode(
            seed
        )

        results.append(
            result
        )

    collisions = sum(
        result[
            "collision"
        ]
        for result in results
    )

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(
        "episodes        =",
        len(
            results
        ),
    )

    print(
        "collisions      =",
        collisions,
    )

    print(
        "collision rate  =",
        collisions
        / len(
            results
        ),
    )

    print(
        "min distance min=",
        min(
            result[
                "min_distance"
            ]
            for result in results
        ),
    )

    print()
    print("TEST PASSED")


if __name__ == "__main__":
    main()
