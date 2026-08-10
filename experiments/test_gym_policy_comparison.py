from __future__ import annotations

import numpy as np

from experiments.envs.commonocean_env import CommonOceanEnv


def policy_collision(step: int) -> np.ndarray:
    """Prueba A: sin maniobra."""
    return np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )


def policy_evasive(step: int) -> np.ndarray:
    """Prueba B: maniobra evasiva fija."""

    if 60 <= step < 100:
        return np.array(
            [0.0, -0.8],
            dtype=np.float32,
        )

    return np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )


def run_episode(
    name: str,
    policy,
    max_episode_steps: int = 220,
):
    env = CommonOceanEnv(
        max_episode_steps=max_episode_steps,
        render_mode=None,
    )

    observation, info = env.reset(
        seed=42
    )

    total_reward = 0.0

    reward_sums = {
        "progress": 0.0,
        "risk": 0.0,
        "time": 0.0,
        "collision": 0.0,
        "goal": 0.0,
    }

    min_distance = float("inf")
    min_distance_step = None

    min_dcpa = float("inf")
    min_dcpa_step = None

    terminated = False
    truncated = False

    for step in range(max_episode_steps):

        action = policy(step)

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += reward

        components = info[
            "reward_components"
        ]

        for key in reward_sums:
            reward_sums[key] += float(
                components[key]
            )

        distance = float(
            info["distance_to_traffic"]
        )

        dcpa = float(
            info["dcpa"]
        )

        if distance < min_distance:
            min_distance = distance
            min_distance_step = step + 1

        if dcpa < min_dcpa:
            min_dcpa = dcpa
            min_dcpa_step = step + 1

        if terminated or truncated:
            break

    result = {
        "name": name,

        "steps": int(
            info["step"]
        ),

        "total_reward": float(
            total_reward
        ),

        "collision": bool(
            info["collision"]
        ),

        "goal_reached": bool(
            info["goal_reached"]
        ),

        "terminated": bool(
            terminated
        ),

        "truncated": bool(
            truncated
        ),

        "min_distance": float(
            min_distance
        ),

        "min_distance_step": int(
            min_distance_step
        ),

        "min_dcpa": float(
            min_dcpa
        ),

        "min_dcpa_step": int(
            min_dcpa_step
        ),

        "final_distance_to_goal": float(
            info["distance_to_goal"]
        ),

        "reward_components": (
            reward_sums
        ),
    }

    env.close()

    return result


def print_result(result):

    print()
    print("=" * 70)
    print(
        f"POLICY: {result['name']}"
    )
    print("=" * 70)

    print(
        f"steps                  = "
        f"{result['steps']}"
    )

    print(
        f"total reward           = "
        f"{result['total_reward']:.3f}"
    )

    print(
        f"collision              = "
        f"{result['collision']}"
    )

    print(
        f"goal reached           = "
        f"{result['goal_reached']}"
    )

    print(
        f"terminated             = "
        f"{result['terminated']}"
    )

    print(
        f"truncated              = "
        f"{result['truncated']}"
    )

    print(
        f"minimum distance       = "
        f"{result['min_distance']:.2f}"
    )

    print(
        f"minimum distance step  = "
        f"{result['min_distance_step']}"
    )

    print(
        f"minimum DCPA           = "
        f"{result['min_dcpa']:.2f}"
    )

    print(
        f"minimum DCPA step      = "
        f"{result['min_dcpa_step']}"
    )

    print(
        f"final distance to goal = "
        f"{result['final_distance_to_goal']:.2f}"
    )

    print()
    print("Reward components:")

    for key, value in (
        result[
            "reward_components"
        ].items()
    ):
        print(
            f"  {key:10s} = "
            f"{value:+10.3f}"
        )


def main():

    collision_result = run_episode(
        name="A - Collision baseline",
        policy=policy_collision,
    )

    evasive_result = run_episode(
        name="B - Fixed evasive maneuver",
        policy=policy_evasive,
    )

    print_result(
        collision_result
    )

    print_result(
        evasive_result
    )

    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    delta_reward = (
        evasive_result["total_reward"]
        - collision_result["total_reward"]
    )

    delta_distance = (
        evasive_result["min_distance"]
        - collision_result["min_distance"]
    )

    print(
        f"Δ total reward "
        f"(B - A) = "
        f"{delta_reward:+.3f}"
    )

    print(
        f"Δ minimum distance "
        f"(B - A) = "
        f"{delta_distance:+.2f} m"
    )

    print(
        f"A collision = "
        f"{collision_result['collision']}"
    )

    print(
        f"B collision = "
        f"{evasive_result['collision']}"
    )


if __name__ == "__main__":
    main()
