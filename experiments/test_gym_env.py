from __future__ import annotations

import numpy as np

from gymnasium.utils.env_checker import (
    check_env,
)

from experiments.envs.commonocean_env import (
    CommonOceanEnv,
)


def main():
    env = CommonOceanEnv(
        max_episode_steps=300
    )

    print(
        "Validando API Gymnasium..."
    )

    check_env(
        env,
        skip_render_check=True,
    )

    print(
        "Gymnasium check_env: OK"
    )

    observation, info = env.reset(
        seed=42
    )

    print("\nReset:")
    print(
        "  observation shape =",
        observation.shape,
    )
    print(
        "  observation dtype =",
        observation.dtype,
    )
    print(
        "  observation =",
        observation,
    )
    print(
        "  info =",
        info,
    )

    print("\nEpisode:")

    total_reward = 0.0

    for step in range(300):
        action = (
            env.action_space.sample()
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += reward

        if (
            step == 0
            or (step + 1) % 10 == 0
            or terminated
            or truncated
        ):
            print(
                f"step={step + 1:03d} "
                f"action="
                f"{np.round(action, 3)} "
                f"distance="
                f"{info['distance_to_traffic']:.2f} "
                f"collision="
                f"{info['collision']} "
                f"terminated="
                f"{terminated} "
                f"truncated="
                f"{truncated}"
            )

        if terminated or truncated:
            break

    print("\nResumen:")
    print(
        "  steps =",
        info["step"],
    )
    print(
        "  total reward =",
        total_reward,
    )
    print(
        "  collision =",
        info["collision"],
    )
    print(
        "  goal reached =",
        info["goal_reached"],
    )
    print(
        "  terminated =",
        terminated,
    )
    print(
        "  truncated =",
        truncated,
    )

    env.close()


if __name__ == "__main__":
    main()
