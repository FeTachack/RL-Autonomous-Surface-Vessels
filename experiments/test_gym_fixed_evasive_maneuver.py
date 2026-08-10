from __future__ import annotations

import numpy as np

from experiments.envs.commonocean_env import (
    CommonOceanEnv,
)


def select_action(step: int) -> np.ndarray:
    """
    Maniobra equivalente a la Prueba B.

    0-59:
        mantener rumbo

    60-99:
        giro a estribor al 80 % del yaw-rate máximo

    >=100:
        mantener el nuevo rumbo
    """

    if 60 <= step < 100:
        return np.array(
            [0.0, -0.8],
            dtype=np.float32,
        )

    return np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )


def main():
    env = CommonOceanEnv(
        max_episode_steps=220,
        render_mode="human",
    )

    observation, info = env.reset(
        seed=42
    )

    print("\nEstado inicial:")
    print(
        f"  ego position       = "
        f"{info['ego_position']}"
    )
    print(
        f"  traffic position   = "
        f"{info['traffic_position']}"
    )
    print(
        f"  distance traffic   = "
        f"{info['distance_to_traffic']:.2f}"
    )
    print(
        f"  DCPA               = "
        f"{info['dcpa']:.2f}"
    )
    print(
        f"  TCPA               = "
        f"{info['tcpa']:.2f}"
    )

    min_distance = float("inf")
    min_distance_step = None

    total_reward = 0.0

    terminated = False
    truncated = False

    for step in range(220):

        action = select_action(step)

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += reward

        distance = float(
            info["distance_to_traffic"]
        )

        if distance < min_distance:
            min_distance = distance
            min_distance_step = step + 1

        if (
            step == 0
            or (step + 1) % 10 == 0
            or terminated
            or truncated
        ):
            physical_action = np.asarray(
                info["physical_action"]
            )

            print(
                f"step={step + 1:04d} "
                f"ego={np.round(info['ego_position'], 2)} "
                f"traffic="
                f"{np.round(info['traffic_position'], 2)} "
                f"distance={distance:7.2f} "
                f"heading={info['ego_heading']:7.4f} "
                f"a={physical_action[0]:+.5f} "
                f"yaw={physical_action[1]:+.5f} "
                f"DCPA={info['dcpa']:7.2f} "
                f"TCPA={info['tcpa']:7.2f} "
                f"R={reward:+7.3f}"
            )

        if terminated or truncated:
            break

    print("\nResumen:")
    print(
        f"  steps ejecutados   = "
        f"{info['step']}"
    )
    print(
        f"  collision          = "
        f"{info['collision']}"
    )
    print(
        f"  goal reached       = "
        f"{info['goal_reached']}"
    )
    print(
        f"  terminated         = "
        f"{terminated}"
    )
    print(
        f"  truncated          = "
        f"{truncated}"
    )
    print(
        f"  min distance       = "
        f"{min_distance:.2f}"
    )
    print(
        f"  min distance step  = "
        f"{min_distance_step}"
    )
    print(
        f"  total reward       = "
        f"{total_reward:.3f}"
    )

    env.close()


if __name__ == "__main__":
    main()
