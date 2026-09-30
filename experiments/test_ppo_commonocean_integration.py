from __future__ import annotations

import numpy as np

from experiments.agents.ppo_continuous import PPOContinuousAgent
from experiments.envs.commonocean_env import CommonOceanEnv


MAX_EPISODE_STEPS = 220


def main():

    env = CommonOceanEnv(
        max_episode_steps=MAX_EPISODE_STEPS,
        render_mode="human",
    )

    agent = PPOContinuousAgent(
        state_size=13,
        action_size=2,
        num_steps=256,
        num_envs=1,
    )

    observation, info = env.reset(seed=42)

    print("\nEstado inicial:")
    print("  observation shape =", observation.shape)
    print("  observation dtype =", observation.dtype)
    print("  ego position      =", info["ego_position"])
    print("  traffic position  =", info["traffic_position"])
    print("  distance traffic  =", f"{info['distance_to_traffic']:.2f}")
    print("  DCPA              =", f"{info['dcpa']:.2f}")
    print("  TCPA              =", f"{info['tcpa']:.2f}")

    assert observation.shape == (13,)
    assert np.all(np.isfinite(observation))

    total_reward = 0.0

    min_distance = float("inf")
    min_distance_step = None

    terminated = False
    truncated = False

    for step in range(MAX_EPISODE_STEPS):

        (
            action,
            pre_tanh_action,
            log_prob,
            value,
        ) = agent.sample_action(observation)


        log_prob = float(log_prob)
        value = float(value)

        assert action.shape == (2,)
        assert pre_tanh_action.shape == (2,)
        assert np.isfinite(log_prob)
        assert np.isfinite(value)
        assert np.all(np.isfinite(action))
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)

        next_observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        assert next_observation.shape == (13,)
        assert np.all(np.isfinite(next_observation))
        assert np.isfinite(reward)

        total_reward += float(reward)

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
                f"a_norm={np.round(action, 3)} "
                f"a_phys={np.round(physical_action, 5)} "
                f"heading={info['ego_heading']:7.4f} "
                f"distance={distance:7.2f} "
                f"DCPA={info['dcpa']:7.2f} "
                f"TCPA={info['tcpa']:7.2f} "
                f"logp={float(log_prob):+8.3f} "
                f"V={float(value):+8.3f} "
                f"R={float(reward):+8.3f}"
            )

        observation = next_observation

        if terminated or truncated:
            break

    print("\nResumen:")
    print("  steps ejecutados   =", info["step"])
    print("  collision          =", info["collision"])
    print("  goal reached       =", info["goal_reached"])
    print("  terminated         =", terminated)
    print("  truncated          =", truncated)
    print("  min distance       =", f"{min_distance:.2f}")
    print("  min distance step  =", min_distance_step)
    print("  total reward       =", f"{total_reward:.3f}")

    env.close()

    print()
    print("======================================")
    print("PPO-COMMONOCEAN INTEGRATION TEST PASSED")
    print("======================================")


if __name__ == "__main__":
    main()
