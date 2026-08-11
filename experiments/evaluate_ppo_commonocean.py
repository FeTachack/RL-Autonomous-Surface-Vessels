from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.agents.ppo_continuous import (
    PPOContinuousAgent,
)
from experiments.envs.commonocean_env import (
    CommonOceanEnv,
)


NUM_EPISODES = 20
MAX_EPISODE_STEPS = 220
BASE_SEED = 2000

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "checkpoints"
    / "ppo_commonocean_best.pt"
)


def run_episode(
    agent: PPOContinuousAgent,
    seed: int,
    render_mode=None,
):
    env = CommonOceanEnv(
        max_episode_steps=MAX_EPISODE_STEPS,
        render_mode=render_mode,
    )

    observation, info = env.reset(
        seed=seed
    )

    total_reward = 0.0
    min_distance = float("inf")

    actions = []

    terminated = False
    truncated = False

    while not (
        terminated
        or truncated
    ):
        action = (
            agent.deterministic_action(
                observation
            )
        )

        actions.append(
            np.asarray(
                action,
                dtype=np.float32,
            )
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

    actions = np.asarray(
        actions,
        dtype=np.float32,
    )

    result = {
        "seed": seed,
        "reward": float(
            total_reward
        ),
        "steps": int(
            info["step"]
        ),
        "collision": bool(
            info["collision"]
        ),
        "goal_reached": bool(
            info["goal_reached"]
        ),
        "truncated": bool(
            truncated
        ),
        "min_distance": float(
            min_distance
        ),
        "final_goal_distance": float(
            info["distance_to_goal"]
        ),
        "mean_abs_accel_action": float(
            np.mean(
                np.abs(
                    actions[:, 0]
                )
            )
        ),
        "mean_abs_yaw_action": float(
            np.mean(
                np.abs(
                    actions[:, 1]
                )
            )
        ),
        "max_abs_accel_action": float(
            np.max(
                np.abs(
                    actions[:, 0]
                )
            )
        ),
        "max_abs_yaw_action": float(
            np.max(
                np.abs(
                    actions[:, 1]
                )
            )
        ),
    }

    env.close()

    return result


def main():

    print("=" * 80)
    print("PPO COMMONOCEAN EVALUATION")
    print("=" * 80)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"No existe checkpoint: "
            f"{CHECKPOINT_PATH}"
        )


    agent = PPOContinuousAgent(
        state_size=13,
        action_size=2,
        num_steps=256,
        num_envs=1,
    )

    checkpoint = agent.load(
        str(
            CHECKPOINT_PATH
        )
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )

    print(
        "Best eval reward saved:",
        checkpoint.get(
            "best_eval_reward",
            "N/A",
        ),
    )

    print(
        "Saved update:",
        checkpoint.get(
            "update",
            "N/A",
        ),
    )

    print(
        "Saved global step:",
        checkpoint.get(
            "global_step",
            "N/A",
        ),
    )

    print()


    results = []

    for ep in range(
        NUM_EPISODES
    ):
        seed = (
            BASE_SEED
            + ep
        )

        result = run_episode(
            agent=agent,
            seed=seed,
            render_mode=None,
        )

        results.append(
            result
        )

        outcome = "TRUNCATED"

        if result["collision"]:
            outcome = "COLLISION"

        elif result["goal_reached"]:
            outcome = "GOAL"

        print(
            f"episode={ep + 1:02d} "
            f"seed={seed} "
            f"steps={result['steps']:03d} "
            f"R={result['reward']:+9.3f} "
            f"min_dist={result['min_distance']:7.2f} "
            f"goal_dist="
            f"{result['final_goal_distance']:8.2f} "
            f"|a|mean="
            f"{result['mean_abs_accel_action']:.3f} "
            f"|yaw|mean="
            f"{result['mean_abs_yaw_action']:.3f} "
            f"{outcome}"
        )


    rewards = np.asarray(
        [
            r["reward"]
            for r in results
        ],
        dtype=np.float64,
    )

    steps = np.asarray(
        [
            r["steps"]
            for r in results
        ],
        dtype=np.float64,
    )

    min_distances = np.asarray(
        [
            r["min_distance"]
            for r in results
        ],
        dtype=np.float64,
    )

    final_goal_distances = np.asarray(
        [
            r["final_goal_distance"]
            for r in results
        ],
        dtype=np.float64,
    )

    mean_accel_actions = np.asarray(
        [
            r[
                "mean_abs_accel_action"
            ]
            for r in results
        ],
        dtype=np.float64,
    )

    mean_yaw_actions = np.asarray(
        [
            r[
                "mean_abs_yaw_action"
            ]
            for r in results
        ],
        dtype=np.float64,
    )

    collisions = sum(
        r["collision"]
        for r in results
    )

    goals = sum(
        r["goal_reached"]
        for r in results
    )

    truncations = sum(
        r["truncated"]
        for r in results
    )

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"episodes              = "
        f"{NUM_EPISODES}"
    )

    print(
        f"collisions            = "
        f"{collisions}"
    )

    print(
        f"collision rate        = "
        f"{collisions / NUM_EPISODES:.3f}"
    )

    print(
        f"goals                 = "
        f"{goals}"
    )

    print(
        f"goal rate             = "
        f"{goals / NUM_EPISODES:.3f}"
    )

    print(
        f"truncations           = "
        f"{truncations}"
    )

    print()

    print(
        f"reward mean           = "
        f"{np.mean(rewards):+.3f}"
    )

    print(
        f"reward std            = "
        f"{np.std(rewards):.3f}"
    )

    print(
        f"reward min            = "
        f"{np.min(rewards):+.3f}"
    )

    print(
        f"reward max            = "
        f"{np.max(rewards):+.3f}"
    )

    print()

    print(
        f"episode length mean   = "
        f"{np.mean(steps):.2f}"
    )

    print(
        f"minimum distance mean = "
        f"{np.mean(min_distances):.2f}"
    )

    print(
        f"minimum distance min  = "
        f"{np.min(min_distances):.2f}"
    )

    print()

    print(
        f"final goal dist mean  = "
        f"{np.mean(final_goal_distances):.2f}"
    )

    print()

    print(
        f"|accel action| mean   = "
        f"{np.mean(mean_accel_actions):.4f}"
    )

    print(
        f"|yaw action| mean     = "
        f"{np.mean(mean_yaw_actions):.4f}"
    )


    print()
    print("=" * 80)
    print("REFERENCE BASELINES")
    print("=" * 80)

    print(
        "Collision baseline reward = "
        "-174.643"
    )

    print(
        "Random policy mean reward = "
        "-71.932"
    )

    print(
        "Fixed evasive reward      = "
        "+19.636"
    )

    print(
        "Current PPO mean reward   = "
        f"{np.mean(rewards):+.3f}"
    )


    print()
    print("=" * 80)
    print("VISIBLE DETERMINISTIC EPISODE")
    print("=" * 80)

    visible_result = run_episode(
        agent=agent,
        seed=BASE_SEED,
        render_mode="human",
    )

    print(
        "Visible episode result:"
    )

    for key, value in (
        visible_result.items()
    ):
        print(
            f"  {key:24s} = "
            f"{value}"
        )


if __name__ == "__main__":
    main()
