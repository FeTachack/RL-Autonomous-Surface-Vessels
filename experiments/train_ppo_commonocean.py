from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch

from experiments.agents.ppo_continuous import (
    PPOContinuousAgent,
)

from experiments.envs.commonocean_env import (
    CommonOceanEnv,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

STATE_SIZE = 13
ACTION_SIZE = 2

NUM_ENVS = 1

ROLLOUT_STEPS = 256

MAX_EPISODE_STEPS = 220

# Primer test de entrenamiento:
# 10 rollouts x 256 transitions = 2560 timesteps
NUM_UPDATES = 10

TOTAL_TIMESTEPS = (
    NUM_UPDATES
    * ROLLOUT_STEPS
    * NUM_ENVS
)

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "ppo_commonocean_best.pt"
)

FINAL_MODEL_PATH = (
    CHECKPOINT_DIR
    / "ppo_commonocean_final.pt"
)


# ============================================================
# Reproducibility
# ============================================================


def set_seed(
    seed: int,
):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# Deterministic evaluation
# ============================================================


def evaluate_policy(
    agent: PPOContinuousAgent,
    max_episode_steps: int,
):
    """
    Evalúa la política determinista:

        action = tanh(mu(s))

    No se utiliza muestreo durante evaluación.
    """

    env = CommonOceanEnv(
        max_episode_steps=max_episode_steps,
        render_mode=None,
    )

    observation, info = env.reset(
        seed=SEED
    )

    total_reward = 0.0

    min_distance = float("inf")

    terminated = False
    truncated = False

    for _ in range(
        max_episode_steps
    ):
        action = (
            agent.deterministic_action(
                observation
            )
        )

        (
            next_observation,
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

        observation = (
            next_observation
        )

        if (
            terminated
            or truncated
        ):
            break

    result = {
        "reward": float(
            total_reward
        ),

        "collision": bool(
            info["collision"]
        ),

        "goal_reached": bool(
            info["goal_reached"]
        ),

        "steps": int(
            info["step"]
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

    env.close()

    return result


# ============================================================
# Training
# ============================================================


def train():
    set_seed(
        SEED
    )

    print(
        "=" * 72
    )

    print(
        "PPO COMMONOCEAN - "
        "FIRST TRAINING TEST"
    )

    print(
        "=" * 72
    )

    print(
        f"Total timesteps : "
        f"{TOTAL_TIMESTEPS}"
    )

    print(
        f"Rollout steps   : "
        f"{ROLLOUT_STEPS}"
    )

    print(
        f"Updates         : "
        f"{NUM_UPDATES}"
    )

    # ========================================================
    # Environment
    # ========================================================

    env = CommonOceanEnv(
        max_episode_steps=(
            MAX_EPISODE_STEPS
        ),
        render_mode=None,
    )

    # ========================================================
    # PPO agent
    # ========================================================

    agent = PPOContinuousAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        num_steps=ROLLOUT_STEPS,
        num_envs=NUM_ENVS,
    )

    print(
        f"Device          : "
        f"{agent.device}"
    )

    # ========================================================
    # Initial state
    # ========================================================

    observation, info = env.reset(
        seed=SEED
    )

    # ========================================================
    # Statistics
    # ========================================================

    global_step = 0

    episode_count = 0

    episode_reward = 0.0

    episode_length = 0

    episode_rewards = []

    episode_lengths = []

    episode_collisions = []

    training_metrics = []

    best_eval_reward = (
        -float("inf")
    )

    # ========================================================
    # PPO updates
    # ========================================================

    for update_index in range(
        1,
        NUM_UPDATES + 1,
    ):
        rollout_collisions = 0
        rollout_episodes = 0

        # ====================================================
        # Collect rollout
        # ====================================================

        for _ in range(
            ROLLOUT_STEPS
        ):
            # ------------------------------------------------
            # PPO action
            # ------------------------------------------------

            (
                action,
                pre_tanh_action,
                log_prob,
                value,
            ) = agent.sample_action(
                observation
            )

            # ------------------------------------------------
            # Environment step
            # ------------------------------------------------

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            episode_end = bool(
                terminated
                or truncated
            )

            # ------------------------------------------------
            # Store transition
            # ------------------------------------------------

            agent.store_transition(
                states=(
                    observation[
                        None,
                        :
                    ]
                ),

                actions=(
                    action[
                        None,
                        :
                    ]
                ),

                pre_tanh_actions=(
                    pre_tanh_action[
                        None,
                        :
                    ]
                ),

                log_probs=np.array(
                    [log_prob],
                    dtype=np.float32,
                ),

                rewards=np.array(
                    [reward],
                    dtype=np.float32,
                ),

                terminated=np.array(
                    [terminated],
                    dtype=np.float32,
                ),

                episode_ends=np.array(
                    [episode_end],
                    dtype=np.float32,
                ),

                next_states=(
                    next_observation[
                        None,
                        :
                    ]
                ),
            )

            global_step += 1

            episode_reward += float(
                reward
            )

            episode_length += 1

            observation = (
                next_observation
            )

            # ------------------------------------------------
            # Episode finished
            # ------------------------------------------------

            if episode_end:
                episode_count += 1

                rollout_episodes += 1

                collision = bool(
                    info["collision"]
                )

                if collision:
                    rollout_collisions += 1

                episode_rewards.append(
                    float(
                        episode_reward
                    )
                )

                episode_lengths.append(
                    int(
                        episode_length
                    )
                )

                episode_collisions.append(
                    collision
                )

                print(
                    f"Episode "
                    f"{episode_count:03d} | "
                    f"steps="
                    f"{episode_length:03d} | "
                    f"R="
                    f"{episode_reward:+9.3f} | "
                    f"collision="
                    f"{collision} | "
                    f"goal="
                    f"{info['goal_reached']}"
                )

                observation, info = (
                    env.reset(
                        seed=(
                            SEED
                            + episode_count
                        )
                    )
                )

                episode_reward = 0.0

                episode_length = 0

        # ====================================================
        # PPO update
        # ====================================================

        metrics = (
            agent.update()
        )

        if metrics is None:
            raise RuntimeError(
                "PPO update no fue ejecutado."
            )

        training_metrics.append(
            metrics
        )

        # ====================================================
        # Deterministic evaluation
        # ====================================================

        eval_result = (
            evaluate_policy(
                agent=agent,
                max_episode_steps=(
                    MAX_EPISODE_STEPS
                ),
            )
        )

        # ====================================================
        # Save best model
        # ====================================================

        if (
            eval_result["reward"]
            > best_eval_reward
        ):
            best_eval_reward = (
                eval_result[
                    "reward"
                ]
            )

            agent.save(
                str(
                    BEST_MODEL_PATH
                ),
                extra={
                    "global_step": (
                        global_step
                    ),

                    "update": (
                        update_index
                    ),

                    "best_eval_reward": (
                        best_eval_reward
                    ),
                },
            )

        # ====================================================
        # Training status
        # ====================================================

        if rollout_episodes > 0:
            rollout_collision_rate = (
                rollout_collisions
                / rollout_episodes
            )
        else:
            rollout_collision_rate = (
                float("nan")
            )

        print()
        print(
            f"UPDATE "
            f"{update_index:02d}/"
            f"{NUM_UPDATES}"
        )

        print(
            f"  global step        = "
            f"{global_step}"
        )

        print(
            f"  loss               = "
            f"{metrics['loss']:+.5f}"
        )

        print(
            f"  actor loss         = "
            f"{metrics['actor_loss']:+.5f}"
        )

        print(
            f"  critic loss        = "
            f"{metrics['critic_loss']:+.5f}"
        )

        print(
            f"  entropy            = "
            f"{metrics['entropy']:+.5f}"
        )

        print(
            f"  rollout episodes   = "
            f"{rollout_episodes}"
        )

        print(
            f"  collision rate     = "
            f"{rollout_collision_rate}"
        )

        print(
            f"  eval reward        = "
            f"{eval_result['reward']:+.3f}"
        )

        print(
            f"  eval collision     = "
            f"{eval_result['collision']}"
        )

        print(
            f"  eval goal          = "
            f"{eval_result['goal_reached']}"
        )

        print(
            f"  eval min distance  = "
            f"{eval_result['min_distance']:.2f}"
        )

        print(
            f"  eval goal distance = "
            f"{eval_result['final_goal_distance']:.2f}"
        )

        print(
            "-" * 72
        )

    # ========================================================
    # Save final model
    # ========================================================

    agent.save(
        str(
            FINAL_MODEL_PATH
        ),
        extra={
            "global_step": (
                global_step
            ),

            "episode_count": (
                episode_count
            ),

            "episode_rewards": (
                episode_rewards
            ),

            "episode_lengths": (
                episode_lengths
            ),

            "episode_collisions": (
                episode_collisions
            ),

            "training_metrics": (
                training_metrics
            ),
        },
    )

    env.close()

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "TRAINING TEST COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Global steps     : "
        f"{global_step}"
    )

    print(
        f"Episodes         : "
        f"{episode_count}"
    )

    if episode_rewards:
        print(
            f"Mean reward      : "
            f"{np.mean(episode_rewards):+.3f}"
        )

        print(
            f"Mean episode len : "
            f"{np.mean(episode_lengths):.2f}"
        )

        print(
            f"Collision rate   : "
            f"{np.mean(episode_collisions):.3f}"
        )

    print(
        f"Best eval reward : "
        f"{best_eval_reward:+.3f}"
    )

    print(
        "Best checkpoint  : "
        f"{BEST_MODEL_PATH}"
    )

    print(
        "Final checkpoint : "
        f"{FINAL_MODEL_PATH}"
    )


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    train()
