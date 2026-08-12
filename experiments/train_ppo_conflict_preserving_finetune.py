from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import json
import random

import numpy as np
import torch

from experiments.agents.ppo_continuous import PPOContinuousAgent
from experiments.envs.commonocean_env import CommonOceanEnv
from experiments.preferences.colreg_reward_wrapper import (
    ColregRewardConfig,
    ColregRewardWrapper,
)
from experiments.scenarios.randomize_conflict_scenario import (
    ConflictPreservingConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "checkpoints"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "ppo_conflict_preserving_finetune"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

INITIAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "ppo_colreg_finetuned_best.pt"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "ppo_conflict_preserving_finetuned_best.pt"
)

FINAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "ppo_conflict_preserving_finetuned_final.pt"
)

TRAINING_HISTORY_PATH = (
    RESULTS_DIR
    / "training_history.json"
)

FINAL_EVALUATION_PATH = (
    RESULTS_DIR
    / "final_evaluation.json"
)

SEED = 321

STATE_SIZE = 13
ACTION_SIZE = 2

NUM_ENVS = 1
ROLLOUT_STEPS = 256
MAX_EPISODE_STEPS = 220

NUM_UPDATES = 50
EVAL_INTERVAL = 5

TRAIN_SEED_BASE = 120000
EVAL_SEEDS = list(
    range(
        96000,
        96020,
    )
)
FINAL_EVAL_SEEDS = list(
    range(
        97000,
        97100,
    )
)

COLREG_WEIGHT = 0.50


def set_global_seed(
    seed: int,
) -> None:
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


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


def make_env() -> ColregRewardWrapper:
    base_env = CommonOceanEnv(
        max_episode_steps=MAX_EPISODE_STEPS,
        render_mode=None,
        randomize_scenario=True,
        randomization_mode="conflict_preserving",
        randomization_config=ConflictPreservingConfig(),
    )

    env = ColregRewardWrapper(
        base_env,
        config=ColregRewardConfig(
            colreg_weight=COLREG_WEIGHT,
        ),
    )

    return env


def load_agent() -> PPOContinuousAgent:
    if not INITIAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"No existe checkpoint inicial: {INITIAL_CHECKPOINT}"
        )

    agent = PPOContinuousAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        num_steps=ROLLOUT_STEPS,
        num_envs=NUM_ENVS,
    )

    checkpoint = agent.load(
        INITIAL_CHECKPOINT
    )

    print(
        f"Loaded initial checkpoint: {INITIAL_CHECKPOINT}"
    )

    if isinstance(
        checkpoint,
        dict,
    ):
        print(
            f"Checkpoint keys: {list(checkpoint.keys())}"
        )

    return agent


def deterministic_policy(
    agent: PPOContinuousAgent,
) -> Callable[[np.ndarray], np.ndarray]:
    def policy(
        observation: np.ndarray,
    ) -> np.ndarray:
        action = agent.deterministic_action(
            observation.reshape(
                1,
                -1,
            )
        )[0]

        return np.asarray(
            action,
            dtype=np.float32,
        )

    return policy


def run_eval_episode(
    agent: PPOContinuousAgent,
    seed: int,
) -> dict[str, Any]:
    env = make_env()

    observation, info = env.reset(
        seed=seed
    )

    total_reward = 0.0
    env_reward_total = 0.0
    colreg_reward_total = 0.0

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

    policy = deterministic_policy(
        agent
    )

    while not (
        terminated
        or truncated
    ):
        action = policy(
            observation
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

        env_reward_total += float(
            info.get(
                "env_reward",
                0.0,
            )
        )

        colreg_reward_total += float(
            info.get(
                "colreg_reward",
                0.0,
            )
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
        "env_reward": float(
            env_reward_total
        ),
        "colreg_reward": float(
            colreg_reward_total
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


def summarize_results(
    name: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    rewards = np.asarray(
        [
            item[
                "reward"
            ]
            for item in results
        ],
        dtype=np.float64,
    )

    env_rewards = np.asarray(
        [
            item[
                "env_reward"
            ]
            for item in results
        ],
        dtype=np.float64,
    )

    colreg_rewards = np.asarray(
        [
            item[
                "colreg_reward"
            ]
            for item in results
        ],
        dtype=np.float64,
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

    goals = np.asarray(
        [
            item[
                "goal_reached"
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

    min_dcpas = np.asarray(
        [
            item[
                "min_dcpa"
            ]
            for item in results
        ],
        dtype=np.float64,
    )

    final_goal_distances = np.asarray(
        [
            item[
                "final_goal_distance"
            ]
            for item in results
        ],
        dtype=np.float64,
    )

    lengths = np.asarray(
        [
            item[
                "steps"
            ]
            for item in results
        ],
        dtype=np.float64,
    )

    summary = {
        "name": name,
        "episodes": int(
            len(
                results
            )
        ),
        "reward_mean": float(
            np.mean(
                rewards
            )
        ),
        "reward_std": float(
            np.std(
                rewards
            )
        ),
        "reward_min": float(
            np.min(
                rewards
            )
        ),
        "reward_max": float(
            np.max(
                rewards
            )
        ),
        "env_reward_mean": float(
            np.mean(
                env_rewards
            )
        ),
        "colreg_reward_mean": float(
            np.mean(
                colreg_rewards
            )
        ),
        "collision_rate": float(
            np.mean(
                collisions
            )
        ),
        "goal_rate": float(
            np.mean(
                goals
            )
        ),
        "min_distance_mean": float(
            np.mean(
                min_distances
            )
        ),
        "min_distance_std": float(
            np.std(
                min_distances
            )
        ),
        "min_distance_min": float(
            np.min(
                min_distances
            )
        ),
        "min_distance_p05": float(
            np.percentile(
                min_distances,
                5,
            )
        ),
        "min_distance_p10": float(
            np.percentile(
                min_distances,
                10,
            )
        ),
        "min_distance_p25": float(
            np.percentile(
                min_distances,
                25,
            )
        ),
        "min_dcpa_mean": float(
            np.mean(
                min_dcpas
            )
        ),
        "min_dcpa_min": float(
            np.min(
                min_dcpas
            )
        ),
        "min_dcpa_p05": float(
            np.percentile(
                min_dcpas,
                5,
            )
        ),
        "min_dcpa_p10": float(
            np.percentile(
                min_dcpas,
                10,
            )
        ),
        "final_goal_distance_mean": float(
            np.mean(
                final_goal_distances
            )
        ),
        "episode_length_mean": float(
            np.mean(
                lengths
            )
        ),
        "results": results,
    }

    summary[
        "selection_score"
    ] = compute_selection_score(
        summary
    )

    return summary


def compute_selection_score(
    summary: dict[str, Any],
) -> float:
    return float(
        summary[
            "reward_mean"
        ]
        - 300.0
        * summary[
            "collision_rate"
        ]
        + 200.0
        * summary[
            "goal_rate"
        ]
        + 0.30
        * summary[
            "min_distance_mean"
        ]
        + 0.50
        * summary[
            "min_distance_p10"
        ]
        - 0.02
        * summary[
            "final_goal_distance_mean"
        ]
    )


def evaluate_agent(
    agent: PPOContinuousAgent,
    seeds: list[int],
    name: str,
) -> dict[str, Any]:
    results = []

    for seed in seeds:
        result = run_eval_episode(
            agent=agent,
            seed=seed,
        )

        results.append(
            result
        )

    summary = summarize_results(
        name=name,
        results=results,
    )

    return summary


def print_eval_summary(
    summary: dict[str, Any],
) -> None:
    print()
    print(
        "="
        * 88
    )

    print(
        summary[
            "name"
        ]
    )

    print(
        "="
        * 88
    )

    keys = [
        "episodes",
        "reward_mean",
        "reward_std",
        "reward_min",
        "reward_max",
        "env_reward_mean",
        "colreg_reward_mean",
        "collision_rate",
        "goal_rate",
        "min_distance_mean",
        "min_distance_min",
        "min_distance_p05",
        "min_distance_p10",
        "min_dcpa_mean",
        "min_dcpa_min",
        "final_goal_distance_mean",
        "episode_length_mean",
        "selection_score",
    ]

    for key in keys:
        value = summary[
            key
        ]

        if isinstance(
            value,
            float,
        ):
            print(
                f"{key:30s}: {value: .6f}"
            )
        else:
            print(
                f"{key:30s}: {value}"
            )


def reset_episode_stats(
    info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "reward": 0.0,
        "env_reward": 0.0,
        "colreg_reward": 0.0,
        "length": 0,
        "min_distance": float(
            info[
                "distance_to_traffic"
            ]
        ),
        "min_dcpa": float(
            info[
                "dcpa"
            ]
        ),
    }


def main() -> None:
    set_global_seed(
        SEED
    )

    agent = load_agent()

    env = make_env()

    training_history: list[dict[str, Any]] = []
    episode_history: list[dict[str, Any]] = []

    best_selection_score = -float(
        "inf"
    )

    global_step = 0
    completed_episodes = 0

    current_seed = TRAIN_SEED_BASE

    observation, info = env.reset(
        seed=current_seed
    )

    episode_stats = reset_episode_stats(
        info
    )

    print()
    print(
        "="
        * 88
    )

    print(
        "PPO CONFLICT-PRESERVING FINE-TUNING"
    )

    print(
        "="
        * 88
    )

    print(
        f"Initial checkpoint : {INITIAL_CHECKPOINT}"
    )

    print(
        f"Best checkpoint    : {BEST_CHECKPOINT}"
    )

    print(
        f"Final checkpoint   : {FINAL_CHECKPOINT}"
    )

    print(
        f"Updates            : {NUM_UPDATES}"
    )

    print(
        f"Rollout steps      : {ROLLOUT_STEPS}"
    )

    print(
        f"Total steps        : {NUM_UPDATES * ROLLOUT_STEPS}"
    )

    print(
        f"COLREG weight      : {COLREG_WEIGHT}"
    )

    for update_index in range(
        1,
        NUM_UPDATES
        + 1,
    ):
        for rollout_step in range(
            ROLLOUT_STEPS
        ):
            state_batch = observation.reshape(
                1,
                -1,
            ).astype(
                np.float32
            )

            (
                action_batch,
                pre_tanh_action_batch,
                log_prob_batch,
                value_batch,
            ) = agent.sample_action(
                state_batch
            )

            action = np.asarray(
                action_batch[0],
                dtype=np.float32,
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

            done = bool(
                terminated
                or truncated
            )

            next_state_batch = next_observation.reshape(
                1,
                -1,
            ).astype(
                np.float32
            )

            reward_batch = np.array(
                [
                    float(
                        reward
                    )
                ],
                dtype=np.float32,
            )

            terminated_batch = np.array(
                [
                    bool(
                        terminated
                    )
                ],
                dtype=np.bool_,
            )

            episode_end_batch = np.array(
                [
                    done
                ],
                dtype=np.bool_,
            )

            agent.store_transition(
                states=state_batch,
                actions=action_batch,
                pre_tanh_actions=pre_tanh_action_batch,
                log_probs=log_prob_batch,
                rewards=reward_batch,
                terminated=terminated_batch,
                episode_ends=episode_end_batch,
                next_states=next_state_batch,
            )

            global_step += 1

            episode_stats[
                "reward"
            ] += float(
                reward
            )

            episode_stats[
                "env_reward"
            ] += float(
                info.get(
                    "env_reward",
                    0.0,
                )
            )

            episode_stats[
                "colreg_reward"
            ] += float(
                info.get(
                    "colreg_reward",
                    0.0,
                )
            )

            episode_stats[
                "length"
            ] += 1

            episode_stats[
                "min_distance"
            ] = min(
                episode_stats[
                    "min_distance"
                ],
                float(
                    info[
                        "distance_to_traffic"
                    ]
                ),
            )

            episode_stats[
                "min_dcpa"
            ] = min(
                episode_stats[
                    "min_dcpa"
                ],
                float(
                    info[
                        "dcpa"
                    ]
                ),
            )

            observation = next_observation

            if done:
                completed_episodes += 1

                episode_record = {
                    "episode": int(
                        completed_episodes
                    ),
                    "seed": int(
                        current_seed
                    ),
                    "global_step": int(
                        global_step
                    ),
                    "reward": float(
                        episode_stats[
                            "reward"
                        ]
                    ),
                    "env_reward": float(
                        episode_stats[
                            "env_reward"
                        ]
                    ),
                    "colreg_reward": float(
                        episode_stats[
                            "colreg_reward"
                        ]
                    ),
                    "length": int(
                        episode_stats[
                            "length"
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
                        episode_stats[
                            "min_distance"
                        ]
                    ),
                    "min_dcpa": float(
                        episode_stats[
                            "min_dcpa"
                        ]
                    ),
                    "final_goal_distance": float(
                        info[
                            "distance_to_goal"
                        ]
                    ),
                }

                episode_history.append(
                    episode_record
                )

                current_seed = (
                    TRAIN_SEED_BASE
                    + completed_episodes
                )

                observation, info = env.reset(
                    seed=current_seed
                )

                episode_stats = reset_episode_stats(
                    info
                )

        update_metrics = agent.update()

        recent_episodes = episode_history[
            -20:
        ]

        if recent_episodes:
            recent_reward_mean = float(
                np.mean(
                    [
                        item[
                            "reward"
                        ]
                        for item in recent_episodes
                    ]
                )
            )

            recent_collision_rate = float(
                np.mean(
                    [
                        item[
                            "collision"
                        ]
                        for item in recent_episodes
                    ]
                )
            )

            recent_goal_rate = float(
                np.mean(
                    [
                        item[
                            "goal_reached"
                        ]
                        for item in recent_episodes
                    ]
                )
            )
        else:
            recent_reward_mean = float(
                "nan"
            )

            recent_collision_rate = float(
                "nan"
            )

            recent_goal_rate = float(
                "nan"
            )

        history_row = {
            "update": int(
                update_index
            ),
            "global_step": int(
                global_step
            ),
            "completed_episodes": int(
                completed_episodes
            ),
            "recent_reward_mean": recent_reward_mean,
            "recent_collision_rate": recent_collision_rate,
            "recent_goal_rate": recent_goal_rate,
            "update_metrics": update_metrics,
        }

        if (
            update_index
            % EVAL_INTERVAL
            == 0
            or update_index
            == 1
        ):
            eval_summary = evaluate_agent(
                agent=agent,
                seeds=EVAL_SEEDS,
                name=(
                    "eval_conflict_preserving_update_"
                    f"{update_index}"
                ),
            )

            history_row[
                "eval_summary"
            ] = eval_summary

            selection_score = float(
                eval_summary[
                    "selection_score"
                ]
            )

            print_eval_summary(
                eval_summary
            )

            if selection_score > best_selection_score:
                best_selection_score = selection_score

                agent.save(
                    BEST_CHECKPOINT,
                    extra={
                        "update": int(
                            update_index
                        ),
                        "global_step": int(
                            global_step
                        ),
                        "selection_score": float(
                            best_selection_score
                        ),
                        "eval_summary": to_serializable(
                            eval_summary
                        ),
                        "initial_checkpoint": str(
                            INITIAL_CHECKPOINT
                        ),
                        "randomization_mode": "conflict_preserving",
                        "colreg_weight": COLREG_WEIGHT,
                    },
                )

                print(
                    f"New best checkpoint saved: {BEST_CHECKPOINT}"
                )

        training_history.append(
            history_row
        )

        save_json(
            TRAINING_HISTORY_PATH,
            {
                "config": {
                    "seed": SEED,
                    "num_updates": NUM_UPDATES,
                    "rollout_steps": ROLLOUT_STEPS,
                    "max_episode_steps": MAX_EPISODE_STEPS,
                    "train_seed_base": TRAIN_SEED_BASE,
                    "eval_seeds": EVAL_SEEDS,
                    "final_eval_seeds": FINAL_EVAL_SEEDS,
                    "colreg_weight": COLREG_WEIGHT,
                    "initial_checkpoint": str(
                        INITIAL_CHECKPOINT
                    ),
                    "best_checkpoint": str(
                        BEST_CHECKPOINT
                    ),
                    "final_checkpoint": str(
                        FINAL_CHECKPOINT
                    ),
                },
                "training_history": training_history,
                "episode_history": episode_history,
            },
        )

        print(
            f"[update {update_index:03d}/{NUM_UPDATES:03d}] "
            f"global_step={global_step} "
            f"episodes={completed_episodes} "
            f"recent_R={recent_reward_mean:+.3f} "
            f"recent_col={recent_collision_rate:.3f} "
            f"recent_goal={recent_goal_rate:.3f}"
        )

    agent.save(
        FINAL_CHECKPOINT,
        extra={
            "global_step": int(
                global_step
            ),
            "completed_episodes": int(
                completed_episodes
            ),
            "initial_checkpoint": str(
                INITIAL_CHECKPOINT
            ),
            "randomization_mode": "conflict_preserving",
            "colreg_weight": COLREG_WEIGHT,
        },
    )

    print(
        f"Final checkpoint saved: {FINAL_CHECKPOINT}"
    )

    if BEST_CHECKPOINT.exists():
        best_agent = PPOContinuousAgent(
            state_size=STATE_SIZE,
            action_size=ACTION_SIZE,
            num_steps=ROLLOUT_STEPS,
            num_envs=NUM_ENVS,
        )

        best_agent.load(
            BEST_CHECKPOINT
        )

        final_eval_summary = evaluate_agent(
            agent=best_agent,
            seeds=FINAL_EVAL_SEEDS,
            name="final_eval_best_conflict_preserving_checkpoint",
        )
    else:
        final_eval_summary = evaluate_agent(
            agent=agent,
            seeds=FINAL_EVAL_SEEDS,
            name="final_eval_final_conflict_preserving_checkpoint",
        )

    print_eval_summary(
        final_eval_summary
    )

    save_json(
        FINAL_EVALUATION_PATH,
        {
            "final_eval_summary": final_eval_summary,
            "best_checkpoint": str(
                BEST_CHECKPOINT
            ),
            "final_checkpoint": str(
                FINAL_CHECKPOINT
            ),
        },
    )

    env.close()


if __name__ == "__main__":
    main()
