from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import random

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.agents.ppo_continuous import PPOContinuousAgent
from experiments.envs.commonocean_env import CommonOceanEnv
from experiments.preferences.colreg_reward_wrapper import (
    ColregRewardConfig,
    ColregRewardWrapper,
)


SEED = 123

STATE_SIZE = 13
ACTION_SIZE = 2

NUM_ENVS = 1
ROLLOUT_STEPS = 256
MAX_EPISODE_STEPS = 220

NUM_UPDATES = 30
EVAL_INTERVAL = 5

COLREG_WEIGHT = 0.50

TRAIN_SEED_BASE = 100_000

EVAL_SEEDS = list(
    range(
        90_000,
        90_050,
    )
)

FINAL_EVAL_SEEDS = list(
    range(
        91_000,
        91_100,
    )
)

PROJECT_ROOT = Path(
    __file__
).resolve().parents[
    1
]

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "checkpoints"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "ppo_colreg_finetune"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

INITIAL_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "ppo_randomized_crossing_best.pt"
)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "ppo_colreg_finetuned_best.pt"
)

FINAL_MODEL_PATH = (
    CHECKPOINT_DIR
    / "ppo_colreg_finetuned_final.pt"
)

TRAINING_HISTORY_PATH = (
    RESULTS_DIR
    / "training_history.json"
)

FINAL_EVALUATION_PATH = (
    RESULTS_DIR
    / "final_evaluation.json"
)

TOTAL_TIMESTEPS = (
    NUM_UPDATES
    * ROLLOUT_STEPS
    * NUM_ENVS
)


BASELINE_RANDOMIZED_PPO = {
    "episodes": 100,
    "reward_mean": 258.4491982713799,
    "reward_std": 14.785982258968325,
    "reward_min": 222.72712747262244,
    "reward_max": 283.47547139396113,
    "collision_rate": 0.0,
    "goal_rate": 1.0,
    "min_distance_mean": 187.27108655825288,
    "min_distance_std": 37.1361906618914,
    "min_distance_min": 107.47248675082686,
    "min_distance_p05": 128.3016036720419,
    "min_distance_p10": 142.8504261483797,
    "min_distance_p25": 162.13017024613066,
    "min_dcpa_mean": 37.2721093448926,
    "min_dcpa_min": 0.18411156203504947,
    "final_goal_distance_mean": 164.51134740751488,
    "episode_length_mean": 204.69,
}


def set_seed(
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


def save_figure(
    fig,
    filename_base: str,
) -> None:
    png_path = (
        RESULTS_DIR
        / f"{filename_base}.png"
    )

    pdf_path = (
        RESULTS_DIR
        / f"{filename_base}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved figure: {png_path}"
    )

    print(
        f"Saved figure: {pdf_path}"
    )


def make_env(
    colreg_weight: float = COLREG_WEIGHT,
) -> ColregRewardWrapper:
    base_env = CommonOceanEnv(
        max_episode_steps=MAX_EPISODE_STEPS,
        render_mode=None,
        randomize_scenario=True,
    )

    wrapper = ColregRewardWrapper(
        base_env,
        config=ColregRewardConfig(
            colreg_weight=colreg_weight,
        ),
    )

    return wrapper


def make_agent() -> PPOContinuousAgent:
    agent = PPOContinuousAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        num_steps=ROLLOUT_STEPS,
        num_envs=NUM_ENVS,
    )

    return agent


def load_initial_agent() -> PPOContinuousAgent:
    if not INITIAL_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el checkpoint inicial para fine-tuning:\n"
            f"{INITIAL_CHECKPOINT_PATH}\n\n"
            "Primero debes entrenar o copiar ppo_randomized_crossing_best.pt."
        )

    agent = make_agent()

    agent.load(
        str(
            INITIAL_CHECKPOINT_PATH
        )
    )

    return agent


def run_deterministic_episode(
    agent: PPOContinuousAgent,
    seed: int,
    colreg_weight: float = COLREG_WEIGHT,
    record_trajectory: bool = False,
) -> dict[str, Any]:
    env = make_env(
        colreg_weight=colreg_weight
    )

    observation, info = env.reset(
        seed=seed
    )

    total_reward = 0.0
    env_reward_sum = 0.0
    colreg_reward_sum = 0.0

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

    ego_positions = []
    traffic_positions = []
    distances = []
    dcpas = []
    tcpas = []
    actions = []

    if record_trajectory:
        ego_positions.append(
            np.asarray(
                info[
                    "ego_position"
                ],
                dtype=np.float64,
            )
        )

        traffic_positions.append(
            np.asarray(
                info[
                    "traffic_position"
                ],
                dtype=np.float64,
            )
        )

        distances.append(
            float(
                info[
                    "distance_to_traffic"
                ]
            )
        )

        dcpas.append(
            float(
                info[
                    "dcpa"
                ]
            )
        )

        tcpas.append(
            float(
                info[
                    "tcpa"
                ]
            )
        )

    terminated = False
    truncated = False

    while not (
        terminated
        or truncated
    ):
        action = np.asarray(
            agent.deterministic_action(
                observation
            ),
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

        env_reward_sum += float(
            info[
                "env_reward"
            ]
        )

        colreg_reward_sum += float(
            info[
                "colreg_reward"
            ]
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

        if record_trajectory:
            ego_positions.append(
                np.asarray(
                    info[
                        "ego_position"
                    ],
                    dtype=np.float64,
                )
            )

            traffic_positions.append(
                np.asarray(
                    info[
                        "traffic_position"
                    ],
                    dtype=np.float64,
                )
            )

            distances.append(
                float(
                    info[
                        "distance_to_traffic"
                    ]
                )
            )

            dcpas.append(
                float(
                    info[
                        "dcpa"
                    ]
                )
            )

            tcpas.append(
                float(
                    info[
                        "tcpa"
                    ]
                )
            )

            actions.append(
                action.copy()
            )

    result = {
        "seed": int(
            seed
        ),
        "reward": float(
            total_reward
        ),
        "env_reward": float(
            env_reward_sum
        ),
        "colreg_reward": float(
            colreg_reward_sum
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
        "scenario": dict(
            info[
                "scenario"
            ]
        ),
    }

    if record_trajectory:
        result[
            "ego_positions"
        ] = np.asarray(
            ego_positions,
            dtype=np.float64,
        )

        result[
            "traffic_positions"
        ] = np.asarray(
            traffic_positions,
            dtype=np.float64,
        )

        result[
            "distances"
        ] = np.asarray(
            distances,
            dtype=np.float64,
        )

        result[
            "dcpas"
        ] = np.asarray(
            dcpas,
            dtype=np.float64,
        )

        result[
            "tcpas"
        ] = np.asarray(
            tcpas,
            dtype=np.float64,
        )

        result[
            "actions"
        ] = np.asarray(
            actions,
            dtype=np.float64,
        )

    env.close()

    return result


def summarize_results(
    results: list[dict[str, Any]],
) -> dict[str, float]:
    rewards = np.asarray(
        [
            result[
                "reward"
            ]
            for result in results
        ],
        dtype=np.float64,
    )

    env_rewards = np.asarray(
        [
            result[
                "env_reward"
            ]
            for result in results
        ],
        dtype=np.float64,
    )

    colreg_rewards = np.asarray(
        [
            result[
                "colreg_reward"
            ]
            for result in results
        ],
        dtype=np.float64,
    )

    collisions = np.asarray(
        [
            result[
                "collision"
            ]
            for result in results
        ],
        dtype=np.float64,
    )

    goals = np.asarray(
        [
            result[
                "goal_reached"
            ]
            for result in results
        ],
        dtype=np.float64,
    )

    min_distances = np.asarray(
        [
            result[
                "min_distance"
            ]
            for result in results
        ],
        dtype=np.float64,
    )

    min_dcpas = np.asarray(
        [
            result[
                "min_dcpa"
            ]
            for result in results
        ],
        dtype=np.float64,
    )

    final_goal_distances = np.asarray(
        [
            result[
                "final_goal_distance"
            ]
            for result in results
        ],
        dtype=np.float64,
    )

    episode_lengths = np.asarray(
        [
            result[
                "steps"
            ]
            for result in results
        ],
        dtype=np.float64,
    )

    summary = {
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
                episode_lengths
            )
        ),
    }

    summary[
        "selection_score"
    ] = compute_selection_score(
        summary
    )

    return summary


def compute_selection_score(
    summary: dict[str, float],
) -> float:
    return float(
        summary[
            "reward_mean"
        ]
        - 100.0
        * summary[
            "collision_rate"
        ]
        + 0.25
        * summary[
            "min_distance_mean"
        ]
        + 0.50
        * summary[
            "min_distance_p10"
        ]
        + 0.25
        * summary[
            "min_dcpa_mean"
        ]
    )


def evaluate_agent(
    agent: PPOContinuousAgent,
    seeds: list[int],
    colreg_weight: float = COLREG_WEIGHT,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    results = []

    for seed in seeds:
        result = run_deterministic_episode(
            agent=agent,
            seed=seed,
            colreg_weight=colreg_weight,
            record_trajectory=False,
        )

        results.append(
            result
        )

    summary = summarize_results(
        results
    )

    return (
        summary,
        results,
    )


def train() -> dict[str, Any]:
    set_seed(
        SEED
    )

    print(
        "="
        * 72
    )

    print(
        "PPO COLREG FINE-TUNING"
    )

    print(
        "="
        * 72
    )

    print(
        "Initial checkpoint:",
        INITIAL_CHECKPOINT_PATH,
    )

    print(
        "Best checkpoint:",
        BEST_MODEL_PATH,
    )

    print(
        "Final checkpoint:",
        FINAL_MODEL_PATH,
    )

    print(
        "Results dir:",
        RESULTS_DIR,
    )

    print(
        "Total timesteps:",
        TOTAL_TIMESTEPS,
    )

    print(
        "COLREG weight:",
        COLREG_WEIGHT,
    )

    agent = load_initial_agent()

    env = make_env(
        colreg_weight=COLREG_WEIGHT
    )

    observation, info = env.reset(
        seed=TRAIN_SEED_BASE
    )

    episode_seed = TRAIN_SEED_BASE

    current_episode_reward = 0.0
    current_episode_env_reward = 0.0
    current_episode_colreg_reward = 0.0
    current_episode_length = 0

    current_episode_min_distance = float(
        info[
            "distance_to_traffic"
        ]
    )

    current_episode_min_dcpa = float(
        info[
            "dcpa"
        ]
    )

    episode_rewards = []
    episode_env_rewards = []
    episode_colreg_rewards = []
    episode_lengths = []
    episode_collisions = []
    episode_goals = []
    episode_min_distances = []
    episode_min_dcpas = []

    training_history = []

    global_step = 0
    best_selection_score = -float(
        "inf"
    )

    initial_eval_summary, _ = evaluate_agent(
        agent=agent,
        seeds=EVAL_SEEDS,
        colreg_weight=COLREG_WEIGHT,
    )

    print()
    print(
        "Initial evaluation:"
    )

    print_summary(
        initial_eval_summary,
        prefix="  ",
    )

    for update_index in range(
        1,
        NUM_UPDATES
        + 1,
    ):
        agent.reset_rollout_buffer()

        for rollout_step in range(
            ROLLOUT_STEPS
        ):
            action, pre_tanh_action, log_prob, value = (
                agent.sample_action(
                    observation
                )
            )

            action = np.asarray(
                action,
                dtype=np.float32,
            )

            pre_tanh_action = np.asarray(
                pre_tanh_action,
                dtype=np.float32,
            )

            next_observation, reward, terminated, truncated, info = (
                env.step(
                    action
                )
            )

            episode_end = bool(
                terminated
                or truncated
            )

            agent.store_transition(
                states=observation.reshape(
                    NUM_ENVS,
                    STATE_SIZE,
                ),
                actions=action.reshape(
                    NUM_ENVS,
                    ACTION_SIZE,
                ),
                pre_tanh_actions=pre_tanh_action.reshape(
                    NUM_ENVS,
                    ACTION_SIZE,
                ),
                log_probs=np.asarray(
                    [
                        log_prob
                    ],
                    dtype=np.float32,
                ),
                rewards=np.asarray(
                    [
                        reward
                    ],
                    dtype=np.float32,
                ),
                terminated=np.asarray(
                    [
                        float(
                            terminated
                        )
                    ],
                    dtype=np.float32,
                ),
                episode_ends=np.asarray(
                    [
                        float(
                            episode_end
                        )
                    ],
                    dtype=np.float32,
                ),
                next_states=next_observation.reshape(
                    NUM_ENVS,
                    STATE_SIZE,
                ),
            )

            global_step += 1

            current_episode_reward += float(
                reward
            )

            current_episode_env_reward += float(
                info[
                    "env_reward"
                ]
            )

            current_episode_colreg_reward += float(
                info[
                    "colreg_reward"
                ]
            )

            current_episode_length += 1

            current_episode_min_distance = min(
                current_episode_min_distance,
                float(
                    info[
                        "distance_to_traffic"
                    ]
                ),
            )

            current_episode_min_dcpa = min(
                current_episode_min_dcpa,
                float(
                    info[
                        "dcpa"
                    ]
                ),
            )

            if episode_end:
                episode_rewards.append(
                    float(
                        current_episode_reward
                    )
                )

                episode_env_rewards.append(
                    float(
                        current_episode_env_reward
                    )
                )

                episode_colreg_rewards.append(
                    float(
                        current_episode_colreg_reward
                    )
                )

                episode_lengths.append(
                    int(
                        current_episode_length
                    )
                )

                episode_collisions.append(
                    bool(
                        info[
                            "collision"
                        ]
                    )
                )

                episode_goals.append(
                    bool(
                        info[
                            "goal_reached"
                        ]
                    )
                )

                episode_min_distances.append(
                    float(
                        current_episode_min_distance
                    )
                )

                episode_min_dcpas.append(
                    float(
                        current_episode_min_dcpa
                    )
                )

                episode_seed += 1

                observation, info = env.reset(
                    seed=episode_seed
                )

                current_episode_reward = 0.0
                current_episode_env_reward = 0.0
                current_episode_colreg_reward = 0.0
                current_episode_length = 0

                current_episode_min_distance = float(
                    info[
                        "distance_to_traffic"
                    ]
                )

                current_episode_min_dcpa = float(
                    info[
                        "dcpa"
                    ]
                )
            else:
                observation = next_observation

        update_metrics = agent.update()

        recent_window = min(
            len(
                episode_rewards
            ),
            20,
        )

        if recent_window > 0:
            recent_rewards = episode_rewards[
                -recent_window:
            ]

            recent_colreg_rewards = episode_colreg_rewards[
                -recent_window:
            ]

            recent_collisions = episode_collisions[
                -recent_window:
            ]

            recent_goals = episode_goals[
                -recent_window:
            ]

            recent_min_distances = episode_min_distances[
                -recent_window:
            ]

            recent_min_dcpas = episode_min_dcpas[
                -recent_window:
            ]

            train_recent_summary = {
                "reward_mean": float(
                    np.mean(
                        recent_rewards
                    )
                ),
                "colreg_reward_mean": float(
                    np.mean(
                        recent_colreg_rewards
                    )
                ),
                "collision_rate": float(
                    np.mean(
                        recent_collisions
                    )
                ),
                "goal_rate": float(
                    np.mean(
                        recent_goals
                    )
                ),
                "min_distance_mean": float(
                    np.mean(
                        recent_min_distances
                    )
                ),
                "min_dcpa_mean": float(
                    np.mean(
                        recent_min_dcpas
                    )
                ),
            }
        else:
            train_recent_summary = {
                "reward_mean": float(
                    "nan"
                ),
                "colreg_reward_mean": float(
                    "nan"
                ),
                "collision_rate": float(
                    "nan"
                ),
                "goal_rate": float(
                    "nan"
                ),
                "min_distance_mean": float(
                    "nan"
                ),
                "min_dcpa_mean": float(
                    "nan"
                ),
            }

        if (
            update_index == 1
            or update_index % EVAL_INTERVAL == 0
            or update_index == NUM_UPDATES
        ):
            eval_summary, _ = evaluate_agent(
                agent=agent,
                seeds=EVAL_SEEDS,
                colreg_weight=COLREG_WEIGHT,
            )
        else:
            eval_summary = None

        history_item = {
            "update": int(
                update_index
            ),
            "global_step": int(
                global_step
            ),
            "update_metrics": update_metrics,
            "train_recent": train_recent_summary,
            "eval": eval_summary,
        }

        training_history.append(
            history_item
        )

        print()
        print(
            f"Update {update_index:03d}/{NUM_UPDATES} "
            f"| step={global_step:06d}"
        )

        print(
            f"  train_recent: "
            f"R={train_recent_summary['reward_mean']:+8.2f} "
            f"R_colreg={train_recent_summary['colreg_reward_mean']:+8.2f} "
            f"col={train_recent_summary['collision_rate']:.2f} "
            f"goal={train_recent_summary['goal_rate']:.2f} "
            f"dmin={train_recent_summary['min_distance_mean']:7.2f} "
            f"DCPA={train_recent_summary['min_dcpa_mean']:7.2f}"
        )

        if update_metrics is not None:
            print(
                f"  losses: "
                f"loss={update_metrics['loss']:+.4f} "
                f"actor={update_metrics['actor_loss']:+.4f} "
                f"critic={update_metrics['critic_loss']:+.4f} "
                f"entropy={update_metrics['entropy']:+.4f}"
            )

        if eval_summary is not None:
            print(
                "  eval:"
            )

            print_summary(
                eval_summary,
                prefix="    ",
            )

            current_selection_score = float(
                eval_summary[
                    "selection_score"
                ]
            )

            if current_selection_score > best_selection_score:
                best_selection_score = current_selection_score

                agent.save(
                    str(
                        BEST_MODEL_PATH
                    ),
                    extra={
                        "type": "ppo_colreg_finetuned_best",
                        "update": int(
                            update_index
                        ),
                        "global_step": int(
                            global_step
                        ),
                        "colreg_weight": float(
                            COLREG_WEIGHT
                        ),
                        "eval_summary": eval_summary,
                        "baseline_randomized_ppo": BASELINE_RANDOMIZED_PPO,
                    },
                )

                print(
                    f"  New best checkpoint saved "
                    f"(selection_score={best_selection_score:+.3f})"
                )

        save_json(
            TRAINING_HISTORY_PATH,
            {
                "config": config_dict(),
                "history": training_history,
                "episode_rewards": episode_rewards,
                "episode_env_rewards": episode_env_rewards,
                "episode_colreg_rewards": episode_colreg_rewards,
                "episode_lengths": episode_lengths,
                "episode_collisions": episode_collisions,
                "episode_goals": episode_goals,
                "episode_min_distances": episode_min_distances,
                "episode_min_dcpas": episode_min_dcpas,
            },
        )

    agent.save(
        str(
            FINAL_MODEL_PATH
        ),
        extra={
            "type": "ppo_colreg_finetuned_final",
            "global_step": int(
                global_step
            ),
            "colreg_weight": float(
                COLREG_WEIGHT
            ),
            "baseline_randomized_ppo": BASELINE_RANDOMIZED_PPO,
        },
    )

    env.close()

    final_eval_summary, final_eval_results = evaluate_agent(
        agent=agent,
        seeds=FINAL_EVAL_SEEDS,
        colreg_weight=COLREG_WEIGHT,
    )

    final_payload = {
        "config": config_dict(),
        "baseline_randomized_ppo": BASELINE_RANDOMIZED_PPO,
        "final_evaluation_summary": final_eval_summary,
        "final_evaluation_results": final_eval_results,
    }

    save_json(
        FINAL_EVALUATION_PATH,
        final_payload,
    )

    plot_training_curves(
        training_history=training_history,
        episode_rewards=episode_rewards,
        episode_colreg_rewards=episode_colreg_rewards,
        episode_min_distances=episode_min_distances,
        episode_min_dcpas=episode_min_dcpas,
        episode_collisions=episode_collisions,
        episode_goals=episode_goals,
    )

    print()
    print(
        "="
        * 72
    )

    print(
        "COLREG FINE-TUNING COMPLETE"
    )

    print(
        "="
        * 72
    )

    print(
        "Global steps        :",
        global_step,
    )

    print(
        "Episodes            :",
        len(
            episode_rewards
        ),
    )

    print(
        "Best selection score:",
        f"{best_selection_score:+.3f}",
    )

    print(
        "Best checkpoint     :",
        BEST_MODEL_PATH,
    )

    print(
        "Final checkpoint    :",
        FINAL_MODEL_PATH,
    )

    print()
    print(
        "Final evaluation:"
    )

    print_summary(
        final_eval_summary,
        prefix="  ",
    )

    return final_payload


def moving_average(
    values: list[float],
    window: int = 10,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.size == 0:
        return array

    if array.size < window:
        return array

    kernel = np.ones(
        window,
        dtype=np.float64,
    ) / float(
        window
    )

    return np.convolve(
        array,
        kernel,
        mode="valid",
    )


def plot_training_curves(
    training_history: list[dict[str, Any]],
    episode_rewards: list[float],
    episode_colreg_rewards: list[float],
    episode_min_distances: list[float],
    episode_min_dcpas: list[float],
    episode_collisions: list[bool],
    episode_goals: list[bool],
) -> None:


    if len(
        episode_rewards
    ) > 0:
        fig, ax = plt.subplots(
            figsize=(
                8,
                4,
            )
        )

        ax.plot(
            episode_rewards,
            linewidth=1.2,
            alpha=0.45,
            label="Episode total reward",
        )

        ma = moving_average(
            episode_rewards,
            window=10,
        )

        if ma.size > 0:
            ax.plot(
                np.arange(
                    len(
                        ma
                    )
                )
                + 9,
                ma,
                linewidth=2.0,
                label="Moving average 10",
            )

        ax.set_xlabel(
            "Episode"
        )

        ax.set_ylabel(
            "Reward"
        )

        ax.set_title(
            "COLREG fine-tuning: episode reward"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        fig.tight_layout()

        save_figure(
            fig,
            "colreg_finetune_episode_reward",
        )


    if len(
        episode_colreg_rewards
    ) > 0:
        fig, ax = plt.subplots(
            figsize=(
                8,
                4,
            )
        )

        ax.plot(
            episode_colreg_rewards,
            linewidth=1.2,
            alpha=0.45,
            label="Episode COLREG reward",
        )

        ma = moving_average(
            episode_colreg_rewards,
            window=10,
        )

        if ma.size > 0:
            ax.plot(
                np.arange(
                    len(
                        ma
                    )
                )
                + 9,
                ma,
                linewidth=2.0,
                label="Moving average 10",
            )

        ax.set_xlabel(
            "Episode"
        )

        ax.set_ylabel(
            "COLREG auxiliary reward"
        )

        ax.set_title(
            "COLREG fine-tuning: auxiliary reward"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        fig.tight_layout()

        save_figure(
            fig,
            "colreg_finetune_auxiliary_reward",
        )


    if len(
        episode_min_distances
    ) > 0:
        fig, ax = plt.subplots(
            figsize=(
                8,
                4,
            )
        )

        ax.plot(
            episode_min_distances,
            linewidth=1.2,
            alpha=0.55,
            label="Min distance",
        )

        ax.plot(
            episode_min_dcpas,
            linewidth=1.2,
            alpha=0.55,
            label="Min DCPA",
        )

        ax.axhline(
            300.0,
            linestyle="--",
            linewidth=1.0,
            label="Safe distance 300 m",
        )

        ax.axhline(
            200.0,
            linestyle=":",
            linewidth=1.0,
            label="Safe DCPA 200 m",
        )

        ax.set_xlabel(
            "Episode"
        )

        ax.set_ylabel(
            "Distance [m]"
        )

        ax.set_title(
            "COLREG fine-tuning: safety margins"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        fig.tight_layout()

        save_figure(
            fig,
            "colreg_finetune_safety_margins",
        )


    eval_items = [
        item
        for item in training_history
        if item[
            "eval"
        ]
        is not None
    ]

    if len(
        eval_items
    ) > 0:
        updates = np.asarray(
            [
                item[
                    "update"
                ]
                for item in eval_items
            ],
            dtype=np.float64,
        )

        reward_mean = np.asarray(
            [
                item[
                    "eval"
                ][
                    "reward_mean"
                ]
                for item in eval_items
            ],
            dtype=np.float64,
        )

        min_distance_mean = np.asarray(
            [
                item[
                    "eval"
                ][
                    "min_distance_mean"
                ]
                for item in eval_items
            ],
            dtype=np.float64,
        )

        min_distance_p10 = np.asarray(
            [
                item[
                    "eval"
                ][
                    "min_distance_p10"
                ]
                for item in eval_items
            ],
            dtype=np.float64,
        )

        min_dcpa_mean = np.asarray(
            [
                item[
                    "eval"
                ][
                    "min_dcpa_mean"
                ]
                for item in eval_items
            ],
            dtype=np.float64,
        )

        collision_rate = np.asarray(
            [
                item[
                    "eval"
                ][
                    "collision_rate"
                ]
                for item in eval_items
            ],
            dtype=np.float64,
        )

        goal_rate = np.asarray(
            [
                item[
                    "eval"
                ][
                    "goal_rate"
                ]
                for item in eval_items
            ],
            dtype=np.float64,
        )

        fig, ax = plt.subplots(
            figsize=(
                8,
                4,
            )
        )

        ax.plot(
            updates,
            reward_mean,
            marker="o",
            label="Reward mean",
        )

        ax.set_xlabel(
            "Update"
        )

        ax.set_ylabel(
            "Reward"
        )

        ax.set_title(
            "COLREG fine-tuning: evaluation reward"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        fig.tight_layout()

        save_figure(
            fig,
            "colreg_finetune_eval_reward",
        )

        fig, ax = plt.subplots(
            figsize=(
                8,
                4,
            )
        )

        ax.plot(
            updates,
            min_distance_mean,
            marker="o",
            label="Min distance mean",
        )

        ax.plot(
            updates,
            min_distance_p10,
            marker="o",
            label="Min distance p10",
        )

        ax.plot(
            updates,
            min_dcpa_mean,
            marker="o",
            label="Min DCPA mean",
        )

        ax.axhline(
            BASELINE_RANDOMIZED_PPO[
                "min_distance_mean"
            ],
            linestyle="--",
            linewidth=1.0,
            label="Baseline min distance mean",
        )

        ax.axhline(
            BASELINE_RANDOMIZED_PPO[
                "min_distance_p10"
            ],
            linestyle=":",
            linewidth=1.0,
            label="Baseline min distance p10",
        )

        ax.set_xlabel(
            "Update"
        )

        ax.set_ylabel(
            "Distance [m]"
        )

        ax.set_title(
            "COLREG fine-tuning: evaluation safety margins"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        fig.tight_layout()

        save_figure(
            fig,
            "colreg_finetune_eval_safety",
        )

        fig, ax = plt.subplots(
            figsize=(
                8,
                4,
            )
        )

        ax.plot(
            updates,
            collision_rate,
            marker="o",
            label="Collision rate",
        )

        ax.plot(
            updates,
            goal_rate,
            marker="o",
            label="Goal rate",
        )

        ax.set_xlabel(
            "Update"
        )

        ax.set_ylabel(
            "Rate"
        )

        ax.set_ylim(
            -0.05,
            1.05,
        )

        ax.set_title(
            "COLREG fine-tuning: success and collision rates"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        fig.tight_layout()

        save_figure(
            fig,
            "colreg_finetune_eval_rates",
        )


def print_summary(
    summary: dict[str, float],
    prefix: str = "",
) -> None:
    print(
        f"{prefix}reward_mean        : "
        f"{summary['reward_mean']:+.3f}"
    )

    print(
        f"{prefix}env_reward_mean    : "
        f"{summary['env_reward_mean']:+.3f}"
    )

    print(
        f"{prefix}colreg_reward_mean : "
        f"{summary['colreg_reward_mean']:+.3f}"
    )

    print(
        f"{prefix}collision_rate     : "
        f"{summary['collision_rate']:.3f}"
    )

    print(
        f"{prefix}goal_rate          : "
        f"{summary['goal_rate']:.3f}"
    )

    print(
        f"{prefix}min_distance_mean  : "
        f"{summary['min_distance_mean']:.3f}"
    )

    print(
        f"{prefix}min_distance_p10   : "
        f"{summary['min_distance_p10']:.3f}"
    )

    print(
        f"{prefix}min_distance_min   : "
        f"{summary['min_distance_min']:.3f}"
    )

    print(
        f"{prefix}min_dcpa_mean      : "
        f"{summary['min_dcpa_mean']:.3f}"
    )

    print(
        f"{prefix}min_dcpa_min       : "
        f"{summary['min_dcpa_min']:.3f}"
    )

    print(
        f"{prefix}selection_score    : "
        f"{summary['selection_score']:+.3f}"
    )


def config_dict() -> dict[str, Any]:
    return {
        "seed": SEED,
        "state_size": STATE_SIZE,
        "action_size": ACTION_SIZE,
        "num_envs": NUM_ENVS,
        "rollout_steps": ROLLOUT_STEPS,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "num_updates": NUM_UPDATES,
        "eval_interval": EVAL_INTERVAL,
        "colreg_weight": COLREG_WEIGHT,
        "train_seed_base": TRAIN_SEED_BASE,
        "eval_seeds": EVAL_SEEDS,
        "final_eval_seeds": FINAL_EVAL_SEEDS,
        "total_timesteps": TOTAL_TIMESTEPS,
        "initial_checkpoint_path": str(
            INITIAL_CHECKPOINT_PATH
        ),
        "best_model_path": str(
            BEST_MODEL_PATH
        ),
        "final_model_path": str(
            FINAL_MODEL_PATH
        ),
        "results_dir": str(
            RESULTS_DIR
        ),
        "baseline_randomized_ppo": BASELINE_RANDOMIZED_PPO,
    }


def main() -> None:
    train()


if __name__ == "__main__":
    main()
