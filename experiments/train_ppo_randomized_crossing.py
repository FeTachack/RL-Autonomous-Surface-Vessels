from __future__ import annotations

from pathlib import Path
import json
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from experiments.agents.ppo_continuous import (
    PPOContinuousAgent,
)

from experiments.envs.commonocean_env import (
    CommonOceanEnv,
)


# ============================================================
# Configuration
# ============================================================

SEED = 123

STATE_SIZE = 13
ACTION_SIZE = 2

NUM_ENVS = 1
ROLLOUT_STEPS = 256
MAX_EPISODE_STEPS = 220

# Primer entrenamiento con generalización.
# Si tarda mucho, baja a 30.
NUM_UPDATES = 50

EVAL_INTERVAL = 5

TRAIN_SEED_BASE = 10_000
EVAL_SEEDS = list(
    range(
        50_000,
        50_010,
    )
)

TRAJECTORY_SEEDS = list(
    range(
        60_000,
        60_005,
    )
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

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "ppo_randomized_crossing"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Checkpoint nominal previo, usado como inicialización.
INITIAL_NOMINAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "ppo_commonocean_best.pt"
)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "ppo_randomized_crossing_best.pt"
)

FINAL_MODEL_PATH = (
    CHECKPOINT_DIR
    / "ppo_randomized_crossing_final.pt"
)

TOTAL_TIMESTEPS = (
    NUM_UPDATES
    * ROLLOUT_STEPS
    * NUM_ENVS
)


# ============================================================
# Baselines
# ============================================================

BASELINES = {
    "nominal_best_reward": 82.921,
    "nominal_previous_reward": 57.483,
    "fixed_evasive_reward": 19.636,
    "random_mean_reward": -71.932,
    "collision_baseline_reward": -174.643,
}


# ============================================================
# Utilities
# ============================================================


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
    obj,
):
    if isinstance(
        obj,
        np.ndarray,
    ):
        return obj.tolist()

    if isinstance(
        obj,
        np.generic,
    ):
        return obj.item()

    if isinstance(
        obj,
        dict,
    ):
        return {
            key: to_serializable(
                value
            )
            for key, value
            in obj.items()
        }

    if isinstance(
        obj,
        list,
    ):
        return [
            to_serializable(
                value
            )
            for value
            in obj
        ]

    return obj


def save_json(
    path: Path,
    data,
) -> None:
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            to_serializable(
                data
            ),
            file,
            indent=2,
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

    print(
        f"Saved figure: {png_path}"
    )

    print(
        f"Saved figure: {pdf_path}"
    )


def summarize_results(
    results: list[dict],
) -> dict:
    rewards = np.asarray(
        [
            result["reward"]
            for result in results
        ],
        dtype=np.float64,
    )

    collisions = np.asarray(
        [
            result["collision"]
            for result in results
        ],
        dtype=np.float64,
    )

    goals = np.asarray(
        [
            result["goal_reached"]
            for result in results
        ],
        dtype=np.float64,
    )

    min_distances = np.asarray(
        [
            result["min_distance"]
            for result in results
        ],
        dtype=np.float64,
    )

    final_goal_distances = np.asarray(
        [
            result["final_goal_distance"]
            for result in results
        ],
        dtype=np.float64,
    )

    steps = np.asarray(
        [
            result["steps"]
            for result in results
        ],
        dtype=np.float64,
    )

    return {
        "episodes": len(
            results
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
        "min_distance_min": float(
            np.min(
                min_distances
            )
        ),
        "final_goal_distance_mean": float(
            np.mean(
                final_goal_distances
            )
        ),
        "episode_length_mean": float(
            np.mean(
                steps
            )
        ),
    }


# ============================================================
# Episode runners
# ============================================================


def make_randomized_env() -> CommonOceanEnv:
    return CommonOceanEnv(
        max_episode_steps=MAX_EPISODE_STEPS,
        render_mode=None,
        randomize_scenario=True,
    )


def run_policy_episode(
    agent: PPOContinuousAgent,
    seed: int,
    record_trajectory: bool = False,
) -> dict:
    env = make_randomized_env()

    observation, info = env.reset(
        seed=seed
    )

    total_reward = 0.0
    min_distance = float(
        "inf"
    )

    terminated = False
    truncated = False

    ego_positions = []
    traffic_positions = []
    distances = []
    dcpas = []
    tcpas = []
    actions = []
    physical_actions = []
    rewards = []

    if record_trajectory:
        ego_positions.append(
            np.asarray(
                info["ego_position"],
                dtype=np.float64,
            )
        )

        traffic_positions.append(
            np.asarray(
                info["traffic_position"],
                dtype=np.float64,
            )
        )

        distances.append(
            float(
                info["distance_to_traffic"]
            )
        )

        dcpas.append(
            float(
                info["dcpa"]
            )
        )

        tcpas.append(
            float(
                info["tcpa"]
            )
        )

    while not (
        terminated
        or truncated
    ):
        action = agent.deterministic_action(
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

        min_distance = min(
            min_distance,
            float(
                info[
                    "distance_to_traffic"
                ]
            ),
        )

        if record_trajectory:
            ego_positions.append(
                np.asarray(
                    info["ego_position"],
                    dtype=np.float64,
                )
            )

            traffic_positions.append(
                np.asarray(
                    info["traffic_position"],
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
                np.asarray(
                    action,
                    dtype=np.float64,
                )
            )

            physical_actions.append(
                np.asarray(
                    info[
                        "physical_action"
                    ],
                    dtype=np.float64,
                )
            )

            rewards.append(
                float(
                    reward
                )
            )

    result = {
        "seed": int(
            seed
        ),
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
        "scenario": dict(
            info["scenario"]
        ),
    }

    if record_trajectory:
        result["trajectory"] = {
            "ego_positions": np.asarray(
                ego_positions,
                dtype=np.float64,
            ),
            "traffic_positions": np.asarray(
                traffic_positions,
                dtype=np.float64,
            ),
            "distances": np.asarray(
                distances,
                dtype=np.float64,
            ),
            "dcpas": np.asarray(
                dcpas,
                dtype=np.float64,
            ),
            "tcpas": np.asarray(
                tcpas,
                dtype=np.float64,
            ),
            "actions": np.asarray(
                actions,
                dtype=np.float64,
            ),
            "physical_actions": np.asarray(
                physical_actions,
                dtype=np.float64,
            ),
            "rewards": np.asarray(
                rewards,
                dtype=np.float64,
            ),
        }

    env.close()

    return result


def run_zero_action_episode(
    seed: int,
) -> dict:
    env = make_randomized_env()

    observation, info = env.reset(
        seed=seed
    )

    zero_action = np.zeros(
        2,
        dtype=np.float32,
    )

    total_reward = 0.0
    min_distance = float(
        "inf"
    )

    terminated = False
    truncated = False

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
        "scenario": dict(
            info["scenario"]
        ),
    }

    env.close()

    return result


def evaluate_policy(
    agent: PPOContinuousAgent,
    seeds: list[int],
) -> tuple[dict, list[dict]]:
    results = []

    for seed in seeds:
        result = run_policy_episode(
            agent=agent,
            seed=seed,
            record_trajectory=False,
        )

        results.append(
            result
        )

    summary = summarize_results(
        results
    )

    # Penalizamos explícitamente colisiones para elegir
    # el mejor checkpoint de generalización.
    summary["score"] = float(
        summary["reward_mean"]
        - 50.0
        * summary["collision_rate"]
    )

    return summary, results


def evaluate_zero_action_baseline(
    seeds: list[int],
) -> tuple[dict, list[dict]]:
    results = []

    for seed in seeds:
        results.append(
            run_zero_action_episode(
                seed
            )
        )

    return (
        summarize_results(
            results
        ),
        results,
    )


# ============================================================
# Plotting
# ============================================================


def plot_training_curves(
    history: dict,
) -> None:
    steps = np.asarray(
        history["global_step"],
        dtype=np.float64,
    )

    reward_mean = np.asarray(
        history["eval_reward_mean"],
        dtype=np.float64,
    )

    reward_std = np.asarray(
        history["eval_reward_std"],
        dtype=np.float64,
    )

    collision_rate = np.asarray(
        history["eval_collision_rate"],
        dtype=np.float64,
    )

    min_distance_mean = np.asarray(
        history["eval_min_distance_mean"],
        dtype=np.float64,
    )

    min_distance_min = np.asarray(
        history["eval_min_distance_min"],
        dtype=np.float64,
    )

    score = np.asarray(
        history["eval_score"],
        dtype=np.float64,
    )

    entropy = np.asarray(
        history["entropy"],
        dtype=np.float64,
    )

    critic_loss = np.asarray(
        history["critic_loss"],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8),
    )

    axes = axes.ravel()

    axes[0].plot(
        steps,
        reward_mean,
        marker="o",
        label="PPO randomized eval",
    )

    axes[0].fill_between(
        steps,
        reward_mean - reward_std,
        reward_mean + reward_std,
        alpha=0.2,
        label="±1 std",
    )

    axes[0].axhline(
        BASELINES["fixed_evasive_reward"],
        linestyle="--",
        linewidth=1.0,
        label="Fixed evasive nominal",
    )

    axes[0].axhline(
        BASELINES["random_mean_reward"],
        linestyle=":",
        linewidth=1.0,
        label="Random nominal mean",
    )

    axes[0].set_title(
        "Reward medio en semillas no vistas"
    )

    axes[0].set_xlabel(
        "Global steps"
    )

    axes[0].set_ylabel(
        "Reward"
    )

    axes[0].grid(
        True,
        alpha=0.3,
    )

    axes[0].legend()

    axes[1].plot(
        steps,
        collision_rate,
        marker="o",
    )

    axes[1].set_title(
        "Tasa de colisión en evaluación"
    )

    axes[1].set_xlabel(
        "Global steps"
    )

    axes[1].set_ylabel(
        "Collision rate"
    )

    axes[1].set_ylim(
        -0.05,
        1.05,
    )

    axes[1].grid(
        True,
        alpha=0.3,
    )

    axes[2].plot(
        steps,
        min_distance_mean,
        marker="o",
        label="Mean",
    )

    axes[2].plot(
        steps,
        min_distance_min,
        marker="x",
        label="Minimum",
    )

    axes[2].axhline(
        300.0,
        linestyle="--",
        linewidth=1.0,
        label="DCPA safe scale",
    )

    axes[2].set_title(
        "Distancia mínima al tráfico"
    )

    axes[2].set_xlabel(
        "Global steps"
    )

    axes[2].set_ylabel(
        "Distance [m]"
    )

    axes[2].grid(
        True,
        alpha=0.3,
    )

    axes[2].legend()

    axes[3].plot(
        steps,
        score,
        marker="o",
    )

    axes[3].set_title(
        "Score de selección de checkpoint"
    )

    axes[3].set_xlabel(
        "Global steps"
    )

    axes[3].set_ylabel(
        "Reward mean - 50·collision_rate"
    )

    axes[3].grid(
        True,
        alpha=0.3,
    )

    axes[4].plot(
        steps,
        critic_loss,
        marker="o",
    )

    axes[4].set_title(
        "Critic loss"
    )

    axes[4].set_xlabel(
        "Global steps"
    )

    axes[4].set_ylabel(
        "Loss"
    )

    axes[4].grid(
        True,
        alpha=0.3,
    )

    axes[5].plot(
        steps,
        entropy,
        marker="o",
    )

    axes[5].set_title(
        "Entropía de la política"
    )

    axes[5].set_xlabel(
        "Global steps"
    )

    axes[5].set_ylabel(
        "Entropy"
    )

    axes[5].grid(
        True,
        alpha=0.3,
    )

    fig.suptitle(
        "PPO con randomización de escenarios de cruce",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout()

    save_figure(
        fig,
        "randomized_training_curves",
    )

    plt.close(
        fig
    )


def plot_policy_comparison(
    zero_summary: dict,
    ppo_summary: dict,
) -> None:
    labels = [
        "Sin acción\nrandomized",
        "PPO\nrandomized",
    ]

    rewards = [
        zero_summary["reward_mean"],
        ppo_summary["reward_mean"],
    ]

    collisions = [
        zero_summary["collision_rate"],
        ppo_summary["collision_rate"],
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4),
    )

    axes[0].bar(
        labels,
        rewards,
    )

    axes[0].axhline(
        0.0,
        linewidth=1.0,
    )

    axes[0].set_title(
        "Reward medio"
    )

    axes[0].set_ylabel(
        "Reward"
    )

    axes[0].grid(
        True,
        axis="y",
        alpha=0.3,
    )

    for idx, value in enumerate(
        rewards
    ):
        axes[0].text(
            idx,
            value,
            f"{value:.1f}",
            ha="center",
            va=(
                "bottom"
                if value >= 0.0
                else "top"
            ),
        )

    axes[1].bar(
        labels,
        collisions,
    )

    axes[1].set_ylim(
        0.0,
        1.05,
    )

    axes[1].set_title(
        "Tasa de colisión"
    )

    axes[1].set_ylabel(
        "Collision rate"
    )

    axes[1].grid(
        True,
        axis="y",
        alpha=0.3,
    )

    for idx, value in enumerate(
        collisions
    ):
        axes[1].text(
            idx,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    fig.suptitle(
        "Evaluación en escenarios randomizados no vistos",
        fontsize=13,
        fontweight="bold",
    )

    fig.tight_layout()

    save_figure(
        fig,
        "randomized_policy_comparison",
    )

    plt.close(
        fig
    )


def plot_trajectory_examples(
    trajectory_results: list[dict],
) -> None:
    n = len(
        trajectory_results
    )

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(
            4 * n,
            4,
        ),
        sharex=False,
        sharey=False,
    )

    if n == 1:
        axes = [
            axes
        ]

    for ax, result in zip(
        axes,
        trajectory_results,
    ):
        trajectory = result[
            "trajectory"
        ]

        ego_positions = np.asarray(
            trajectory[
                "ego_positions"
            ],
            dtype=np.float64,
        )

        traffic_positions = np.asarray(
            trajectory[
                "traffic_positions"
            ],
            dtype=np.float64,
        )

        distances = np.asarray(
            trajectory[
                "distances"
            ],
            dtype=np.float64,
        )

        min_idx = int(
            np.argmin(
                distances
            )
        )

        ax.plot(
            ego_positions[:, 0],
            ego_positions[:, 1],
            linewidth=2.0,
            label="Ego",
        )

        ax.plot(
            traffic_positions[:, 0],
            traffic_positions[:, 1],
            linestyle="--",
            linewidth=1.5,
            label="Traffic",
        )

        ax.scatter(
            ego_positions[0, 0],
            ego_positions[0, 1],
            marker="o",
            s=50,
        )

        ax.scatter(
            traffic_positions[0, 0],
            traffic_positions[0, 1],
            marker="o",
            s=50,
        )

        ax.scatter(
            ego_positions[min_idx, 0],
            ego_positions[min_idx, 1],
            marker="x",
            s=70,
        )

        ax.scatter(
            traffic_positions[min_idx, 0],
            traffic_positions[min_idx, 1],
            marker="x",
            s=70,
        )

        ax.set_title(
            f"seed={result['seed']}\n"
            f"R={result['reward']:.1f}, "
            f"col={result['collision']}"
        )

        ax.set_xlabel(
            "X [m]"
        )

        ax.set_ylabel(
            "Y [m]"
        )

        ax.axis(
            "equal"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

    axes[0].legend(
        loc="best"
    )

    fig.suptitle(
        "Trayectorias PPO en escenarios randomizados no vistos",
        fontsize=13,
        fontweight="bold",
    )

    fig.tight_layout()

    save_figure(
        fig,
        "randomized_trajectory_examples",
    )

    plt.close(
        fig
    )


# ============================================================
# Training
# ============================================================


def train() -> None:
    set_seed(
        SEED
    )

    print("=" * 72)
    print("PPO RANDOMIZED CROSSING TRAINING")
    print("=" * 72)
    print(f"Total timesteps : {TOTAL_TIMESTEPS}")
    print(f"Rollout steps   : {ROLLOUT_STEPS}")
    print(f"Updates         : {NUM_UPDATES}")
    print(f"Eval interval   : {EVAL_INTERVAL}")
    print(f"Eval seeds      : {EVAL_SEEDS}")
    print(f"Results dir     : {RESULTS_DIR}")

    env = make_randomized_env()

    agent = PPOContinuousAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        num_steps=ROLLOUT_STEPS,
        num_envs=NUM_ENVS,
    )

    print(
        f"Device          : {agent.device}"
    )

    if INITIAL_NOMINAL_CHECKPOINT.exists():
        print(
            "Loading nominal checkpoint:",
            INITIAL_NOMINAL_CHECKPOINT,
        )

        agent.load(
            str(
                INITIAL_NOMINAL_CHECKPOINT
            )
        )
    else:
        print(
            "Nominal checkpoint not found. "
            "Training from scratch."
        )

    observation, info = env.reset(
        seed=TRAIN_SEED_BASE
    )

    global_step = 0
    episode_count = 0

    episode_reward = 0.0
    episode_length = 0

    episode_rewards = []
    episode_lengths = []
    episode_collisions = []

    best_eval_score = -float(
        "inf"
    )

    history = {
        "update": [],
        "global_step": [],

        "loss": [],
        "actor_loss": [],
        "critic_loss": [],
        "entropy": [],

        "rollout_episodes": [],
        "rollout_collision_rate": [],

        "eval_reward_mean": [],
        "eval_reward_std": [],
        "eval_collision_rate": [],
        "eval_goal_rate": [],
        "eval_min_distance_mean": [],
        "eval_min_distance_min": [],
        "eval_final_goal_distance_mean": [],
        "eval_score": [],
    }

    for update_index in range(
        1,
        NUM_UPDATES + 1,
    ):
        rollout_collisions = 0
        rollout_episodes = 0

        for _ in range(
            ROLLOUT_STEPS
        ):
            (
                action,
                pre_tanh_action,
                log_prob,
                value,
            ) = agent.sample_action(
                observation
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

            episode_end = bool(
                terminated
                or truncated
            )

            agent.store_transition(
                states=observation[
                    None,
                    :
                ],
                actions=action[
                    None,
                    :
                ],
                pre_tanh_actions=pre_tanh_action[
                    None,
                    :
                ],
                log_probs=np.array(
                    [
                        log_prob
                    ],
                    dtype=np.float32,
                ),
                rewards=np.array(
                    [
                        reward
                    ],
                    dtype=np.float32,
                ),
                terminated=np.array(
                    [
                        terminated
                    ],
                    dtype=np.float32,
                ),
                episode_ends=np.array(
                    [
                        episode_end
                    ],
                    dtype=np.float32,
                ),
                next_states=next_observation[
                    None,
                    :
                ],
            )

            global_step += 1

            episode_reward += float(
                reward
            )

            episode_length += 1

            observation = (
                next_observation
            )

            if episode_end:
                episode_count += 1
                rollout_episodes += 1

                collision = bool(
                    info[
                        "collision"
                    ]
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
                    bool(
                        collision
                    )
                )

                print(
                    f"Episode {episode_count:04d} | "
                    f"steps={episode_length:03d} | "
                    f"R={episode_reward:+9.3f} | "
                    f"collision={collision} | "
                    f"goal={info['goal_reached']} | "
                    f"scenario_seed="
                    f"{info['scenario'].get('seed', 'N/A')}"
                )

                observation, info = env.reset(
                    seed=(
                        TRAIN_SEED_BASE
                        + episode_count
                    )
                )

                episode_reward = 0.0
                episode_length = 0

        metrics = agent.update()

        if metrics is None:
            raise RuntimeError(
                "PPO update no fue ejecutado."
            )

        if rollout_episodes > 0:
            rollout_collision_rate = (
                rollout_collisions
                / rollout_episodes
            )
        else:
            rollout_collision_rate = float(
                "nan"
            )

        should_evaluate = (
            update_index == 1
            or update_index % EVAL_INTERVAL == 0
            or update_index == NUM_UPDATES
        )

        if should_evaluate:
            eval_summary, eval_results = evaluate_policy(
                agent=agent,
                seeds=EVAL_SEEDS,
            )

            if (
                eval_summary["score"]
                > best_eval_score
            ):
                best_eval_score = float(
                    eval_summary[
                        "score"
                    ]
                )

                agent.save(
                    str(
                        BEST_MODEL_PATH
                    ),
                    extra={
                        "global_step": global_step,
                        "update": update_index,
                        "best_eval_score": best_eval_score,
                        "eval_summary": eval_summary,
                    },
                )
        else:
            eval_summary = {
                "reward_mean": float(
                    "nan"
                ),
                "reward_std": float(
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
                "min_distance_min": float(
                    "nan"
                ),
                "final_goal_distance_mean": float(
                    "nan"
                ),
                "score": float(
                    "nan"
                ),
            }

        history["update"].append(
            update_index
        )

        history["global_step"].append(
            global_step
        )

        history["loss"].append(
            float(
                metrics[
                    "loss"
                ]
            )
        )

        history["actor_loss"].append(
            float(
                metrics[
                    "actor_loss"
                ]
            )
        )

        history["critic_loss"].append(
            float(
                metrics[
                    "critic_loss"
                ]
            )
        )

        history["entropy"].append(
            float(
                metrics[
                    "entropy"
                ]
            )
        )

        history["rollout_episodes"].append(
            int(
                rollout_episodes
            )
        )

        history["rollout_collision_rate"].append(
            float(
                rollout_collision_rate
            )
        )

        history["eval_reward_mean"].append(
            float(
                eval_summary[
                    "reward_mean"
                ]
            )
        )

        history["eval_reward_std"].append(
            float(
                eval_summary[
                    "reward_std"
                ]
            )
        )

        history["eval_collision_rate"].append(
            float(
                eval_summary[
                    "collision_rate"
                ]
            )
        )

        history["eval_goal_rate"].append(
            float(
                eval_summary[
                    "goal_rate"
                ]
            )
        )

        history["eval_min_distance_mean"].append(
            float(
                eval_summary[
                    "min_distance_mean"
                ]
            )
        )

        history["eval_min_distance_min"].append(
            float(
                eval_summary[
                    "min_distance_min"
                ]
            )
        )

        history[
            "eval_final_goal_distance_mean"
        ].append(
            float(
                eval_summary[
                    "final_goal_distance_mean"
                ]
            )
        )

        history["eval_score"].append(
            float(
                eval_summary[
                    "score"
                ]
            )
        )

        print()
        print(
            f"UPDATE {update_index:03d}/{NUM_UPDATES}"
        )
        print(
            f"  global step              = {global_step}"
        )
        print(
            f"  loss                     = {metrics['loss']:+.5f}"
        )
        print(
            f"  actor loss               = {metrics['actor_loss']:+.5f}"
        )
        print(
            f"  critic loss              = {metrics['critic_loss']:+.5f}"
        )
        print(
            f"  entropy                  = {metrics['entropy']:+.5f}"
        )
        print(
            f"  rollout episodes         = {rollout_episodes}"
        )
        print(
            f"  rollout collision rate   = {rollout_collision_rate}"
        )

        if should_evaluate:
            print(
                f"  eval reward mean         = "
                f"{eval_summary['reward_mean']:+.3f}"
            )
            print(
                f"  eval reward std          = "
                f"{eval_summary['reward_std']:.3f}"
            )
            print(
                f"  eval collision rate      = "
                f"{eval_summary['collision_rate']:.3f}"
            )
            print(
                f"  eval min distance mean   = "
                f"{eval_summary['min_distance_mean']:.2f}"
            )
            print(
                f"  eval min distance min    = "
                f"{eval_summary['min_distance_min']:.2f}"
            )
            print(
                f"  eval score               = "
                f"{eval_summary['score']:+.3f}"
            )
        else:
            print(
                "  eval                     = skipped"
            )

        print(
            "-" * 72
        )

    env.close()

    agent.save(
        str(
            FINAL_MODEL_PATH
        ),
        extra={
            "global_step": global_step,
            "episode_count": episode_count,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "episode_collisions": episode_collisions,
            "history": history,
        },
    )

    history_path = (
        RESULTS_DIR
        / "training_history.json"
    )

    save_json(
        history_path,
        history,
    )

    # ========================================================
    # Final evaluation using best checkpoint
    # ========================================================

    best_agent = PPOContinuousAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        num_steps=ROLLOUT_STEPS,
        num_envs=NUM_ENVS,
    )

    best_agent.load(
        str(
            BEST_MODEL_PATH
        )
    )

    final_eval_summary, final_eval_results = (
        evaluate_policy(
            agent=best_agent,
            seeds=EVAL_SEEDS,
        )
    )

    zero_summary, zero_results = (
        evaluate_zero_action_baseline(
            seeds=EVAL_SEEDS,
        )
    )

    final_eval_path = (
        RESULTS_DIR
        / "final_evaluation.json"
    )

    save_json(
        final_eval_path,
        {
            "ppo_summary": final_eval_summary,
            "ppo_results": final_eval_results,
            "zero_action_summary": zero_summary,
            "zero_action_results": zero_results,
            "eval_seeds": EVAL_SEEDS,
            "trajectory_seeds": TRAJECTORY_SEEDS,
        },
    )

    trajectory_results = []

    for seed in TRAJECTORY_SEEDS:
        trajectory_results.append(
            run_policy_episode(
                agent=best_agent,
                seed=seed,
                record_trajectory=True,
            )
        )

    trajectory_path = (
        RESULTS_DIR
        / "trajectory_examples.json"
    )

    save_json(
        trajectory_path,
        trajectory_results,
    )

    # ========================================================
    # Figures
    # ========================================================

    # Filtramos NaN para la curva de evaluación.
    plot_training_history = {
        key: []
        for key in history
    }

    for idx in range(
        len(
            history[
                "update"
            ]
        )
    ):
        if not np.isfinite(
            history[
                "eval_reward_mean"
            ][idx]
        ):
            continue

        for key in history:
            plot_training_history[
                key
            ].append(
                history[
                    key
                ][idx]
            )

    plot_training_curves(
        plot_training_history
    )

    plot_policy_comparison(
        zero_summary=zero_summary,
        ppo_summary=final_eval_summary,
    )

    plot_trajectory_examples(
        trajectory_results
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 72)
    print("RANDOMIZED TRAINING COMPLETE")
    print("=" * 72)

    print(
        f"Global steps       : {global_step}"
    )

    print(
        f"Episodes           : {episode_count}"
    )

    if episode_rewards:
        print(
            f"Train mean reward  : "
            f"{np.mean(episode_rewards):+.3f}"
        )

        print(
            f"Train collision rate: "
            f"{np.mean(episode_collisions):.3f}"
        )

    print()
    print("Final PPO evaluation:")
    print(
        f"  reward mean       : "
        f"{final_eval_summary['reward_mean']:+.3f}"
    )
    print(
        f"  reward std        : "
        f"{final_eval_summary['reward_std']:.3f}"
    )
    print(
        f"  collision rate    : "
        f"{final_eval_summary['collision_rate']:.3f}"
    )
    print(
        f"  min distance mean : "
        f"{final_eval_summary['min_distance_mean']:.2f}"
    )
    print(
        f"  min distance min  : "
        f"{final_eval_summary['min_distance_min']:.2f}"
    )
    print(
        f"  score             : "
        f"{final_eval_summary['score']:+.3f}"
    )

    print()
    print("Zero-action baseline:")
    print(
        f"  reward mean       : "
        f"{zero_summary['reward_mean']:+.3f}"
    )
    print(
        f"  collision rate    : "
        f"{zero_summary['collision_rate']:.3f}"
    )

    print()
    print(
        f"Best checkpoint    : {BEST_MODEL_PATH}"
    )

    print(
        f"Final checkpoint   : {FINAL_MODEL_PATH}"
    )

    print(
        f"Results directory  : {RESULTS_DIR}"
    )


if __name__ == "__main__":
    train()
