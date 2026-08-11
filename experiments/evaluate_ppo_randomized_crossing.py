from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

from experiments.agents.ppo_continuous import (
    PPOContinuousAgent,
)

from experiments.envs.commonocean_env import (
    CommonOceanEnv,
)


STATE_SIZE = 13
ACTION_SIZE = 2

MAX_EPISODE_STEPS = 220
ROLLOUT_STEPS = 256
NUM_ENVS = 1

EVAL_SEEDS = list(
    range(
        70_000,
        70_100,
    )
)

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "checkpoints"
    / "ppo_randomized_crossing_best.pt"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "ppo_randomized_crossing_eval"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
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
            key: to_serializable(value)
            for key, value in obj.items()
        }

    if isinstance(
        obj,
        list,
    ):
        return [
            to_serializable(value)
            for value in obj
        ]

    return obj


def save_json(
    path: Path,
    data,
):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            to_serializable(data),
            file,
            indent=2,
        )

    print(
        f"Saved JSON: {path}"
    )


def save_figure(
    fig,
    filename_base: str,
):
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


def make_env():
    return CommonOceanEnv(
        max_episode_steps=MAX_EPISODE_STEPS,
        render_mode=None,
        randomize_scenario=True,
    )


def run_episode(
    agent: PPOContinuousAgent,
    seed: int,
    record_trajectory: bool = False,
):
    env = make_env()

    observation, info = env.reset(
        seed=seed
    )

    total_reward = 0.0
    min_distance = float("inf")
    min_dcpa = float("inf")

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
        "seed": int(seed),
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
        "min_dcpa": float(
            min_dcpa
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


def summarize_results(
    results: list[dict],
):
    rewards = np.asarray(
        [
            r["reward"]
            for r in results
        ],
        dtype=np.float64,
    )

    collisions = np.asarray(
        [
            r["collision"]
            for r in results
        ],
        dtype=np.float64,
    )

    goals = np.asarray(
        [
            r["goal_reached"]
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

    min_dcpas = np.asarray(
        [
            r["min_dcpa"]
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

    steps = np.asarray(
        [
            r["steps"]
            for r in results
        ],
        dtype=np.float64,
    )

    return {
        "episodes": len(results),

        "reward_mean": float(
            np.mean(rewards)
        ),
        "reward_std": float(
            np.std(rewards)
        ),
        "reward_min": float(
            np.min(rewards)
        ),
        "reward_max": float(
            np.max(rewards)
        ),

        "collision_rate": float(
            np.mean(collisions)
        ),
        "goal_rate": float(
            np.mean(goals)
        ),

        "min_distance_mean": float(
            np.mean(min_distances)
        ),
        "min_distance_std": float(
            np.std(min_distances)
        ),
        "min_distance_min": float(
            np.min(min_distances)
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
            np.mean(min_dcpas)
        ),
        "min_dcpa_min": float(
            np.min(min_dcpas)
        ),

        "final_goal_distance_mean": float(
            np.mean(
                final_goal_distances
            )
        ),

        "episode_length_mean": float(
            np.mean(steps)
        ),
    }


def plot_histograms(
    results: list[dict],
):
    rewards = np.asarray(
        [
            r["reward"]
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

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4),
    )

    axes[0].hist(
        rewards,
        bins=15,
    )

    axes[0].set_title(
        "Distribución de reward"
    )

    axes[0].set_xlabel(
        "Reward"
    )

    axes[0].set_ylabel(
        "Frecuencia"
    )

    axes[0].grid(
        True,
        alpha=0.3,
    )

    axes[1].hist(
        min_distances,
        bins=15,
    )

    axes[1].axvline(
        300.0,
        linestyle="--",
        linewidth=1.0,
        label="300 m"
    )

    axes[1].set_title(
        "Distribución de distancia mínima"
    )

    axes[1].set_xlabel(
        "Distancia mínima [m]"
    )

    axes[1].set_ylabel(
        "Frecuencia"
    )

    axes[1].grid(
        True,
        alpha=0.3,
    )

    axes[1].legend()

    fig.suptitle(
        "Evaluación robusta PPO en 100 escenarios randomizados",
        fontsize=13,
        fontweight="bold",
    )

    fig.tight_layout()

    save_figure(
        fig,
        "robust_eval_histograms",
    )

    plt.close(fig)


def plot_worst_trajectories(
    worst_results: list[dict],
):
    n = len(
        worst_results
    )

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(
            4 * n,
            4,
        ),
    )

    if n == 1:
        axes = [
            axes
        ]

    for ax, result in zip(
        axes,
        worst_results,
    ):
        trajectory = result[
            "trajectory"
        ]

        ego_positions = np.asarray(
            trajectory["ego_positions"],
            dtype=np.float64,
        )

        traffic_positions = np.asarray(
            trajectory["traffic_positions"],
            dtype=np.float64,
        )

        distances = np.asarray(
            trajectory["distances"],
            dtype=np.float64,
        )

        min_idx = int(
            np.argmin(distances)
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
            f"d_min={result['min_distance']:.1f} m, "
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
        "Peores trayectorias según distancia mínima",
        fontsize=13,
        fontweight="bold",
    )

    fig.tight_layout()

    save_figure(
        fig,
        "robust_eval_worst_trajectories",
    )

    plt.close(fig)


def main():
    print("=" * 72)
    print("ROBUST PPO RANDOMIZED CROSSING EVALUATION")
    print("=" * 72)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"No existe checkpoint: {CHECKPOINT_PATH}"
        )

    agent = PPOContinuousAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        num_steps=ROLLOUT_STEPS,
        num_envs=NUM_ENVS,
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
        "Checkpoint metadata:",
        {
            "global_step": checkpoint.get(
                "global_step"
            ),
            "update": checkpoint.get(
                "update"
            ),
            "best_eval_score": checkpoint.get(
                "best_eval_score"
            ),
        },
    )

    results = []

    for idx, seed in enumerate(
        EVAL_SEEDS,
        start=1,
    ):
        result = run_episode(
            agent=agent,
            seed=seed,
            record_trajectory=False,
        )

        results.append(
            result
        )

        print(
            f"{idx:03d}/{len(EVAL_SEEDS)} "
            f"seed={seed} "
            f"R={result['reward']:+8.2f} "
            f"col={result['collision']} "
            f"dmin={result['min_distance']:7.2f} "
            f"dcpa_min={result['min_dcpa']:7.2f} "
            f"goal_dist={result['final_goal_distance']:7.2f}"
        )

    summary = summarize_results(
        results
    )


    worst_base = sorted(
        results,
        key=lambda item: item[
            "min_distance"
        ],
    )[:5]

    worst_results = []

    for item in worst_base:
        worst_results.append(
            run_episode(
                agent=agent,
                seed=item["seed"],
                record_trajectory=True,
            )
        )

    save_json(
        RESULTS_DIR
        / "robust_evaluation.json",
        {
            "summary": summary,
            "results": results,
            "worst_seeds": [
                item["seed"]
                for item in worst_results
            ],
        },
    )

    save_json(
        RESULTS_DIR
        / "worst_trajectory_examples.json",
        worst_results,
    )

    plot_histograms(
        results
    )

    plot_worst_trajectories(
        worst_results
    )

    print()
    print("=" * 72)
    print("ROBUST EVALUATION SUMMARY")
    print("=" * 72)

    for key, value in summary.items():
        print(
            f"{key:28s}: {value}"
        )

    print()
    print(
        "Results directory:",
        RESULTS_DIR,
    )


if __name__ == "__main__":
    main()
