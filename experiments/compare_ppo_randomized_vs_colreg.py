from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.agents.ppo_continuous import PPOContinuousAgent
from experiments.envs.commonocean_env import CommonOceanEnv
from experiments.preferences.colreg_reward_wrapper import (
    ColregRewardConfig,
    ColregRewardWrapper,
)


STATE_SIZE = 13
ACTION_SIZE = 2
ROLLOUT_STEPS = 256
NUM_ENVS = 1
MAX_EPISODE_STEPS = 220

COLREG_WEIGHT = 0.50

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
    / "ppo_colreg_comparison"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOMIZED_CHECKPOINT = (
    CHECKPOINT_DIR
    / "ppo_randomized_crossing_best.pt"
)

COLREG_CHECKPOINT = (
    CHECKPOINT_DIR
    / "ppo_colreg_finetuned_best.pt"
)

OUTPUT_JSON = (
    RESULTS_DIR
    / "randomized_vs_colreg_comparison.json"
)

EVAL_SEEDS = list(
    range(
        91_000,
        91_100,
    )
)

POLICIES = {
    "ppo_randomized": RANDOMIZED_CHECKPOINT,
    "ppo_colreg_best": COLREG_CHECKPOINT,
}


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


def make_env() -> ColregRewardWrapper:
    base_env = CommonOceanEnv(
        max_episode_steps=MAX_EPISODE_STEPS,
        render_mode=None,
        randomize_scenario=True,
    )

    return ColregRewardWrapper(
        base_env,
        config=ColregRewardConfig(
            colreg_weight=COLREG_WEIGHT,
        ),
    )


def load_agent(
    checkpoint_path: Path,
) -> PPOContinuousAgent:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No existe el checkpoint: {checkpoint_path}"
        )

    agent = PPOContinuousAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        num_steps=ROLLOUT_STEPS,
        num_envs=NUM_ENVS,
    )

    agent.load(
        str(
            checkpoint_path
        )
    )

    return agent


def run_episode(
    agent: PPOContinuousAgent,
    seed: int,
) -> dict[str, Any]:
    env = make_env()

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

    env.close()

    return result


def evaluate_policy(
    policy_name: str,
    checkpoint_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print()
    print(
        "="
        * 72
    )

    print(
        f"Evaluating {policy_name}"
    )

    print(
        "="
        * 72
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    agent = load_agent(
        checkpoint_path
    )

    results = []

    for index, seed in enumerate(
        EVAL_SEEDS,
        start=1,
    ):
        result = run_episode(
            agent=agent,
            seed=seed,
        )

        results.append(
            result
        )

        print(
            f"[{index:03d}/{len(EVAL_SEEDS):03d}] "
            f"seed={seed} "
            f"R={result['reward']:+8.2f} "
            f"R_env={result['env_reward']:+8.2f} "
            f"R_colreg={result['colreg_reward']:+8.2f} "
            f"col={result['collision']} "
            f"goal={result['goal_reached']} "
            f"dmin={result['min_distance']:7.2f} "
            f"DCPA={result['min_dcpa']:7.2f}"
        )

    summary = summarize_results(
        results
    )

    return (
        summary,
        results,
    )


def summarize_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
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
    summary: dict[str, Any],
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


def compute_deltas(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float]:
    keys = [
        "reward_mean",
        "env_reward_mean",
        "colreg_reward_mean",
        "collision_rate",
        "goal_rate",
        "min_distance_mean",
        "min_distance_p05",
        "min_distance_p10",
        "min_distance_min",
        "min_dcpa_mean",
        "min_dcpa_min",
        "final_goal_distance_mean",
        "episode_length_mean",
        "selection_score",
    ]

    return {
        key: float(
            candidate[
                key
            ]
            - reference[
                key
            ]
        )
        for key in keys
    }


def print_summary(
    policy_name: str,
    summary: dict[str, Any],
) -> None:
    print()
    print(
        "-" * 72
    )

    print(
        policy_name
    )

    print(
        "-" * 72
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
        "min_distance_p05",
        "min_distance_p10",
        "min_distance_min",
        "min_dcpa_mean",
        "min_dcpa_min",
        "final_goal_distance_mean",
        "episode_length_mean",
        "selection_score",
    ]

    for key in keys:
        print(
            f"{key:<28s}: {summary[key]}"
        )


def print_delta(
    delta: dict[str, float],
) -> None:
    print()
    print(
        "="
        * 72
    )

    print(
        "DELTA: ppo_colreg_best - ppo_randomized"
    )

    print(
        "="
        * 72
    )

    for key, value in delta.items():
        print(
            f"{key:<28s}: {value:+.6f}"
        )


def plot_metric_bars(
    summaries: dict[str, dict[str, Any]],
) -> None:
    labels = [
        "PPO randomized",
        "PPO + COLREG",
    ]

    policy_keys = [
        "ppo_randomized",
        "ppo_colreg_best",
    ]

    metrics = [
        (
            "reward_mean",
            "Mean total reward",
            "comparison_reward_mean",
        ),
        (
            "env_reward_mean",
            "Mean environment reward",
            "comparison_env_reward_mean",
        ),
        (
            "colreg_reward_mean",
            "Mean COLREG auxiliary reward",
            "comparison_colreg_reward_mean",
        ),
        (
            "min_distance_mean",
            "Mean minimum separation [m]",
            "comparison_min_distance_mean",
        ),
        (
            "min_distance_p10",
            "P10 minimum separation [m]",
            "comparison_min_distance_p10",
        ),
        (
            "min_distance_min",
            "Worst-case minimum separation [m]",
            "comparison_min_distance_min",
        ),
        (
            "min_dcpa_mean",
            "Mean minimum DCPA [m]",
            "comparison_min_dcpa_mean",
        ),
        (
            "final_goal_distance_mean",
            "Mean final distance to goal [m]",
            "comparison_final_goal_distance",
        ),
    ]

    for metric_key, title, filename_base in metrics:
        values = [
            summaries[
                policy_key
            ][
                metric_key
            ]
            for policy_key in policy_keys
        ]

        fig, ax = plt.subplots(
            figsize=(
                6,
                4,
            )
        )

        ax.bar(
            labels,
            values,
        )

        ax.set_title(
            title
        )

        ax.set_ylabel(
            metric_key
        )

        ax.grid(
            True,
            axis="y",
            alpha=0.3,
        )

        for index, value in enumerate(
            values
        ):
            ax.text(
                index,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom"
                if value >= 0.0
                else "top",
            )

        fig.tight_layout()

        save_figure(
            fig,
            filename_base,
        )


    rate_metrics = [
        "collision_rate",
        "goal_rate",
    ]

    x = np.arange(
        len(
            rate_metrics
        )
    )

    width = 0.35

    randomized_values = [
        summaries[
            "ppo_randomized"
        ][
            key
        ]
        for key in rate_metrics
    ]

    colreg_values = [
        summaries[
            "ppo_colreg_best"
        ][
            key
        ]
        for key in rate_metrics
    ]

    fig, ax = plt.subplots(
        figsize=(
            6,
            4,
        )
    )

    ax.bar(
        x
        - width
        / 2,
        randomized_values,
        width,
        label="PPO randomized",
    )

    ax.bar(
        x
        + width
        / 2,
        colreg_values,
        width,
        label="PPO + COLREG",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            "Collision",
            "Goal",
        ]
    )

    ax.set_ylim(
        -0.05,
        1.05,
    )

    ax.set_ylabel(
        "Rate"
    )

    ax.set_title(
        "Collision and goal rates"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    save_figure(
        fig,
        "comparison_rates",
    )


def plot_histograms(
    results_by_policy: dict[str, list[dict[str, Any]]],
) -> None:
    label_map = {
        "ppo_randomized": "PPO randomized",
        "ppo_colreg_best": "PPO + COLREG",
    }

    histogram_specs = [
        (
            "min_distance",
            "Minimum separation [m]",
            "comparison_hist_min_distance",
        ),
        (
            "min_dcpa",
            "Minimum DCPA [m]",
            "comparison_hist_min_dcpa",
        ),
        (
            "env_reward",
            "Environment reward",
            "comparison_hist_env_reward",
        ),
        (
            "colreg_reward",
            "COLREG auxiliary reward",
            "comparison_hist_colreg_reward",
        ),
    ]

    for metric_key, xlabel, filename_base in histogram_specs:
        fig, ax = plt.subplots(
            figsize=(
                7,
                4,
            )
        )

        for policy_key, results in results_by_policy.items():
            values = np.asarray(
                [
                    result[
                        metric_key
                    ]
                    for result in results
                ],
                dtype=np.float64,
            )

            ax.hist(
                values,
                bins=16,
                alpha=0.45,
                label=label_map[
                    policy_key
                ],
            )

        ax.set_xlabel(
            xlabel
        )

        ax.set_ylabel(
            "Count"
        )

        ax.set_title(
            xlabel
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        fig.tight_layout()

        save_figure(
            fig,
            filename_base,
        )


def main() -> None:
    print(
        "="
        * 72
    )

    print(
        "PPO RANDOMIZED VS PPO COLREG COMPARISON"
    )

    print(
        "="
        * 72
    )

    print(
        "Eval seeds:",
        EVAL_SEEDS[
            0
        ],
        "to",
        EVAL_SEEDS[
            -1
        ],
    )

    print(
        "Results dir:",
        RESULTS_DIR,
    )

    summaries = {}
    results_by_policy = {}

    for policy_name, checkpoint_path in POLICIES.items():
        summary, results = evaluate_policy(
            policy_name=policy_name,
            checkpoint_path=checkpoint_path,
        )

        summaries[
            policy_name
        ] = summary

        results_by_policy[
            policy_name
        ] = results

        print_summary(
            policy_name,
            summary,
        )

    delta = compute_deltas(
        reference=summaries[
            "ppo_randomized"
        ],
        candidate=summaries[
            "ppo_colreg_best"
        ],
    )

    print_delta(
        delta
    )

    payload = {
        "config": {
            "state_size": STATE_SIZE,
            "action_size": ACTION_SIZE,
            "rollout_steps": ROLLOUT_STEPS,
            "num_envs": NUM_ENVS,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "colreg_weight": COLREG_WEIGHT,
            "eval_seeds": EVAL_SEEDS,
            "randomized_checkpoint": str(
                RANDOMIZED_CHECKPOINT
            ),
            "colreg_checkpoint": str(
                COLREG_CHECKPOINT
            ),
        },
        "summaries": summaries,
        "delta_colreg_minus_randomized": delta,
        "results_by_policy": results_by_policy,
    }

    save_json(
        OUTPUT_JSON,
        payload,
    )

    plot_metric_bars(
        summaries
    )

    plot_histograms(
        results_by_policy
    )

    print()
    print(
        "="
        * 72
    )

    print(
        "DONE"
    )

    print(
        "="
        * 72
    )

    print(
        "Saved comparison JSON:",
        OUTPUT_JSON,
    )

    print(
        "Saved figures in:",
        RESULTS_DIR,
    )


if __name__ == "__main__":
    main()
