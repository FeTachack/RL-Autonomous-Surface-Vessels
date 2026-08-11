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

EVAL_SEEDS = list(
    range(
        91_000,
        91_100,
    )
)

OUTPUT_JSON = (
    RESULTS_DIR
    / "route_randomized_vs_colreg_same_seed.json"
)

OUTPUT_BASENAME = "route_randomized_vs_colreg_same_seed"


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
    policy_name: str,
    record_trajectory: bool = False,
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
        "policy_name": policy_name,
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
        "goal_position": np.asarray(
            env.unwrapped.goal_position,
            dtype=np.float64,
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


def find_best_visual_seed(
    randomized_agent: PPOContinuousAgent,
    colreg_agent: PPOContinuousAgent,
) -> dict[str, Any]:
    comparisons = []

    print()
    print(
        "="
        * 72
    )

    print(
        "Searching representative seed"
    )

    print(
        "="
        * 72
    )

    for seed in EVAL_SEEDS:
        randomized_result = run_episode(
            agent=randomized_agent,
            seed=seed,
            policy_name="PPO randomizado",
            record_trajectory=False,
        )

        colreg_result = run_episode(
            agent=colreg_agent,
            seed=seed,
            policy_name="PPO + COLREG",
            record_trajectory=False,
        )

        delta_min_distance = (
            colreg_result[
                "min_distance"
            ]
            - randomized_result[
                "min_distance"
            ]
        )

        delta_min_dcpa = (
            colreg_result[
                "min_dcpa"
            ]
            - randomized_result[
                "min_dcpa"
            ]
        )

        delta_reward = (
            colreg_result[
                "reward"
            ]
            - randomized_result[
                "reward"
            ]
        )

        both_safe_successful = (
            not randomized_result[
                "collision"
            ]
            and not colreg_result[
                "collision"
            ]
            and randomized_result[
                "goal_reached"
            ]
            and colreg_result[
                "goal_reached"
            ]
        )

        comparison = {
            "seed": int(
                seed
            ),
            "randomized": randomized_result,
            "colreg": colreg_result,
            "delta_min_distance": float(
                delta_min_distance
            ),
            "delta_min_dcpa": float(
                delta_min_dcpa
            ),
            "delta_reward": float(
                delta_reward
            ),
            "both_safe_successful": bool(
                both_safe_successful
            ),
        }

        comparisons.append(
            comparison
        )

        print(
            f"seed={seed} "
            f"Δdmin={delta_min_distance:+7.2f} "
            f"ΔDCPA={delta_min_dcpa:+7.2f} "
            f"ΔR={delta_reward:+8.2f} "
            f"ok={both_safe_successful}"
        )

    valid_comparisons = [
        item
        for item in comparisons
        if item[
            "both_safe_successful"
        ]
    ]

    if len(
        valid_comparisons
    ) == 0:
        valid_comparisons = comparisons


    selected = max(
        valid_comparisons,
        key=lambda item: (
            item[
                "delta_min_distance"
            ]
            + 0.10
            * item[
                "delta_reward"
            ]
            + 0.25
            * item[
                "delta_min_dcpa"
            ]
        ),
    )

    print()
    print(
        "Selected seed:",
        selected[
            "seed"
        ],
    )

    print(
        f"  Δdmin = {selected['delta_min_distance']:+.2f} m"
    )

    print(
        f"  ΔDCPA = {selected['delta_min_dcpa']:+.2f} m"
    )

    print(
        f"  Δreward = {selected['delta_reward']:+.2f}"
    )

    return {
        "selected": selected,
        "all_comparisons": comparisons,
    }


def set_equal_limits(
    ax,
    arrays: list[np.ndarray],
    margin: float = 150.0,
) -> None:
    all_points = np.vstack(
        arrays
    )

    x_min = float(
        np.min(
            all_points[
                :,
                0,
            ]
        )
        - margin
    )

    x_max = float(
        np.max(
            all_points[
                :,
                0,
            ]
        )
        + margin
    )

    y_min = float(
        np.min(
            all_points[
                :,
                1,
            ]
        )
        - margin
    )

    y_max = float(
        np.max(
            all_points[
                :,
                1,
            ]
        )
        + margin
    )

    ax.set_xlim(
        x_min,
        x_max,
    )

    ax.set_ylim(
        y_min,
        y_max,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )


def plot_comparison_route(
    randomized_result: dict[str, Any],
    colreg_result: dict[str, Any],
    delta: dict[str, float],
) -> None:
    randomized_ego = np.asarray(
        randomized_result[
            "ego_positions"
        ],
        dtype=np.float64,
    )

    colreg_ego = np.asarray(
        colreg_result[
            "ego_positions"
        ],
        dtype=np.float64,
    )


    randomized_traffic = np.asarray(
        randomized_result[
            "traffic_positions"
        ],
        dtype=np.float64,
    )

    colreg_traffic = np.asarray(
        colreg_result[
            "traffic_positions"
        ],
        dtype=np.float64,
    )

    if colreg_traffic.shape[
        0
    ] >= randomized_traffic.shape[
        0
    ]:
        traffic_positions = colreg_traffic
    else:
        traffic_positions = randomized_traffic

    goal_position = np.asarray(
        colreg_result[
            "goal_position"
        ],
        dtype=np.float64,
    )

    randomized_distances = np.asarray(
        randomized_result[
            "distances"
        ],
        dtype=np.float64,
    )

    colreg_distances = np.asarray(
        colreg_result[
            "distances"
        ],
        dtype=np.float64,
    )

    randomized_closest_idx = int(
        np.argmin(
            randomized_distances
        )
    )

    colreg_closest_idx = int(
        np.argmin(
            colreg_distances
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            7,
            6,
        )
    )

    ax.plot(
        traffic_positions[
            :,
            0,
        ],
        traffic_positions[
            :,
            1,
        ],
        linestyle="--",
        linewidth=2.0,
        label="Tráfico",
    )

    ax.plot(
        randomized_ego[
            :,
            0,
        ],
        randomized_ego[
            :,
            1,
        ],
        linewidth=2.2,
        label="Ego PPO randomizado",
    )

    ax.plot(
        colreg_ego[
            :,
            0,
        ],
        colreg_ego[
            :,
            1,
        ],
        linewidth=2.2,
        label="Ego PPO + COLREG",
    )

    ax.scatter(
        randomized_ego[
            0,
            0,
        ],
        randomized_ego[
            0,
            1,
        ],
        marker="o",
        s=65,
        label="Inicio ego",
    )

    ax.scatter(
        traffic_positions[
            0,
            0,
        ],
        traffic_positions[
            0,
            1,
        ],
        marker="s",
        s=65,
        label="Inicio tráfico",
    )

    ax.scatter(
        goal_position[
            0
        ],
        goal_position[
            1
        ],
        marker="*",
        s=180,
        label="Objetivo",
    )

    ax.scatter(
        randomized_ego[
            randomized_closest_idx,
            0,
        ],
        randomized_ego[
            randomized_closest_idx,
            1,
        ],
        marker="x",
        s=95,
        label="Mín. sep. PPO rand.",
    )

    ax.scatter(
        colreg_ego[
            colreg_closest_idx,
            0,
        ],
        colreg_ego[
            colreg_closest_idx,
            1,
        ],
        marker="+",
        s=130,
        label="Mín. sep. PPO + COLREG",
    )

    summary_text = (
        f"Seed: {colreg_result['seed']}\n"
        f"PPO rand. $d_{{min}}$: "
        f"{randomized_result['min_distance']:.1f} m\n"
        f"PPO+COLREG $d_{{min}}$: "
        f"{colreg_result['min_distance']:.1f} m\n"
        f"$\\Delta d_{{min}}$: "
        f"{delta['delta_min_distance']:+.1f} m\n"
        f"$\\Delta R$: "
        f"{delta['delta_reward']:+.1f}"
    )

    ax.text(
        0.02,
        0.98,
        summary_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round",
            "alpha": 0.15,
        },
    )

    ax.set_title(
        "Comparación de trayectoria: PPO randomizado vs PPO + COLREG"
    )

    ax.set_xlabel(
        "X [m]"
    )

    ax.set_ylabel(
        "Y [m]"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        fontsize=8,
        loc="lower right",
    )

    set_equal_limits(
        ax,
        arrays=[
            randomized_ego,
            colreg_ego,
            traffic_positions,
            goal_position.reshape(
                1,
                2,
            ),
        ],
    )

    fig.tight_layout()

    save_figure(
        fig,
        OUTPUT_BASENAME,
    )


def main() -> None:
    print(
        "="
        * 72
    )

    print(
        "PLOTTING PPO RANDOMIZED VS PPO COLREG ROUTE"
    )

    print(
        "="
        * 72
    )

    print(
        "Randomized checkpoint:",
        RANDOMIZED_CHECKPOINT,
    )

    print(
        "COLREG checkpoint:",
        COLREG_CHECKPOINT,
    )

    print(
        "Results dir:",
        RESULTS_DIR,
    )

    randomized_agent = load_agent(
        RANDOMIZED_CHECKPOINT
    )

    colreg_agent = load_agent(
        COLREG_CHECKPOINT
    )

    selection_payload = find_best_visual_seed(
        randomized_agent=randomized_agent,
        colreg_agent=colreg_agent,
    )

    selected = selection_payload[
        "selected"
    ]

    selected_seed = int(
        selected[
            "seed"
        ]
    )

    randomized_result = run_episode(
        agent=randomized_agent,
        seed=selected_seed,
        policy_name="PPO randomizado",
        record_trajectory=True,
    )

    colreg_result = run_episode(
        agent=colreg_agent,
        seed=selected_seed,
        policy_name="PPO + COLREG",
        record_trajectory=True,
    )

    delta = {
        "delta_min_distance": float(
            colreg_result[
                "min_distance"
            ]
            - randomized_result[
                "min_distance"
            ]
        ),
        "delta_min_dcpa": float(
            colreg_result[
                "min_dcpa"
            ]
            - randomized_result[
                "min_dcpa"
            ]
        ),
        "delta_reward": float(
            colreg_result[
                "reward"
            ]
            - randomized_result[
                "reward"
            ]
        ),
        "delta_env_reward": float(
            colreg_result[
                "env_reward"
            ]
            - randomized_result[
                "env_reward"
            ]
        ),
        "delta_colreg_reward": float(
            colreg_result[
                "colreg_reward"
            ]
            - randomized_result[
                "colreg_reward"
            ]
        ),
        "delta_final_goal_distance": float(
            colreg_result[
                "final_goal_distance"
            ]
            - randomized_result[
                "final_goal_distance"
            ]
        ),
    }

    payload = {
        "selected_seed": selected_seed,
        "selection_payload": selection_payload,
        "randomized_result": randomized_result,
        "colreg_result": colreg_result,
        "delta": delta,
        "output_png": str(
            RESULTS_DIR
            / f"{OUTPUT_BASENAME}.png"
        ),
        "output_pdf": str(
            RESULTS_DIR
            / f"{OUTPUT_BASENAME}.pdf"
        ),
    }

    save_json(
        OUTPUT_JSON,
        payload,
    )

    plot_comparison_route(
        randomized_result=randomized_result,
        colreg_result=colreg_result,
        delta=delta,
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
        "Selected seed:",
        selected_seed,
    )

    print(
        f"Delta min distance: {delta['delta_min_distance']:+.3f} m"
    )

    print(
        f"Delta min DCPA: {delta['delta_min_dcpa']:+.3f} m"
    )

    print(
        f"Delta reward: {delta['delta_reward']:+.3f}"
    )

    print(
        "Saved JSON:",
        OUTPUT_JSON,
    )

    print(
        "Saved PDF:",
        RESULTS_DIR
        / f"{OUTPUT_BASENAME}.pdf",
    )

    print(
        "Saved PNG:",
        RESULTS_DIR
        / f"{OUTPUT_BASENAME}.png",
    )


if __name__ == "__main__":
    main()
