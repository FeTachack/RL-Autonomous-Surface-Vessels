from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.agents.ppo_continuous import PPOContinuousAgent
from experiments.envs.commonocean_env import CommonOceanEnv


STATE_SIZE = 13
ACTION_SIZE = 2
ROLLOUT_STEPS = 256
NUM_ENVS = 1
MAX_EPISODE_STEPS = 220

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
    / "report_routes"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EVAL_SEEDS = list(
    range(
        70000,
        70100,
    )
)


def make_env() -> CommonOceanEnv:
    return CommonOceanEnv(
        max_episode_steps=MAX_EPISODE_STEPS,
        render_mode=None,
        randomize_scenario=True,
    )


def load_agent() -> PPOContinuousAgent:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"No existe el checkpoint randomizado: {CHECKPOINT_PATH}"
        )

    agent = PPOContinuousAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        num_steps=ROLLOUT_STEPS,
        num_envs=NUM_ENVS,
    )

    agent.load(
        str(
            CHECKPOINT_PATH
        )
    )

    return agent


def run_episode(
    agent: PPOContinuousAgent,
    seed: int,
    record_trajectory: bool = False,
) -> dict:
    env = make_env()

    observation, info = env.reset(
        seed=seed
    )

    ego_positions = []
    traffic_positions = []
    distances = []
    dcpas = []
    tcpas = []

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

    total_reward = 0.0
    min_distance = float("inf")
    min_dcpa = float("inf")

    terminated = False
    truncated = False

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
                info["distance_to_traffic"]
            ),
        )

        min_dcpa = min(
            min_dcpa,
            float(
                info["dcpa"]
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
        "min_dcpa": float(
            min_dcpa
        ),
        "final_goal_distance": float(
            info["distance_to_goal"]
        ),
        "scenario": dict(
            info["scenario"]
        ),
        "goal_position": np.asarray(
            env.goal_position,
            dtype=np.float64,
        ),
    }

    if record_trajectory:
        result["ego_positions"] = np.asarray(
            ego_positions,
            dtype=np.float64,
        )

        result["traffic_positions"] = np.asarray(
            traffic_positions,
            dtype=np.float64,
        )

        result["distances"] = np.asarray(
            distances,
            dtype=np.float64,
        )

        result["dcpas"] = np.asarray(
            dcpas,
            dtype=np.float64,
        )

        result["tcpas"] = np.asarray(
            tcpas,
            dtype=np.float64,
        )

    env.close()

    return result


def select_best_result(
    results: list[dict],
) -> dict:
    successful_results = [
        result
        for result in results
        if (
            not result["collision"]
            and result["goal_reached"]
        )
    ]

    if successful_results:
        candidates = successful_results
    else:
        candidates = results

    return max(
        candidates,
        key=lambda result: result["reward"],
    )


def set_equal_limits(
    ax,
    ego_positions: np.ndarray,
    traffic_positions: np.ndarray,
    goal_position: np.ndarray,
    margin: float = 150.0,
) -> None:
    all_positions = np.vstack(
        [
            ego_positions,
            traffic_positions,
            goal_position.reshape(
                1,
                2,
            ),
        ]
    )

    ax.set_xlim(
        float(
            np.min(
                all_positions[:, 0]
            )
            - margin
        ),
        float(
            np.max(
                all_positions[:, 0]
            )
            + margin
        ),
    )

    ax.set_ylim(
        float(
            np.min(
                all_positions[:, 1]
            )
            - margin
        ),
        float(
            np.max(
                all_positions[:, 1]
            )
            + margin
        ),
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )


def plot_route(
    result: dict,
) -> None:
    ego_positions = np.asarray(
        result["ego_positions"],
        dtype=np.float64,
    )

    traffic_positions = np.asarray(
        result["traffic_positions"],
        dtype=np.float64,
    )

    distances = np.asarray(
        result["distances"],
        dtype=np.float64,
    )

    goal_position = np.asarray(
        result["goal_position"],
        dtype=np.float64,
    )

    closest_idx = int(
        np.argmin(
            distances
        )
    )

    scenario = result["scenario"]

    fig, ax = plt.subplots(
        figsize=(
            6,
            6,
        )
    )

    ax.plot(
        ego_positions[:, 0],
        ego_positions[:, 1],
        linewidth=2.2,
        label="Ego",
    )

    ax.plot(
        traffic_positions[:, 0],
        traffic_positions[:, 1],
        linestyle="--",
        linewidth=2.0,
        label="Tráfico",
    )

    ax.scatter(
        ego_positions[0, 0],
        ego_positions[0, 1],
        marker="o",
        s=70,
        label="Inicio ego",
    )

    ax.scatter(
        traffic_positions[0, 0],
        traffic_positions[0, 1],
        marker="s",
        s=70,
        label="Inicio tráfico",
    )

    ax.scatter(
        goal_position[0],
        goal_position[1],
        marker="*",
        s=180,
        label="Objetivo",
    )

    ax.scatter(
        ego_positions[closest_idx, 0],
        ego_positions[closest_idx, 1],
        marker="x",
        s=100,
        label="Mínima separación",
    )

    ax.scatter(
        traffic_positions[closest_idx, 0],
        traffic_positions[closest_idx, 1],
        marker="x",
        s=100,
    )

    summary_text = (
        f"Seed: {result['seed']}\n"
        f"Reward: {result['reward']:+.2f}\n"
        f"Steps: {result['steps']}\n"
        f"Collision: {result['collision']}\n"
        f"Goal: {result['goal_reached']}\n"
        f"$d_{{min}}$: {result['min_distance']:.1f} m"
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

    parameters = scenario.get(
        "parameters",
        scenario,
    )

    scenario_text = (
        f"$x_0^o$: {parameters['traffic_x0']:.1f} m\n"
        f"$y_0^o$: {parameters['traffic_y0']:.1f} m\n"
        f"$v_o$: {parameters['traffic_speed']:.2f} m/s\n"
        f"$\\psi_o$: {parameters['traffic_heading']:.2f} rad\n"
        f"$v_e$: {parameters['ego_speed']:.2f} m/s"
    )

    ax.text(
        0.98,
        0.02,
        scenario_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={
            "boxstyle": "round",
            "alpha": 0.15,
        },
    )

    ax.set_title(
        "Mejor ruta PPO en escenario randomizado no visto"
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
        loc="lower right",
        fontsize=8,
    )

    set_equal_limits(
        ax,
        ego_positions,
        traffic_positions,
        goal_position,
    )

    fig.tight_layout()

    png_path = (
        RESULTS_DIR
        / "route_best_ppo_randomized.png"
    )

    pdf_path = (
        RESULTS_DIR
        / "route_best_ppo_randomized.pdf"
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
        f"Saved: {png_path}"
    )

    print(
        f"Saved: {pdf_path}"
    )


def main() -> None:
    print("=" * 72)
    print("BEST PPO ROUTE AFTER RANDOMIZED TRAINING")
    print("=" * 72)

    agent = load_agent()

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )

    results = []

    for seed in EVAL_SEEDS:
        result = run_episode(
            agent=agent,
            seed=seed,
            record_trajectory=False,
        )

        results.append(
            result
        )

    best_result = select_best_result(
        results
    )

    print()
    print("Best randomized scenario:")
    print(
        {
            "seed": best_result["seed"],
            "reward": best_result["reward"],
            "steps": best_result["steps"],
            "collision": best_result["collision"],
            "goal_reached": best_result["goal_reached"],
            "min_distance": best_result["min_distance"],
            "min_dcpa": best_result["min_dcpa"],
            "final_goal_distance": best_result["final_goal_distance"],
            "scenario": best_result["scenario"],
        }
    )

    best_result = run_episode(
        agent=agent,
        seed=best_result["seed"],
        record_trajectory=True,
    )

    plot_route(
        best_result
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
