from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

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
    / "report_routes"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


PPO_RANDOMIZED_CHECKPOINT = (
    CHECKPOINT_DIR
    / "ppo_randomized_crossing_best.pt"
)

PPO_NOMINAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "ppo_commonocean_best.pt"
)

NOMINAL_SEED = 123
RANDOM_CANDIDATE_SEEDS = list(
    range(
        20
    )
)


def make_nominal_env() -> CommonOceanEnv:


    return CommonOceanEnv(
        max_episode_steps=MAX_EPISODE_STEPS,
        render_mode=None,
        randomize_scenario=False,
    )


def load_best_ppo_agent() -> tuple[PPOContinuousAgent, Path]:
    agent = PPOContinuousAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        num_steps=ROLLOUT_STEPS,
        num_envs=NUM_ENVS,
    )

    if PPO_RANDOMIZED_CHECKPOINT.exists():
        checkpoint_path = PPO_RANDOMIZED_CHECKPOINT
    elif PPO_NOMINAL_CHECKPOINT.exists():
        checkpoint_path = PPO_NOMINAL_CHECKPOINT
    else:
        raise FileNotFoundError(
            "No se encontró ningún checkpoint PPO. "
            f"Busqué:\n"
            f"  {PPO_RANDOMIZED_CHECKPOINT}\n"
            f"  {PPO_NOMINAL_CHECKPOINT}"
        )

    agent.load(
        str(
            checkpoint_path
        )
    )

    return (
        agent,
        checkpoint_path,
    )


def run_episode(
    policy_name: str,
    action_fn,
    seed: int,
) -> dict:
    env = make_nominal_env()

    observation, info = env.reset(
        seed=seed
    )

    ego_positions = [
        np.asarray(
            info["ego_position"],
            dtype=np.float64,
        )
    ]

    traffic_positions = [
        np.asarray(
            info["traffic_position"],
            dtype=np.float64,
        )
    ]

    distances = [
        float(
            info["distance_to_traffic"]
        )
    ]

    dcpas = [
        float(
            info["dcpa"]
        )
    ]

    tcpas = [
        float(
            info["tcpa"]
        )
    ]

    rewards = []
    actions = []

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
        action = action_fn(
            observation,
            info,
        )

        action = np.asarray(
            action,
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

        min_distance = min(
            min_distance,
            float(
                info[
                    "distance_to_traffic"
                ]
            ),
        )

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

        rewards.append(
            float(
                reward
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
        "rewards": np.asarray(
            rewards,
            dtype=np.float64,
        ),
        "actions": np.asarray(
            actions,
            dtype=np.float64,
        ),
        "goal_position": np.asarray(
            env.goal_position,
            dtype=np.float64,
        ),
    }

    env.close()

    return result


def zero_action_policy(
    observation,
    info,
):
    return np.zeros(
        2,
        dtype=np.float32,
    )


def make_random_policy(
    seed: int,
):
    rng = np.random.default_rng(
        seed
    )

    def policy(
        observation,
        info,
    ):
        return rng.uniform(
            low=-1.0,
            high=1.0,
            size=(
                2,
            ),
        ).astype(
            np.float32
        )

    return policy


def make_ppo_policy(
    agent: PPOContinuousAgent,
):
    def policy(
        observation,
        info,
    ):
        return agent.deterministic_action(
            observation
        )

    return policy


def select_representative_random_route() -> dict:


    random_results = []

    for seed in RANDOM_CANDIDATE_SEEDS:
        result = run_episode(
            policy_name="Política aleatoria",
            action_fn=make_random_policy(
                seed
            ),
            seed=NOMINAL_SEED,
        )

        result[
            "random_policy_seed"
        ] = seed

        random_results.append(
            result
        )

    rewards = np.asarray(
        [
            result[
                "reward"
            ]
            for result in random_results
        ],
        dtype=np.float64,
    )

    median_reward = float(
        np.median(
            rewards
        )
    )

    selected = min(
        random_results,
        key=lambda result: abs(
            result[
                "reward"
            ]
            - median_reward
        ),
    )

    print()
    print("Random policy candidates:")
    print(
        f"  reward mean   = {np.mean(rewards):+.3f}"
    )
    print(
        f"  reward median = {median_reward:+.3f}"
    )
    print(
        f"  selected seed = {selected['random_policy_seed']}"
    )
    print(
        f"  selected R    = {selected['reward']:+.3f}"
    )
    print(
        f"  selected col  = {selected['collision']}"
    )

    return selected


def _set_equal_limits(
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

    x_min = float(
        np.min(
            all_positions[:, 0]
        )
        - margin
    )

    x_max = float(
        np.max(
            all_positions[:, 0]
        )
        + margin
    )

    y_min = float(
        np.min(
            all_positions[:, 1]
        )
        - margin
    )

    y_max = float(
        np.max(
            all_positions[:, 1]
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


def plot_route(
    result: dict,
    filename_base: str,
    title: str,
) -> None:
    ego_positions = np.asarray(
        result[
            "ego_positions"
        ],
        dtype=np.float64,
    )

    traffic_positions = np.asarray(
        result[
            "traffic_positions"
        ],
        dtype=np.float64,
    )

    distances = np.asarray(
        result[
            "distances"
        ],
        dtype=np.float64,
    )

    goal_position = np.asarray(
        result[
            "goal_position"
        ],
        dtype=np.float64,
    )

    closest_idx = int(
        np.argmin(
            distances
        )
    )

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

    if result[
        "collision"
    ]:
        ax.scatter(
            ego_positions[-1, 0],
            ego_positions[-1, 1],
            marker="X",
            s=180,
            label="Colisión",
        )

    summary_text = (
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

    ax.set_title(
        title
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

    _set_equal_limits(
        ax,
        ego_positions,
        traffic_positions,
        goal_position,
    )

    fig.tight_layout()

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
        f"Saved: {png_path}"
    )

    print(
        f"Saved: {pdf_path}"
    )


def main() -> None:
    print("=" * 72)
    print("REPORT ROUTE FIGURES")
    print("=" * 72)

    print(
        "Results directory:",
        RESULTS_DIR,
    )


    collision_result = run_episode(
        policy_name="Sin evasión",
        action_fn=zero_action_policy,
        seed=NOMINAL_SEED,
    )


    random_result = select_representative_random_route()


    agent, checkpoint_path = load_best_ppo_agent()

    print()
    print(
        "PPO checkpoint:",
        checkpoint_path,
    )

    ppo_result = run_episode(
        policy_name="PPO",
        action_fn=make_ppo_policy(
            agent
        ),
        seed=NOMINAL_SEED,
    )

    print()
    print("Collision baseline:")
    print(
        {
            "reward": collision_result["reward"],
            "steps": collision_result["steps"],
            "collision": collision_result["collision"],
            "goal": collision_result["goal_reached"],
            "min_distance": collision_result["min_distance"],
        }
    )

    print()
    print("Random policy:")
    print(
        {
            "reward": random_result["reward"],
            "steps": random_result["steps"],
            "collision": random_result["collision"],
            "goal": random_result["goal_reached"],
            "min_distance": random_result["min_distance"],
            "random_policy_seed": random_result.get(
                "random_policy_seed"
            ),
        }
    )

    print()
    print("Best PPO:")
    print(
        {
            "reward": ppo_result["reward"],
            "steps": ppo_result["steps"],
            "collision": ppo_result["collision"],
            "goal": ppo_result["goal_reached"],
            "min_distance": ppo_result["min_distance"],
        }
    )

    plot_route(
        collision_result,
        "route_collision_baseline",
        "Ruta sin evasión: colisión",
    )

    plot_route(
        random_result,
        "route_random_policy",
        "Ruta con política aleatoria",
    )

    plot_route(
        ppo_result,
        "route_best_ppo",
        "Ruta con mejor política PPO",
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
