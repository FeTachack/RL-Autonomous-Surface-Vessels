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


SEED = 42

STATE_SIZE = 13
ACTION_SIZE = 2

NUM_ENVS = 1
ROLLOUT_STEPS = 256
MAX_EPISODE_STEPS = 220


NUM_UPDATES = 50

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

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "ppo_nominal"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
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


BASELINES = {
    "collision": -174.643,
    "random": -71.932,
    "fixed_evasive": 19.636,
}


def set_seed(
    seed: int,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def moving_average(
    values,
    window: int = 5,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) < window:
        return values

    kernel = (
        np.ones(window)
        / window
    )

    return np.convolve(
        values,
        kernel,
        mode="valid",
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


def evaluate_policy(
    agent: PPOContinuousAgent,
    max_episode_steps: int,
    seed: int = SEED,
    render_mode=None,
    record_trajectory: bool = False,
):
\
\
\
\
\
\
\


    env = CommonOceanEnv(
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
    )

    observation, info = env.reset(
        seed=seed
    )

    total_reward = 0.0
    min_distance = float("inf")

    terminated = False
    truncated = False

    ego_positions = []
    traffic_positions = []
    actions = []
    physical_actions = []
    rewards = []
    distances = []
    dcpas = []
    tcpas = []
    headings = []

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

        headings.append(
            float(
                info["ego_heading"]
            )
        )

    for _ in range(
        max_episode_steps
    ):
        action = agent.deterministic_action(
            observation
        )

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += float(
            reward
        )

        min_distance = min(
            min_distance,
            float(
                info["distance_to_traffic"]
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

            actions.append(
                np.asarray(
                    action,
                    dtype=np.float64,
                )
            )

            physical_actions.append(
                np.asarray(
                    info["physical_action"],
                    dtype=np.float64,
                )
            )

            rewards.append(
                float(reward)
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

            headings.append(
                float(
                    info["ego_heading"]
                )
            )

        observation = next_observation

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

        "truncated": bool(
            truncated
        ),

        "steps": int(
            info["step"]
        ),

        "min_distance": float(
            min_distance
        ),

        "final_goal_distance": float(
            info["distance_to_goal"]
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

            "headings": np.asarray(
                headings,
                dtype=np.float64,
            ),

            "goal_position": np.asarray(
                env.goal_position,
                dtype=np.float64,
            ).copy(),
        }

    env.close()

    return result


def plot_training_curves(
    history: dict,
):
    updates = np.asarray(
        history["update"],
        dtype=np.int64,
    )

    global_steps = np.asarray(
        history["global_step"],
        dtype=np.int64,
    )

    eval_rewards = np.asarray(
        history["eval_reward"],
        dtype=np.float64,
    )

    eval_collision = np.asarray(
        history["eval_collision"],
        dtype=np.float64,
    )

    eval_min_distance = np.asarray(
        history["eval_min_distance"],
        dtype=np.float64,
    )

    eval_goal_distance = np.asarray(
        history["eval_goal_distance"],
        dtype=np.float64,
    )

    losses = np.asarray(
        history["loss"],
        dtype=np.float64,
    )

    entropy = np.asarray(
        history["entropy"],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8),
    )

    axes = axes.ravel()


    axes[0].plot(
        global_steps,
        eval_rewards,
        marker="o",
        linewidth=1.5,
        label="PPO eval",
    )

    axes[0].axhline(
        BASELINES["fixed_evasive"],
        linestyle="--",
        linewidth=1.0,
        label="Fixed evasive",
    )

    axes[0].axhline(
        BASELINES["random"],
        linestyle=":",
        linewidth=1.0,
        label="Random mean",
    )

    axes[0].set_title(
        "Reward de evaluación"
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
        global_steps,
        eval_collision,
        marker="o",
        linewidth=1.5,
    )

    axes[1].set_title(
        "Colisión en evaluación"
    )

    axes[1].set_xlabel(
        "Global steps"
    )

    axes[1].set_ylabel(
        "0 = seguro, 1 = colisión"
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
        global_steps,
        eval_min_distance,
        marker="o",
        linewidth=1.5,
    )

    axes[2].axhline(
        281.42,
        linestyle="--",
        linewidth=1.0,
        label="Fixed evasive",
    )

    axes[2].set_title(
        "Distancia mínima al tráfico"
    )

    axes[2].set_xlabel(
        "Global steps"
    )

    axes[2].set_ylabel(
        "Distancia [m]"
    )

    axes[2].grid(
        True,
        alpha=0.3,
    )

    axes[2].legend()


    axes[3].plot(
        global_steps,
        eval_goal_distance,
        marker="o",
        linewidth=1.5,
    )

    axes[3].axhline(
        1485.71,
        linestyle="--",
        linewidth=1.0,
        label="Fixed evasive",
    )

    axes[3].set_title(
        "Distancia final al objetivo"
    )

    axes[3].set_xlabel(
        "Global steps"
    )

    axes[3].set_ylabel(
        "Distancia [m]"
    )

    axes[3].grid(
        True,
        alpha=0.3,
    )

    axes[3].legend()


    axes[4].plot(
        global_steps,
        losses,
        marker="o",
        linewidth=1.5,
    )

    axes[4].set_title(
        "Pérdida PPO"
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
        global_steps,
        entropy,
        marker="o",
        linewidth=1.5,
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
        "Entrenamiento PPO en escenario nominal de cruce",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout()

    save_figure(
        fig,
        "training_curves",
    )

    plt.close(fig)


def plot_nominal_trajectory(
    eval_result: dict,
):
    trajectory = eval_result[
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

    goal_position = np.asarray(
        trajectory["goal_position"],
        dtype=np.float64,
    )

    distances = np.asarray(
        trajectory["distances"],
        dtype=np.float64,
    )

    min_idx = int(
        np.argmin(distances)
    )

    fig, ax = plt.subplots(
        figsize=(8, 7),
    )

    ax.plot(
        ego_positions[:, 0],
        ego_positions[:, 1],
        linewidth=2.0,
        label="Ruta ego PPO",
    )

    ax.plot(
        traffic_positions[:, 0],
        traffic_positions[:, 1],
        linestyle="--",
        linewidth=1.5,
        label="Ruta tráfico",
    )

    ax.scatter(
        ego_positions[0, 0],
        ego_positions[0, 1],
        marker="o",
        s=80,
        label="Inicio ego",
    )

    ax.scatter(
        traffic_positions[0, 0],
        traffic_positions[0, 1],
        marker="o",
        s=80,
        label="Inicio tráfico",
    )

    ax.scatter(
        goal_position[0],
        goal_position[1],
        marker="*",
        s=160,
        label="Objetivo",
    )

    ax.scatter(
        ego_positions[min_idx, 0],
        ego_positions[min_idx, 1],
        marker="x",
        s=100,
        label="Ego en distancia mínima",
    )

    ax.scatter(
        traffic_positions[min_idx, 0],
        traffic_positions[min_idx, 1],
        marker="x",
        s=100,
        label="Tráfico en distancia mínima",
    )

    ax.set_title(
        "Ruta aprendida por PPO en escenario nominal"
    )

    ax.set_xlabel(
        "Posición X [m]"
    )

    ax.set_ylabel(
        "Posición Y [m]"
    )

    ax.axis(
        "equal"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        loc="best",
    )

    text = (
        f"Reward: {eval_result['reward']:.2f}\n"
        f"Colisión: {eval_result['collision']}\n"
        f"Dist. mínima: {eval_result['min_distance']:.2f} m\n"
        f"Dist. final objetivo: "
        f"{eval_result['final_goal_distance']:.2f} m"
    )

    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        fontsize=9,
        bbox={
            "boxstyle": "round",
            "alpha": 0.85,
        },
    )

    fig.tight_layout()

    save_figure(
        fig,
        "nominal_trajectory",
    )

    plt.close(fig)


def plot_policy_comparison(
    ppo_reward: float,
):
    labels = [
        "Sin maniobra\n(colisión)",
        "Política\naleatoria",
        "Evasiva\nfija",
        "PPO\nnominal",
    ]

    values = [
        BASELINES["collision"],
        BASELINES["random"],
        BASELINES["fixed_evasive"],
        ppo_reward,
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5),
    )

    bars = ax.bar(
        labels,
        values,
    )

    ax.axhline(
        0.0,
        linewidth=1.0,
    )

    ax.set_title(
        "Comparación de políticas en escenario nominal"
    )

    ax.set_ylabel(
        "Reward medio"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    for bar, value in zip(
        bars,
        values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2.0,
            value,
            f"{value:.1f}",
            ha="center",
            va=(
                "bottom"
                if value >= 0.0
                else "top"
            ),
        )

    fig.tight_layout()

    save_figure(
        fig,
        "policy_comparison",
    )

    plt.close(fig)


def train():
    set_seed(
        SEED
    )

    print(
        "=" * 72
    )

    print(
        "PPO COMMONOCEAN - NOMINAL TRAINING"
    )

    print(
        "=" * 72
    )

    print(
        f"Total timesteps : {TOTAL_TIMESTEPS}"
    )

    print(
        f"Rollout steps   : {ROLLOUT_STEPS}"
    )

    print(
        f"Updates         : {NUM_UPDATES}"
    )

    print(
        f"Results dir     : {RESULTS_DIR}"
    )


    env = CommonOceanEnv(
        max_episode_steps=MAX_EPISODE_STEPS,
        render_mode=None,
    )


    agent = PPOContinuousAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        num_steps=ROLLOUT_STEPS,
        num_envs=NUM_ENVS,
    )

    print(
        f"Device          : {agent.device}"
    )


    observation, info = env.reset(
        seed=SEED
    )


    global_step = 0
    episode_count = 0

    episode_reward = 0.0
    episode_length = 0

    episode_rewards = []
    episode_lengths = []
    episode_collisions = []

    best_eval_reward = -float("inf")

    history = {
        "update": [],
        "global_step": [],

        "loss": [],
        "actor_loss": [],
        "critic_loss": [],
        "entropy": [],

        "rollout_episodes": [],
        "rollout_collision_rate": [],

        "eval_reward": [],
        "eval_collision": [],
        "eval_goal_reached": [],
        "eval_steps": [],
        "eval_min_distance": [],
        "eval_goal_distance": [],
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

            observation = next_observation

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
                    f"steps={episode_length:03d} | "
                    f"R={episode_reward:+9.3f} | "
                    f"collision={collision} | "
                    f"goal={info['goal_reached']}"
                )

                observation, info = env.reset(
                    seed=(
                        SEED
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


        eval_result = evaluate_policy(
            agent=agent,
            max_episode_steps=MAX_EPISODE_STEPS,
            seed=SEED,
            render_mode=None,
            record_trajectory=False,
        )


        if (
            eval_result["reward"]
            > best_eval_reward
        ):
            best_eval_reward = eval_result[
                "reward"
            ]

            agent.save(
                str(
                    BEST_MODEL_PATH
                ),
                extra={
                    "global_step": global_step,
                    "update": update_index,
                    "best_eval_reward": best_eval_reward,
                },
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

        history["update"].append(
            update_index
        )

        history["global_step"].append(
            global_step
        )

        history["loss"].append(
            float(metrics["loss"])
        )

        history["actor_loss"].append(
            float(metrics["actor_loss"])
        )

        history["critic_loss"].append(
            float(metrics["critic_loss"])
        )

        history["entropy"].append(
            float(metrics["entropy"])
        )

        history["rollout_episodes"].append(
            int(rollout_episodes)
        )

        history["rollout_collision_rate"].append(
            float(
                rollout_collision_rate
            )
        )

        history["eval_reward"].append(
            float(
                eval_result["reward"]
            )
        )

        history["eval_collision"].append(
            float(
                eval_result["collision"]
            )
        )

        history["eval_goal_reached"].append(
            float(
                eval_result["goal_reached"]
            )
        )

        history["eval_steps"].append(
            int(
                eval_result["steps"]
            )
        )

        history["eval_min_distance"].append(
            float(
                eval_result["min_distance"]
            )
        )

        history["eval_goal_distance"].append(
            float(
                eval_result[
                    "final_goal_distance"
                ]
            )
        )

        print()
        print(
            f"UPDATE {update_index:03d}/{NUM_UPDATES}"
        )

        print(
            f"  global step        = {global_step}"
        )

        print(
            f"  loss               = {metrics['loss']:+.5f}"
        )

        print(
            f"  actor loss         = {metrics['actor_loss']:+.5f}"
        )

        print(
            f"  critic loss        = {metrics['critic_loss']:+.5f}"
        )

        print(
            f"  entropy            = {metrics['entropy']:+.5f}"
        )

        print(
            f"  rollout episodes   = {rollout_episodes}"
        )

        print(
            f"  collision rate     = {rollout_collision_rate}"
        )

        print(
            f"  eval reward        = {eval_result['reward']:+.3f}"
        )

        print(
            f"  eval collision     = {eval_result['collision']}"
        )

        print(
            f"  eval min distance  = {eval_result['min_distance']:.2f}"
        )

        print(
            f"  eval goal distance = "
            f"{eval_result['final_goal_distance']:.2f}"
        )

        print(
            "-" * 72
        )


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

    env.close()


    history_path = (
        RESULTS_DIR
        / "training_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            to_serializable(history),
            file,
            indent=2,
        )

    print(
        f"Saved history: {history_path}"
    )


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

    best_eval_result = evaluate_policy(
        agent=best_agent,
        max_episode_steps=MAX_EPISODE_STEPS,
        seed=SEED,
        render_mode=None,
        record_trajectory=True,
    )


    trajectory_npz_path = (
        RESULTS_DIR
        / "nominal_trajectory.npz"
    )

    np.savez(
        trajectory_npz_path,
        **best_eval_result[
            "trajectory"
        ],
    )

    print(
        f"Saved trajectory data: {trajectory_npz_path}"
    )


    plot_training_curves(
        history
    )

    plot_nominal_trajectory(
        best_eval_result
    )

    plot_policy_comparison(
        ppo_reward=best_eval_result[
            "reward"
        ]
    )


    print()
    print(
        "=" * 72
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Global steps       : {global_step}"
    )

    print(
        f"Episodes           : {episode_count}"
    )

    if episode_rewards:
        print(
            f"Mean reward        : {np.mean(episode_rewards):+.3f}"
        )

        print(
            f"Mean episode len   : {np.mean(episode_lengths):.2f}"
        )

        print(
            f"Collision rate     : {np.mean(episode_collisions):.3f}"
        )

    print(
        f"Best eval reward   : {best_eval_reward:+.3f}"
    )

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
