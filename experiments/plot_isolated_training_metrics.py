from __future__ import annotations

from pathlib import Path
import json
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "ppo_randomized_crossing"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "checkpoints"
)

TRAINING_HISTORY_PATH = (
    RESULTS_DIR
    / "training_history.json"
)

TRAINING_LOG_PATH = (
    RESULTS_DIR
    / "training_run.log"
)

FINAL_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "ppo_randomized_crossing_final.pt"
)


# ============================================================
# IO
# ============================================================


def load_training_history(
    path: Path,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe training_history.json: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def load_episode_rewards_from_checkpoint(
    path: Path,
) -> list[float]:
    """
    Lee episode_rewards desde el checkpoint final.

    Soporta dos formatos posibles:

    1. checkpoint["episode_rewards"]
    2. checkpoint["extra"]["episode_rewards"]
    """

    if not path.exists():
        raise FileNotFoundError(
            f"No existe checkpoint final: {path}"
        )

    checkpoint = torch.load(
        str(path),
        map_location="cpu",
    )

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and "episode_rewards" in checkpoint
    ):
        rewards = checkpoint[
            "episode_rewards"
        ]

        return [
            float(
                reward
            )
            for reward in rewards
        ]

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and "extra" in checkpoint
        and isinstance(
            checkpoint["extra"],
            dict,
        )
        and "episode_rewards" in checkpoint["extra"]
    ):
        rewards = checkpoint[
            "extra"
        ][
            "episode_rewards"
        ]

        return [
            float(
                reward
            )
            for reward in rewards
        ]

    available_keys = (
        list(
            checkpoint.keys()
        )
        if isinstance(
            checkpoint,
            dict,
        )
        else []
    )

    raise RuntimeError(
        "El checkpoint existe, pero no contiene episode_rewards. "
        f"Keys disponibles: {available_keys}"
    )


def parse_episode_rewards_from_log(
    path: Path,
) -> list[float]:
    """
    Fallback: extrae rewards por episodio desde training_run.log.

    Ejemplo esperado:

        Episode 0001 | steps=220 | R=  +85.123 | collision=False
    """

    if not path.exists():
        raise FileNotFoundError(
            f"No existe training_run.log: {path}"
        )

    pattern = re.compile(
        r"Episode\s+\d+.*?\|\s*R=\s*([+-]?\s*\d+(?:\.\d+)?)"
    )

    rewards = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            match = pattern.search(
                line
            )

            if match is None:
                continue

            value = (
                match.group(1)
                .replace(
                    " ",
                    "",
                )
            )

            rewards.append(
                float(
                    value
                )
            )

    if not rewards:
        raise RuntimeError(
            "No se encontraron rewards de episodio en "
            f"{path}."
        )

    return rewards


def load_episode_rewards() -> tuple[list[float], str]:
    """
    Fuente preferida:
        checkpoint final.

    Fuente alternativa:
        training_run.log.
    """

    try:
        rewards = load_episode_rewards_from_checkpoint(
            FINAL_CHECKPOINT_PATH
        )

        return (
            rewards,
            str(
                FINAL_CHECKPOINT_PATH
            ),
        )

    except Exception as checkpoint_error:
        print(
            "No se pudieron leer episode_rewards "
            "desde el checkpoint final."
        )

        print(
            "Detalle:",
            checkpoint_error,
        )

    try:
        rewards = parse_episode_rewards_from_log(
            TRAINING_LOG_PATH
        )

        return (
            rewards,
            str(
                TRAINING_LOG_PATH
            ),
        )

    except Exception as log_error:
        print(
            "No se pudieron leer episode_rewards "
            "desde training_run.log."
        )

        print(
            "Detalle:",
            log_error,
        )

    raise RuntimeError(
        "No se pudo obtener episode_rewards. "
        "Necesitas al menos uno de estos archivos:\n"
        f"1. {FINAL_CHECKPOINT_PATH}\n"
        f"2. {TRAINING_LOG_PATH}"
    )


def save_figure(
    fig,
    filename_base: str,
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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


# ============================================================
# Plotting
# ============================================================


def plot_loss_curve(
    history: dict,
) -> None:
    global_steps = np.asarray(
        history[
            "global_step"
        ],
        dtype=np.float64,
    )

    loss = np.asarray(
        history[
            "loss"
        ],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(
        figsize=(
            7,
            4,
        )
    )

    ax.plot(
        global_steps,
        loss,
        marker="o",
        linewidth=1.8,
        label="Pérdida PPO",
    )

    ax.set_title(
        "Pérdida durante el entrenamiento PPO"
    )

    ax.set_xlabel(
        "Pasos globales"
    )

    ax.set_ylabel(
        "Pérdida"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    save_figure(
        fig,
        "isolated_loss_curve",
    )

    plt.close(
        fig
    )


def plot_loss_components(
    history: dict,
) -> None:
    """
    Figura opcional con pérdida total, actor y critic.
    """

    global_steps = np.asarray(
        history[
            "global_step"
        ],
        dtype=np.float64,
    )

    loss = np.asarray(
        history[
            "loss"
        ],
        dtype=np.float64,
    )

    actor_loss = np.asarray(
        history[
            "actor_loss"
        ],
        dtype=np.float64,
    )

    critic_loss = np.asarray(
        history[
            "critic_loss"
        ],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(
        figsize=(
            7,
            4,
        )
    )

    ax.plot(
        global_steps,
        loss,
        marker="o",
        linewidth=1.8,
        label="Pérdida total",
    )

    ax.plot(
        global_steps,
        critic_loss,
        marker="s",
        linewidth=1.2,
        linestyle="--",
        label="Pérdida critic",
    )

    ax.plot(
        global_steps,
        actor_loss,
        marker="^",
        linewidth=1.2,
        linestyle=":",
        label="Pérdida actor",
    )

    ax.set_title(
        "Componentes de pérdida PPO"
    )

    ax.set_xlabel(
        "Pasos globales"
    )

    ax.set_ylabel(
        "Pérdida"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    save_figure(
        fig,
        "isolated_loss_components",
    )

    plt.close(
        fig
    )


def plot_cumulative_mean_reward(
    episode_rewards: list[float],
) -> None:
    rewards = np.asarray(
        episode_rewards,
        dtype=np.float64,
    )

    episode_index = np.arange(
        1,
        len(
            rewards
        )
        + 1,
        dtype=np.int64,
    )

    cumulative_mean_reward = (
        np.cumsum(
            rewards
        )
        / episode_index
    )

    fig, ax = plt.subplots(
        figsize=(
            7,
            4,
        )
    )

    ax.plot(
        episode_index,
        cumulative_mean_reward,
        marker="o",
        linewidth=1.8,
        label="Reward promedio acumulado",
    )

    ax.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )

    ax.set_title(
        "Reward promedio acumulado durante el entrenamiento"
    )

    ax.set_xlabel(
        "Episodio"
    )

    ax.set_ylabel(
        "Reward promedio acumulado"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    final_value = float(
        cumulative_mean_reward[
            -1
        ]
    )

    ax.text(
        0.98,
        0.05,
        f"Valor final: {final_value:+.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={
            "boxstyle": "round",
            "alpha": 0.15,
        },
    )

    fig.tight_layout()

    save_figure(
        fig,
        "isolated_cumulative_mean_reward",
    )

    plt.close(
        fig
    )


def plot_episode_reward_with_cumulative_mean(
    episode_rewards: list[float],
) -> None:
    """
    Figura opcional: reward por episodio + promedio acumulado.
    """

    rewards = np.asarray(
        episode_rewards,
        dtype=np.float64,
    )

    episode_index = np.arange(
        1,
        len(
            rewards
        )
        + 1,
        dtype=np.int64,
    )

    cumulative_mean_reward = (
        np.cumsum(
            rewards
        )
        / episode_index
    )

    fig, ax = plt.subplots(
        figsize=(
            8,
            4.5,
        )
    )

    ax.plot(
        episode_index,
        rewards,
        linewidth=1.0,
        alpha=0.45,
        label="Reward por episodio",
    )

    ax.plot(
        episode_index,
        cumulative_mean_reward,
        linewidth=2.2,
        label="Reward promedio acumulado",
    )

    ax.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )

    ax.set_title(
        "Reward de entrenamiento PPO"
    )

    ax.set_xlabel(
        "Episodio"
    )

    ax.set_ylabel(
        "Reward"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    save_figure(
        fig,
        "isolated_episode_reward_with_cumulative_mean",
    )

    plt.close(
        fig
    )


# ============================================================
# Main
# ============================================================


def main() -> None:
    print("=" * 72)
    print("ISOLATED TRAINING METRICS")
    print("=" * 72)

    print(
        "Results dir:",
        RESULTS_DIR,
    )

    history = load_training_history(
        TRAINING_HISTORY_PATH
    )

    episode_rewards, reward_source = (
        load_episode_rewards()
    )

    print(
        "Training history:",
        TRAINING_HISTORY_PATH,
    )

    print(
        "Episode reward source:",
        reward_source,
    )

    print(
        "Updates found:",
        len(
            history[
                "global_step"
            ]
        ),
    )

    print(
        "Episode rewards found:",
        len(
            episode_rewards
        ),
    )

    print(
        "Final cumulative mean reward:",
        float(
            np.mean(
                episode_rewards
            )
        ),
    )

    plot_loss_curve(
        history
    )

    plot_loss_components(
        history
    )

    plot_cumulative_mean_reward(
        episode_rewards
    )

    plot_episode_reward_with_cumulative_mean(
        episode_rewards
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
