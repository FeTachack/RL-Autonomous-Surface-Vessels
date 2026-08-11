from __future__ import annotations

from pathlib import Path
from typing import Any
import json

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

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "checkpoints"
    / "ppo_colreg_finetuned_best.pt"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "ppo_colreg_finetune"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "best_checkpoint_evaluation.json"
)

EVAL_SEEDS = list(
    range(
        91_000,
        91_100,
    )
)


def to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            key: to_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            to_serializable(item)
            for item in value
        ]

    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            to_serializable(payload),
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved JSON: {path}")


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


def load_agent() -> PPOContinuousAgent:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"No existe el checkpoint: {CHECKPOINT_PATH}"
        )

    agent = PPOContinuousAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        num_steps=ROLLOUT_STEPS,
        num_envs=NUM_ENVS,
    )

    agent.load(
        str(CHECKPOINT_PATH)
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
        info["distance_to_traffic"]
    )

    min_dcpa = float(
        info["dcpa"]
    )

    terminated = False
    truncated = False

    while not (terminated or truncated):
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

        total_reward += float(reward)
        env_reward_sum += float(info["env_reward"])
        colreg_reward_sum += float(info["colreg_reward"])

        min_distance = min(
            min_distance,
            float(info["distance_to_traffic"]),
        )

        min_dcpa = min(
            min_dcpa,
            float(info["dcpa"]),
        )

    result = {
        "seed": int(seed),
        "reward": float(total_reward),
        "env_reward": float(env_reward_sum),
        "colreg_reward": float(colreg_reward_sum),
        "steps": int(info["step"]),
        "collision": bool(info["collision"]),
        "goal_reached": bool(info["goal_reached"]),
        "truncated": bool(truncated),
        "min_distance": float(min_distance),
        "min_dcpa": float(min_dcpa),
        "final_goal_distance": float(info["distance_to_goal"]),
        "scenario": dict(info["scenario"]),
    }

    env.close()

    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = np.asarray(
        [r["reward"] for r in results],
        dtype=np.float64,
    )

    env_rewards = np.asarray(
        [r["env_reward"] for r in results],
        dtype=np.float64,
    )

    colreg_rewards = np.asarray(
        [r["colreg_reward"] for r in results],
        dtype=np.float64,
    )

    collisions = np.asarray(
        [r["collision"] for r in results],
        dtype=np.float64,
    )

    goals = np.asarray(
        [r["goal_reached"] for r in results],
        dtype=np.float64,
    )

    min_distances = np.asarray(
        [r["min_distance"] for r in results],
        dtype=np.float64,
    )

    min_dcpas = np.asarray(
        [r["min_dcpa"] for r in results],
        dtype=np.float64,
    )

    final_goal_distances = np.asarray(
        [r["final_goal_distance"] for r in results],
        dtype=np.float64,
    )

    steps = np.asarray(
        [r["steps"] for r in results],
        dtype=np.float64,
    )

    return {
        "episodes": int(len(results)),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "env_reward_mean": float(np.mean(env_rewards)),
        "colreg_reward_mean": float(np.mean(colreg_rewards)),
        "collision_rate": float(np.mean(collisions)),
        "goal_rate": float(np.mean(goals)),
        "min_distance_mean": float(np.mean(min_distances)),
        "min_distance_std": float(np.std(min_distances)),
        "min_distance_min": float(np.min(min_distances)),
        "min_distance_p05": float(np.percentile(min_distances, 5)),
        "min_distance_p10": float(np.percentile(min_distances, 10)),
        "min_distance_p25": float(np.percentile(min_distances, 25)),
        "min_dcpa_mean": float(np.mean(min_dcpas)),
        "min_dcpa_min": float(np.min(min_dcpas)),
        "min_dcpa_p05": float(np.percentile(min_dcpas, 5)),
        "min_dcpa_p10": float(np.percentile(min_dcpas, 10)),
        "final_goal_distance_mean": float(np.mean(final_goal_distances)),
        "episode_length_mean": float(np.mean(steps)),
    }


def main() -> None:
    print("=" * 72)
    print("EVALUATING COLREG FINE-TUNED BEST CHECKPOINT")
    print("=" * 72)

    print("Checkpoint:", CHECKPOINT_PATH)
    print("Seeds:", EVAL_SEEDS[0], "to", EVAL_SEEDS[-1])

    agent = load_agent()

    results = []

    for seed in EVAL_SEEDS:
        result = run_episode(
            agent=agent,
            seed=seed,
        )

        results.append(result)

    summary = summarize(results)

    payload = {
        "checkpoint": str(CHECKPOINT_PATH),
        "colreg_weight": COLREG_WEIGHT,
        "eval_seeds": EVAL_SEEDS,
        "summary": summary,
        "results": results,
    }

    save_json(
        OUTPUT_PATH,
        payload,
    )

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    for key, value in summary.items():
        print(f"{key:<28s}: {value}")


if __name__ == "__main__":
    main()
