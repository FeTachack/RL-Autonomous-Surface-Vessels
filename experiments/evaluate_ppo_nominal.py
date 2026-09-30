from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from experiments.agents.ppo_continuous import PPOContinuousAgent
from experiments.train_ppo_commonocean import (
    STATE_SIZE,
    ACTION_SIZE,
    ROLLOUT_STEPS,
    NUM_ENVS,
    MAX_EPISODE_STEPS,
    evaluate_policy,
    to_serializable,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "checkpoints"
    / "ppo_commonocean_best.pt"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "ppo_nominal_eval"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


EVAL_SEEDS = list(range(1000, 1100))


def summarize(results: list[dict]) -> dict:
    rewards = np.asarray([r["reward"] for r in results], dtype=np.float64)
    collisions = np.asarray([r["collision"] for r in results], dtype=np.float64)
    goals = np.asarray([r["goal_reached"] for r in results], dtype=np.float64)
    steps = np.asarray([r["steps"] for r in results], dtype=np.float64)
    min_distances = np.asarray([r["min_distance"] for r in results], dtype=np.float64)
    final_goal_distances = np.asarray(
        [r["final_goal_distance"] for r in results],
        dtype=np.float64,
    )

    return {
        "num_episodes": len(results),

        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),

        "collision_rate": float(np.mean(collisions)),
        "goal_rate": float(np.mean(goals)),

        "steps_mean": float(np.mean(steps)),
        "steps_std": float(np.std(steps)),

        "min_distance_mean": float(np.mean(min_distances)),
        "min_distance_std": float(np.std(min_distances)),
        "min_distance_min": float(np.min(min_distances)),
        "min_distance_p05": float(np.percentile(min_distances, 5)),

        "final_goal_distance_mean": float(np.mean(final_goal_distances)),
        "final_goal_distance_std": float(np.std(final_goal_distances)),
        "final_goal_distance_min": float(np.min(final_goal_distances)),
        "final_goal_distance_max": float(np.max(final_goal_distances)),
    }


def main():
    print("=" * 72)
    print("PPO COMMONOCEAN - NOMINAL EVALUATION")
    print("=" * 72)
    print(f"Checkpoint : {CHECKPOINT_PATH}")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Episodes   : {len(EVAL_SEEDS)}")
    print()

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

    checkpoint = agent.load(
        str(CHECKPOINT_PATH)
    )

    print("Loaded checkpoint")
    print(f"global_step      : {checkpoint.get('global_step')}")
    print(f"update           : {checkpoint.get('update')}")
    print(f"best_eval_reward : {checkpoint.get('best_eval_reward')}")
    print()

    results = []

    for seed in EVAL_SEEDS:
        result = evaluate_policy(
            agent=agent,
            max_episode_steps=MAX_EPISODE_STEPS,
            seed=seed,
            render_mode=None,
            record_trajectory=False,
        )

        result["seed"] = seed
        results.append(result)

        print(
            f"seed={seed} | "
            f"reward={result['reward']:+9.3f} | "
            f"collision={result['collision']} | "
            f"goal={result['goal_reached']} | "
            f"min_dist={result['min_distance']:.2f} | "
            f"final_goal_dist={result['final_goal_distance']:.2f} | "
            f"steps={result['steps']}"
        )

    summary = summarize(results)

    output = {
        "checkpoint": str(CHECKPOINT_PATH),
        "eval_seeds": EVAL_SEEDS,
        "summary": summary,
        "episodes": results,
    }

    output_path = RESULTS_DIR / "nominal_evaluation.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            to_serializable(output),
            file,
            indent=2,
        )

    print()
    print("=" * 72)
    print("NOMINAL EVALUATION SUMMARY")
    print("=" * 72)

    for key, value in summary.items():
        print(f"{key:30s}: {value}")

    print()
    print(f"Saved evaluation: {output_path}")


if __name__ == "__main__":
    main()
