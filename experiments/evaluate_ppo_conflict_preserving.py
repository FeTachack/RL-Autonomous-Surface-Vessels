from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import json

import numpy as np

from experiments.agents.ppo_continuous import PPOContinuousAgent
from experiments.envs.commonocean_env import CommonOceanEnv
from experiments.preferences.colreg_reward_wrapper import (
    ColregRewardConfig,
    ColregRewardWrapper,
)
from experiments.scenarios.randomize_conflict_scenario import (
    ConflictPreservingConfig,
)


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
    / "ppo_conflict_preserving_eval"
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

OUTPUT_PATH = (
    RESULTS_DIR
    / "summary.json"
)

MAX_EPISODE_STEPS = 220
NUM_EPISODES = 100
SEED_START = 94000
ROLLOUT_STEPS = 256
STATE_SIZE = 13
ACTION_SIZE = 2
COLREG_WEIGHT = 0.50


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


def make_env() -> ColregRewardWrapper:
    base_env = CommonOceanEnv(
        max_episode_steps=MAX_EPISODE_STEPS,
        render_mode=None,
        randomize_scenario=True,
        randomization_mode="conflict_preserving",
        randomization_config=ConflictPreservingConfig(),
    )

    env = ColregRewardWrapper(
        base_env,
        config=ColregRewardConfig(
            colreg_weight=COLREG_WEIGHT,
        ),
    )

    return env


def load_agent(
    checkpoint_path: Path,
) -> PPOContinuousAgent:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No existe checkpoint: {checkpoint_path}"
        )

    agent = PPOContinuousAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        num_steps=ROLLOUT_STEPS,
        num_envs=1,
    )

    agent.load(
        checkpoint_path
    )

    return agent


def zero_action_policy(
    observation: np.ndarray,
) -> np.ndarray:
    return np.zeros(
        ACTION_SIZE,
        dtype=np.float32,
    )


def make_agent_policy(
    agent: PPOContinuousAgent,
) -> Callable[[np.ndarray], np.ndarray]:
    def policy(
        observation: np.ndarray,
    ) -> np.ndarray:
        action = agent.deterministic_action(
            observation.reshape(
                1,
                -1,
            )
        )[0]

        return np.asarray(
            action,
            dtype=np.float32,
        )

    return policy


def run_episode(
    policy: Callable[[np.ndarray], np.ndarray],
    seed: int,
) -> dict[str, Any]:
    env = make_env()

    observation, info = env.reset(
        seed=seed
    )

    metadata = dict(
        env.unwrapped.scenario_metadata
    )

    total_reward = 0.0
    env_reward_total = 0.0
    colreg_reward_total = 0.0

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
        action = policy(
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

        env_reward_total += float(
            info.get(
                "env_reward",
                0.0,
            )
        )

        colreg_reward_total += float(
            info.get(
                "colreg_reward",
                0.0,
            )
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
            env_reward_total
        ),
        "colreg_reward": float(
            colreg_reward_total
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
        "steps": int(
            info[
                "step"
            ]
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
        "metadata": metadata,
    }

    env.close()

    return result


def summarize(
    name: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    rewards = np.asarray(
        [
            item[
                "reward"
            ]
            for item in results
        ],
        dtype=np.float64,
    )

    env_rewards = np.asarray(
        [
            item[
                "env_reward"
            ]
            for item in results
        ],
        dtype=np.float64,
    )

    colreg_rewards = np.asarray(
        [
            item[
                "colreg_reward"
            ]
            for item in results
        ],
        dtype=np.float64,
    )

    collisions = np.asarray(
        [
            item[
                "collision"
            ]
            for item in results
        ],
        dtype=np.float64,
    )

    goals = np.asarray(
        [
            item[
                "goal_reached"
            ]
            for item in results
        ],
        dtype=np.float64,
    )

    min_distances = np.asarray(
        [
            item[
                "min_distance"
            ]
            for item in results
        ],
        dtype=np.float64,
    )

    min_dcpas = np.asarray(
        [
            item[
                "min_dcpa"
            ]
            for item in results
        ],
        dtype=np.float64,
    )

    final_goal_distances = np.asarray(
        [
            item[
                "final_goal_distance"
            ]
            for item in results
        ],
        dtype=np.float64,
    )

    lengths = np.asarray(
        [
            item[
                "steps"
            ]
            for item in results
        ],
        dtype=np.float64,
    )

    summary = {
        "name": name,
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
                lengths
            )
        ),
        "results": results,
    }

    return summary


def print_summary(
    summary: dict[str, Any],
) -> None:
    print()
    print(
        "="
        * 88
    )

    print(
        summary[
            "name"
        ]
    )

    print(
        "="
        * 88
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
        "min_distance_min",
        "min_distance_p05",
        "min_distance_p10",
        "min_dcpa_mean",
        "min_dcpa_min",
        "final_goal_distance_mean",
        "episode_length_mean",
    ]

    for key in keys:
        value = summary[
            key
        ]

        if isinstance(
            value,
            float,
        ):
            print(
                f"{key:30s}: {value: .6f}"
            )
        else:
            print(
                f"{key:30s}: {value}"
            )


def evaluate_policy(
    name: str,
    policy: Callable[[np.ndarray], np.ndarray],
    seeds: list[int],
) -> dict[str, Any]:
    print()
    print(
        "="
        * 88
    )

    print(
        f"Evaluating {name}"
    )

    print(
        "="
        * 88
    )

    results = []

    for index, seed in enumerate(
        seeds,
        start=1,
    ):
        result = run_episode(
            policy=policy,
            seed=seed,
        )

        results.append(
            result
        )

        print(
            f"[{index:03d}/{len(seeds):03d}] "
            f"seed={seed} "
            f"R={result['reward']:+8.2f} "
            f"env={result['env_reward']:+8.2f} "
            f"colreg={result['colreg_reward']:+8.2f} "
            f"col={result['collision']} "
            f"goal={result['goal_reached']} "
            f"dmin={result['min_distance']:7.2f} "
            f"DCPA={result['min_dcpa']:7.2f} "
            f"steps={result['steps']:3d}"
        )

    summary = summarize(
        name=name,
        results=results,
    )

    print_summary(
        summary
    )

    return summary


def main() -> None:
    seeds = list(
        range(
            SEED_START,
            SEED_START
            + NUM_EPISODES,
        )
    )

    randomized_agent = load_agent(
        RANDOMIZED_CHECKPOINT
    )

    colreg_agent = load_agent(
        COLREG_CHECKPOINT
    )

    summaries = {}

    summaries[
        "zero_action"
    ] = evaluate_policy(
        name="zero_action_conflict_preserving",
        policy=zero_action_policy,
        seeds=seeds,
    )

    summaries[
        "ppo_randomized"
    ] = evaluate_policy(
        name="ppo_randomized_conflict_preserving",
        policy=make_agent_policy(
            randomized_agent
        ),
        seeds=seeds,
    )

    summaries[
        "ppo_colreg"
    ] = evaluate_policy(
        name="ppo_colreg_conflict_preserving",
        policy=make_agent_policy(
            colreg_agent
        ),
        seeds=seeds,
    )

    delta = {}

    for key in [
        "reward_mean",
        "env_reward_mean",
        "colreg_reward_mean",
        "collision_rate",
        "goal_rate",
        "min_distance_mean",
        "min_distance_min",
        "min_distance_p05",
        "min_distance_p10",
        "min_dcpa_mean",
        "min_dcpa_min",
        "final_goal_distance_mean",
        "episode_length_mean",
    ]:
        delta[
            key
        ] = (
            summaries[
                "ppo_colreg"
            ][
                key
            ]
            - summaries[
                "ppo_randomized"
            ][
                key
            ]
        )

    payload = {
        "seeds": seeds,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "num_episodes": NUM_EPISODES,
        "randomized_checkpoint": str(
            RANDOMIZED_CHECKPOINT
        ),
        "colreg_checkpoint": str(
            COLREG_CHECKPOINT
        ),
        "colreg_weight": COLREG_WEIGHT,
        "summaries": summaries,
        "delta_colreg_minus_randomized": delta,
    }

    save_json(
        OUTPUT_PATH,
        payload,
    )

    print()
    print(
        "="
        * 88
    )

    print(
        "DELTA: PPO COLREG - PPO RANDOMIZED"
    )

    print(
        "="
        * 88
    )

    for key, value in delta.items():
        print(
            f"{key:30s}: {value:+.6f}"
        )


if __name__ == "__main__":
    main()
