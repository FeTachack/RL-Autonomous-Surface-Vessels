from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
from typing import Any, Callable
import json

import numpy as np

from experiments.agents.ppo_continuous import (
    PPOContinuousAgent,
)

from experiments.envs.commonocean_env import (
    CommonOceanEnv,
)

from experiments.preferences.colreg_preference_scorer import (
    ColregPreferenceScorer,
    ColregPreferenceConfig,
)


# ============================================================
# Configuration
# ============================================================

STATE_SIZE = 13
ACTION_SIZE = 2
ROLLOUT_STEPS = 256
NUM_ENVS = 1
MAX_EPISODE_STEPS = 220

PROJECT_ROOT = Path(
    __file__
).resolve().parents[
    1
]

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
    / "colreg_preferences"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREFERENCES_JSONL = (
    RESULTS_DIR
    / "preferences.jsonl"
)

CANDIDATE_SCORES_JSON = (
    RESULTS_DIR
    / "candidate_scores.json"
)

SUMMARY_JSON = (
    RESULTS_DIR
    / "summary.json"
)

# Semillas no vistas para generar preferencias.
PREFERENCE_SEEDS = list(
    range(
        80000,
        80030,
    )
)

CANDIDATE_POLICY_NAMES = [
    "ppo",
    "ppo_noise_0.10",
    "ppo_noise_0.25",
    "ppo_starboard_bias",
    "ppo_port_bias",
    "zero_action",
    "random",
]


# ============================================================
# JSON helpers
# ============================================================


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
        (
            np.integer,
        ),
    ):
        return int(
            value
        )

    if isinstance(
        value,
        (
            np.floating,
        ),
    ):
        return float(
            value
        )

    if isinstance(
        value,
        (
            np.bool_,
        ),
    ):
        return bool(
            value
        )

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


def append_jsonl(
    path: Path,
    payload: Any,
) -> None:
    with open(
        path,
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                to_serializable(
                    payload
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# Environment and agent
# ============================================================


def make_env() -> CommonOceanEnv:
    return CommonOceanEnv(
        max_episode_steps=MAX_EPISODE_STEPS,
        render_mode=None,
        randomize_scenario=True,
    )


def load_agent() -> PPOContinuousAgent:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el checkpoint PPO randomizado:\n"
            f"{CHECKPOINT_PATH}"
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


# ============================================================
# Policies
# ============================================================


def make_policy(
    policy_name: str,
    agent: PPOContinuousAgent,
    rng: np.random.Generator,
) -> Callable:
    def ppo_policy(
        observation,
        info,
    ):
        return np.asarray(
            agent.deterministic_action(
                observation
            ),
            dtype=np.float32,
        )

    def ppo_noise_policy(
        observation,
        info,
        sigma: float,
    ):
        base_action = np.asarray(
            agent.deterministic_action(
                observation
            ),
            dtype=np.float32,
        )

        noisy_action = (
            base_action
            + rng.normal(
                loc=0.0,
                scale=sigma,
                size=(
                    ACTION_SIZE,
                ),
            ).astype(
                np.float32
            )
        )

        return np.clip(
            noisy_action,
            -1.0,
            1.0,
        ).astype(
            np.float32
        )

    def ppo_starboard_bias_policy(
        observation,
        info,
    ):
        action = np.asarray(
            agent.deterministic_action(
                observation
            ),
            dtype=np.float32,
        )

        # En nuestro marco de acción, yaw negativo corresponde
        # a giro hacia estribor para el escenario nominal.
        action[
            1
        ] = np.clip(
            action[
                1
            ]
            - 0.20,
            -1.0,
            1.0,
        )

        return action

    def ppo_port_bias_policy(
        observation,
        info,
    ):
        action = np.asarray(
            agent.deterministic_action(
                observation
            ),
            dtype=np.float32,
        )

        action[
            1
        ] = np.clip(
            action[
                1
            ]
            + 0.20,
            -1.0,
            1.0,
        )

        return action

    def zero_action_policy(
        observation,
        info,
    ):
        return np.zeros(
            ACTION_SIZE,
            dtype=np.float32,
        )

    def random_policy(
        observation,
        info,
    ):
        return rng.uniform(
            low=-1.0,
            high=1.0,
            size=(
                ACTION_SIZE,
            ),
        ).astype(
            np.float32,
        )

    if policy_name == "ppo":
        return ppo_policy

    if policy_name == "ppo_noise_0.10":
        return lambda observation, info: ppo_noise_policy(
            observation,
            info,
            sigma=0.10,
        )

    if policy_name == "ppo_noise_0.25":
        return lambda observation, info: ppo_noise_policy(
            observation,
            info,
            sigma=0.25,
        )

    if policy_name == "ppo_starboard_bias":
        return ppo_starboard_bias_policy

    if policy_name == "ppo_port_bias":
        return ppo_port_bias_policy

    if policy_name == "zero_action":
        return zero_action_policy

    if policy_name == "random":
        return random_policy

    raise ValueError(
        f"Política candidata no soportada: {policy_name}"
    )


# ============================================================
# Rollout
# ============================================================


def run_episode(
    policy_name: str,
    action_fn: Callable,
    seed: int,
) -> dict[str, Any]:
    env = make_env()

    observation, info = env.reset(
        seed=seed
    )

    ego_positions = [
        np.asarray(
            info[
                "ego_position"
            ],
            dtype=np.float64,
        )
    ]

    traffic_positions = [
        np.asarray(
            info[
                "traffic_position"
            ],
            dtype=np.float64,
        )
    ]

    ego_headings = [
        float(
            info[
                "ego_heading"
            ]
        )
    ]

    traffic_headings = [
        float(
            info[
                "traffic_heading"
            ]
        )
    ]

    distances = [
        float(
            info[
                "distance_to_traffic"
            ]
        )
    ]

    dcpas = [
        float(
            info[
                "dcpa"
            ]
        )
    ]

    tcpas = [
        float(
            info[
                "tcpa"
            ]
        )
    ]

    actions = []
    physical_actions = []
    rewards = []

    total_reward = 0.0

    terminated = False
    truncated = False

    while not (
        terminated
        or truncated
    ):
        action = np.asarray(
            action_fn(
                observation,
                info,
            ),
            dtype=np.float32,
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
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

        actions.append(
            action.copy()
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

        ego_headings.append(
            float(
                info[
                    "ego_heading"
                ]
            )
        )

        traffic_headings.append(
            float(
                info[
                    "traffic_heading"
                ]
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

    result = {
        "policy_name": policy_name,
        "seed": int(
            seed
        ),
        "reward": float(
            total_reward
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
        "ego_positions": np.asarray(
            ego_positions,
            dtype=np.float64,
        ),
        "traffic_positions": np.asarray(
            traffic_positions,
            dtype=np.float64,
        ),
        "ego_headings": np.asarray(
            ego_headings,
            dtype=np.float64,
        ),
        "traffic_headings": np.asarray(
            traffic_headings,
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


# ============================================================
# Preference generation
# ============================================================


def evaluate_candidates_for_seed(
    seed: int,
    agent: PPOContinuousAgent,
    scorer: ColregPreferenceScorer,
) -> list[dict[str, Any]]:
    candidates = []

    for policy_index, policy_name in enumerate(
        CANDIDATE_POLICY_NAMES
    ):
        rng = np.random.default_rng(
            seed
            + 1000
            * (
                policy_index
                + 1
            )
        )

        policy = make_policy(
            policy_name=policy_name,
            agent=agent,
            rng=rng,
        )

        trajectory = run_episode(
            policy_name=policy_name,
            action_fn=policy,
            seed=seed,
        )

        features = scorer.evaluate(
            trajectory
        )

        candidate = {
            "seed": int(
                seed
            ),
            "policy_name": policy_name,
            "reward": float(
                trajectory[
                    "reward"
                ]
            ),
            "collision": bool(
                trajectory[
                    "collision"
                ]
            ),
            "goal_reached": bool(
                trajectory[
                    "goal_reached"
                ]
            ),
            "steps": int(
                trajectory[
                    "steps"
                ]
            ),
            "scenario": trajectory[
                "scenario"
            ],
            "features": asdict(
                features
            ),
        }

        candidates.append(
            candidate
        )

    return candidates


def make_pairs_for_seed(
    seed: int,
    candidates: list[dict[str, Any]],
    scorer: ColregPreferenceScorer,
) -> list[dict[str, Any]]:
    pairs = []

    sorted_candidates = sorted(
        candidates,
        key=lambda item: item[
            "features"
        ][
            "colreg_score"
        ],
        reverse=True,
    )

    for i in range(
        len(
            sorted_candidates
        )
    ):
        for j in range(
            i
            + 1,
            len(
                sorted_candidates
            ),
        ):
            preferred = sorted_candidates[
                i
            ]

            rejected = sorted_candidates[
                j
            ]

            preferred_score = float(
                preferred[
                    "features"
                ][
                    "colreg_score"
                ]
            )

            rejected_score = float(
                rejected[
                    "features"
                ][
                    "colreg_score"
                ]
            )

            score_margin = (
                preferred_score
                - rejected_score
            )

            if (
                score_margin
                < scorer.config.minimum_score_margin
            ):
                continue

            pair = {
                "seed": int(
                    seed
                ),
                "preferred_policy": preferred[
                    "policy_name"
                ],
                "rejected_policy": rejected[
                    "policy_name"
                ],
                "preferred_score": preferred_score,
                "rejected_score": rejected_score,
                "score_margin": float(
                    score_margin
                ),
                "preferred_features": preferred[
                    "features"
                ],
                "rejected_features": rejected[
                    "features"
                ],
            }

            pairs.append(
                pair
            )

    return pairs


def main() -> None:
    print(
        "="
        * 72
    )

    print(
        "COLREG PREFERENCE DATASET GENERATION"
    )

    print(
        "="
        * 72
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )

    print(
        "Results dir:",
        RESULTS_DIR,
    )

    agent = load_agent()

    config = ColregPreferenceConfig()

    scorer = ColregPreferenceScorer(
        config=config
    )

    if PREFERENCES_JSONL.exists():
        PREFERENCES_JSONL.unlink()

    all_candidates = []
    all_pairs = []

    for seed in PREFERENCE_SEEDS:
        print()
        print(
            f"Seed {seed}"
        )

        candidates = evaluate_candidates_for_seed(
            seed=seed,
            agent=agent,
            scorer=scorer,
        )

        candidates_sorted = sorted(
            candidates,
            key=lambda item: item[
                "features"
            ][
                "colreg_score"
            ],
            reverse=True,
        )

        for rank, candidate in enumerate(
            candidates_sorted,
            start=1,
        ):
            features = candidate[
                "features"
            ]

            print(
                f"  {rank:02d}. "
                f"{candidate['policy_name']:<20s} "
                f"score={features['colreg_score']:+8.2f} "
                f"R={candidate['reward']:+8.2f} "
                f"col={candidate['collision']} "
                f"goal={candidate['goal_reached']} "
                f"dmin={features['min_distance']:7.2f} "
                f"DCPA={features['min_dcpa']:7.2f} "
                f"astern={features['pass_astern_at_closest']}"
            )

        pairs = make_pairs_for_seed(
            seed=seed,
            candidates=candidates,
            scorer=scorer,
        )

        for pair in pairs:
            append_jsonl(
                PREFERENCES_JSONL,
                pair,
            )

        all_candidates.extend(
            candidates
        )

        all_pairs.extend(
            pairs
        )

        print(
            f"  pairs generated: {len(pairs)}"
        )

    save_json(
        CANDIDATE_SCORES_JSON,
        all_candidates,
    )

    summary = {
        "num_seeds": len(
            PREFERENCE_SEEDS
        ),
        "num_candidate_policies": len(
            CANDIDATE_POLICY_NAMES
        ),
        "num_candidates": len(
            all_candidates
        ),
        "num_pairs": len(
            all_pairs
        ),
        "candidate_policy_names": CANDIDATE_POLICY_NAMES,
        "config": asdict(
            config
        ),
        "outputs": {
            "preferences_jsonl": str(
                PREFERENCES_JSONL
            ),
            "candidate_scores_json": str(
                CANDIDATE_SCORES_JSON
            ),
        },
    }

    save_json(
        SUMMARY_JSON,
        summary,
    )

    print()
    print(
        "="
        * 72
    )

    print(
        "SUMMARY"
    )

    print(
        "="
        * 72
    )

    print(
        f"Seeds evaluated      : {summary['num_seeds']}"
    )

    print(
        f"Candidates evaluated : {summary['num_candidates']}"
    )

    print(
        f"Preference pairs     : {summary['num_pairs']}"
    )

    print(
        f"Preferences JSONL    : {PREFERENCES_JSONL}"
    )

    print(
        f"Candidate scores JSON: {CANDIDATE_SCORES_JSON}"
    )

    print(
        f"Summary JSON         : {SUMMARY_JSON}"
    )


if __name__ == "__main__":
    main()
