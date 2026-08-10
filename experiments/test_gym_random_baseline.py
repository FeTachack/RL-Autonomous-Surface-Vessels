from __future__ import annotations

import numpy as np

from experiments.envs.commonocean_env import CommonOceanEnv


NUM_EPISODES = 20
MAX_EPISODE_STEPS = 220
BASE_SEED = 1000


def run_random_episode(
    episode_index: int,
    seed: int,
) -> dict:
    """
    Ejecuta un episodio completo usando una política
    completamente aleatoria.

    Las acciones se generan de forma reproducible:
        action[0] ~ Uniform(-1, 1)
        action[1] ~ Uniform(-1, 1)
    """

    env = CommonOceanEnv(
        max_episode_steps=MAX_EPISODE_STEPS,
        render_mode=None,
    )

    observation, info = env.reset(
        seed=seed
    )

    rng = np.random.default_rng(seed)

    total_reward = 0.0

    reward_sums = {
        "progress": 0.0,
        "risk": 0.0,
        "time": 0.0,
        "collision": 0.0,
        "goal": 0.0,
    }

    min_distance = float("inf")
    min_distance_step = None

    terminated = False
    truncated = False

    for step in range(MAX_EPISODE_STEPS):

        # Acción Gymnasium normalizada.
        action = rng.uniform(
            low=-1.0,
            high=1.0,
            size=2,
        ).astype(np.float32)

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += float(reward)

        for key, value in (
            info["reward_components"].items()
        ):
            reward_sums[key] += float(value)

        distance = float(
            info["distance_to_traffic"]
        )

        if distance < min_distance:
            min_distance = distance
            min_distance_step = step + 1

        if terminated or truncated:
            break

    result = {
        "episode": episode_index,
        "seed": seed,

        "steps": int(
            info["step"]
        ),

        "total_reward": float(
            total_reward
        ),

        "collision": bool(
            info["collision"]
        ),

        "goal_reached": bool(
            info["goal_reached"]
        ),

        "terminated": bool(
            terminated
        ),

        "truncated": bool(
            truncated
        ),

        "min_distance": float(
            min_distance
        ),

        "min_distance_step": int(
            min_distance_step
        ),

        "final_distance_to_goal": float(
            info["distance_to_goal"]
        ),

        "reward_progress": float(
            reward_sums["progress"]
        ),

        "reward_risk": float(
            reward_sums["risk"]
        ),

        "reward_time": float(
            reward_sums["time"]
        ),

        "reward_collision": float(
            reward_sums["collision"]
        ),

        "reward_goal": float(
            reward_sums["goal"]
        ),
    }

    env.close()

    return result


def print_episode(result: dict) -> None:

    outcome = "SAFE"

    if result["collision"]:
        outcome = "COLLISION"

    elif result["goal_reached"]:
        outcome = "GOAL"

    elif result["truncated"]:
        outcome = "TRUNCATED"

    print(
        f"episode={result['episode']:02d} "
        f"seed={result['seed']} "
        f"steps={result['steps']:03d} "
        f"R={result['total_reward']:+9.3f} "
        f"min_dist={result['min_distance']:7.2f} "
        f"goal_dist="
        f"{result['final_distance_to_goal']:8.2f} "
        f"{outcome}"
    )


def main():

    results = []

    print("=" * 90)
    print(
        f"RANDOM POLICY BASELINE "
        f"({NUM_EPISODES} episodes)"
    )
    print("=" * 90)

    for episode in range(NUM_EPISODES):

        seed = BASE_SEED + episode

        result = run_random_episode(
            episode_index=episode + 1,
            seed=seed,
        )

        results.append(result)

        print_episode(result)

    # ========================================================
    # Aggregate statistics
    # ========================================================

    rewards = np.array(
        [
            r["total_reward"]
            for r in results
        ],
        dtype=np.float64,
    )

    min_distances = np.array(
        [
            r["min_distance"]
            for r in results
        ],
        dtype=np.float64,
    )

    final_goal_distances = np.array(
        [
            r["final_distance_to_goal"]
            for r in results
        ],
        dtype=np.float64,
    )

    progresses = np.array(
        [
            r["reward_progress"]
            for r in results
        ],
        dtype=np.float64,
    )

    risks = np.array(
        [
            r["reward_risk"]
            for r in results
        ],
        dtype=np.float64,
    )

    collisions = sum(
        r["collision"]
        for r in results
    )

    goals = sum(
        r["goal_reached"]
        for r in results
    )

    truncations = sum(
        r["truncated"]
        for r in results
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 90)
    print("RANDOM POLICY SUMMARY")
    print("=" * 90)

    print(
        f"episodes               = "
        f"{NUM_EPISODES}"
    )

    print(
        f"collisions             = "
        f"{collisions}"
    )

    print(
        f"collision rate         = "
        f"{collisions / NUM_EPISODES:.3f}"
    )

    print(
        f"goals                  = "
        f"{goals}"
    )

    print(
        f"goal rate              = "
        f"{goals / NUM_EPISODES:.3f}"
    )

    print(
        f"truncations            = "
        f"{truncations}"
    )

    print()

    print(
        f"reward mean            = "
        f"{np.mean(rewards):+.3f}"
    )

    print(
        f"reward std             = "
        f"{np.std(rewards):.3f}"
    )

    print(
        f"reward min             = "
        f"{np.min(rewards):+.3f}"
    )

    print(
        f"reward max             = "
        f"{np.max(rewards):+.3f}"
    )

    print(
        f"reward median          = "
        f"{np.median(rewards):+.3f}"
    )

    print()

    print(
        f"minimum distance mean  = "
        f"{np.mean(min_distances):.2f} m"
    )

    print(
        f"minimum distance min   = "
        f"{np.min(min_distances):.2f} m"
    )

    print()

    print(
        f"final goal dist mean   = "
        f"{np.mean(final_goal_distances):.2f} m"
    )

    print()

    print(
        f"progress reward mean   = "
        f"{np.mean(progresses):+.3f}"
    )

    print(
        f"risk penalty mean      = "
        f"{np.mean(risks):+.3f}"
    )

    # ========================================================
    # Reference baselines already measured
    # ========================================================

    collision_baseline_reward = -174.643
    safe_baseline_reward = 19.636

    print()
    print("=" * 90)
    print("REFERENCE COMPARISON")
    print("=" * 90)

    print(
        f"A - collision baseline = "
        f"{collision_baseline_reward:+.3f}"
    )

    print(
        f"Random mean            = "
        f"{np.mean(rewards):+.3f}"
    )

    print(
        f"B - safe maneuver      = "
        f"{safe_baseline_reward:+.3f}"
    )

    print()

    if (
        collision_baseline_reward
        < np.mean(rewards)
        < safe_baseline_reward
    ):
        print(
            "Reward ordering: "
            "A < Random < B  [OK]"
        )
    else:
        print(
            "Reward ordering: "
            "A < Random < B  [NOT SATISFIED]"
        )


if __name__ == "__main__":
    main()
