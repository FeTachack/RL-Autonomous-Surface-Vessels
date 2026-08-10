import numpy as np

from experiments.agents.ppo_continuous import (
    PPOContinuousAgent,
)


def main():

    agent = PPOContinuousAgent(
        state_size=13,
        action_size=2,
        num_steps=8,
        num_envs=4,
    )

    print(
        "Device:",
        agent.device,
    )

    states = np.random.uniform(
        low=-1.0,
        high=1.0,
        size=(4, 13),
    ).astype(np.float32)

    (
        actions,
        pre_tanh_actions,
        log_probs,
        values,
    ) = agent.sample_action(
        states
    )

    print()
    print(
        "states shape:",
        states.shape,
    )

    print(
        "actions shape:",
        actions.shape,
    )

    print(
        "pre_tanh shape:",
        pre_tanh_actions.shape,
    )

    print(
        "log_probs shape:",
        log_probs.shape,
    )

    print(
        "values shape:",
        values.shape,
    )

    print()
    print(
        "actions min:",
        actions.min(),
    )

    print(
        "actions max:",
        actions.max(),
    )

    assert (
        actions.shape
        == (4, 2)
    )

    assert (
        pre_tanh_actions.shape
        == (4, 2)
    )

    assert (
        log_probs.shape
        == (4,)
    )

    assert (
        values.shape
        == (4,)
    )

    assert np.all(
        actions >= -1.0
    )

    assert np.all(
        actions <= 1.0
    )

    print()
    print(
        "Action sampling: OK"
    )

    # ========================================================
    # Fill artificial rollout
    # ========================================================

    current_states = states.copy()

    for step in range(
        agent.num_steps
    ):
        (
            actions,
            pre_tanh_actions,
            log_probs,
            values,
        ) = agent.sample_action(
            current_states
        )

        rewards = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=agent.num_envs,
        ).astype(np.float32)

        terminated = np.zeros(
            agent.num_envs,
            dtype=np.float32,
        )

        next_states = np.random.uniform(
            low=-1.0,
            high=1.0,
            size=(
                agent.num_envs,
                agent.state_size,
            ),
        ).astype(np.float32)

        agent.store_transition(
            states=current_states,
            actions=actions,
            pre_tanh_actions=(
                pre_tanh_actions
            ),
            log_probs=log_probs,
            rewards=rewards,
            terminated=terminated,
            next_states=next_states,
        )

        current_states = (
            next_states
        )

    assert (
        agent.ptr
        == agent.num_steps
    )

    print(
        "Rollout buffer: OK"
    )

    advantages, returns = (
        agent.compute_gae()
    )

    print()
    print(
        "advantages shape:",
        advantages.shape,
    )

    print(
        "returns shape:",
        returns.shape,
    )

    assert (
        advantages.shape
        == (
            agent.num_steps,
            agent.num_envs,
        )
    )

    assert (
        returns.shape
        == (
            agent.num_steps,
            agent.num_envs,
        )
    )

    print(
        "GAE: OK"
    )

    metrics = agent.update()

    print()
    print(
        "Update metrics:",
        metrics,
    )

    assert metrics is not None

    assert agent.ptr == 0

    assert np.isfinite(
        metrics["loss"]
    )

    assert np.isfinite(
        metrics["actor_loss"]
    )

    assert np.isfinite(
        metrics["critic_loss"]
    )

    assert np.isfinite(
        metrics["entropy"]
    )

    print()
    print(
        "PPO update: OK"
    )

    # ========================================================
    # Deterministic action
    # ========================================================

    deterministic = (
        agent.deterministic_action(
            states
        )
    )

    print()
    print(
        "deterministic shape:",
        deterministic.shape,
    )

    print(
        "deterministic min:",
        deterministic.min(),
    )

    print(
        "deterministic max:",
        deterministic.max(),
    )

    assert (
        deterministic.shape
        == (4, 2)
    )

    assert np.all(
        deterministic >= -1.0
    )

    assert np.all(
        deterministic <= 1.0
    )

    print()
    print(
        "Deterministic policy: OK"
    )

    print()
    print(
        "=================================="
    )

    print(
        "PPO CONTINUOUS TEST PASSED"
    )

    print(
        "=================================="
    )


if __name__ == "__main__":
    main()
