from __future__ import annotations

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Normal


# ============================================================
# Numerical constants
# ============================================================

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

TANH_EPS = 1e-6


# ============================================================
# Actor
# ============================================================


class Actor(nn.Module):
    """
    Política Gaussiana continua con squashing tanh.

    La política genera:

        z ~ Normal(mu(s), sigma)

    y posteriormente:

        action = tanh(z)

    Por construcción:

        action in (-1, 1)

    Esto coincide con el action_space normalizado de
    CommonOceanEnv:

        [longitudinal acceleration, yaw rate]
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                state_size,
                128,
            ),
            nn.Tanh(),

            nn.Linear(
                128,
                128,
            ),
            nn.Tanh(),

            nn.Linear(
                128,
                action_size,
            ),
        )

        # Desviación estándar aprendible.
        #
        # Se mantiene independiente del estado en esta primera
        # implementación.
        self.log_std = nn.Parameter(
            torch.zeros(
                action_size,
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        states: torch.Tensor,
    ):
        mean = self.net(
            states
        )

        log_std = torch.clamp(
            self.log_std,
            LOG_STD_MIN,
            LOG_STD_MAX,
        )

        std = torch.exp(
            log_std
        )

        return (
            mean,
            std,
        )


# ============================================================
# Critic
# ============================================================


class Critic(nn.Module):
    """
    Aproximador de la función de valor:

        V(s)
    """

    def __init__(
        self,
        state_size: int,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                state_size,
                128,
            ),
            nn.Tanh(),

            nn.Linear(
                128,
                128,
            ),
            nn.Tanh(),

            nn.Linear(
                128,
                1,
            ),
        )

    def forward(
        self,
        states: torch.Tensor,
    ):
        return self.net(
            states
        )


# ============================================================
# PPO Continuous Agent
# ============================================================


class PPOContinuousAgent:
    """
    PPO para espacios de acción continuos.

    Diseñado para CommonOceanEnv:

        observation shape = (13,)
        action shape      = (2,)

    action:
        [acceleration, yaw_rate]

    ambas dimensiones se entregan normalizadas a [-1, 1].
    """

    def __init__(
        self,
        state_size: int = 13,
        action_size: int = 2,
        num_steps: int = 256,
        num_envs: int = 4,
    ):
        # ====================================================
        # Dimensions
        # ====================================================

        self.state_size = int(
            state_size
        )

        self.action_size = int(
            action_size
        )

        self.num_steps = int(
            num_steps
        )

        self.num_envs = int(
            num_envs
        )

        # ====================================================
        # Device
        # ====================================================

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # ====================================================
        # PPO hyperparameters
        # ====================================================

        self.gamma = 0.99

        self.gae_lambda = 0.95

        self.clip_eps = 0.20

        self.learning_rate = 3e-4

        self.update_epochs = 4

        self.minibatch_size = 256

        self.value_coef = 0.5

        self.entropy_coef = 0.01

        self.max_grad_norm = 0.5

        # ====================================================
        # Networks
        # ====================================================

        self.actor = Actor(
            state_size=self.state_size,
            action_size=self.action_size,
        ).to(
            self.device
        )

        self.critic = Critic(
            state_size=self.state_size,
        ).to(
            self.device
        )

        # ====================================================
        # Optimizer
        # ====================================================

        self.optimizer = optim.Adam(
            list(
                self.actor.parameters()
            )
            + list(
                self.critic.parameters()
            ),
            lr=self.learning_rate,
            eps=1e-5,
        )

        # ====================================================
        # Rollout buffer
        # ====================================================

        self.reset_rollout_buffer()

    # ========================================================
    # Rollout buffer
    # ========================================================

    def reset_rollout_buffer(
        self,
    ):
        self.ptr = 0

        # ----------------------------------------------------
        # States
        # ----------------------------------------------------

        self.states = torch.zeros(
            (
                self.num_steps,
                self.num_envs,
                self.state_size,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        # ----------------------------------------------------
        # Actions after tanh
        # ----------------------------------------------------

        self.actions = torch.zeros(
            (
                self.num_steps,
                self.num_envs,
                self.action_size,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        # ----------------------------------------------------
        # Actions before tanh
        # ----------------------------------------------------
        #
        # Se guardan explícitamente para poder evaluar de nuevo
        # exactamente la densidad de probabilidad de la acción
        # durante la actualización PPO.
        # ----------------------------------------------------

        self.pre_tanh_actions = torch.zeros(
            (
                self.num_steps,
                self.num_envs,
                self.action_size,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        # ----------------------------------------------------
        # Log probabilities
        # ----------------------------------------------------

        self.log_probs = torch.zeros(
            (
                self.num_steps,
                self.num_envs,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        # ----------------------------------------------------
        # Rewards
        # ----------------------------------------------------

        self.rewards = torch.zeros(
            (
                self.num_steps,
                self.num_envs,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        # ----------------------------------------------------
        # True terminal states
        # ----------------------------------------------------
        #
        # terminated:
        #     collision
        #     goal reached
        #
        # truncated:
        #     NO se guarda aquí como terminal verdadero.
        # ----------------------------------------------------

        self.terminated = torch.zeros(
            (
                self.num_steps,
                self.num_envs,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        # ----------------------------------------------------
        # V(s_t)
        # ----------------------------------------------------

        self.values = torch.zeros(
            (
                self.num_steps,
                self.num_envs,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        # ----------------------------------------------------
        # V(s_{t+1})
        # ----------------------------------------------------

        self.next_values = torch.zeros(
            (
                self.num_steps,
                self.num_envs,
            ),
            dtype=torch.float32,
            device=self.device,
        )

    # ========================================================
    # Distribution
    # ========================================================

    def _distribution(
        self,
        states: torch.Tensor,
    ):
        mean, std = self.actor(
            states
        )

        return Normal(
            mean,
            std,
        )

    # ========================================================
    # Squashed Gaussian log probability
    # ========================================================

    def _squashed_log_prob(
        self,
        dist: Normal,
        pre_tanh_action: torch.Tensor,
        action: torch.Tensor,
    ):
        """
        Calcula:

            log pi(a | s)

        para:

            z ~ Normal(mu, sigma)
            a = tanh(z)

        incluyendo la corrección del Jacobiano de tanh.
        """

        gaussian_log_prob = (
            dist.log_prob(
                pre_tanh_action
            )
            .sum(
                dim=-1
            )
        )

        correction = torch.log(
            1.0
            - action.pow(2)
            + TANH_EPS
        ).sum(
            dim=-1
        )

        log_prob = (
            gaussian_log_prob
            - correction
        )

        return log_prob

    # ========================================================
    # Sample stochastic action
    # ========================================================

    @torch.no_grad()
    def sample_action(
        self,
        states,
    ):
        """
        Selecciona acciones estocásticas.

        IMPORTANTE:
        No se utiliza Tensor.numpy().

        En este entorno existe actualmente una incompatibilidad
        PyTorch/NumPy en la conversión directa de memoria.

        Por ello se utiliza:

            tensor.cpu().tolist()
                ->
            np.asarray(...)

        lo cual realiza una copia segura a través de objetos
        Python.
        """

        states_np = np.asarray(
            states,
            dtype=np.float32,
        )

        single_state = (
            states_np.ndim == 1
        )

        if single_state:
            states_np = (
                states_np[
                    None,
                    :
                ]
            )

        states_t = torch.as_tensor(
            states_np,
            dtype=torch.float32,
            device=self.device,
        )

        # ----------------------------------------------------
        # Distribution
        # ----------------------------------------------------

        dist = self._distribution(
            states_t
        )

        # rsample permite reparameterization.
        pre_tanh_action = (
            dist.rsample()
        )

        action = torch.tanh(
            pre_tanh_action
        )

        # ----------------------------------------------------
        # Log probability
        # ----------------------------------------------------

        log_prob = (
            self._squashed_log_prob(
                dist=dist,
                pre_tanh_action=pre_tanh_action,
                action=action,
            )
        )

        # ----------------------------------------------------
        # Value estimate
        # ----------------------------------------------------

        value = (
            self.critic(
                states_t
            )
            .squeeze(
                -1
            )
        )

        # ====================================================
        # SAFE conversions
        # ====================================================

        action_np = np.asarray(
            action
            .detach()
            .cpu()
            .tolist(),
            dtype=np.float32,
        )

        pre_tanh_np = np.asarray(
            pre_tanh_action
            .detach()
            .cpu()
            .tolist(),
            dtype=np.float32,
        )

        log_prob_np = np.asarray(
            log_prob
            .detach()
            .cpu()
            .tolist(),
            dtype=np.float32,
        )

        value_np = np.asarray(
            value
            .detach()
            .cpu()
            .tolist(),
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Single environment
        # ----------------------------------------------------

        if single_state:
            return (
                action_np[0],
                pre_tanh_np[0],
                float(
                    log_prob_np[0]
                ),
                float(
                    value_np[0]
                ),
            )

        # ----------------------------------------------------
        # Vector environments
        # ----------------------------------------------------

        return (
            action_np,
            pre_tanh_np,
            log_prob_np,
            value_np,
        )

    # ========================================================
    # Deterministic policy
    # ========================================================

    @torch.no_grad()
    def deterministic_action(
        self,
        states,
    ):
        """
        Acción usada durante evaluación:

            action = tanh(mu(s))

        No existe muestreo.
        """

        states_np = np.asarray(
            states,
            dtype=np.float32,
        )

        single_state = (
            states_np.ndim == 1
        )

        if single_state:
            states_np = (
                states_np[
                    None,
                    :
                ]
            )

        states_t = torch.as_tensor(
            states_np,
            dtype=torch.float32,
            device=self.device,
        )

        mean, _ = self.actor(
            states_t
        )

        action = torch.tanh(
            mean
        )

        # Evitamos Tensor.numpy().
        action_np = np.asarray(
            action
            .detach()
            .cpu()
            .tolist(),
            dtype=np.float32,
        )

        if single_state:
            return action_np[0]

        return action_np

    # ========================================================
    # Value prediction
    # ========================================================

    @torch.no_grad()
    def predict_value(
        self,
        states,
    ):
        states_np = np.asarray(
            states,
            dtype=np.float32,
        )

        single_state = (
            states_np.ndim == 1
        )

        if single_state:
            states_np = (
                states_np[
                    None,
                    :
                ]
            )

        states_t = torch.as_tensor(
            states_np,
            dtype=torch.float32,
            device=self.device,
        )

        values = (
            self.critic(
                states_t
            )
            .squeeze(
                -1
            )
        )

        # Evitamos Tensor.numpy().
        values_np = np.asarray(
            values
            .detach()
            .cpu()
            .tolist(),
            dtype=np.float32,
        )

        if single_state:
            return float(
                values_np[0]
            )

        return values_np

    # ========================================================
    # Store transition
    # ========================================================

    def store_transition(
        self,
        states,
        actions,
        pre_tanh_actions,
        log_probs,
        rewards,
        terminated,
        next_states,
    ):
        """
        Guarda una transición para todos los entornos.

        terminated:
            verdadero terminal del MDP.

        Ejemplos:
            collision
            goal reached

        Las truncaciones temporales NO deben convertirse
        automáticamente en terminated.
        """

        if self.ptr >= self.num_steps:
            raise RuntimeError(
                "El rollout buffer está lleno."
            )

        # ====================================================
        # Input tensors
        # ====================================================

        states_t = torch.as_tensor(
            np.asarray(
                states,
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        actions_t = torch.as_tensor(
            np.asarray(
                actions,
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        pre_tanh_t = torch.as_tensor(
            np.asarray(
                pre_tanh_actions,
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        log_probs_t = torch.as_tensor(
            np.asarray(
                log_probs,
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        rewards_t = torch.as_tensor(
            np.asarray(
                rewards,
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        terminated_t = torch.as_tensor(
            np.asarray(
                terminated,
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        next_states_np = np.asarray(
            next_states,
            dtype=np.float32,
        )

        next_states_t = torch.as_tensor(
            next_states_np,
            dtype=torch.float32,
            device=self.device,
        )

        # ====================================================
        # Value predictions
        # ====================================================

        with torch.no_grad():

            values_t = (
                self.critic(
                    states_t
                )
                .squeeze(
                    -1
                )
            )

            next_values_t = (
                self.critic(
                    next_states_t
                )
                .squeeze(
                    -1
                )
            )

        # ====================================================
        # Store
        # ====================================================

        self.states[
            self.ptr
        ] = states_t

        self.actions[
            self.ptr
        ] = actions_t

        self.pre_tanh_actions[
            self.ptr
        ] = pre_tanh_t

        self.log_probs[
            self.ptr
        ] = log_probs_t

        self.rewards[
            self.ptr
        ] = rewards_t

        self.terminated[
            self.ptr
        ] = terminated_t

        self.values[
            self.ptr
        ] = values_t

        self.next_values[
            self.ptr
        ] = next_values_t

        self.ptr += 1

    # ========================================================
    # Generalized Advantage Estimation
    # ========================================================

    def compute_gae(
        self,
    ):
        """
        Generalized Advantage Estimation.

        Para cada transición:

            delta_t =
                r_t
                + gamma V(s_{t+1})
                - V(s_t)

        si el episodio terminó verdaderamente:

            V(s_{t+1}) = 0

        Una truncación temporal no se trata como terminal
        verdadero.
        """

        if self.ptr != self.num_steps:
            raise RuntimeError(
                "compute_gae requiere "
                "un rollout completo."
            )

        advantages = torch.zeros_like(
            self.rewards
        )

        last_gae = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )

        # ----------------------------------------------------
        # Backward GAE recursion
        # ----------------------------------------------------

        for t in reversed(
            range(
                self.num_steps
            )
        ):
            non_terminal = (
                1.0
                - self.terminated[t]
            )

            delta = (
                self.rewards[t]
                + self.gamma
                * self.next_values[t]
                * non_terminal
                - self.values[t]
            )

            last_gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * non_terminal
                * last_gae
            )

            advantages[t] = (
                last_gae
            )

        returns = (
            advantages
            + self.values
        )

        return (
            advantages,
            returns,
        )

    # ========================================================
    # PPO update
    # ========================================================

    def update(
        self,
    ):
        """
        Ejecuta una actualización PPO cuando el rollout
        contiene num_steps transiciones.
        """

        if self.ptr < self.num_steps:
            return None

        # ====================================================
        # Advantages / returns
        # ====================================================

        with torch.no_grad():
            (
                advantages,
                returns,
            ) = self.compute_gae()

        # ====================================================
        # Flatten rollout
        # ====================================================

        batch_size = (
            self.num_steps
            * self.num_envs
        )

        b_states = (
            self.states.reshape(
                batch_size,
                self.state_size,
            )
        )

        b_actions = (
            self.actions.reshape(
                batch_size,
                self.action_size,
            )
        )

        b_pre_tanh = (
            self.pre_tanh_actions.reshape(
                batch_size,
                self.action_size,
            )
        )

        b_old_log_probs = (
            self.log_probs.reshape(
                batch_size
            )
        )

        b_advantages = (
            advantages.reshape(
                batch_size
            )
        )

        b_returns = (
            returns.reshape(
                batch_size
            )
        )

        # ====================================================
        # Advantage normalization
        # ====================================================

        advantage_std = (
            b_advantages.std()
        )

        b_advantages = (
            b_advantages
            - b_advantages.mean()
        ) / (
            advantage_std
            + 1e-8
        )

        # ====================================================
        # Mini-batch PPO
        # ====================================================

        indices = np.arange(
            batch_size
        )

        total_loss = 0.0

        total_actor_loss = 0.0

        total_critic_loss = 0.0

        total_entropy = 0.0

        total_updates = 0

        # ----------------------------------------------------
        # Epochs
        # ----------------------------------------------------

        for _ in range(
            self.update_epochs
        ):
            np.random.shuffle(
                indices
            )

            # ------------------------------------------------
            # Mini-batches
            # ------------------------------------------------

            for start in range(
                0,
                batch_size,
                self.minibatch_size,
            ):
                end = min(
                    start
                    + self.minibatch_size,
                    batch_size,
                )

                mb_idx_np = (
                    indices[
                        start:end
                    ]
                )

                mb_idx = torch.as_tensor(
                    mb_idx_np,
                    dtype=torch.long,
                    device=self.device,
                )

                # ============================================
                # New policy distribution
                # ============================================

                dist = self._distribution(
                    b_states[
                        mb_idx
                    ]
                )

                # ============================================
                # New log probabilities
                # ============================================

                new_log_probs = (
                    self._squashed_log_prob(
                        dist=dist,
                        pre_tanh_action=(
                            b_pre_tanh[
                                mb_idx
                            ]
                        ),
                        action=(
                            b_actions[
                                mb_idx
                            ]
                        ),
                    )
                )

                # ============================================
                # Entropy
                # ============================================
                #
                # Usamos la entropía de la Normal pre-tanh
                # como aproximación para regularizar la
                # exploración.
                # ============================================

                entropy = (
                    dist.entropy()
                    .sum(
                        dim=-1
                    )
                    .mean()
                )

                # ============================================
                # New value
                # ============================================

                new_values = (
                    self.critic(
                        b_states[
                            mb_idx
                        ]
                    )
                    .squeeze(
                        -1
                    )
                )

                # ============================================
                # PPO ratio
                # ============================================

                ratio = torch.exp(
                    new_log_probs
                    - b_old_log_probs[
                        mb_idx
                    ]
                )

                # ============================================
                # Actor loss
                # ============================================

                pg_loss_unclipped = (
                    -b_advantages[
                        mb_idx
                    ]
                    * ratio
                )

                pg_loss_clipped = (
                    -b_advantages[
                        mb_idx
                    ]
                    * torch.clamp(
                        ratio,
                        1.0
                        - self.clip_eps,
                        1.0
                        + self.clip_eps,
                    )
                )

                actor_loss = (
                    torch.max(
                        pg_loss_unclipped,
                        pg_loss_clipped,
                    )
                    .mean()
                )

                # ============================================
                # Critic loss
                # ============================================

                critic_loss = (
                    F.mse_loss(
                        new_values,
                        b_returns[
                            mb_idx
                        ],
                    )
                )

                # ============================================
                # Total loss
                # ============================================

                loss = (
                    actor_loss
                    + self.value_coef
                    * critic_loss
                    - self.entropy_coef
                    * entropy
                )

                # ============================================
                # Optimization
                # ============================================

                self.optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(
                        self.actor.parameters()
                    )
                    + list(
                        self.critic.parameters()
                    ),
                    self.max_grad_norm,
                )

                self.optimizer.step()

                # ============================================
                # Metrics
                # ============================================

                total_loss += float(
                    loss.item()
                )

                total_actor_loss += float(
                    actor_loss.item()
                )

                total_critic_loss += float(
                    critic_loss.item()
                )

                total_entropy += float(
                    entropy.item()
                )

                total_updates += 1

        # ====================================================
        # Metrics
        # ====================================================

        denominator = max(
            total_updates,
            1,
        )

        metrics = {
            "loss": (
                total_loss
                / denominator
            ),

            "actor_loss": (
                total_actor_loss
                / denominator
            ),

            "critic_loss": (
                total_critic_loss
                / denominator
            ),

            "entropy": (
                total_entropy
                / denominator
            ),
        }

        # ====================================================
        # Reset rollout
        # ====================================================

        self.reset_rollout_buffer()

        return metrics

    # ========================================================
    # Save checkpoint
    # ========================================================

    def save(
        self,
        path: str,
        extra: dict | None = None,
    ):
        checkpoint = {
            # Networks
            "actor_state_dict": (
                self.actor.state_dict()
            ),

            "critic_state_dict": (
                self.critic.state_dict()
            ),

            # Optimizer
            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),

            # Dimensions
            "state_size": (
                self.state_size
            ),

            "action_size": (
                self.action_size
            ),

            # Rollout configuration
            "num_steps": (
                self.num_steps
            ),

            "num_envs": (
                self.num_envs
            ),

            # PPO parameters
            "gamma": (
                self.gamma
            ),

            "gae_lambda": (
                self.gae_lambda
            ),

            "clip_eps": (
                self.clip_eps
            ),

            "learning_rate": (
                self.learning_rate
            ),

            "update_epochs": (
                self.update_epochs
            ),

            "minibatch_size": (
                self.minibatch_size
            ),

            "value_coef": (
                self.value_coef
            ),

            "entropy_coef": (
                self.entropy_coef
            ),

            "max_grad_norm": (
                self.max_grad_norm
            ),
        }

        if extra is not None:
            checkpoint.update(
                extra
            )

        torch.save(
            checkpoint,
            path,
        )

    # ========================================================
    # Load checkpoint
    # ========================================================

    def load(
        self,
        path: str,
    ):
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        self.actor.load_state_dict(
            checkpoint[
                "actor_state_dict"
            ]
        )

        self.critic.load_state_dict(
            checkpoint[
                "critic_state_dict"
            ]
        )

        if (
            "optimizer_state_dict"
            in checkpoint
        ):
            self.optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

        return checkpoint
