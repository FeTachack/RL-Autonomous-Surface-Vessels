from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import math
import gymnasium as gym
import numpy as np


def global_to_ego_frame(
    vector: np.ndarray,
    heading: float,
) -> np.ndarray:


    c = math.cos(
        heading
    )

    s = math.sin(
        heading
    )

    rotation = np.array(
        [
            [
                c,
                s,
            ],
            [
                -s,
                c,
            ],
        ],
        dtype=np.float64,
    )

    return rotation @ vector


@dataclass
class ColregRewardConfig:


    colreg_weight: float = 0.10


    risk_distance: float = 800.0
    safe_distance: float = 300.0
    safe_dcpa: float = 200.0


    early_action_horizon_steps: int = 90


    starboard_bonus: float = 1.50
    port_penalty: float = -1.50
    early_starboard_bonus: float = 1.00


    close_distance_penalty: float = -2.50
    low_dcpa_penalty: float = -0.75


    pass_astern_bonus: float = 0.60
    pass_ahead_penalty: float = -0.60


    smoothness_penalty: float = -0.25
    action_magnitude_penalty: float = -0.05


    goal_bonus: float = 2.00
    collision_penalty: float = -6.00


    terminal_min_distance_bonus: float = 80.0
    terminal_min_dcpa_bonus: float = 30.0
    terminal_low_distance_penalty: float = -60.0
    terminal_low_dcpa_penalty: float = -20.0


class ColregRewardWrapper(
    gym.Wrapper,
):


    def __init__(
        self,
        env: gym.Env,
        config: ColregRewardConfig | None = None,
    ) -> None:
        super().__init__(
            env
        )

        self.config = (
            config
            if config is not None
            else ColregRewardConfig()
        )

        self._step_index = 0

        self._last_action = np.zeros(
            2,
            dtype=np.float64,
        )

        self._last_info: dict[str, Any] | None = None

        self._min_distance = float(
            "inf"
        )

        self._min_dcpa = float(
            "inf"
        )


    def reset(
        self,
        **kwargs,
    ):
        observation, info = self.env.reset(
            **kwargs
        )

        self._step_index = 0

        self._last_action = np.zeros(
            2,
            dtype=np.float64,
        )

        self._last_info = dict(
            info
        )

        self._min_distance = float(
            "inf"
        )

        self._min_dcpa = float(
            "inf"
        )

        info[
            "colreg_reward"
        ] = 0.0

        info[
            "env_reward"
        ] = 0.0

        info[
            "total_reward"
        ] = 0.0

        info[
            "colreg_weight"
        ] = float(
            self.config.colreg_weight
        )

        info[
            "colreg_reward_components"
        ] = {}

        return (
            observation,
            info,
        )


    def step(
        self,
        action,
    ):
        action_np = np.asarray(
            action,
            dtype=np.float64,
        )

        previous_info = (
            dict(
                self._last_info
            )
            if self._last_info is not None
            else None
        )

        (
            observation,
            env_reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(
            action
        )

        colreg_reward, components = (
            self._compute_colreg_reward(
                action=action_np,
                previous_action=self._last_action,
                previous_info=previous_info,
                current_info=info,
                terminated=terminated,
            )
        )

        total_reward = (
            float(
                env_reward
            )
            + self.config.colreg_weight
            * float(
                colreg_reward
            )
        )

        info[
            "env_reward"
        ] = float(
            env_reward
        )

        info[
            "colreg_reward"
        ] = float(
            colreg_reward
        )

        info[
            "colreg_weight"
        ] = float(
            self.config.colreg_weight
        )

        info[
            "total_reward"
        ] = float(
            total_reward
        )

        info[
            "colreg_reward_components"
        ] = components

        self._last_action = action_np.copy()

        self._last_info = dict(
            info
        )

        self._step_index += 1

        return (
            observation,
            float(
                total_reward
            ),
            terminated,
            truncated,
            info,
        )


    def _compute_colreg_reward(
        self,
        action: np.ndarray,
        previous_action: np.ndarray,
        previous_info: dict[str, Any] | None,
        current_info: dict[str, Any],
        terminated: bool,
    ) -> tuple[float, dict[str, float | bool | str]]:
        cfg = self.config


        distance = float(
            current_info[
                "distance_to_traffic"
            ]
        )

        dcpa = float(
            current_info[
                "dcpa"
            ]
        )

        tcpa = float(
            current_info[
                "tcpa"
            ]
        )

        collision = bool(
            current_info[
                "collision"
            ]
        )

        goal_reached = bool(
            current_info[
                "goal_reached"
            ]
        )

        ego_position = np.asarray(
            current_info[
                "ego_position"
            ],
            dtype=np.float64,
        )

        traffic_position = np.asarray(
            current_info[
                "traffic_position"
            ],
            dtype=np.float64,
        )

        ego_heading = float(
            current_info[
                "ego_heading"
            ]
        )

        traffic_heading = float(
            current_info[
                "traffic_heading"
            ]
        )


        self._min_distance = min(
            self._min_distance,
            distance,
        )

        self._min_dcpa = min(
            self._min_dcpa,
            dcpa,
        )


        relative_position = (
            traffic_position
            - ego_position
        )

        relative_ego = global_to_ego_frame(
            vector=relative_position,
            heading=ego_heading,
        )

        if relative_ego[
            1
        ] < 0.0:
            bearing_side = "starboard"
        else:
            bearing_side = "port"

        give_way_crossing = bool(
            bearing_side == "starboard"
            and tcpa > 0.0
            and distance < cfg.risk_distance
        )

        if give_way_crossing:
            risk_factor_distance = max(
                0.0,
                1.0
                - distance
                / cfg.risk_distance,
            )
        else:
            risk_factor_distance = 0.0


        yaw_action = float(
            action[
                1
            ]
        )

        starboard_action = max(
            0.0,
            -yaw_action,
        )

        port_action = max(
            0.0,
            yaw_action,
        )


        if (
            tcpa > 0.0
            and distance < cfg.safe_distance
        ):
            close_ratio = (
                1.0
                - distance
                / cfg.safe_distance
            )

            close_distance_term = (
                cfg.close_distance_penalty
                * close_ratio
                * close_ratio
            )
        else:
            close_distance_term = 0.0

        if (
            tcpa > 0.0
            and dcpa < cfg.safe_dcpa
        ):
            dcpa_ratio = (
                1.0
                - dcpa
                / cfg.safe_dcpa
            )

            low_dcpa_term = (
                cfg.low_dcpa_penalty
                * dcpa_ratio
                * dcpa_ratio
            )
        else:
            low_dcpa_term = 0.0


        starboard_term = (
            cfg.starboard_bonus
            * risk_factor_distance
            * starboard_action
        )

        port_term = (
            cfg.port_penalty
            * risk_factor_distance
            * port_action
        )

        if self._step_index < cfg.early_action_horizon_steps:
            early_factor = (
                1.0
                - self._step_index
                / float(
                    cfg.early_action_horizon_steps
                )
            )
        else:
            early_factor = 0.0

        early_starboard_term = (
            cfg.early_starboard_bonus
            * early_factor
            * risk_factor_distance
            * starboard_action
        )


        traffic_forward = np.array(
            [
                math.cos(
                    traffic_heading
                ),
                math.sin(
                    traffic_heading
                ),
            ],
            dtype=np.float64,
        )

        ego_from_traffic = (
            ego_position
            - traffic_position
        )

        along_track = float(
            np.dot(
                ego_from_traffic,
                traffic_forward,
            )
        )

        if give_way_crossing:
            if along_track < 0.0:
                pass_term = (
                    cfg.pass_astern_bonus
                    * risk_factor_distance
                )
            else:
                pass_term = (
                    cfg.pass_ahead_penalty
                    * risk_factor_distance
                )
        else:
            pass_term = 0.0


        action_delta = float(
            np.linalg.norm(
                action
                - previous_action
            )
        )

        action_norm = float(
            np.linalg.norm(
                action
            )
        )

        smoothness_term = (
            cfg.smoothness_penalty
            * action_delta
        )

        action_magnitude_term = (
            cfg.action_magnitude_penalty
            * action_norm
        )


        if terminated and goal_reached:
            goal_term = cfg.goal_bonus
        else:
            goal_term = 0.0

        if terminated and collision:
            collision_term = cfg.collision_penalty
        else:
            collision_term = 0.0


        if terminated and not collision:
            min_distance_ratio = min(
                self._min_distance
                / cfg.safe_distance,
                1.0,
            )

            terminal_min_distance_term = (
                cfg.terminal_min_distance_bonus
                * min_distance_ratio
            )

            min_dcpa_ratio = min(
                self._min_dcpa
                / cfg.safe_dcpa,
                1.0,
            )

            terminal_min_dcpa_term = (
                cfg.terminal_min_dcpa_bonus
                * min_dcpa_ratio
            )
        else:
            terminal_min_distance_term = 0.0
            terminal_min_dcpa_term = 0.0

        if (
            terminated
            and self._min_distance < cfg.safe_distance
        ):
            low_distance_ratio = (
                1.0
                - self._min_distance
                / cfg.safe_distance
            )

            terminal_low_distance_term = (
                cfg.terminal_low_distance_penalty
                * low_distance_ratio
                * low_distance_ratio
            )
        else:
            terminal_low_distance_term = 0.0

        if (
            terminated
            and self._min_dcpa < cfg.safe_dcpa
        ):
            low_dcpa_ratio = (
                1.0
                - self._min_dcpa
                / cfg.safe_dcpa
            )

            terminal_low_dcpa_term = (
                cfg.terminal_low_dcpa_penalty
                * low_dcpa_ratio
                * low_dcpa_ratio
            )
        else:
            terminal_low_dcpa_term = 0.0


        colreg_reward = (
            close_distance_term
            + low_dcpa_term
            + starboard_term
            + port_term
            + early_starboard_term
            + pass_term
            + smoothness_term
            + action_magnitude_term
            + goal_term
            + collision_term
            + terminal_min_distance_term
            + terminal_min_dcpa_term
            + terminal_low_distance_term
            + terminal_low_dcpa_term
        )

        components = {
            "bearing_side": bearing_side,
            "give_way_crossing": give_way_crossing,
            "risk_factor_distance": float(
                risk_factor_distance
            ),
            "distance": float(
                distance
            ),
            "dcpa": float(
                dcpa
            ),
            "tcpa": float(
                tcpa
            ),
            "min_distance_so_far": float(
                self._min_distance
            ),
            "min_dcpa_so_far": float(
                self._min_dcpa
            ),
            "yaw_action": float(
                yaw_action
            ),
            "starboard_action": float(
                starboard_action
            ),
            "port_action": float(
                port_action
            ),
            "along_track": float(
                along_track
            ),
            "close_distance_term": float(
                close_distance_term
            ),
            "low_dcpa_term": float(
                low_dcpa_term
            ),
            "starboard_term": float(
                starboard_term
            ),
            "port_term": float(
                port_term
            ),
            "early_starboard_term": float(
                early_starboard_term
            ),
            "pass_term": float(
                pass_term
            ),
            "smoothness_term": float(
                smoothness_term
            ),
            "action_magnitude_term": float(
                action_magnitude_term
            ),
            "goal_term": float(
                goal_term
            ),
            "collision_term": float(
                collision_term
            ),
            "terminal_min_distance_term": float(
                terminal_min_distance_term
            ),
            "terminal_min_dcpa_term": float(
                terminal_min_dcpa_term
            ),
            "terminal_low_distance_term": float(
                terminal_low_distance_term
            ),
            "terminal_low_dcpa_term": float(
                terminal_low_dcpa_term
            ),
            "colreg_reward": float(
                colreg_reward
            ),
        }

        return (
            float(
                colreg_reward
            ),
            components,
        )


    def config_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(
            self.config
        )
