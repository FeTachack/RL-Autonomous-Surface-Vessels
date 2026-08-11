from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import math
import numpy as np


def angle_wrap(
    angle: float,
) -> float:
    return float(
        math.atan2(
            math.sin(angle),
            math.cos(angle),
        )
    )


def global_to_body_frame(
    vector: np.ndarray,
    heading: float,
) -> np.ndarray:
\
\
\
\
\
\
\
\
\
\


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


def compute_tcpa_dcpa(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    eps: float = 1.0e-8,
) -> tuple[float, float]:
    speed_sq = float(
        np.dot(
            relative_velocity,
            relative_velocity,
        )
    )

    if speed_sq < eps:
        return (
            0.0,
            float(
                np.linalg.norm(
                    relative_position
                )
            ),
        )

    tcpa = -float(
        np.dot(
            relative_position,
            relative_velocity,
        )
        / speed_sq
    )

    closest_relative_position = (
        relative_position
        + tcpa
        * relative_velocity
    )

    dcpa = float(
        np.linalg.norm(
            closest_relative_position
        )
    )

    return (
        tcpa,
        dcpa,
    )


def first_index_where(
    mask: np.ndarray,
) -> int | None:
    indices = np.flatnonzero(
        mask
    )

    if indices.size == 0:
        return None

    return int(
        indices[
            0
        ]
    )


def safe_float(
    value: Any,
) -> float:
    if value is None:
        return float(
            "nan"
        )

    return float(
        value
    )


@dataclass
class ColregPreferenceConfig:
\
\
\
\
\
\


    safe_distance: float = 300.0
    safe_dcpa: float = 200.0
    risk_distance: float = 800.0


    early_action_horizon_steps: int = 90
    substantial_yaw_threshold: float = 0.20


    collision_penalty: float = -1200.0
    no_goal_penalty: float = -250.0
    goal_bonus: float = 250.0

    min_distance_weight: float = 0.80
    min_dcpa_weight: float = 1.20

    pass_astern_bonus: float = 180.0
    pass_ahead_penalty: float = -180.0

    early_action_bonus: float = 120.0
    starboard_action_bonus: float = 100.0
    port_action_penalty: float = -100.0

    smoothness_penalty: float = -35.0
    action_magnitude_penalty: float = -10.0


    minimum_score_margin: float = 25.0


@dataclass
class ColregFeatures:
    collision: bool
    goal_reached: bool
    total_reward: float
    steps: int

    min_distance: float
    min_dcpa: float
    min_tcpa: float

    initial_bearing_side: str
    crossing_give_way: bool

    closest_index: int
    pass_astern_at_closest: bool
    along_track_at_closest: float

    first_substantial_action_step: int | None
    early_action_score: float

    starboard_action_ratio: float
    port_action_ratio: float

    mean_action_norm: float
    mean_abs_yaw_action: float
    action_smoothness: float

    colreg_score: float


class ColregPreferenceScorer:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\


    def __init__(
        self,
        config: ColregPreferenceConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else ColregPreferenceConfig()
        )


    def evaluate(
        self,
        trajectory: dict[str, Any],
    ) -> ColregFeatures:
        ego_positions = np.asarray(
            trajectory[
                "ego_positions"
            ],
            dtype=np.float64,
        )

        traffic_positions = np.asarray(
            trajectory[
                "traffic_positions"
            ],
            dtype=np.float64,
        )

        ego_headings = np.asarray(
            trajectory[
                "ego_headings"
            ],
            dtype=np.float64,
        )

        traffic_headings = np.asarray(
            trajectory[
                "traffic_headings"
            ],
            dtype=np.float64,
        )

        distances = np.asarray(
            trajectory[
                "distances"
            ],
            dtype=np.float64,
        )

        dcpas = np.asarray(
            trajectory[
                "dcpas"
            ],
            dtype=np.float64,
        )

        tcpas = np.asarray(
            trajectory[
                "tcpas"
            ],
            dtype=np.float64,
        )

        actions = np.asarray(
            trajectory[
                "actions"
            ],
            dtype=np.float64,
        )

        collision = bool(
            trajectory[
                "collision"
            ]
        )

        goal_reached = bool(
            trajectory[
                "goal_reached"
            ]
        )

        total_reward = float(
            trajectory[
                "reward"
            ]
        )

        steps = int(
            trajectory[
                "steps"
            ]
        )

        min_distance = float(
            np.min(
                distances
            )
        )

        min_dcpa = float(
            np.min(
                dcpas
            )
        )

        min_tcpa = float(
            np.min(
                tcpas
            )
        )

        closest_index = int(
            np.argmin(
                distances
            )
        )

        initial_relative_position = (
            traffic_positions[
                0
            ]
            - ego_positions[
                0
            ]
        )

        initial_relative_ego = global_to_body_frame(
            initial_relative_position,
            float(
                ego_headings[
                    0
                ]
            ),
        )

        if initial_relative_ego[
            1
        ] < 0.0:
            initial_bearing_side = "starboard"
        else:
            initial_bearing_side = "port"

        initial_tcpa = float(
            tcpas[
                0
            ]
        )

        initial_dcpa = float(
            dcpas[
                0
            ]
        )

        crossing_give_way = bool(
            initial_bearing_side == "starboard"
            and initial_tcpa > 0.0
            and initial_dcpa
            < self.config.risk_distance
        )


        traffic_heading_at_closest = float(
            traffic_headings[
                closest_index
            ]
        )

        traffic_forward = np.array(
            [
                math.cos(
                    traffic_heading_at_closest
                ),
                math.sin(
                    traffic_heading_at_closest
                ),
            ],
            dtype=np.float64,
        )

        relative_ego_from_traffic = (
            ego_positions[
                closest_index
            ]
            - traffic_positions[
                closest_index
            ]
        )

        along_track_at_closest = float(
            np.dot(
                relative_ego_from_traffic,
                traffic_forward,
            )
        )


        pass_astern_at_closest = bool(
            along_track_at_closest < 0.0
        )


        if actions.size == 0:
            action_norms = np.zeros(
                1,
                dtype=np.float64,
            )

            yaw_actions = np.zeros(
                1,
                dtype=np.float64,
            )
        else:
            action_norms = np.linalg.norm(
                actions,
                axis=1,
            )

            yaw_actions = actions[
                :,
                1
            ]

        substantial_action_mask = (
            np.abs(
                yaw_actions
            )
            >= self.config.substantial_yaw_threshold
        )

        first_substantial_action_step = first_index_where(
            substantial_action_mask
        )

        if first_substantial_action_step is None:
            early_action_score = 0.0
        else:
            early_action_score = max(
                0.0,
                1.0
                - (
                    first_substantial_action_step
                    / float(
                        self.config.early_action_horizon_steps
                    )
                ),
            )

        if actions.shape[
            0
        ] >= 2:
            action_smoothness = float(
                np.mean(
                    np.linalg.norm(
                        np.diff(
                            actions,
                            axis=0,
                        ),
                        axis=1,
                    )
                )
            )
        else:
            action_smoothness = 0.0

        mean_action_norm = float(
            np.mean(
                action_norms
            )
        )

        mean_abs_yaw_action = float(
            np.mean(
                np.abs(
                    yaw_actions
                )
            )
        )


        if actions.shape[
            0
        ] > 0:
            action_distances = distances[
                1:
                1
                + actions.shape[
                    0
                ]
            ]

            action_tcpas = tcpas[
                1:
                1
                + actions.shape[
                    0
                ]
            ]

            critical_mask = np.logical_and(
                action_distances
                < self.config.risk_distance,
                action_tcpas > 0.0,
            )

            if np.any(
                critical_mask
            ):
                critical_yaw = yaw_actions[
                    critical_mask
                ]

                starboard_action_ratio = float(
                    np.mean(
                        critical_yaw < -0.05
                    )
                )

                port_action_ratio = float(
                    np.mean(
                        critical_yaw > 0.05
                    )
                )
            else:
                starboard_action_ratio = 0.0
                port_action_ratio = 0.0
        else:
            starboard_action_ratio = 0.0
            port_action_ratio = 0.0

        colreg_score = self._compute_score(
            collision=collision,
            goal_reached=goal_reached,
            min_distance=min_distance,
            min_dcpa=min_dcpa,
            crossing_give_way=crossing_give_way,
            pass_astern_at_closest=pass_astern_at_closest,
            early_action_score=early_action_score,
            starboard_action_ratio=starboard_action_ratio,
            port_action_ratio=port_action_ratio,
            mean_action_norm=mean_action_norm,
            action_smoothness=action_smoothness,
        )

        return ColregFeatures(
            collision=collision,
            goal_reached=goal_reached,
            total_reward=total_reward,
            steps=steps,
            min_distance=min_distance,
            min_dcpa=min_dcpa,
            min_tcpa=min_tcpa,
            initial_bearing_side=initial_bearing_side,
            crossing_give_way=crossing_give_way,
            closest_index=closest_index,
            pass_astern_at_closest=pass_astern_at_closest,
            along_track_at_closest=along_track_at_closest,
            first_substantial_action_step=first_substantial_action_step,
            early_action_score=early_action_score,
            starboard_action_ratio=starboard_action_ratio,
            port_action_ratio=port_action_ratio,
            mean_action_norm=mean_action_norm,
            mean_abs_yaw_action=mean_abs_yaw_action,
            action_smoothness=action_smoothness,
            colreg_score=colreg_score,
        )


    def _compute_score(
        self,
        collision: bool,
        goal_reached: bool,
        min_distance: float,
        min_dcpa: float,
        crossing_give_way: bool,
        pass_astern_at_closest: bool,
        early_action_score: float,
        starboard_action_ratio: float,
        port_action_ratio: float,
        mean_action_norm: float,
        action_smoothness: float,
    ) -> float:
        cfg = self.config

        score = 0.0

        if collision:
            score += cfg.collision_penalty

        if goal_reached:
            score += cfg.goal_bonus
        else:
            score += cfg.no_goal_penalty

        score += cfg.min_distance_weight * min_distance
        score += cfg.min_dcpa_weight * min_dcpa

        if crossing_give_way:
            if pass_astern_at_closest:
                score += cfg.pass_astern_bonus
            else:
                score += cfg.pass_ahead_penalty

            score += (
                cfg.early_action_bonus
                * early_action_score
            )

            score += (
                cfg.starboard_action_bonus
                * starboard_action_ratio
            )

            score += (
                cfg.port_action_penalty
                * port_action_ratio
            )

        score += (
            cfg.smoothness_penalty
            * action_smoothness
        )

        score += (
            cfg.action_magnitude_penalty
            * mean_action_norm
        )

        return float(
            score
        )


    def compare(
        self,
        trajectory_a: dict[str, Any],
        trajectory_b: dict[str, Any],
    ) -> dict[str, Any] | None:
        features_a = self.evaluate(
            trajectory_a
        )

        features_b = self.evaluate(
            trajectory_b
        )

        score_a = features_a.colreg_score
        score_b = features_b.colreg_score

        margin = abs(
            score_a
            - score_b
        )

        if margin < self.config.minimum_score_margin:
            return None

        if score_a > score_b:
            preferred = trajectory_a
            rejected = trajectory_b
            preferred_features = features_a
            rejected_features = features_b
        else:
            preferred = trajectory_b
            rejected = trajectory_a
            preferred_features = features_b
            rejected_features = features_a

        return {
            "preferred_policy": preferred[
                "policy_name"
            ],
            "rejected_policy": rejected[
                "policy_name"
            ],
            "preferred_seed": int(
                preferred[
                    "seed"
                ]
            ),
            "rejected_seed": int(
                rejected[
                    "seed"
                ]
            ),
            "score_margin": float(
                margin
            ),
            "preferred_features": asdict(
                preferred_features
            ),
            "rejected_features": asdict(
                rejected_features
            ),
        }
