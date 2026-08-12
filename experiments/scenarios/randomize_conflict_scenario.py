from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import math
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class ConflictPreservingConfig:
    ego_x0_min: float = -1150.0
    ego_x0_max: float = -650.0
    ego_y0_min: float = -950.0
    ego_y0_max: float = -650.0

    ego_heading_center: float = math.pi / 2.0
    ego_heading_delta: float = 0.20

    ego_speed_min: float = 4.0
    ego_speed_max: float = 4.6

    traffic_speed_min: float = 4.6
    traffic_speed_max: float = 5.4
    traffic_heading_delta: float = 0.12

    conflict_time_min: float = 100.0
    conflict_time_max: float = 160.0

    conflict_lateral_offset_min: float = -45.0
    conflict_lateral_offset_max: float = 45.0

    route_length: float = 1700.0

    min_initial_tcpa: float = 60.0
    max_initial_tcpa: float = 190.0
    max_initial_dcpa: float = 160.0
    min_initial_distance: float = 350.0
    max_initial_distance: float = 1600.0

    max_sampling_attempts: int = 200


def angle_wrap(
    angle: float,
) -> float:
    return float(
        (
            angle
            + math.pi
        )
        % (
            2.0
            * math.pi
        )
        - math.pi
    )


def heading_to_unit(
    heading: float,
) -> np.ndarray:
    return np.array(
        [
            math.cos(
                heading
            ),
            math.sin(
                heading
            ),
        ],
        dtype=np.float64,
    )


def left_normal(
    vector: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            -vector[1],
            vector[0],
        ],
        dtype=np.float64,
    )


def global_to_body_frame(
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


def compute_tcpa_dcpa(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
) -> tuple[float, float]:
    relative_position = np.asarray(
        relative_position,
        dtype=np.float64,
    )

    relative_velocity = np.asarray(
        relative_velocity,
        dtype=np.float64,
    )

    velocity_norm_squared = float(
        np.dot(
            relative_velocity,
            relative_velocity,
        )
    )

    if velocity_norm_squared < 1e-9:
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
        / velocity_norm_squared
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


def _require(
    parent: ET.Element,
    path: str,
) -> ET.Element:
    element = parent.find(
        path
    )

    if element is None:
        raise RuntimeError(
            f"No se encontró el elemento XML: {path}"
        )

    return element


def _set_text_float(
    parent: ET.Element,
    path: str,
    value: float,
    decimals: int = 6,
) -> None:
    element = _require(
        parent,
        path,
    )

    element.text = (
        f"{float(value):.{decimals}f}"
    )


def _set_state_position(
    state: ET.Element,
    x: float,
    y: float,
) -> None:
    _set_text_float(
        state,
        "position/point/x",
        x,
    )

    _set_text_float(
        state,
        "position/point/y",
        y,
    )


def _set_state_orientation(
    state: ET.Element,
    heading: float,
) -> None:
    _set_text_float(
        state,
        "orientation/exact",
        heading,
    )


def _set_state_velocity(
    state: ET.Element,
    speed: float,
) -> None:
    _set_text_float(
        state,
        "velocity/exact",
        speed,
    )


def _set_rectangle_center(
    parent: ET.Element,
    x: float,
    y: float,
) -> None:
    _set_text_float(
        parent,
        "position/rectangle/center/x",
        x,
    )

    _set_text_float(
        parent,
        "position/rectangle/center/y",
        y,
    )


def _set_rectangle_orientation(
    parent: ET.Element,
    heading: float,
) -> None:
    element = parent.find(
        "position/rectangle/orientation"
    )

    if element is not None:
        element.text = (
            f"{float(heading):.6f}"
        )


def _set_orientation_interval_full(
    parent: ET.Element,
) -> None:
    interval_start = parent.find(
        "orientation/intervalStart"
    )

    interval_end = parent.find(
        "orientation/intervalEnd"
    )

    if interval_start is not None:
        interval_start.text = "-3.141500"

    if interval_end is not None:
        interval_end.text = "3.141500"


def _get_state_time(
    state: ET.Element,
) -> int:
    value = _require(
        state,
        "time/exact",
    ).text

    return int(
        float(
            value
        )
    )


def _coerce_config(
    config: (
        ConflictPreservingConfig
        | dict[str, Any]
        | None
    ),
) -> ConflictPreservingConfig:
    if config is None:
        return ConflictPreservingConfig()

    if isinstance(
        config,
        ConflictPreservingConfig,
    ):
        return config

    if isinstance(
        config,
        dict,
    ):
        return ConflictPreservingConfig(
            **config
        )

    raise TypeError(
        "config debe ser None, dict o ConflictPreservingConfig."
    )


def _is_valid_conflict(
    params: dict[str, Any],
    cfg: ConflictPreservingConfig,
) -> bool:
    tcpa = float(
        params[
            "initial_tcpa"
        ]
    )

    dcpa = float(
        params[
            "initial_dcpa"
        ]
    )

    initial_distance = float(
        params[
            "initial_distance"
        ]
    )

    side = str(
        params[
            "initial_bearing_side"
        ]
    )

    if side != "starboard":
        return False

    if tcpa < cfg.min_initial_tcpa:
        return False

    if tcpa > cfg.max_initial_tcpa:
        return False

    if dcpa > cfg.max_initial_dcpa:
        return False

    if initial_distance < cfg.min_initial_distance:
        return False

    if initial_distance > cfg.max_initial_distance:
        return False

    return True


def sample_conflict_preserving_parameters(
    seed: int,
    config: (
        ConflictPreservingConfig
        | dict[str, Any]
        | None
    ) = None,
) -> dict[str, Any]:
    cfg = _coerce_config(
        config
    )

    rng = np.random.default_rng(
        int(
            seed
        )
    )

    last_params: dict[str, Any] | None = None

    for attempt in range(
        1,
        cfg.max_sampling_attempts
        + 1,
    ):
        ego_position = np.array(
            [
                rng.uniform(
                    cfg.ego_x0_min,
                    cfg.ego_x0_max,
                ),
                rng.uniform(
                    cfg.ego_y0_min,
                    cfg.ego_y0_max,
                ),
            ],
            dtype=np.float64,
        )

        ego_heading = angle_wrap(
            cfg.ego_heading_center
            + rng.uniform(
                -cfg.ego_heading_delta,
                cfg.ego_heading_delta,
            )
        )

        ego_speed = float(
            rng.uniform(
                cfg.ego_speed_min,
                cfg.ego_speed_max,
            )
        )

        ego_direction = heading_to_unit(
            ego_heading
        )

        conflict_time = float(
            rng.uniform(
                cfg.conflict_time_min,
                cfg.conflict_time_max,
            )
        )

        traffic_speed = float(
            rng.uniform(
                cfg.traffic_speed_min,
                cfg.traffic_speed_max,
            )
        )

        traffic_heading = angle_wrap(
            ego_heading
            + math.pi
            / 2.0
            + rng.uniform(
                -cfg.traffic_heading_delta,
                cfg.traffic_heading_delta,
            )
        )

        traffic_direction = heading_to_unit(
            traffic_heading
        )

        traffic_normal = left_normal(
            traffic_direction
        )

        lateral_offset = float(
            rng.uniform(
                cfg.conflict_lateral_offset_min,
                cfg.conflict_lateral_offset_max,
            )
        )

        conflict_point = (
            ego_position
            + ego_speed
            * conflict_time
            * ego_direction
        )

        traffic_conflict_point = (
            conflict_point
            + lateral_offset
            * traffic_normal
        )

        traffic_position = (
            traffic_conflict_point
            - traffic_speed
            * conflict_time
            * traffic_direction
        )

        goal_position = (
            ego_position
            + cfg.route_length
            * ego_direction
        )

        ego_velocity = (
            ego_speed
            * ego_direction
        )

        traffic_velocity = (
            traffic_speed
            * traffic_direction
        )

        relative_position = (
            traffic_position
            - ego_position
        )

        relative_velocity = (
            traffic_velocity
            - ego_velocity
        )

        tcpa, dcpa = compute_tcpa_dcpa(
            relative_position=relative_position,
            relative_velocity=relative_velocity,
        )

        relative_body = global_to_body_frame(
            vector=relative_position,
            heading=ego_heading,
        )

        if relative_body[1] < 0.0:
            bearing_side = "starboard"
        else:
            bearing_side = "port"

        initial_distance = float(
            np.linalg.norm(
                relative_position
            )
        )

        params = {
            "attempt": int(
                attempt
            ),
            "ego_x0": float(
                ego_position[0]
            ),
            "ego_y0": float(
                ego_position[1]
            ),
            "ego_heading": float(
                ego_heading
            ),
            "ego_speed": float(
                ego_speed
            ),
            "goal_x": float(
                goal_position[0]
            ),
            "goal_y": float(
                goal_position[1]
            ),
            "traffic_x0": float(
                traffic_position[0]
            ),
            "traffic_y0": float(
                traffic_position[1]
            ),
            "traffic_heading": float(
                traffic_heading
            ),
            "traffic_speed": float(
                traffic_speed
            ),
            "conflict_x": float(
                conflict_point[0]
            ),
            "conflict_y": float(
                conflict_point[1]
            ),
            "traffic_conflict_x": float(
                traffic_conflict_point[0]
            ),
            "traffic_conflict_y": float(
                traffic_conflict_point[1]
            ),
            "conflict_time": float(
                conflict_time
            ),
            "conflict_lateral_offset": float(
                lateral_offset
            ),
            "initial_tcpa": float(
                tcpa
            ),
            "initial_dcpa": float(
                dcpa
            ),
            "initial_distance": float(
                initial_distance
            ),
            "initial_bearing_side": bearing_side,
            "encounter_type": "crossing_starboard",
        }

        last_params = params

        if _is_valid_conflict(
            params=params,
            cfg=cfg,
        ):
            return params

    raise RuntimeError(
        "No fue posible generar un escenario con conflicto válido. "
        f"Últimos parámetros: {last_params}"
    )


def _update_traffic_trajectory(
    dynamic_obstacle: ET.Element,
    params: dict[str, Any],
) -> None:
    traffic_initial = _require(
        dynamic_obstacle,
        "initialState",
    )

    _set_state_position(
        traffic_initial,
        params[
            "traffic_x0"
        ],
        params[
            "traffic_y0"
        ],
    )

    _set_state_orientation(
        traffic_initial,
        params[
            "traffic_heading"
        ],
    )

    _set_state_velocity(
        traffic_initial,
        params[
            "traffic_speed"
        ],
    )

    trajectory = _require(
        dynamic_obstacle,
        "trajectory",
    )

    direction = heading_to_unit(
        float(
            params[
                "traffic_heading"
            ]
        )
    )

    p0 = np.array(
        [
            params[
                "traffic_x0"
            ],
            params[
                "traffic_y0"
            ],
        ],
        dtype=np.float64,
    )

    speed = float(
        params[
            "traffic_speed"
        ]
    )

    for state in trajectory.findall(
        "state"
    ):
        time_step = _get_state_time(
            state
        )

        position = (
            p0
            + speed
            * direction
            * float(
                time_step
            )
        )

        _set_state_position(
            state,
            position[0],
            position[1],
        )

        _set_state_orientation(
            state,
            params[
                "traffic_heading"
            ],
        )

        _set_state_velocity(
            state,
            speed,
        )


def _update_ego_planning_problem(
    planning_problem: ET.Element,
    params: dict[str, Any],
) -> None:
    ego_initial = _require(
        planning_problem,
        "initialState",
    )

    ego_position = np.array(
        [
            params[
                "ego_x0"
            ],
            params[
                "ego_y0"
            ],
        ],
        dtype=np.float64,
    )

    goal_position = np.array(
        [
            params[
                "goal_x"
            ],
            params[
                "goal_y"
            ],
        ],
        dtype=np.float64,
    )

    ego_heading = float(
        params[
            "ego_heading"
        ]
    )

    _set_state_position(
        ego_initial,
        ego_position[0],
        ego_position[1],
    )

    _set_state_orientation(
        ego_initial,
        ego_heading,
    )

    _set_state_velocity(
        ego_initial,
        params[
            "ego_speed"
        ],
    )

    goal_state = _require(
        planning_problem,
        "goalState",
    )

    _set_rectangle_center(
        goal_state,
        goal_position[0],
        goal_position[1],
    )

    _set_rectangle_orientation(
        goal_state,
        ego_heading,
    )

    _set_orientation_interval_full(
        goal_state
    )

    waypoints = planning_problem.findall(
        "waypoint"
    )

    if not waypoints:
        raise RuntimeError(
            "El planningProblem no contiene waypoints."
        )

    for index, waypoint in enumerate(
        waypoints
    ):
        if len(
            waypoints
        ) == 1:
            alpha = 1.0
        else:
            alpha = float(
                index
            ) / float(
                len(
                    waypoints
                )
                - 1
            )

        waypoint_position = (
            ego_position
            + alpha
            * (
                goal_position
                - ego_position
            )
        )

        _set_rectangle_center(
            waypoint,
            waypoint_position[0],
            waypoint_position[1],
        )

        _set_rectangle_orientation(
            waypoint,
            ego_heading,
        )

        _set_orientation_interval_full(
            waypoint
        )


def generate_conflict_preserving_scenario(
    base_xml_path: str | Path,
    output_xml_path: str | Path,
    seed: int,
    config: (
        ConflictPreservingConfig
        | dict[str, Any]
        | None
    ) = None,
) -> dict[str, Any]:
    cfg = _coerce_config(
        config
    )

    base_xml_path = Path(
        base_xml_path
    )

    output_xml_path = Path(
        output_xml_path
    )

    if not base_xml_path.exists():
        raise FileNotFoundError(
            f"No existe escenario base: {base_xml_path}"
        )

    output_xml_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    params = sample_conflict_preserving_parameters(
        seed=seed,
        config=cfg,
    )

    tree = ET.parse(
        base_xml_path
    )

    root = tree.getroot()

    root.set(
        "source",
        "conflict_preserving_randomization",
    )

    root.set(
        "benchmarkID",
        f"ZAM_AAA-2_20250129_T-{int(seed)}",
    )

    dynamic_obstacle = _require(
        root,
        "dynamicObstacle",
    )

    planning_problem = _require(
        root,
        "planningProblem",
    )

    _update_traffic_trajectory(
        dynamic_obstacle=dynamic_obstacle,
        params=params,
    )

    _update_ego_planning_problem(
        planning_problem=planning_problem,
        params=params,
    )

    tree.write(
        output_xml_path,
        encoding="UTF-8",
        xml_declaration=True,
    )

    return {
        "randomized": True,
        "randomization_type": "conflict_preserving",
        "seed": int(
            seed
        ),
        "base_xml_path": str(
            base_xml_path
        ),
        "output_xml_path": str(
            output_xml_path
        ),
        "config": asdict(
            cfg
        ),
        "parameters": params,
    }
