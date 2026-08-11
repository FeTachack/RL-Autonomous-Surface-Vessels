from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import math
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class RandomizedCrossingConfig:
\
\


    traffic_x0_min: float = 80.0
    traffic_x0_max: float = 180.0

    traffic_y0_min: float = -60.0
    traffic_y0_max: float = 60.0

    traffic_speed_min: float = 4.6
    traffic_speed_max: float = 5.4

    traffic_heading_center: float = math.pi
    traffic_heading_delta: float = 0.10

    ego_speed_min: float = 4.0
    ego_speed_max: float = 4.6


def _require(
    parent: ET.Element,
    path: str,
) -> ET.Element:
    element = parent.find(path)

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


def _get_state_time(
    state: ET.Element,
) -> int:
    value = _require(
        state,
        "time/exact",
    ).text

    return int(
        float(value)
    )


def _coerce_config(
    config: (
        RandomizedCrossingConfig
        | dict[str, Any]
        | None
    ),
) -> RandomizedCrossingConfig:
    if config is None:
        return RandomizedCrossingConfig()

    if isinstance(
        config,
        RandomizedCrossingConfig,
    ):
        return config

    if isinstance(
        config,
        dict,
    ):
        return RandomizedCrossingConfig(
            **config
        )

    raise TypeError(
        "config debe ser None, dict o RandomizedCrossingConfig."
    )


def sample_crossing_parameters(
    seed: int,
    config: (
        RandomizedCrossingConfig
        | dict[str, Any]
        | None
    ) = None,
) -> dict[str, float]:
\
\
\
\
\
\


    cfg = _coerce_config(
        config
    )

    rng = np.random.default_rng(
        int(seed)
    )

    traffic_x0 = float(
        rng.uniform(
            cfg.traffic_x0_min,
            cfg.traffic_x0_max,
        )
    )

    traffic_y0 = float(
        rng.uniform(
            cfg.traffic_y0_min,
            cfg.traffic_y0_max,
        )
    )

    traffic_speed = float(
        rng.uniform(
            cfg.traffic_speed_min,
            cfg.traffic_speed_max,
        )
    )

    traffic_heading = float(
        cfg.traffic_heading_center
        + rng.uniform(
            -cfg.traffic_heading_delta,
            cfg.traffic_heading_delta,
        )
    )

    ego_speed = float(
        rng.uniform(
            cfg.ego_speed_min,
            cfg.ego_speed_max,
        )
    )

    return {
        "traffic_x0": traffic_x0,
        "traffic_y0": traffic_y0,
        "traffic_speed": traffic_speed,
        "traffic_heading": traffic_heading,
        "ego_speed": ego_speed,
    }


def generate_randomized_crossing_scenario(
    base_xml_path: str | Path,
    output_xml_path: str | Path,
    seed: int,
    config: (
        RandomizedCrossingConfig
        | dict[str, Any]
        | None
    ) = None,
) -> dict[str, Any]:
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

    params = sample_crossing_parameters(
        seed=seed,
        config=cfg,
    )

    tree = ET.parse(
        base_xml_path
    )

    root = tree.getroot()

    root.set(
        "source",
        "randomized_crossing",
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


    traffic_initial = _require(
        dynamic_obstacle,
        "initialState",
    )

    _set_state_position(
        traffic_initial,
        params["traffic_x0"],
        params["traffic_y0"],
    )

    _set_state_orientation(
        traffic_initial,
        params["traffic_heading"],
    )

    _set_state_velocity(
        traffic_initial,
        params["traffic_speed"],
    )


    trajectory = _require(
        dynamic_obstacle,
        "trajectory",
    )

    direction = np.array(
        [
            math.cos(
                params["traffic_heading"]
            ),
            math.sin(
                params["traffic_heading"]
            ),
        ],
        dtype=np.float64,
    )

    p0 = np.array(
        [
            params["traffic_x0"],
            params["traffic_y0"],
        ],
        dtype=np.float64,
    )

    for state in trajectory.findall(
        "state"
    ):
        time_step = _get_state_time(
            state
        )

        position = (
            p0
            + params["traffic_speed"]
            * direction
            * float(time_step)
        )

        _set_state_position(
            state,
            position[0],
            position[1],
        )

        _set_state_orientation(
            state,
            params["traffic_heading"],
        )

        _set_state_velocity(
            state,
            params["traffic_speed"],
        )


    ego_initial = _require(
        planning_problem,
        "initialState",
    )

    _set_state_velocity(
        ego_initial,
        params["ego_speed"],
    )

    tree.write(
        output_xml_path,
        encoding="UTF-8",
        xml_declaration=True,
    )

    return {
        "randomized": True,
        "seed": int(seed),
        "base_xml_path": str(
            base_xml_path
        ),
        "output_xml_path": str(
            output_xml_path
        ),
        "config": asdict(cfg),
        "parameters": params,
    }
