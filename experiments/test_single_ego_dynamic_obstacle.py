from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from Pipeline.SimulationIO import SimulationIO
from Simulator.SimulatorFactory import SimulatorFactory

from commonocean.common.solution import VesselModel
from commonocean.prediction.prediction import TrajectoryPrediction
from commonocean.scenario.obstacle import DynamicObstacle
from commonocean.scenario.state import YPState
from commonocean.scenario.trajectory import Trajectory

from rules.common.helper import load_yaml

from experiments.controllers.external_action_controller import (
    ExternalActionController,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGURATION_PATH = PROJECT_ROOT / "src" / "configuration.yaml"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prueba una arquitectura con una única embarcación ego "
            "controlada externamente y tráfico como DynamicObstacle."
        )
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--acceleration-fraction",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--yaw-rate-fraction",
        type=float,
        default=0.0,
    )

    return parser.parse_args()


def make_predefined_dynamic_obstacle(
    source_vessel,
    dt: float,
    steps: int,
) -> DynamicObstacle:
\
\
\
\
\


    source_controller = source_vessel.get_controller()

    desired_velocity = getattr(
        source_controller,
        "desired_velocity",
        None,
    )

    if desired_velocity is None or desired_velocity <= 0.0:
        desired_velocity = max(
            float(source_vessel.velocity),
            1.0,
        )

    speed = float(desired_velocity)

    initial_position = np.asarray(
        source_vessel.position,
        dtype=np.float64,
    ).copy()

    heading = float(source_vessel.heading)

    direction = np.array(
        [
            np.cos(heading),
            np.sin(heading),
        ],
        dtype=np.float64,
    )

    initial_state = YPState(
        position=initial_position.copy(),
        orientation=heading,
        velocity=speed,
        time_step=0,
    )

    state_list = []

    for k in range(1, steps + 1):
        elapsed_time = k * dt

        position = (
            initial_position
            + direction * speed * elapsed_time
        )

        state = YPState(
            position=position,
            orientation=heading,
            velocity=speed,
            time_step=k,
        )

        state_list.append(state)

    trajectory = Trajectory(
        1,
        state_list,
    )

    shape = copy.deepcopy(
        source_vessel.dynamic_obstacle.obstacle_shape
    )

    prediction = TrajectoryPrediction(
        trajectory=trajectory,
        shape=shape,
    )

    return DynamicObstacle(
        obstacle_id=source_vessel.dynamic_obstacle.obstacle_id,
        obstacle_type=source_vessel.dynamic_obstacle.obstacle_type,
        obstacle_shape=shape,
        initial_state=initial_state,
        prediction=prediction,
    )


def build_single_ego_simulation(steps: int):
    configuration = load_yaml(
        str(CONFIGURATION_PATH)
    )


    configuration["general_simulator"][
        "using_collision_avoider"
    ] = False


    configuration["general_simulator"][
        "using_collision_detection"
    ] = False

    dt = float(
        configuration["general_simulator"]["dt"]
    )

    simulation_factory = SimulatorFactory(dt)
    simulation_io = SimulationIO(simulation_factory)

    simulation_factory.current_configuration = configuration

    simulation_io.configure_simfac_from_config_dict(
        current_configuration_input=configuration
    )

    if len(simulation_factory.models) < 2:
        raise RuntimeError(
            "El escenario inicial debe contener al menos "
            "dos SurfaceVessels para realizar esta conversión."
        )

    print("\nArquitectura original:")
    print(
        f"  SurfaceVessels = "
        f"{len(simulation_factory.models)}"
    )
    print(
        f"  DynamicObstacles = "
        f"{len(simulation_factory.dynamic_obstacles)}"
    )

    ego_template = simulation_factory.models[0]
    traffic_template = simulation_factory.models[1]

    traffic_obstacle = make_predefined_dynamic_obstacle(
        source_vessel=traffic_template,
        dt=dt,
        steps=steps,
    )


    simulation_factory.models = [
        ego_template,
    ]

    simulation_factory.dynamic_obstacles = [
        traffic_obstacle,
    ]

    simulation_factory.with_avoidance = False
    simulation_factory.with_collision = False

    simulator = simulation_factory.generate_scenario()

    return simulator


def install_external_controller(
    simulator,
    acceleration_fraction: float,
    yaw_rate_fraction: float,
):
    if len(simulator.models) != 1:
        raise RuntimeError(
            "La prueba espera exactamente una SurfaceVessel."
        )

    ego = simulator.models[0]

    if ego.vessel_dynamics.vessel_model != VesselModel.YP:
        raise RuntimeError(
            "La embarcación ego debe utilizar el modelo YP."
        )

    original_controller = ego.get_controller()

    desired_velocity = getattr(
        original_controller,
        "desired_velocity",
        None,
    )

    if desired_velocity is None:
        desired_velocity = (
            0.5 * float(ego.parameters.v_max)
        )

    acceleration = (
        acceleration_fraction
        * float(ego.parameters.a_max)
    )

    yaw_rate = (
        yaw_rate_fraction
        * float(ego.maximum_yaw_rate)
    )

    fixed_action = np.array(
        [
            acceleration,
            yaw_rate,
        ],
        dtype=np.float64,
    )

    controller = ExternalActionController(
        vessel=ego,
        initial_action=fixed_action,
        desired_velocity=desired_velocity,
    )

    controller.sim = simulator
    controller.initialise()

    ego.set_controller(controller)

    return ego, controller, fixed_action


def main() -> None:
    args = parse_arguments()

    simulator = build_single_ego_simulation(
        steps=args.steps
    )

    ego, controller, fixed_action = (
        install_external_controller(
            simulator,
            acceleration_fraction=(
                args.acceleration_fraction
            ),
            yaw_rate_fraction=(
                args.yaw_rate_fraction
            ),
        )
    )

    print("\nArquitectura final:")
    print(
        f"  SurfaceVessels controladas = "
        f"{len(simulator.models)}"
    )
    print(
        f"  DynamicObstacles = "
        f"{len(simulator.dynamic_obstacles)}"
    )
    print(
        f"  controlador ego = "
        f"{type(ego.get_controller()).__name__}"
    )

    print("\nAcción ego:")
    print(
        f"  acceleration = "
        f"{fixed_action[0]:.6f}"
    )
    print(
        f"  yaw_rate     = "
        f"{fixed_action[1]:.6f}"
    )

    traffic = simulator.dynamic_obstacles[0]

    print("\nEstado inicial:")
    print(
        f"  ego position    = "
        f"{np.asarray(ego.position)}"
    )
    print(
        f"  traffic position = "
        f"{traffic.initial_state.position}"
    )
    print(
        f"  traffic velocity = "
        f"{traffic.initial_state.velocity:.6f}"
    )

    executed_steps = 0

    try:
        for step in range(args.steps):
            if not simulator.is_running:
                break

            simulator.compute_next_state()
            executed_steps += 1

            time_step = step + 1

            traffic_state = traffic.state_at_time(
                time_step
            )

            if (
                step == 0
                or time_step % 10 == 0
            ):
                print(
                    f"step={time_step:04d} "
                    f"ego={np.asarray(ego.position)} "
                    f"traffic="
                    f"{np.asarray(traffic_state.position)}"
                )

    finally:
        if simulator.displayer is not None:
            simulator.displayer.close()

    final_traffic_state = traffic.state_at_time(
        executed_steps
    )

    print("\nResumen:")
    print(
        f"  steps ejecutados       = "
        f"{executed_steps}"
    )
    print(
        f"  llamadas controller    = "
        f"{controller.stepped}"
    )
    print(
        f"  SurfaceVessels         = "
        f"{len(simulator.models)}"
    )
    print(
        f"  DynamicObstacles       = "
        f"{len(simulator.dynamic_obstacles)}"
    )
    print(
        f"  controlador ego        = "
        f"{type(ego.get_controller()).__name__}"
    )
    print(
        f"  posición final ego     = "
        f"{np.asarray(ego.position)}"
    )
    print(
        f"  posición final traffic = "
        f"{np.asarray(final_traffic_state.position)}"
    )


if __name__ == "__main__":
    main()
