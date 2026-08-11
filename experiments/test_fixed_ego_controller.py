from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from Pipeline.SimulationIO import SimulationIO
from Simulator.SimulatorFactory import SimulatorFactory
from commonocean.common.solution import VesselModel
from rules.common.helper import load_yaml

from experiments.controllers.external_action_controller import (
    ExternalActionController,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGURATION_PATH = PROJECT_ROOT / "src" / "configuration.yaml"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reemplaza únicamente el controlador de la embarcación ego "
            "por una acción externa fija."
        )
    )

    parser.add_argument(
        "--ego-index",
        type=int,
        default=0,
        help="Índice de la embarcación ego dentro de sim.models.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Número máximo de pasos de simulación.",
    )
    parser.add_argument(
        "--acceleration-fraction",
        type=float,
        default=0.10,
        help="Fracción de la aceleración máxima de ego.",
    )
    parser.add_argument(
        "--yaw-rate-fraction",
        type=float,
        default=0.0,
        help="Fracción del yaw rate máximo de ego.",
    )

    return parser.parse_args()


def build_simulation():
    print(f"Loading configuration: {CONFIGURATION_PATH}")

    configuration = load_yaml(str(CONFIGURATION_PATH))
    dt = float(configuration["general_simulator"]["dt"])

    simulation_factory = SimulatorFactory(dt)
    simulation_io = SimulationIO(simulation_factory)

    simulation_factory.current_configuration = configuration

    simulation_io.configure_simfac_from_config_dict(
        current_configuration_input=configuration
    )

    simulator = simulation_factory.generate_scenario()

    return simulator, simulation_io, configuration


def print_controllers(simulator, title: str) -> None:
    print(f"\n{title}")

    for index, vessel in enumerate(simulator.models):
        controller_name = type(vessel.get_controller()).__name__

        print(
            f"  model[{index}] "
            f"name={vessel} "
            f"controller={controller_name}"
        )

def detach_ego_from_builtin_collision_avoider(ego) -> bool:
\
\
\
\
\


    collision_state = getattr(
        ego,
        "reference_to_collisionstate",
        None,
    )

    if collision_state is None:
        print(
            "Advertencia: ego no tiene un CollisionState asociado. "
            "Probablemente el CollisionAvoider no está habilitado."
        )
        return False

    collision_state.sim_controlled = False
    collision_state.currently_avoiding = False

    print(
        "Ego fue excluida del CollisionAvoider interno; "
        "su movimiento depende únicamente del controlador externo."
    )

    return True


def main() -> None:
    args = parse_arguments()

    simulator, _, _ = build_simulation()

    if not simulator.models:
        raise RuntimeError(
            "La simulación no contiene embarcaciones controlables."
        )

    if not 0 <= args.ego_index < len(simulator.models):
        raise IndexError(
            f"ego-index={args.ego_index} no es válido. "
            f"La simulación contiene {len(simulator.models)} modelos."
        )

    print_controllers(
        simulator,
        title="Controladores antes del reemplazo:",
    )

    ego = simulator.models[args.ego_index]
    original_controller = ego.get_controller()

    desired_velocity = getattr(
        original_controller,
        "desired_velocity",
        None,
    )

    if desired_velocity is None:
        desired_velocity = 0.5 * float(ego.parameters.v_max)
    if ego.vessel_dynamics.vessel_model != VesselModel.YP:
        raise RuntimeError(
            "Este experimento espera que ego utilice el modelo YP. "
            f"Modelo encontrado: {ego.vessel_dynamics.vessel_model}"
        )

    acceleration = (
        args.acceleration_fraction * float(ego.parameters.a_max)
    )
    yaw_rate = (
        args.yaw_rate_fraction * float(ego.maximum_yaw_rate)
    )

    fixed_action = np.array(
        [acceleration, yaw_rate],
        dtype=np.float64,
    )

    external_controller = ExternalActionController(
        vessel=ego,
        initial_action=fixed_action,
        desired_velocity=desired_velocity,
    )


    external_controller.sim = simulator
    external_controller.initialise()


    ego.set_controller(external_controller)
    detached_from_avoider = (
        detach_ego_from_builtin_collision_avoider(ego)
    )

    print(
        "Built-in avoidance activo para ego:",
        not detached_from_avoider,
    )
    print_controllers(
        simulator,
        title="Controladores después del reemplazo:",
    )

    print("\nAcción física fija:")
    print(f"  acceleration = {fixed_action[0]:.6f}")
    print(f"  yaw_rate     = {fixed_action[1]:.6f}")

    initial_position = np.asarray(
        ego.position,
        dtype=np.float64,
    ).copy()
    initial_heading = float(ego.heading)
    initial_velocity = float(ego.velocity)

    print("\nEstado inicial de ego:")
    print(f"  position = {initial_position}")
    print(f"  heading  = {initial_heading:.6f}")
    print(f"  velocity = {initial_velocity:.6f}")

    executed_steps = 0

    try:
        for step in range(args.steps):
            if not simulator.is_running:
                print("\nEl simulador alcanzó una condición terminal.")
                break

            simulator.compute_next_state()
            executed_steps += 1

            if step == 0 or (step + 1) % 10 == 0:
                print(
                    f"step={step + 1:04d} "
                    f"time={simulator.time:8.2f} "
                    f"position={np.asarray(ego.position)} "
                    f"heading={ego.heading:8.4f} "
                    f"velocity={ego.velocity:8.4f}"
                )

    finally:
        if simulator.displayer is not None:
            simulator.displayer.close()

    final_position = np.asarray(
        ego.position,
        dtype=np.float64,
    ).copy()

    displacement = float(
        np.linalg.norm(final_position - initial_position)
    )

    print("\nResumen:")
    print(f"  steps ejecutados  = {executed_steps}")
    print(f"  llamadas controller = {external_controller.stepped}")
    print(f"  posición inicial  = {initial_position}")
    print(f"  posición final    = {final_position}")
    print(f"  desplazamiento    = {displacement:.6f}")
    print(f"  heading inicial   = {initial_heading:.6f}")
    print(f"  heading final     = {ego.heading:.6f}")
    print(f"  velocidad inicial = {initial_velocity:.6f}")
    print(f"  velocidad final   = {ego.velocity:.6f}")


if __name__ == "__main__":
    main()
