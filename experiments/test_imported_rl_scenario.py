from __future__ import annotations

from pathlib import Path
import copy

from commonocean.prediction.prediction import TrajectoryPrediction
from commonocean.scenario.obstacle import DynamicObstacle
from commonocean.scenario.state import YPState
from commonocean.scenario.trajectory import Trajectory
import numpy as np

from Pipeline.SimulationIO import SimulationIO
from Simulator.SimulatorFactory import SimulatorFactory
from commonocean.common.solution import VesselModel
from rules.common.helper import load_yaml
from experiments.controllers.external_action_controller import (
    ExternalActionController,
)
from CollisionHandling.CollisionDetector import CollisionDetector

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIGURATION_PATH = (
    PROJECT_ROOT / "src" / "configuration.yaml"
)

SCENARIO_PATH = (
        "/experiments/scenarios/one_ego_one_traffic.xml"
)



def force_collision_scenario(
    factory,
    trajectory_steps: int = 100,
) -> None:
    """
    Sustituye temporalmente el tráfico importado por un
    DynamicObstacle estacionario situado sobre la posición
    inicial de ego.

    Se usa únicamente para validar CollisionDetector.
    """

    if len(factory.models) != 1:
        raise RuntimeError(
            "Se esperaba exactamente una ego."
        )

    if len(factory.dynamic_obstacles) < 1:
        raise RuntimeError(
            "Se esperaba al menos un DynamicObstacle."
        )

    ego = factory.models[0]
    traffic_template = factory.dynamic_obstacles[0]

    position = np.asarray(
        ego.position,
        dtype=np.float64,
    ).copy()

    heading = float(ego.heading)

    shape = copy.deepcopy(
        traffic_template.obstacle_shape
    )

    initial_state = YPState(
        position=position.copy(),
        orientation=heading,
        velocity=0.0,
        time_step=0,
    )

    state_list = []

    for k in range(1, trajectory_steps + 1):
        state_list.append(
            YPState(
                position=position.copy(),
                orientation=heading,
                velocity=0.0,
                time_step=k,
            )
        )

    trajectory = Trajectory(
        1,
        state_list,
    )

    prediction = TrajectoryPrediction(
        trajectory=trajectory,
        shape=shape,
    )

    forced_obstacle = DynamicObstacle(
        obstacle_id=traffic_template.obstacle_id,
        obstacle_type=traffic_template.obstacle_type,
        obstacle_shape=shape,
        initial_state=initial_state,
        prediction=prediction,
    )

    factory.dynamic_obstacles = [
        forced_obstacle
    ]

    print("\nEscenario de colisión forzada:")
    print(
        "  ego position    =",
        np.asarray(ego.position),
    )
    print(
        "  traffic position =",
        np.asarray(initial_state.position),
    )

def build_simulation():
    configuration = load_yaml(
        str(CONFIGURATION_PATH)
    )

    # -----------------------------------------------------
    # Importar nuestro escenario CommonOcean
    # -----------------------------------------------------

    import_config = configuration[
        "scenario_selection"
    ]["import_scenario"]

    import_config["use_imported_scenario"] = True
    import_config["scenario_filepath"] = SCENARIO_PATH

    # Usar el mismo conjunto de parámetros para ego.
    import_config["vessel_type"] = 1
    import_config["vessel_type_by_id"] = None
    import_config["controller_type"] = "mpc"

    # RL tomará las decisiones de evasión.
    configuration["general_simulator"][
        "using_collision_avoider"
    ] = False

    # Primero validamos únicamente importación + arquitectura.
    configuration["general_simulator"][
        "using_collision_detection"
    ] = True

    configuration["general_simulator"][
        "using_displayer"
    ] = False

    configuration["general_simulator"][
        "plotting"
    ]["do_plotting"] = False

    dt = float(
        configuration["general_simulator"]["dt"]
    )

    factory = SimulatorFactory(dt)

    simulation_io = SimulationIO(factory)

    factory.current_configuration = configuration

    simulation_io.configure_simfac_from_config_dict(
        current_configuration_input=configuration
    )

    force_collision_scenario(factory)

    print("\nDespués de importar XML:")
    print(
        f"  factory.models = "
        f"{len(factory.models)}"
    )
    print(
        f"  factory.dynamic_obstacles = "
        f"{len(factory.dynamic_obstacles)}"
    )
    print("\nDEBUG antes de generate_scenario:")
    print(
        "  config collision detection =",
        configuration["general_simulator"][
            "using_collision_detection"
    ]   ,
    )
    print(
        "  factory.with_collision =",
        factory.with_collision,
    )
    print(
        "  factory models =",
        len(factory.models),
    )
    print(
        "  factory dynamic obstacles =",
        len(factory.dynamic_obstacles),
    )
    print(
        "  factory static obstacles =",
        len(factory.obstacles),
    )

    def mark_collision(vehicle, other_object, sim):
        sim.rl_collision_occurred = True
        sim.rl_collision_vehicle = vehicle
        sim.rl_collision_object = other_object


    factory.collision_methods.append(mark_collision)
    simulator = factory.generate_scenario()
    
    simulator.rl_collision_occurred = False
    simulator.rl_collision_vehicle = None
    simulator.rl_collision_object = None





    print("\nDEBUG después de generate_scenario:")
    print(
        "  listeners =",
        [type(listener).__name__ for listener in simulator.listeners],
    )
    print(
        "  dynamic obstacles =",
        len(simulator.dynamic_obstacles),
    )




    return simulator


def install_external_controller(simulator):
    if len(simulator.models) != 1:
        raise RuntimeError(
            "Se esperaba exactamente una SurfaceVessel."
        )

    ego = simulator.models[0]

    if ego.vessel_dynamics.vessel_model != VesselModel.YP:
        raise RuntimeError(
            "Ego debe utilizar el modelo YP."
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

    action = np.array(
        [
            0.05 * float(ego.parameters.a_max),
            0.0,
        ],
        dtype=np.float64,
    )

    controller = ExternalActionController(
        vessel=ego,
        initial_action=action,
        desired_velocity=desired_velocity,
    )

    controller.sim = simulator
    controller.initialise()

    ego.set_controller(controller)

    return ego, controller


def main():
    
    simulator = build_simulation()

    ego, controller = install_external_controller(
        simulator
    )
    collision_detector = next(
        (
            listener
            for listener in simulator.listeners
                if type(listener).__name__ == "CollisionDetector"
        ),
        None,
    )

    if collision_detector is None:
        raise RuntimeError(
            "CollisionDetector no fue instalado en el simulador."
        )

    print("\nCollision detection:")
    print("  CollisionDetector = activo")
    print("\nArquitectura importada:")
    print(
        f"  SurfaceVessels = "
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

    print("\nEstado inicial:")
    print(
        f"  ego = "
        f"{np.asarray(ego.position)}"
    )

    for index, obstacle in enumerate(
        simulator.dynamic_obstacles
    ):
        print(
            f"  traffic[{index}] = "
            f"{np.asarray(obstacle.initial_state.position)}"
        )

    executed_steps = 0
    collision_step = None
    for step in range(100):
        if not simulator.is_running:
            break

        simulator.compute_next_state()
        executed_steps += 1

        if (
                simulator.rl_collision_occurred
                and collision_step is None
        ):
            collision_step = step + 1

            print(
                "\nCOLISIÓN DETECTADA "
                f"en step={collision_step}"
            )


        if (
            step == 0
            or (step + 1) % 10 == 0
        ):
            print(
                f"step={step + 1:04d} "
                f"ego={np.asarray(ego.position)}"
            )

    print("\nResumen:")
    print(f"  steps ejecutados    = {executed_steps}")
    print(f"  controller calls    = {controller.stepped}")
    print(f"  SurfaceVessels      = {len(simulator.models)}")
    print(f"  DynamicObstacles    = {len(simulator.dynamic_obstacles)}")
    print(f"  controlador ego     = {type(ego.get_controller()).__name__}")
    print(f"  collision step     = {collision_step}")
    print(f"  simulator running  = {simulator.is_running}")
    print(
        f"  collision detected = "
        f"{simulator.rl_collision_occurred}"
    )

if __name__ == "__main__":
    main()
