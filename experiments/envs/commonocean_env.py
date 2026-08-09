from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from CollisionHandling.CollisionDetector import CollisionDetector
from Pipeline.SimulationIO import SimulationIO
from Simulator.SimulatorFactory import SimulatorFactory

from commonocean.common.solution import VesselModel

from rules.common.helper import load_yaml

from experiments.controllers.external_action_controller import (
    ExternalActionController,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIGURATION_PATH = (
    PROJECT_ROOT
    / "src"
    / "configuration.yaml"
)

# SimulationIO construye internamente la ruta al escenario.
SCENARIO_PATH = (
    "/experiments/scenarios/one_ego_one_traffic.xml"
)


# ============================================================
# Environment
# ============================================================


class CommonOceanEnv(gym.Env):
    """
    Entorno Gymnasium para navegación de una embarcación ego
    usando CommonOcean-Sim.

    Arquitectura
    ------------
    planningProblem
        -> SurfaceVessel ego
        -> ExternalActionController
        -> política RL

    dynamicObstacle
        -> tráfico marítimo
        -> trayectoria predefinida

    CollisionDetector
        -> detección de colisiones

    CollisionAvoider
        -> desactivado

    Acción
    ------
    action[0] ∈ [-1, 1]
        aceleración longitudinal normalizada

    action[1] ∈ [-1, 1]
        yaw rate normalizado

    Observación
    -----------
    Vector de 13 variables:

    0   ego_speed
    1   goal_distance
    2   sin(goal_bearing)
    3   cos(goal_bearing)

    4   traffic_distance
    5   sin(traffic_bearing)
    6   cos(traffic_bearing)

    7   relative_velocity_x
    8   relative_velocity_y

    9   sin(relative_heading)
    10  cos(relative_heading)

    11  DCPA
    12  TCPA

    Todas las variables entregadas a la política son
    normalizadas/clipped en [-1, 1].
    """

    metadata = {
        "render_modes": [],
    }

    # ========================================================
    # Initialization
    # ========================================================

    def __init__(
        self,
        max_episode_steps: int = 300,
        collision_penalty: float = -100.0,
        goal_reward: float = 100.0,
        distance_scale: float = 2500.0,
        tcpa_scale: float = 300.0,
    ) -> None:
        super().__init__()

        # ----------------------------------------------------
        # Episode configuration
        # ----------------------------------------------------

        self.max_episode_steps = int(
            max_episode_steps
        )

        self.collision_penalty = float(
            collision_penalty
        )

        self.goal_reward = float(
            goal_reward
        )

        # ----------------------------------------------------
        # Normalization scales
        # ----------------------------------------------------

        self.distance_scale = float(
            distance_scale
        )

        self.tcpa_scale = float(
            tcpa_scale
        )

        # ----------------------------------------------------
        # Gymnasium action space
        # ----------------------------------------------------

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Gymnasium observation space
        # ----------------------------------------------------

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(13,),
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # CommonOcean runtime objects
        # ----------------------------------------------------

        self.factory = None
        self.simulation_io = None
        self.simulator = None

        self.ego = None
        self.traffic = None

        self.controller = None
        self.collision_detector = None

        self.goal_position = None

        # ----------------------------------------------------
        # Episode state
        # ----------------------------------------------------

        self._elapsed_steps = 0
        self._episode_done = False

        self._last_physical_action = np.zeros(
            2,
            dtype=np.float64,
        )

        # Necesaria para calcular progreso hacia el objetivo.
        self._previous_distance_to_goal = None

    # ========================================================
    # Gymnasium API
    # ========================================================

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """
        Reinicia completamente CommonOcean y crea un nuevo
        episodio.
        """

        super().reset(seed=seed)

        # Limpiar simulación anterior.
        self.close()

        self._elapsed_steps = 0
        self._episode_done = False

        # El episodio siempre comienza sin acción previa.
        self._last_physical_action = np.zeros(
            2,
            dtype=np.float64,
        )

        # Crear nuevamente CommonOcean.
        self._build_simulator()

        # ----------------------------------------------------
        # Inicializar distancia anterior al objetivo
        # ----------------------------------------------------

        self._previous_distance_to_goal = float(
            np.linalg.norm(
                self.goal_position
                - np.asarray(
                    self.ego.position,
                    dtype=np.float64,
                )
            )
        )

        observation = self._get_observation()
        info = self._get_info()

        if not self.observation_space.contains(
            observation
        ):
            raise RuntimeError(
                "La observación inicial no pertenece "
                "al observation_space.\n"
                f"Observation = {observation}"
            )

        return observation, info

    # --------------------------------------------------------

    def step(
        self,
        action: np.ndarray,
    ):
        """
        Ejecuta exactamente un paso del simulador.
        """

        if self.simulator is None:
            raise RuntimeError(
                "Debe llamar env.reset() antes de env.step()."
            )

        if self._episode_done:
            raise RuntimeError(
                "El episodio ya terminó. "
                "Debe llamar env.reset() antes de continuar."
            )

        # ====================================================
        # Validate action
        # ====================================================

        normalized_action = np.asarray(
            action,
            dtype=np.float32,
        )

        if normalized_action.shape != (2,):
            raise ValueError(
                "La acción debe tener shape (2,), "
                f"pero se recibió "
                f"{normalized_action.shape}."
            )

        if not np.all(
            np.isfinite(normalized_action)
        ):
            raise ValueError(
                "La acción contiene valores no finitos."
            )

        normalized_action = np.clip(
            normalized_action,
            -1.0,
            1.0,
        )

        # ====================================================
        # RL action -> physical action
        # ====================================================

        physical_action = (
            self._map_action_to_physical(
                normalized_action
            )
        )

        self._last_physical_action = (
            physical_action.copy()
        )

        self.controller.set_action(
            physical_action
        )

        # ====================================================
        # CommonOcean step
        # ====================================================

        self.simulator.compute_next_state()

        self._elapsed_steps += 1

        # ====================================================
        # Termination
        # ====================================================

        collision = bool(
            self.simulator.rl_collision_occurred
        )

        goal_reached = bool(
            self.ego.journey_finished
        )

        # Final natural del problema.
        terminated = bool(
            collision
            or goal_reached
        )

        # Final por horizonte temporal.
        truncated = bool(
            self._elapsed_steps
            >= self.max_episode_steps
            and not terminated
        )

        # ====================================================
        # Observation and info
        # ====================================================

        observation = self._get_observation()

        info = self._get_info()

        # ====================================================
        # Reward
        # ====================================================

        reward, reward_components = (
            self._compute_reward(
                collision=collision,
                goal_reached=goal_reached,
                info=info,
            )
        )

        info["reward_components"] = (
            reward_components
        )

        # ====================================================
        # Episode status
        # ====================================================

        self._episode_done = bool(
            terminated
            or truncated
        )

        if not self.observation_space.contains(
            observation
        ):
            raise RuntimeError(
                "La observación producida por step() "
                "no pertenece al observation_space.\n"
                f"Observation = {observation}"
            )

        return (
            observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    # --------------------------------------------------------

    def close(self):
        """
        Libera referencias de la simulación actual.
        """

        if self.simulator is not None:
            displayer = getattr(
                self.simulator,
                "displayer",
                None,
            )

            if displayer is not None:
                try:
                    displayer.close()
                except Exception:
                    pass

        self.factory = None
        self.simulation_io = None
        self.simulator = None

        self.ego = None
        self.traffic = None

        self.controller = None
        self.collision_detector = None

        self.goal_position = None

        self._previous_distance_to_goal = None

    # ========================================================
    # CommonOcean setup
    # ========================================================

    def _build_simulator(
        self,
    ) -> None:
        """
        Construye una simulación CommonOcean nueva para el
        episodio.
        """

        configuration = load_yaml(
            str(CONFIGURATION_PATH)
        )

        # ====================================================
        # Imported scenario
        # ====================================================

        import_config = configuration[
            "scenario_selection"
        ]["import_scenario"]

        import_config[
            "use_imported_scenario"
        ] = True

        import_config[
            "scenario_filepath"
        ] = SCENARIO_PATH

        import_config[
            "vessel_type"
        ] = 1

        import_config[
            "vessel_type_by_id"
        ] = None

        # Se crea inicialmente con MPC, pero inmediatamente
        # después lo sustituimos por ExternalActionController.
        import_config[
            "controller_type"
        ] = "mpc"

        # ====================================================
        # Simulator configuration
        # ====================================================

        configuration["general_simulator"][
            "using_collision_avoider"
        ] = False

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
            configuration[
                "general_simulator"
            ]["dt"]
        )

        # ====================================================
        # Factory
        # ====================================================

        factory = SimulatorFactory(
            dt
        )

        simulation_io = SimulationIO(
            factory
        )

        factory.current_configuration = (
            configuration
        )

        # ====================================================
        # Collision callback for RL
        # ====================================================

        def mark_collision(
            vehicle,
            other_object,
            sim,
        ):
            sim.rl_collision_occurred = True
            sim.rl_collision_vehicle = vehicle
            sim.rl_collision_object = (
                other_object
            )

        factory.collision_methods.append(
            mark_collision
        )

        # ====================================================
        # Load scenario
        # ====================================================

        simulation_io.configure_simfac_from_config_dict(
            current_configuration_input=configuration
        )

        # ====================================================
        # Validate architecture
        # ====================================================

        if len(factory.models) != 1:
            raise RuntimeError(
                "El entorno espera exactamente "
                "1 SurfaceVessel ego, pero encontró "
                f"{len(factory.models)}."
            )

        if len(factory.dynamic_obstacles) != 1:
            raise RuntimeError(
                "Esta versión espera exactamente "
                "1 DynamicObstacle, pero encontró "
                f"{len(factory.dynamic_obstacles)}."
            )

        # ====================================================
        # Generate simulator
        # ====================================================

        simulator = (
            factory.generate_scenario()
        )

        simulator.rl_collision_occurred = False
        simulator.rl_collision_vehicle = None
        simulator.rl_collision_object = None

        # ====================================================
        # Ego
        # ====================================================

        ego = simulator.models[0]

        if (
            ego.vessel_dynamics.vessel_model
            != VesselModel.YP
        ):
            raise RuntimeError(
                "La embarcación ego debe utilizar "
                "VesselModel.YP."
            )

        # ====================================================
        # Goal
        # ====================================================

        if (
            ego.waypoints is None
            or len(ego.waypoints) == 0
        ):
            raise RuntimeError(
                "La embarcación ego no tiene waypoints."
            )

        goal_position = np.asarray(
            ego.waypoints[-1],
            dtype=np.float64,
        ).copy()

        # ====================================================
        # Replace MPC controller
        # ====================================================

        original_controller = (
            ego.get_controller()
        )

        desired_velocity = getattr(
            original_controller,
            "desired_velocity",
            None,
        )

        if desired_velocity is None:
            desired_velocity = (
                0.5
                * float(
                    ego.parameters.v_max
                )
            )

        external_controller = (
            ExternalActionController(
                vessel=ego,
                initial_action=np.zeros(
                    2,
                    dtype=np.float64,
                ),
                desired_velocity=(
                    desired_velocity
                ),
            )
        )

        external_controller.sim = simulator

        external_controller.initialise()

        ego.set_controller(
            external_controller
        )

        # ====================================================
        # Collision detector
        # ====================================================

        collision_detector = next(
            (
                listener
                for listener
                in simulator.listeners
                if isinstance(
                    listener,
                    CollisionDetector,
                )
            ),
            None,
        )

        if collision_detector is None:
            raise RuntimeError(
                "CollisionDetector no fue instalado."
            )

        # ====================================================
        # Traffic
        # ====================================================

        traffic = (
            simulator.dynamic_obstacles[0]
        )

        if traffic.prediction is None:
            raise RuntimeError(
                "El DynamicObstacle no tiene "
                "una trayectoria prediction."
            )

        traffic_states = (
            traffic
            .prediction
            .trajectory
            .state_list
        )

        if not traffic_states:
            raise RuntimeError(
                "La trayectoria del DynamicObstacle "
                "está vacía."
            )

        last_traffic_time_step = int(
            traffic_states[-1].time_step
        )

        # No permitimos que el episodio continúe después
        # de que termine la trayectoria conocida del tráfico.
        if (
            self.max_episode_steps
            > last_traffic_time_step
        ):
            raise RuntimeError(
                "max_episode_steps excede la trayectoria "
                "del DynamicObstacle: "
                f"{self.max_episode_steps} > "
                f"{last_traffic_time_step}"
            )

        # ====================================================
        # Save references
        # ====================================================

        self.factory = factory

        self.simulation_io = (
            simulation_io
        )

        self.simulator = simulator

        self.ego = ego

        self.traffic = traffic

        self.controller = (
            external_controller
        )

        self.collision_detector = (
            collision_detector
        )

        self.goal_position = (
            goal_position
        )

    # ========================================================
    # Actions
    # ========================================================

    def _map_action_to_physical(
        self,
        action: np.ndarray,
    ) -> np.ndarray:
        """
        Convierte la acción RL normalizada a unidades físicas.

        action[0]
            -> aceleración longitudinal [m/s²]

        action[1]
            -> yaw rate [rad/s]
        """

        acceleration = (
            float(action[0])
            * float(
                self.ego.parameters.a_max
            )
        )

        yaw_rate = (
            float(action[1])
            * float(
                self.ego.maximum_yaw_rate
            )
        )

        return np.array(
            [
                acceleration,
                yaw_rate,
            ],
            dtype=np.float64,
        )

    # ========================================================
    # Coordinate transformations
    # ========================================================

    def _global_to_ego_frame(
        self,
        vector: np.ndarray,
    ) -> np.ndarray:
        """
        Transforma un vector global al marco local de ego.

        Marco local:

            +x = delante de la embarcación
            +y = babor
        """

        psi = float(
            self.ego.heading
        )

        c = np.cos(psi)
        s = np.sin(psi)

        rotation = np.array(
            [
                [c, s],
                [-s, c],
            ],
            dtype=np.float64,
        )

        vector = np.asarray(
            vector,
            dtype=np.float64,
        )

        return rotation @ vector

    # ========================================================
    # CPA / TCPA
    # ========================================================

    def _compute_cpa(
        self,
        relative_position: np.ndarray,
        relative_velocity: np.ndarray,
    ) -> tuple[float, float]:
        """
        Calcula Distance at Closest Point of Approach (DCPA)
        y Time to Closest Point of Approach (TCPA).

        Se asume velocidad constante instantánea.
        """

        relative_position = np.asarray(
            relative_position,
            dtype=np.float64,
        )

        relative_velocity = np.asarray(
            relative_velocity,
            dtype=np.float64,
        )

        relative_speed_squared = float(
            np.dot(
                relative_velocity,
                relative_velocity,
            )
        )

        # Velocidad relativa prácticamente nula.
        if relative_speed_squared < 1e-8:
            return (
                float(
                    np.linalg.norm(
                        relative_position
                    )
                ),
                0.0,
            )

        tcpa = -float(
            np.dot(
                relative_position,
                relative_velocity,
            )
        ) / relative_speed_squared

        closest_position = (
            relative_position
            + tcpa * relative_velocity
        )

        dcpa = float(
            np.linalg.norm(
                closest_position
            )
        )

        return dcpa, tcpa

    # ========================================================
    # Traffic
    # ========================================================

    def _get_traffic_state(
        self,
    ):
        """
        Obtiene el estado del DynamicObstacle correspondiente
        al time_step actual.

        No congelamos artificialmente el tráfico si la
        trayectoria se termina.
        """

        time_step = int(
            self.ego.state.time_step
        )

        state = (
            self.traffic.state_at_time(
                time_step
            )
        )

        if state is not None:
            return state

        prediction = (
            self.traffic.prediction
        )

        if prediction is None:
            raise RuntimeError(
                "DynamicObstacle no tiene prediction."
            )

        states = (
            prediction
            .trajectory
            .state_list
        )

        if not states:
            raise RuntimeError(
                "La trayectoria del DynamicObstacle "
                "está vacía."
            )

        raise RuntimeError(
            "La simulación alcanzó un time_step "
            "para el cual el DynamicObstacle no "
            "tiene trayectoria. "
            f"time_step={time_step}, "
            f"último time_step="
            f"{states[-1].time_step}"
        )

    # ========================================================
    # Observations
    # ========================================================

    def _get_observation(
        self,
    ) -> np.ndarray:
        """
        Construye la observación egocéntrica de 13 variables.
        """

        # ====================================================
        # Ego state
        # ====================================================

        ego_position = np.asarray(
            self.ego.position,
            dtype=np.float64,
        )

        ego_heading = float(
            self.ego.heading
        )

        ego_speed = float(
            self.ego.velocity
        )

        # ====================================================
        # Traffic state
        # ====================================================

        traffic_state = (
            self._get_traffic_state()
        )

        traffic_position = np.asarray(
            traffic_state.position,
            dtype=np.float64,
        )

        traffic_heading = float(
            traffic_state.orientation
        )

        traffic_speed = float(
            traffic_state.velocity
        )

        # ====================================================
        # Goal relative to ego
        # ====================================================

        goal_vector_global = (
            self.goal_position
            - ego_position
        )

        goal_vector_local = (
            self._global_to_ego_frame(
                goal_vector_global
            )
        )

        goal_distance = float(
            np.linalg.norm(
                goal_vector_local
            )
        )

        goal_bearing = float(
            np.arctan2(
                goal_vector_local[1],
                goal_vector_local[0],
            )
        )

        # ====================================================
        # Traffic relative position
        # ====================================================

        relative_position_global = (
            traffic_position
            - ego_position
        )

        relative_position_local = (
            self._global_to_ego_frame(
                relative_position_global
            )
        )

        traffic_distance = float(
            np.linalg.norm(
                relative_position_local
            )
        )

        traffic_bearing = float(
            np.arctan2(
                relative_position_local[1],
                relative_position_local[0],
            )
        )

        # ====================================================
        # Global velocity vectors
        # ====================================================

        ego_velocity_global = np.array(
            [
                ego_speed
                * np.cos(
                    ego_heading
                ),

                ego_speed
                * np.sin(
                    ego_heading
                ),
            ],
            dtype=np.float64,
        )

        traffic_velocity_global = np.array(
            [
                traffic_speed
                * np.cos(
                    traffic_heading
                ),

                traffic_speed
                * np.sin(
                    traffic_heading
                ),
            ],
            dtype=np.float64,
        )

        relative_velocity_global = (
            traffic_velocity_global
            - ego_velocity_global
        )

        relative_velocity_local = (
            self._global_to_ego_frame(
                relative_velocity_global
            )
        )

        # ====================================================
        # CPA / TCPA
        # ====================================================

        dcpa, tcpa = self._compute_cpa(
            relative_position_global,
            relative_velocity_global,
        )

        # ====================================================
        # Relative heading
        # ====================================================

        relative_heading = float(
            np.arctan2(
                np.sin(
                    traffic_heading
                    - ego_heading
                ),
                np.cos(
                    traffic_heading
                    - ego_heading
                ),
            )
        )

        # ====================================================
        # Normalization
        # ====================================================

        speed_scale = max(
            float(
                self.ego.parameters.v_max
            ),
            1e-6,
        )

        relative_speed_scale = (
            2.0 * speed_scale
        )

        observation = np.array(
            [
                # 0 - ego speed
                ego_speed
                / speed_scale,

                # 1 - goal distance
                goal_distance
                / self.distance_scale,

                # 2
                np.sin(
                    goal_bearing
                ),

                # 3
                np.cos(
                    goal_bearing
                ),

                # 4 - traffic distance
                traffic_distance
                / self.distance_scale,

                # 5
                np.sin(
                    traffic_bearing
                ),

                # 6
                np.cos(
                    traffic_bearing
                ),

                # 7
                relative_velocity_local[0]
                / relative_speed_scale,

                # 8
                relative_velocity_local[1]
                / relative_speed_scale,

                # 9
                np.sin(
                    relative_heading
                ),

                # 10
                np.cos(
                    relative_heading
                ),

                # 11
                dcpa
                / self.distance_scale,

                # 12
                tcpa
                / self.tcpa_scale,
            ],
            dtype=np.float64,
        )

        observation = np.clip(
            observation,
            -1.0,
            1.0,
        )

        return observation.astype(
            np.float32
        )

    # ========================================================
    # Reward
    # ========================================================

    def _compute_reward(
        self,
        collision: bool,
        goal_reached: bool,
        info: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:
        """
        Primera recompensa funcional.

        Componentes:

        1. Progreso hacia el objetivo
        2. Riesgo DCPA/TCPA
        3. Coste temporal
        4. Colisión
        5. Llegada al objetivo

        Todavía NO incluye COLREGs.
        """

        current_distance = float(
            info["distance_to_goal"]
        )

        # ====================================================
        # 1. Progress reward
        # ====================================================

        if self._previous_distance_to_goal is None:
            progress_m = 0.0
        else:
            progress_m = (
                self._previous_distance_to_goal
                - current_distance
            )

        # Un progreso de aproximadamente 10 m produce
        # reward ≈ +1.
        progress_reward = (
            progress_m / 10.0
        )

        self._previous_distance_to_goal = (
            current_distance
        )

        # ====================================================
        # 2. Collision-risk penalty
        # ====================================================

        dcpa = float(
            info["dcpa"]
        )

        tcpa = float(
            info["tcpa"]
        )

        # Escalas iniciales.
        #
        # Posteriormente las ajustaremos con experimentación.
        dcpa_safe = 300.0
        tcpa_safe = 180.0

        dcpa_risk = float(
            np.clip(
                1.0
                - dcpa / dcpa_safe,
                0.0,
                1.0,
            )
        )

        # TCPA <= 0 significa que el CPA ya ocurrió
        # según la aproximación cinemática actual.
        if tcpa > 0.0:
            tcpa_risk = float(
                np.clip(
                    1.0
                    - tcpa / tcpa_safe,
                    0.0,
                    1.0,
                )
            )
        else:
            tcpa_risk = 0.0

        risk_penalty = (
            -2.0
            * dcpa_risk
            * tcpa_risk
        )

        # ====================================================
        # 3. Time penalty
        # ====================================================

        time_penalty = -0.01

        # ====================================================
        # 4. Collision
        # ====================================================

        collision_reward = (
            self.collision_penalty
            if collision
            else 0.0
        )

        # ====================================================
        # 5. Goal
        # ====================================================

        terminal_goal_reward = (
            self.goal_reward
            if goal_reached
            else 0.0
        )

        # ====================================================
        # Total
        # ====================================================

        reward = (
            progress_reward
            + risk_penalty
            + time_penalty
            + collision_reward
            + terminal_goal_reward
        )

        components = {
            "progress": float(
                progress_reward
            ),

            "risk": float(
                risk_penalty
            ),

            "time": float(
                time_penalty
            ),

            "collision": float(
                collision_reward
            ),

            "goal": float(
                terminal_goal_reward
            ),
        }

        return (
            float(reward),
            components,
        )

    # ========================================================
    # Info
    # ========================================================

    def _get_info(
        self,
    ) -> dict[str, Any]:
        """
        Información física no normalizada.

        No forma parte de la observación entregada al agente.
        Se utiliza para:

        - debugging
        - evaluación
        - plots
        - diseño de reward
        """

        # ====================================================
        # Ego
        # ====================================================

        ego_position = np.asarray(
            self.ego.position,
            dtype=np.float64,
        )

        ego_heading = float(
            self.ego.heading
        )

        ego_speed = float(
            self.ego.velocity
        )

        # ====================================================
        # Traffic
        # ====================================================

        traffic_state = (
            self._get_traffic_state()
        )

        traffic_position = np.asarray(
            traffic_state.position,
            dtype=np.float64,
        )

        traffic_heading = float(
            traffic_state.orientation
        )

        traffic_speed = float(
            traffic_state.velocity
        )

        # ====================================================
        # Relative position
        # ====================================================

        relative_position = (
            traffic_position
            - ego_position
        )

        distance_to_traffic = float(
            np.linalg.norm(
                relative_position
            )
        )

        distance_to_goal = float(
            np.linalg.norm(
                self.goal_position
                - ego_position
            )
        )

        # ====================================================
        # Velocity vectors
        # ====================================================

        ego_velocity = np.array(
            [
                ego_speed
                * np.cos(
                    ego_heading
                ),

                ego_speed
                * np.sin(
                    ego_heading
                ),
            ],
            dtype=np.float64,
        )

        traffic_velocity = np.array(
            [
                traffic_speed
                * np.cos(
                    traffic_heading
                ),

                traffic_speed
                * np.sin(
                    traffic_heading
                ),
            ],
            dtype=np.float64,
        )

        relative_velocity = (
            traffic_velocity
            - ego_velocity
        )

        # ====================================================
        # CPA
        # ====================================================

        dcpa, tcpa = self._compute_cpa(
            relative_position,
            relative_velocity,
        )

        # ====================================================
        # Return
        # ====================================================

        return {
            "step": self._elapsed_steps,

            "simulation_time": float(
                self.simulator.time
            ),

            "collision": bool(
                self.simulator
                .rl_collision_occurred
            ),

            "goal_reached": bool(
                self.ego.journey_finished
            ),

            "simulator_running": bool(
                self.simulator.is_running
            ),

            "distance_to_goal": float(
                distance_to_goal
            ),

            "distance_to_traffic": float(
                distance_to_traffic
            ),

            "dcpa": float(
                dcpa
            ),

            "tcpa": float(
                tcpa
            ),

            "ego_position": (
                ego_position.copy()
            ),

            "ego_velocity": float(
                ego_speed
            ),

            "ego_heading": float(
                ego_heading
            ),

            "traffic_position": (
                traffic_position.copy()
            ),

            "traffic_velocity": float(
                traffic_speed
            ),

            "traffic_heading": float(
                traffic_heading
            ),

            "physical_action": (
                self._last_physical_action.copy()
            ),
        }
