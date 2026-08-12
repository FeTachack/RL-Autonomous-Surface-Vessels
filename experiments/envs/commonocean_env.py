from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from CollisionHandling.CollisionDetector import (
    CollisionDetector,
)
from Pipeline.SimulationIO import (
    SimulationIO,
)
from Simulator.SimulatorFactory import (
    SimulatorFactory,
)

from commonocean.common.solution import (
    VesselModel,
)

from rules.common.helper import (
    load_yaml,
)

from experiments.controllers.external_action_controller import (
    ExternalActionController,
)

from experiments.scenarios.randomize_crossing_scenario import (
    RandomizedCrossingConfig,
    generate_randomized_crossing_scenario,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIGURATION_PATH = (
    PROJECT_ROOT
    / "src"
    / "configuration.yaml"
)

SCENARIO_PATH = (
    "/experiments/scenarios/one_ego_one_traffic.xml"
)


class CommonOceanEnv(gym.Env):


    metadata = {
        "render_modes": [
            "human",
        ],
        "render_fps": 30,
    }


    def __init__(
        self,
        max_episode_steps: int = 300,
        collision_penalty: float = -100.0,
        goal_reward: float = 100.0,
        distance_scale: float = 2500.0,
        tcpa_scale: float = 300.0,
        render_mode: str | None = None,
        scenario_path: str = SCENARIO_PATH,
        randomize_scenario: bool = False,
        randomization_config: (
            RandomizedCrossingConfig
            | dict[str, float]
            | None
        ) = None,
        randomized_scenario_dir: str | Path | None = None,
    ) -> None:
        super().__init__()

        if render_mode not in (
            None,
            "human",
        ):
            raise ValueError(
                "render_mode debe ser None o 'human', "
                f"pero se recibió {render_mode!r}."
            )

        self.render_mode = render_mode

        self.max_episode_steps = int(
            max_episode_steps
        )

        self.collision_penalty = float(
            collision_penalty
        )

        self.goal_reward = float(
            goal_reward
        )

        self.distance_scale = float(
            distance_scale
        )

        self.tcpa_scale = float(
            tcpa_scale
        )


        self.base_scenario_path = str(
            scenario_path
        )

        self.scenario_path = str(
            scenario_path
        )

        self.randomize_scenario = bool(
            randomize_scenario
        )

        self.randomization_config = (
            randomization_config
        )

        if randomized_scenario_dir is None:
            self.randomized_scenario_dir = (
                PROJECT_ROOT
                / "experiments"
                / "scenarios"
                / "generated"
            )
        else:
            self.randomized_scenario_dir = Path(
                randomized_scenario_dir
            )

        self.scenario_metadata = {
            "randomized": False,
            "scenario_path": self.scenario_path,
        }


        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(13,),
            dtype=np.float32,
        )


        self.factory = None
        self.simulation_io = None
        self.simulator = None

        self.ego = None
        self.traffic = None

        self.controller = None
        self.collision_detector = None

        self.goal_position = None


        self._elapsed_steps = 0
        self._episode_done = False

        self._last_physical_action = np.zeros(
            2,
            dtype=np.float64,
        )

        self._previous_distance_to_goal = None


    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):


        super().reset(
            seed=seed
        )

        self.close()

        self._elapsed_steps = 0
        self._episode_done = False

        self._last_physical_action = np.zeros(
            2,
            dtype=np.float64,
        )

        self._prepare_scenario_for_reset(
            seed=seed
        )

        self._build_simulator()

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


    def step(
        self,
        action: np.ndarray,
    ):


        if self.simulator is None:
            raise RuntimeError(
                "Debe llamar env.reset() antes de env.step()."
            )

        if self._episode_done:
            raise RuntimeError(
                "El episodio ya terminó. Debe llamar env.reset() "
                "antes de continuar."
            )

        normalized_action = np.asarray(
            action,
            dtype=np.float32,
        )

        if normalized_action.shape != (2,):
            raise ValueError(
                "La acción debe tener shape (2,), "
                f"pero se recibió {normalized_action.shape}."
            )

        if not np.all(
            np.isfinite(
                normalized_action
            )
        ):
            raise ValueError(
                "La acción contiene valores no finitos."
            )

        normalized_action = np.clip(
            normalized_action,
            -1.0,
            1.0,
        )

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

        self.simulator.compute_next_state()

        self._elapsed_steps += 1

        collision = bool(
            self.simulator.rl_collision_occurred
        )

        goal_reached = bool(
            self.ego.journey_finished
        )

        terminated = bool(
            collision
            or goal_reached
        )

        truncated = bool(
            self._elapsed_steps
            >= self.max_episode_steps
            and not terminated
        )

        observation = self._get_observation()
        info = self._get_info()

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


    def render(
        self,
    ):


        return None


    def close(
        self,
    ):


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


    def _prepare_scenario_for_reset(
        self,
        seed: int | None,
    ) -> None:


        self.scenario_path = (
            self.base_scenario_path
        )

        self.scenario_metadata = {
            "randomized": False,
            "scenario_path": self.scenario_path,
        }

        if not self.randomize_scenario:
            return

        if seed is None:
            effective_seed = int(
                self.np_random.integers(
                    0,
                    np.iinfo(
                        np.uint32
                    ).max,
                )
            )
        else:
            effective_seed = int(
                seed
            )

        base_xml_path = (
            PROJECT_ROOT
            / self.base_scenario_path.lstrip(
                "/"
            )
        )

        output_xml_path = (
            self.randomized_scenario_dir
            / (
                "randomized_crossing_seed_"
                f"{effective_seed}.xml"
            )
        )

        metadata = (
            generate_randomized_crossing_scenario(
                base_xml_path=base_xml_path,
                output_xml_path=output_xml_path,
                seed=effective_seed,
                config=self.randomization_config,
            )
        )

        try:
            relative_output_path = (
                output_xml_path
                .resolve()
                .relative_to(
                    PROJECT_ROOT.resolve()
                )
                .as_posix()
            )
        except ValueError as exc:
            raise RuntimeError(
                "randomized_scenario_dir debe estar dentro "
                f"de PROJECT_ROOT={PROJECT_ROOT}"
            ) from exc

        self.scenario_path = (
            "/"
            + relative_output_path
        )

        metadata["scenario_path"] = (
            self.scenario_path
        )

        self.scenario_metadata = metadata


    def _build_simulator(
        self,
    ) -> None:


        configuration = load_yaml(
            str(
                CONFIGURATION_PATH
            )
        )

        import_config = configuration[
            "scenario_selection"
        ][
            "import_scenario"
        ]

        import_config[
            "use_imported_scenario"
        ] = True

        import_config[
            "scenario_filepath"
        ] = self.scenario_path

        import_config[
            "vessel_type"
        ] = 1

        import_config[
            "vessel_type_by_id"
        ] = None

        import_config[
            "controller_type"
        ] = "mpc"

        configuration[
            "general_simulator"
        ][
            "using_collision_avoider"
        ] = False

        configuration[
            "general_simulator"
        ][
            "using_collision_detection"
        ] = True

        configuration[
            "general_simulator"
        ][
            "using_displayer"
        ] = (
            self.render_mode
            == "human"
        )

        configuration[
            "general_simulator"
        ][
            "mark_mpc_states"
        ] = False

        configuration[
            "general_simulator"
        ][
            "plotting"
        ][
            "do_plotting"
        ] = False

        dt = float(
            configuration[
                "general_simulator"
            ][
                "dt"
            ]
        )

        factory = SimulatorFactory(
            dt
        )

        simulation_io = SimulationIO(
            factory
        )

        factory.current_configuration = (
            configuration
        )

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

        simulation_io.configure_simfac_from_config_dict(
            current_configuration_input=configuration
        )

        if len(
            factory.models
        ) != 1:
            raise RuntimeError(
                "El entorno espera exactamente 1 SurfaceVessel ego, "
                f"pero encontró {len(factory.models)}."
            )

        if len(
            factory.dynamic_obstacles
        ) != 1:
            raise RuntimeError(
                "Esta versión espera exactamente 1 DynamicObstacle, "
                f"pero encontró {len(factory.dynamic_obstacles)}."
            )

        simulator = (
            factory.generate_scenario()
        )

        simulator.rl_collision_occurred = False
        simulator.rl_collision_vehicle = None
        simulator.rl_collision_object = None

        ego = simulator.models[0]

        if (
            ego.vessel_dynamics.vessel_model
            != VesselModel.YP
        ):
            raise RuntimeError(
                "La embarcación ego debe utilizar VesselModel.YP."
            )

        if (
            ego.waypoints is None
            or len(
                ego.waypoints
            ) == 0
        ):
            raise RuntimeError(
                "La embarcación ego no tiene waypoints."
            )

        goal_position = np.asarray(
            ego.waypoints[-1],
            dtype=np.float64,
        ).copy()

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
                desired_velocity=desired_velocity,
            )
        )

        external_controller.sim = simulator
        external_controller.initialise()

        ego.set_controller(
            external_controller
        )

        collision_detector = next(
            (
                listener
                for listener in simulator.listeners
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

        traffic = simulator.dynamic_obstacles[0]

        if traffic.prediction is None:
            raise RuntimeError(
                "El DynamicObstacle no tiene una trayectoria prediction."
            )

        traffic_states = (
            traffic
            .prediction
            .trajectory
            .state_list
        )

        if not traffic_states:
            raise RuntimeError(
                "La trayectoria del DynamicObstacle está vacía."
            )

        first_traffic_time_step = int(
            traffic_states[0].time_step
        )

        last_traffic_time_step = int(
            traffic_states[-1].time_step
        )

        if first_traffic_time_step > 1:
            raise RuntimeError(
                "La trayectoria del DynamicObstacle comienza "
                "demasiado tarde: "
                f"primer time_step={first_traffic_time_step}."
            )

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

        self.factory = factory
        self.simulation_io = simulation_io
        self.simulator = simulator

        self.ego = ego
        self.traffic = traffic

        self.controller = external_controller
        self.collision_detector = collision_detector

        self.goal_position = goal_position


    def _map_action_to_physical(
        self,
        action: np.ndarray,
    ) -> np.ndarray:


        acceleration = (
            float(
                action[0]
            )
            * float(
                self.ego.parameters.a_max
            )
        )

        yaw_rate = (
            float(
                action[1]
            )
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


    def _global_to_ego_frame(
        self,
        vector: np.ndarray,
    ) -> np.ndarray:


        psi = float(
            self.ego.heading
        )

        c = np.cos(
            psi
        )

        s = np.sin(
            psi
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

        return (
            rotation
            @ np.asarray(
                vector,
                dtype=np.float64,
            )
        )


    def _compute_cpa(
        self,
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

        relative_speed_squared = float(
            np.dot(
                relative_velocity,
                relative_velocity,
            )
        )

        if relative_speed_squared < 1e-8:
            return (
                float(
                    np.linalg.norm(
                        relative_position
                    )
                ),
                0.0,
            )

        tcpa = (
            -float(
                np.dot(
                    relative_position,
                    relative_velocity,
                )
            )
            / relative_speed_squared
        )

        closest_position = (
            relative_position
            + tcpa
            * relative_velocity
        )

        dcpa = float(
            np.linalg.norm(
                closest_position
            )
        )

        return (
            dcpa,
            tcpa,
        )


    def _get_traffic_state(
        self,
    ):


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
                "La trayectoria del DynamicObstacle está vacía."
            )

        raise RuntimeError(
            "La simulación alcanzó un time_step para el cual "
            "el DynamicObstacle no tiene trayectoria. "
            f"time_step={time_step}, "
            f"último time_step={states[-1].time_step}"
        )


    def _get_observation(
        self,
    ) -> np.ndarray:


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

        dcpa, tcpa = (
            self._compute_cpa(
                relative_position_global,
                relative_velocity_global,
            )
        )

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

        speed_scale = max(
            float(
                self.ego.parameters.v_max
            ),
            1e-6,
        )

        relative_speed_scale = (
            2.0
            * speed_scale
        )

        observation = np.array(
            [
                ego_speed / speed_scale,
                goal_distance / self.distance_scale,
                np.sin(
                    goal_bearing
                ),
                np.cos(
                    goal_bearing
                ),
                traffic_distance / self.distance_scale,
                np.sin(
                    traffic_bearing
                ),
                np.cos(
                    traffic_bearing
                ),
                relative_velocity_local[0]
                / relative_speed_scale,
                relative_velocity_local[1]
                / relative_speed_scale,
                np.sin(
                    relative_heading
                ),
                np.cos(
                    relative_heading
                ),
                dcpa / self.distance_scale,
                tcpa / self.tcpa_scale,
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


    def _compute_reward(
        self,
        collision: bool,
        goal_reached: bool,
        info: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:


        current_distance = float(
            info[
                "distance_to_goal"
            ]
        )

        if self._previous_distance_to_goal is None:
            progress_m = 0.0
        else:
            progress_m = (
                self._previous_distance_to_goal
                - current_distance
            )

        progress_reward = (
            progress_m
            / 10.0
        )

        self._previous_distance_to_goal = (
            current_distance
        )

        dcpa = float(
            info[
                "dcpa"
            ]
        )

        tcpa = float(
            info[
                "tcpa"
            ]
        )

        dcpa_safe = 300.0
        tcpa_safe = 180.0

        dcpa_risk = float(
            np.clip(
                1.0
                - dcpa
                / dcpa_safe,
                0.0,
                1.0,
            )
        )

        if tcpa > 0.0:
            tcpa_risk = float(
                np.clip(
                    1.0
                    - tcpa
                    / tcpa_safe,
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

        time_penalty = -0.01

        collision_reward = (
            self.collision_penalty
            if collision
            else 0.0
        )

        terminal_goal_reward = (
            self.goal_reward
            if goal_reached
            else 0.0
        )

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
            float(
                reward
            ),
            components,
        )


    def _get_info(
        self,
    ) -> dict[str, Any]:


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

        dcpa, tcpa = (
            self._compute_cpa(
                relative_position,
                relative_velocity,
            )
        )

        return {
            "step": int(
                self._elapsed_steps
            ),
            "simulation_time": float(
                self.simulator.time
            ),
            "collision": bool(
                self.simulator.rl_collision_occurred
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
            "scenario": dict(
                self.scenario_metadata
            ),
        }
