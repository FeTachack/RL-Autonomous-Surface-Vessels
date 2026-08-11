from __future__ import annotations

import copy
from typing import Any

import numpy as np

from Controller.VesselController import VesselController
from Environment.SurfaceVessel import SurfaceVessel


class ExternalActionController(VesselController):
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
        vessel: SurfaceVessel,
        initial_action: np.ndarray | None = None,
        desired_velocity: float | None = None,
        ) -> None:

        super().__init__(vessel)


        self.dt = float(vessel.dt)
        self.control_time_step = self.dt

        if desired_velocity is None:
            desired_velocity = 0.5 * float(vessel.parameters.v_max)

        self.desired_velocity = float(desired_velocity)

        self.sim: Any | None = None
        self.stepped = 0
        self.waypoints = None

        self._action = np.zeros(2, dtype=np.float64)

        if initial_action is not None:
            self.set_action(initial_action)

    def initialise(self) -> None:

        return None

    def set_action(self, action: np.ndarray) -> None:

        array = np.asarray(action, dtype=np.float64)

        if array.shape != (2,):
            raise ValueError(
                "La acción debe tener forma (2,), pero se recibió "
                f"{array.shape}."
            )

        if not np.all(np.isfinite(array)):
            raise ValueError("La acción contiene NaN o valores infinitos.")

        self._action = array.copy()

    def control_input(self, *args: Any, **kwargs: Any) -> np.ndarray:

        del args, kwargs

        self.stepped += 1
        return self._action.copy()

    def update_after_waypoint(self) -> None:

        return None

    def set_waypoints(self, waypoints: Any) -> None:

        self.waypoints = waypoints

    def equals(self, controller: Any) -> bool:
        return (
            isinstance(controller, ExternalActionController)
            and np.allclose(self._action, controller._action)
            and self.dt == controller.dt
        )

    def deep_copy(self) -> "ExternalActionController":

        controlled_vessel = self.controlled_vessel
        self.controlled_vessel = None

        try:
            result = copy.deepcopy(self)
        finally:
            self.controlled_vessel = controlled_vessel

        return result
