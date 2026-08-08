# Autonomous Surface Vessel Navigation with Reinforcement Learning

Reinforcement Learning project for autonomous navigation of surface
vessels using CommonOcean-Sim.

## Objective

The objective is to develop a reinforcement learning agent capable of
controlling an ego surface vessel while navigating in dynamic maritime
environments.

The agent controls only the ego vessel. Other vessels are considered
part of the environment.

## Simulator

The project uses:

- CommonOcean-Sim
- CommonOcean scenarios
- COLREGs-based maritime environments

Repository:

https://github.com/CommonOcean/commonocean-sim

## Control architecture

The original CommonOcean-Sim MPC controller of the ego vessel is
replaced by an external controller:

RL Policy
    -> ExternalActionController
    -> SurfaceVessel dynamics

The action for the YP vessel model is:

    [longitudinal acceleration, yaw rate]

Other vessels remain controlled by the simulator or follow predefined
dynamic-obstacle trajectories.


## Current status

Validated:

- CommonOcean-Sim installation
- OSQP as experimental MPC solver
- External controller for ego vessel
- Independent control of ego vessel
- Fixed longitudinal acceleration
- Positive yaw-rate control
- Negative yaw-rate control
- Symmetric steering response
- One external action applied per simulator step


## Next steps

1. Create a scenario with one planningProblem for the ego vessel.
2. Represent surrounding traffic using dynamicObstacle.
3. Implement Gymnasium environment.
4. Define observations, action normalization and reward.
5. Train RL baseline.






