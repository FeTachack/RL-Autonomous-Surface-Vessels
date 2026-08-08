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
- Fixed longitudinal acceleration experiment

Test result:

- initial velocity: 0.0 m/s
- acceleration: 0.024 m/s²
- simulation time: 100 s
- final velocity: 2.4 m/s
- displacement: 120 m

## Next steps

1. Validate yaw-rate control.
2. Create a scenario with one planningProblem for the ego vessel.
3. Represent surrounding traffic using dynamicObstacle.
4. Implement Gymnasium environment.
5. Define observations, actions and reward.
6. Train RL baseline.
