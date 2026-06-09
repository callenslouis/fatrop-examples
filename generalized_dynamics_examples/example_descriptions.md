# Example Descriptions

This folder contains three illustrative optimal control examples. Each example shows a different type of dynamic system and trajectory optimization problem.

The <code>Dockerfile</code> builds a version of FATROP that allows to exploit structure in the equality constraint Jacobian arising from rewriting general dynamics and a <code>CasADi</code> installation to specify the OCPs

## usage
Each subfolder contains a main.py that runs a problem a few times (to average timing results) both with the normal FATROP version and the Accelerated version. The script <code>run_new_docker.sh</code> installs requirements and ports this repo as a volume.

## hello_world

This is a small, pedagogical multi-agent optimal control problem in the plane.

- System: `nb_agents` point-mass agents with planar position and velocity states.
- Control: planar accelerations for each agent.
- Dynamics: a simple backward-Euler discretization of double-integrator motion, enforced using gap-closing constraints.
- Constraint: each agent is constrained to remain on a circle of radius 1 around the origin at every time step.
- Objective: minimize quadratic control effort over the horizon.
- Task: move the agents from an initial circular formation to a rotated terminal formation while keeping the formation constraint and reaching zero terminal velocity.

This example is meant to demonstrate a minimal optimal control problem that is still rich enough to exercise dynamics constraints, path constraints, and terminal boundary conditions.

## pendulum_3d

This example solves a trajectory optimization problem for a chain of 3D pendulum links.

- System: a serial chain of pendulums represented as free-moving point masses in 3D space.
- Dynamics: an index-1 DAE formulation where the masses move freely and rigid-length constraints enforce the pendulum lengths between joints.
- Control: force inputs applied at a small set of actuated joints specified in `config.yaml`.
- Objective: typically minimize control effort while either stabilizing the chain or reaching a nominal rest pose at the final time.
- Task: compute a dynamically feasible swing/trajectory for the articulated pendulum under gravity.

The pendulum model supports optional stabilizer terms and optional joint stiffness/damping, and it can be configured for path tracking or a free swing depending on `config.yaml`.

## walking_gait

This example is a direct-collocation optimal control formulation for a planar walking gait.

- System: a musculoskeletal biped model with pelvis translation, pelvis tilt, left/right hip, knee, and ankle joints.
- Dynamics: collocation-based discretization over a half gait cycle, with equality and inequality constraints encoding joint motion, contact structure, and muscle-actuation-like control variables.
- Problem configuration: the walking problem is configured through `config.yaml` and can be adapted by changing problem type, solver options, and visualization settings.
- Origin: the formulation is taken from the MATLAB script `test_gait.m` by Lars D'Hondt and translated into this Python/CasADi implementation.
- Task: solve for a half-step of walking and then reconstruct a full gait cycle by mirroring the left and right leg motions.

The example illustrates how a gait optimization can be expressed as a structured optimal control problem and how the resulting solution can be visualized as a walking skeleton.
