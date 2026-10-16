import numpy as np
from systems import DynamicalSystem
from controllers import Controller
from .utils import integrate


def run_simulation(
        system: DynamicalSystem,
        controller: Controller,
        initial_state: np.ndarray,
        reference_trajectory: np.ndarray,
        simulation_time=10.0,
        control_rate=50.0,
        integration_method='rk4'
    ):
    Nx, Nu, Ny = system.Nx, system.Nu, system.Ny
    if initial_state.shape[0] != Nx:
        raise ValueError("Initial state dimension must match the system state dimension.")

    dt = 1.0 / control_rate
    numstep = int(simulation_time * control_rate) + 1
    if reference_trajectory.shape == (Ny,):
        reference_trajectory = np.tile(reference_trajectory, (numstep, 1))
    if reference_trajectory.shape[0] != numstep:
        raise ValueError("Reference trajectory length must match the number of simulation steps.")

    tvec = np.linspace(0, simulation_time, numstep)
    xvec = np.zeros((numstep, Nx))
    uvec = np.zeros((numstep, Nu))
    yvec = np.zeros((numstep, Ny))

    # For adaptive control
    uvec_b = np.zeros((numstep, Nu))
    uvec_ad = np.zeros((numstep, Nu))

    curr_state = initial_state
    curr_output = system.get_output(curr_state)
    for k in range(numstep):
        xvec[k] = curr_state
        yvec[k] = curr_output

        # Control
        curr_control = controller.compute_control(
            current_state=curr_state,
            current_reference=reference_trajectory[k],
            dt=dt
        )
        uvec[k] = curr_control['u']
        if 'mrac' in curr_control:
            uvec_b[k] = curr_control['mrac']['u_b']
            uvec_ad[k] = curr_control['mrac']['u_ad']

        # Next time step
        curr_state = integrate(system.state_dot, curr_state, curr_control['u'], dt, method=integration_method)
        curr_output = system.get_output(curr_state)
    
    return {
        'tvec': tvec,
        'rvec': reference_trajectory,
        'xvec': xvec,
        'uvec': uvec,
        'yvec': yvec,
        'mrac': {
            'uvec_b': uvec_b,
            'uvec_ad': uvec_ad,
        },
    }
