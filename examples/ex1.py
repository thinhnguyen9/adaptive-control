"""
Example 1: direct MRAC on a double integrator.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src import *
from systems import LinearSystem
from controllers import DirectMRAC


def main():

    """
    Given the linear system
        xdot = A*x + B*u,
        y = C*x,
    where A, B are unknown, C is known.
    
    Goal: design a control u to drive the output y to some reference r.
    """


    """Step 1: model - our guess of the true system (could be wrong)."""
    A = np.array([[0.0, 1.0],
                  [0.0, 0.0]])
    B = np.array([[0.0],
                  [1.0]])
    C = np.array([[1.0, 0.0]])


    """Step 2: design a nominal control u = -Kx*x + Kr*r."""
    Kx = np.array([[2.0, 3.0]])
    Kr = np.array([[2.0]])
    # Nominal closed-loop system: xdot = Am*x + Bm*r, where
    Am = A - B @ Kx
    Bm = B @ Kr


    """Step 3: given the true system with unknown parameters."""
    A_true = np.array([[0.0, 1.0],
                       [0.0, 0.0]])
    B_true = np.array([[0.5],
                       [1.5]])
    true_sys = LinearSystem(A_true, B_true, C)
    Nx, Nu, Ny = true_sys.Nx, true_sys.Nu, true_sys.Ny


    """Step 4: augment the baseline control with MRAC to handle uncertainties.
    MRAC will try to compensate for uncertainties to make the closed-loop system
    behave like the nominal system xdot = Am*x + Bm*r."""
    mrac = DirectMRAC(
        baseline_controller_Kx=Kx,
        baseline_controller_Kr=Kr,
        Am=Am,
        Bm=Bm,
        B1=B,
        omega_guess=np.eye(Nu),
        omega_sign=np.eye(Nu),
        adaptive_gain=20000.0,
        integration_method='rk4',
    )

    # Simulation
    sim = run_simulation(
        system=true_sys,
        controller=mrac,
        initial_state=np.array([0, 0]),
        reference_trajectory=np.array([1]),
        simulation_time=10.0,
        control_rate=100.0,
        integration_method='rk4'
    )
    tvec, rvec, xvec, uvec, yvec = sim['tvec'], sim['rvec'], sim['xvec'], sim['uvec'], sim['yvec']
    uvec_b, uvec_ad = sim['mrac']['uvec_b'], sim['mrac']['uvec_ad']

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    state_axis, control_axis = axes
    for state_idx in range(xvec.shape[1]):
        state_axis.plot(tvec, xvec[:, state_idx], lw=1.5, label=f'x[{state_idx}]')
        if state_idx < rvec.shape[1]:
            state_axis.plot(
                tvec,
                rvec[:, state_idx],
                '--',
                lw=1.,
                label=f'r[{state_idx}]'
            )
    state_axis.set_title('States and references')
    state_axis.set_xlabel('Time (s)')
    state_axis.set_ylabel('State value')
    state_axis.grid(True)
    state_axis.legend()

    for control_idx in range(uvec_b.shape[1]):
        control_axis.plot(
            tvec,
            uvec_b[:, control_idx],
            '--',
            lw=1.,
            label=f'u_b[{control_idx}]'
        )
        control_axis.plot(
            tvec,
            uvec_ad[:, control_idx],
            '-',
            lw=1.5,
            label=f'u_ad[{control_idx}]'
        )
    control_axis.set_title('Control inputs')
    control_axis.set_xlabel('Time (s)')
    control_axis.set_ylabel('Control value')
    control_axis.grid(True)
    control_axis.legend()

    fig.suptitle('MRAC simulation')
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()