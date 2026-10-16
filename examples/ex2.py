"""
Example 2: LQR control on a 2D Quadrotor.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src import *
from systems import Quadrotor_2D
from controllers import LQR_Controller


def main():

    # Quadrotor
    true_quad = Quadrotor_2D(mass=0.5, J=0.1)
    known_quad = Quadrotor_2D(mass=0.5*1.3, J=0.1*2)     # Best-known model, can be wrong
    Nx, Nu, Ny = true_quad.Nx, true_quad.Nu, true_quad.Ny

    # Control - only use known model
    lqr = LQR_Controller(
        system=known_quad,
        nominal_state=np.array([0, 0, 0, 0, 0, 0]),
        nominal_control=np.array([known_quad.m*known_quad.g, 0]),   # gravity compensation
        Q=np.diag([10., 10., 1., 5., 5., 1.]),
        R=np.diag([1., 1.])
    )

    # Simulation
    sim = run_simulation(
        system=true_quad,
        controller=lqr,
        initial_state=np.array([0, 0, 0, 0, 0, 0]),
        reference_trajectory=np.array([1, 1]),
        simulation_time=5.0,
        control_rate=100.0,
        integration_method='rk4'
    )
    tvec, rvec, xvec, uvec, yvec = sim['tvec'], sim['rvec'], sim['xvec'], sim['uvec'], sim['yvec']
    uvec_b, uvec_ad = sim['mrac']['uvec_b'], sim['mrac']['uvec_ad']

    plot_info = [(0, 0,    'x (m)',            1.0,        0),   # x_idx, r_idx, label, unit scaling, subplot idx
                 (1, 1,    'z (m)',            1.0,        2),
                 (2, None, 'theta (deg)',      180./np.pi, 4),
                 (3, None, 'xdot (m/s)',       1.0,        1),
                 (4, None, 'zdot (m/s)',       1.0,        3),
                 (5, None, 'thetadot (deg/s)', 180./np.pi, 5)]
    fig, axes = plt.subplots(4, 2, figsize=(9, 8), sharex=True)
    for x_idx, r_idx, label, unit_scaling, subplot_idx in plot_info:
        axis = axes.flat[subplot_idx]
        if r_idx is not None:
            axis.plot(tvec, rvec[:, r_idx]*unit_scaling, 'k--', lw=1., label='Reference')
        axis.plot(tvec, xvec[:, x_idx]*unit_scaling, 'k-', lw=1.5, label='True state')
        axis.set_ylabel(label)
        axis.grid(True)
        if subplot_idx==0:
            axis.legend()

    axes[3, 0].plot(tvec, uvec_b[:, 0], 'b--', lw=1., label='F_b')
    axes[3, 0].plot(tvec, uvec_ad[:, 0], 'r--', lw=1., label='F_ad')
    axes[3, 0].plot(tvec, uvec[:, 0], 'k-', lw=1.5, label='F')
    axes[3, 0].set_ylabel('Thrust (N)')
    axes[3, 0].legend()
    axes[3, 0].grid(True)

    axes[3, 1].plot(tvec, uvec_b[:, 1], 'b--', lw=1., label='M_b')
    axes[3, 1].plot(tvec, uvec_ad[:, 1], 'r--', lw=1., label='M_ad')
    axes[3, 1].plot(tvec, uvec[:, 1], 'k-', lw=1.5, label='M')
    axes[3, 1].set_ylabel('Moment (N m)')
    axes[3, 1].legend()
    axes[3, 1].grid(True)

    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    fig.suptitle('LQR control on a planar quadrotor')
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()