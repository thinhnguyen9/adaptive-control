from .dynamical_system import DynamicalSystem
import numpy as np
from math import sqrt, sin, cos


class Quadrotor_2D(DynamicalSystem):
    
    def __init__(self, mass=0.5, J=0.1, g=9.81):
        """
        Initialize 2D quadrotor model parameters.
            States  : [x, z, theta, xdot, zdot, thetadot]
            Control : [thrust F, moment M]
        
        Args:
            mass : Quadrotor mass (kg)
            J    : Moments of inertia (kg*m^2)
            g    : Gravitational acceleration (m/s^2)
        """
        # Physical parameters
        self.m = mass
        self.J = J
        self.g = g

        # Model dimensions
        self.Nx = 6
        self.Nu = 2
        self.Ny = 2

        # Output matrix - gets [x, z]
        self.C = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        
    def state_dot(self, current_state, current_input):
        x, z, theta, xdot, zdot, thetadot = current_state
        F, M = current_input
        return np.array([
            xdot,
            zdot,
            thetadot,
            -F/self.m * sin(theta),
            F/self.m * cos(theta) - self.g,
            M/self.J
        ])

    def linearize(self, nominal_state, nominal_input):
        x, z, theta, xdot, zdot, thetadot = nominal_state
        F, M = nominal_input
        A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, -F/self.m * cos(theta), 0, 0, 0],
            [0, 0, -F/self.m * sin(theta), 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        B = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [-sin(theta)/self.m, 0],
            [cos(theta)/self.m, 0],
            [0, 1/self.J]
        ])
        return A, B, self.C

    def get_output(self, current_state):
        return self.C @ current_state