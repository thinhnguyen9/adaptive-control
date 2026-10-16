from .dynamical_system import DynamicalSystem
import numpy as np


class LinearSystem(DynamicalSystem):
    
    def __init__(
            self,
            A=np.array([[0.0, 1.0],
                        [0.0, 0.0]]),
            B=np.array([[0.0],
                        [1.0]]),
            C=np.array([[1.0, 0.0]])
        ):
        self.A = A
        self.B = B
        self.C = C

        # Model dimensions
        self.Nx = A.shape[0]
        self.Nu = B.shape[1]
        self.Ny = C.shape[0]

        if self.A.shape[0] != self.Nx or self.A.shape[1] != self.Nx:
            raise ValueError("System matrix A shape does not match system state dimension.")
        if self.B.shape[0] != self.Nx or self.B.shape[1] != self.Nu:
            raise ValueError("Input matrix B shape does not match system dimensions.")
        if self.C.shape[0] != self.Ny or self.C.shape[1] != self.Nx:
            raise ValueError("Output matrix C shape does not match system dimensions.")
        
    def state_dot(self, current_state, current_input):
        return self.A @ current_state + self.B @ current_input

    def linearize(self, nominal_state, nominal_input):
        return self.A, self.B, self.C

    def get_output(self, current_state):
        return self.C @ current_state