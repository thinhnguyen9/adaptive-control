from .controller import Controller
from systems import DynamicalSystem
import numpy as np


class LQR_Controller(Controller):
    """
    Continuous-time LQR controller for nonlinear systems around a nominal operating point.
    Drive the output y to some reference r.

    Original system:
        xdot = f(x, u)
        y = h(x)
    Linearized system around (xs, us):
        (x - xs)dot = A(x - xs) + B(u - us)
        y - h(xs) = C(x - xs)
    Control:
        u = us - Kx x + Kr r
    """
    def __init__(
            self,
            system: DynamicalSystem,
            nominal_state: np.ndarray,
            nominal_control: np.ndarray,
            Q: np.ndarray,
            R: np.ndarray
        ):
        self.system = system
        self.xs = nominal_state
        self.us = nominal_control
        self.ys = self.system.get_output(self.xs)
        self.Nx, self.Nu, self.Ny = system.Nx, system.Nu, system.Ny
        if self.xs.shape[0] != self.Nx:
            raise ValueError("Nominal state shape does not match system state dimension.")
        if self.us.shape[0] != self.Nu:
            raise ValueError("Nominal control shape does not match system control dimension.")
        
        A, B, C = system.linearize(self.xs, self.us)
        self.update_model(A, B, C)
        self.update_weights(Q, R)

    def update_model(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C
        self.recalculate_gain = True

    def update_weights(self, Q, R):
        self.Q = Q
        self.R = R
        self.R_inv = np.linalg.inv(R)
        if self.Q.shape[0] != self.Nx or self.Q.shape[1] != self.Nx:
            raise ValueError("Q matrix shape does not match system state dimension.")
        if self.R.shape[0] != self.Nu or self.R.shape[1] != self.Nu:
            raise ValueError("R matrix shape does not match system control dimension.")
        self.recalculate_gain = True
    
    def get_lqr_gains(self):
        if self.recalculate_gain:
            self._calculate_lqr_gain()
        return self.Kx, self.Kr

    def compute_control(self, current_state, current_reference, dt=None):
        Kx, Kr = self.get_lqr_gains()
        control = self.us - Kx @ current_state + Kr @ current_reference
        return {
            'u': control
        }

    def _calculate_lqr_gain(self):
        H = np.vstack((np.hstack(( self.A, -self.B @ self.R_inv @ self.B.T)),
                       np.hstack((-self.Q, -self.A.T))))
        eigVal, eigVec = np.linalg.eig(H)
        U = eigVec[:, np.argsort(eigVal)]
        U = U[:, :self.Nx]       # 'stable' subspace - Nx smallest eigenvalues
        U11 = U[:self.Nx, :]
        U21 = U[self.Nx:, :]
        try:
            P = U21 @ np.linalg.inv(U11)
        except:
            print('LQR warning: U11 is not invertible!')
            P = U21 @ np.linalg.pinv(U11)
        self.P = P.real
        self.Kx = self.R_inv @ self.B.T @ self.P

        # Feedforward gain
        self.Am = self.A - self.B @ self.Kx
        self.Kr = - np.linalg.pinv(self.C @ np.linalg.pinv(self.Am) @ self.B)
        # print("LQR feedback gain Kx:", self.Kx)
        # print("LQR feedforward gain Kr:", self.Kr)
        self.recalculate_gain = False
    
    def get_P_matrix(self):
        if self.recalculate_gain:
            self._calculate_lqr_gain()
        return self.P
