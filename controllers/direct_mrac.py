from .controller import Controller
from systems import DynamicalSystem
from src.utils import integrate
import numpy as np
from scipy.linalg import solve_continuous_lyapunov


class DirectMRAC(Controller):
    """
    Direct MRAC augmentation on a baseline controller.
    During simulation, must call compute_control() at each time step to update internal properties.

    System:
        xdot = A*x + B1*(omega*u + sigma1),
        y = Cx,
        u = u_b + u_ad,     u_b = -Kb_x*x + Kb_r*r.
    Known: B1/B2, sign(omega);  Unknown: A, omega, sigma1.

    Desired closed-loop system (omega=I, sigma1=0):
        xdot = Am*x + Bm*r,
        Am = A - B1*Kb_x,      Bm = B1*Kb_r.

    True closed-loop system:
        xdot = Am*x + Bm*r + B1*[omega*u_ad - (omega-I)*Kb_x*x + (omega-I)*Kb_r*r + sigma1],
        u_ad = Kx*x + Kr*r (ignore sigma1 for now).

    Args:
        Am: desired closed-loop system matrix.
        Bm: desired closed-loop input matrix.
        B1/B2: Matched/unmatched input matrices (B = [B1 B2] full rank).
        omega: unknown input gain.
    """
    def __init__(
            self,
            # system: DynamicalSystem,
            # baseline_controller: Controller,  # doesn't work for arbitrary baseline controllers yet?
            baseline_controller_Kx: np.ndarray,
            baseline_controller_Kr: np.ndarray,
            Am: np.ndarray,
            Bm: np.ndarray,
            B1: np.ndarray,
            omega_guess: np.ndarray,
            omega_sign: np.ndarray,
            adaptive_gain: float,
            B2=None,
            integration_method='rk4',
        ):
        self.Nx = Am.shape[0]
        self.Nu = B1.shape[1]
        self.Ny = Bm.shape[1]
        # self.system = system
        # self.Nx, self.Nu, self.Ny = system.Nx, system.Nu, system.Ny
        self.Kb_x, self.Kb_r = baseline_controller_Kx, baseline_controller_Kr
        self.Am, self.Bm, self.B1, self.B2 = Am, Bm, B1, B2
        self.omega_guess = omega_guess
        self.omega_sign = omega_sign
        self.Gamma = adaptive_gain
        if self.Gamma < 0:
            raise ValueError("Adaptive gain must be nonnegative.")
        self.integration_method = integration_method
        if self.integration_method not in ['rk4', 'euler']:
            raise ValueError("Integration method must be 'rk4' or 'euler'.")

        # With baseline controller, initialize adaptive control at 0
        self.Kx = np.zeros((self.Nu, self.Nx))
        self.Kr = np.zeros((self.Nu, self.Ny))
        # Adaptation uses trapezoidal rule: Kx(k) = Kx(k-1) + dt/2 * (Kx_dot(k-1) + Kx_dot(k))
        self.prev_Kx_dot = np.zeros((self.Nu, self.Nx))
        self.prev_Kr_dot = np.zeros((self.Nu, self.Ny))

        # Verify shapes
        if self.Kb_x.shape[0] != self.Nu or self.Kb_x.shape[1] != self.Nx:
            raise ValueError("Baseline controller Kx shape does not match system dimensions.")
        if self.Kb_r.shape[0] != self.Nu or self.Kb_r.shape[1] != self.Ny:
            raise ValueError("Baseline controller Kr shape does not match system dimensions.")
        if self.Am.shape[0] != self.Nx or self.Am.shape[1] != self.Nx:
            raise ValueError("Desired closed-loop system matrix Am shape does not match system state dimension.")
        if self.Bm.shape[0] != self.Nx or self.Bm.shape[1] != self.Ny:
            raise ValueError("Desired closed-loop input matrix Bm shape does not match system dimensions.")
        if self.B1.shape[0] != self.Nx or self.B1.shape[1] != self.Nu:
            raise ValueError("Matched input matrix B1 shape does not match system dimensions.")
        if self.omega_guess.shape[0] != self.Nu or self.omega_guess.shape[1] != self.Nu:
            raise ValueError("Omega guess shape does not match system input dimension.")
        if self.omega_sign.shape[0] != self.Nu or self.omega_sign.shape[1] != self.Nu:
            raise ValueError("Omega sign shape does not match system input dimension.")

        # Verify stability
        eigvals = np.linalg.eigvals(self.Am)
        if np.any(np.real(eigvals) >= 0):
            raise ValueError("Am is not Hurwitz!")
        self.P = solve_continuous_lyapunov(self.Am.T, -np.eye(self.Nx))
        self.P_B1_sgn_omega = self.P @ self.B1 @ self.omega_sign

        # Internal properties
        self.xm = None
        self.prev_r = None
        self.prev_u = None
        self.prev_u_b = None
        self.prev_u_ad = None

    # ---------------------------------------------------- Reference model
    def _xm_dot(self, xm, r):
        return self.Am @ xm + self.Bm @ r

    def _update_xm(self, curr_x, dt):
        if self.xm is None:     # initialize xm(0)=x(0)
            self.xm = curr_x
        else:
            self.xm = integrate(self._xm_dot, self.xm, self.prev_r, dt, self.integration_method)

    # ---------------------------------------------------- Adaptation law
    def _adaptation_law(self, x, xm, r):
        x = x.reshape(-1, 1)
        xm = xm.reshape(-1, 1)
        r = r.reshape(-1, 1)
        e = x - xm

        temp = -self.Gamma * e.T @ self.P_B1_sgn_omega
        Kx_dot = x @ temp
        Kr_dot = r @ temp
        return Kx_dot.T, Kr_dot.T

    def _run_adaptation(self, curr_x, curr_r, dt):
        curr_Kx_dot, curr_Kr_dot = self._adaptation_law(curr_x, self.xm, curr_r)
        self.Kx = self.Kx + dt/2.0 * (self.prev_Kx_dot + curr_Kx_dot)
        self.Kr = self.Kr + dt/2.0 * (self.prev_Kr_dot + curr_Kr_dot)
        self.prev_Kx_dot, self.prev_Kr_dot = curr_Kx_dot, curr_Kr_dot

    # ---------------------------------------------------- Compute control
    def _control_law(self, x, r):
        u_b = -self.Kb_x @ x + self.Kb_r @ r
        u_ad = self.Kx @ x + self.Kr @ r
        return u_b, u_ad
    
    def compute_control(self, current_state, current_reference, dt):
        """
        Update Kx,Kr only when this function is called (at discrete time steps).

        current_state: x(k)
        current_reference: r(k)
        dt: t(k) - t(k-1)
        """
        # Update reference state: self.xm <- xm(k)
        self._update_xm(current_state, dt)  

        # Run adaptation:
        #     self.Kx, self.Kr <- Kx(k), Kr(k)
        #     self.prev_Kx_dot, self.prev_Kr_dot <- Kx_dot(k), Kr_dot(k)
        self._run_adaptation(current_state, current_reference, dt)

        # Control signal
        u_b, u_ad = self._control_law(current_state, current_reference)
        u = u_b + u_ad

        # Save for next time step use
        self.prev_r = current_reference
        self.prev_u, self.prev_u_b, self.prev_u_ad = u, u_b, u_ad

        return {
            'u': u,
            'mrac': {
                'u_b'  : u_b,
                'u_ad' : u_ad,
                'x_m'  : self.xm,
                'K_x'  : self.Kx,
                'K_r'  : self.Kr,
            }
        }
