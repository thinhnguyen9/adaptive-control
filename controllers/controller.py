from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class Controller(ABC):
    
    @abstractmethod
    def compute_control(
        self,
        current_state: np.ndarray,
        current_reference: np.ndarray,
        dt: float,
    ) -> dict[str, Any]:
        """Compute the control input at each time step"""
        pass

    def _projection(self, theta, y, theta_max, tol):
        """
        Apply projection operator on each column of theta.

        Args:
            theta, y: (column) vectors in R^n
            theta_max: ||theta|| <= thetamax
            tol: convergence tolerance
        """

        m = theta.shape[1]
        out = np.zeros((theta.shape[0], m))
        for col in range(m):
            theta_col = theta[:, col]
            y_col = y[:, col]
            thetamax = theta_max[col]
            tol_col = tol[col]

            f = (theta_col @ theta_col - thetamax**2) / (tol_col * thetamax**2)
            grad_f = 2*theta_col / (tol_col * thetamax**2)
            grad_f_y = grad_f @ y_col
            
            if f >= 0 and grad_f_y > 0:
                out[:, col] = y_col - grad_f / (np.linalg.norm(grad_f)**2) * grad_f_y * f
            else:
                out[:, col] = y_col
        return out
