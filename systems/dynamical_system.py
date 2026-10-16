from abc import ABC, abstractmethod
import numpy as np


class DynamicalSystem(ABC):
    
    @abstractmethod
    def state_dot(
        self,
        current_state: np.ndarray,
        current_input: np.ndarray
    ) -> np.ndarray:
        """Time derivative of the state vector"""
        pass

    @abstractmethod
    def linearize(
        self,
        nominal_state: np.ndarray,
        nominal_input: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Linearize the system around a nominal trajectory"""
        pass

    @abstractmethod
    def get_output(
        self,
        current_state: np.ndarray
    ) -> np.ndarray:
        """Get the output vector"""
        pass
