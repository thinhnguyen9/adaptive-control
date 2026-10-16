"""
Basic functions only, should not import other modules.
"""
import numpy as np


def integrate(fun, x, u, dt, method='euler'):
    if method == 'euler':
        x_next = x + dt * fun(x, u)
        return x_next
    elif method == 'rk4':
        k1 = fun(x, u)
        k2 = fun(x + dt/2 * k1, u)
        k3 = fun(x + dt/2 * k2, u)
        k4 = fun(x + dt * k3, u)
        x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        return x_next
    else:
        raise ValueError(f"Unknown integration method: {method}") 

