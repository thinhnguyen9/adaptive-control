# adaptive-control
Implementation of MRAC and $\mathcal{L}_1$ adaptive controllers for multiple applications. Test the controllers on a [`LinearSystem`](systems/linear_system.py), [`Quadrotor_2D`](systems/quadrotor_2d.py), or define a custom [`DynamicalSystem`](systems/dynamical_system.py) of your own.


# examples
- [`examples/ex1.py`](examples/ex1.py): direct MRAC on a double integrator.
- [`examples/ex2.py`](examples/ex2.py): LQR control on a 2D quadrotor.

Run an example from the repository root:
```bash
python examples/ex1.py
python examples/ex2.py
```
