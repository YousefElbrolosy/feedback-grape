"""
Module for solving the time-dependent Schrödinger equation and master equation
"""

# TODO: IMPRORTANT: THIS CURRENTLY ALLOWS ONLY FOR UNITARY EVOLUTION
# --> NEED TO ADD MASTER EQUATION EVOLUTION

# ruff: noqa N8
import jax
from feedback_grape.utils.superoperator import lindblad


# TODO: make it more efficient (using ODE methods maybe?)
def sesolve(Hs, initial_state, delta_ts, type="density"):
    """
    Find evolution operator for piecewise Hs on time intervals delta_ts

    Args:
        Hs: List of Hamiltonians for each time interval.
        (time-dependent Hamiltonian)
        initial_state: Initial state.
        delta_ts: List of time intervals.
    Returns:
        U: Evolved state after applying the time-dependent Hamiltonians.

    """

    U_final = initial_state
    if type == "density":
        for _, (H, delta_t) in enumerate(zip(Hs, delta_ts)):
            U_final = (
                jax.scipy.linalg.expm(-1j * delta_t * (H))
                @ U_final
                @ jax.scipy.linalg.expm(-1j * delta_t * (H)).conj().T
            )
        return U_final
    else:
        for _, (H, delta_t) in enumerate(zip(Hs, delta_ts)):
            U_final = jax.scipy.linalg.expm(-1j * delta_t * (H)) @ U_final
        return U_final


# TODO: Add functionality for Liouvillian
def mesolve(Hs, c_ops, rho_0, delta_ts):
    """
    Master equation evolution of a density matrix for a given Hamiltonian and
    an optional set of collapse operators, or a Liouvillian. A Liouvillian is a
    superoperator that accounts for hamiltonian and collapse operators.

    Args:
        Hs: List of Hamiltonians for each time interval.
        (time-dependent Hamiltonian)
        c_ops: List of collapse operators.
        rho_0: Initial density matrix.
        delta_ts: List of time intervals.
    Returns:
        rho_final: Evolved density matrix after applying the time-dependent Hamiltonians.
    """

    def RK4_step(rho, H, c_ops, delta_t):
        """
        Perform a single RK4 step for lindblad master equation evolution.
        """
        k1 = lindblad(H, c_ops, rho)
        k2 = lindblad(H, c_ops, rho + 0.5 * delta_t * k1)
        k3 = lindblad(H, c_ops, rho + 0.5 * delta_t * k2)
        k4 = lindblad(H, c_ops, rho + delta_t * k3)
        return rho + (delta_t / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    rho = rho_0
    for H, delta_t in zip(Hs, delta_ts):
        rho = RK4_step(rho, H, c_ops, delta_t)
    return rho
