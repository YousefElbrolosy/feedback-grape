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


# TODO: Add functionality for supplying H and c_ops and then doing the evolution
def mesolve_1(Hs, c_ops, rho_0, delta_ts):
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


def mesolve(L, rho_0, delta_ts):
    """
    Master equation evolution of a density matrix for a given Liouvillian.

    Args:
        L: Liouvillian superoperator.
        rho_0: Initial density matrix.
        delta_ts: List of time intervals.
    Returns:
        rho_final: Evolved density matrix after applying the Liouvillian.
    """

    def RK4_step(rho, L, delta_t):
        """
        Perform a single RK4 step for lindblad master equation evolution.
        """
        k1 = L @ rho
        k2 = L @ (rho + 0.5 * delta_t * k1)
        k3 = L @ (rho + 0.5 * delta_t * k2)
        k4 = L @ (rho + delta_t * k3)
        return rho + (delta_t / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    rho = rho_0
    for Ls, delta_t in zip(L, delta_ts):
        rho = RK4_step(rho, Ls, delta_t)
    return rho
