"""
Module for solving the time-dependent Schrödinger equation and master equation
"""

# TODO: IMPRORTANT: THIS CURRENTLY ALLOWS ONLY FOR UNITARY EVOLUTION
# --> NEED TO ADD MASTER EQUATION EVOLUTION

# ruff: noqa N8
import jax
from feedback_grape.utils.fidelity import _isket


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
    if(type == "density"):
        for _, (H, delta_t) in enumerate(zip(Hs, delta_ts)):
            U_final = jax.scipy.linalg.expm(-1j * delta_t * (H)) @ U_final @ jax.scipy.linalg.expm(-1j * delta_t * (H)).conj().T
        return U_final
    else:
        for _, (H, delta_t) in enumerate(zip(Hs, delta_ts)):
            U_final = jax.scipy.linalg.expm(-1j * delta_t * (H)) @ U_final
        return U_final

# TODO: Add functionality for Liouvillian
def mesolve(Hs, rho_0, delta_ts):
    """
    Master equation evolution of a density matrix for a given Hamiltonian and
    an optional set of collapse operators, or a Liouvillian. A Liouvillian is a
    superoperator that accounts for hamiltonian and collapse operators.

    Args:
        Hs: List of Hamiltonians for each time interval.
        (time-dependent Hamiltonian)
        rho_0: Initial density matrix.
        delta_ts: List of time intervals.
    Returns:
        rho_final: Evolved density matrix after applying the time-dependent Hamiltonians.
    """
    if _isket(rho_0):
        raise ValueError(
            "rho_0 should be a density matrix, not a state vector."
        )
    rho_final = rho_0
    for _, (H, delta_t) in enumerate(zip(Hs, delta_ts)):
        rho_final = (
            jax.scipy.linalg.expm(-1j * delta_t * (H))
            @ rho_final
            @ jax.scipy.linalg.expm(-1j * delta_t * (H)).conj().T
        )
        print("shape of rho_final: ", rho_final.shape)
    return rho_final
