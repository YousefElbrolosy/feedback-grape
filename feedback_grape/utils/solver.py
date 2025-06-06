"""
Module for solving the time-dependent Schrödinger equation and master equation
"""


# ruff: noqa N8
import jax
import jax.numpy as jnp
from feedback_grape.utils.superoperator import lindblad
from dynamiqs import mesolve as mesolve_dynamiqs

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
def mesolve(H, jump_ops, rho0, tsave):
    """
    Master equation evolution of a density matrix for a given Hamiltonian and
    an optional set of collapse operators, or a Liouvillian. A Liouvillian is a
    superoperator that accounts for hamiltonian and collapse operators.

    Args:
        H: List of Hamiltonians for each time interval.
        (time-dependent Hamiltonian)
        jump_ops: List of collapse operators.
        rho0: Initial density matrix.
        tsave: List of time intervals.
    Returns:
        rho_final: Evolved density matrix after applying the time-dependent Hamiltonians.
    """
    return mesolve_dynamiqs(
        H=H,
        jump_ops=jump_ops,
        rho0=rho0,
        tsave=tsave,
    ).states[-1].data