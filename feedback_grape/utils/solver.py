"""
Module for solving the time-dependent Schrödinger equation and master equation
"""

# ruff: noqa N8
import jax
import jax.numpy as jnp

# from feedback_grape.utils.superoperator import lindblad
from dynamiqs import mesolve as mesolve_dynamiqs
from .operators import identity
import dynamiqs as dq


def sesolve(Hs, initial_state, delta_ts, type="density"):
    """

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
    # for state vectors and unitary operators
    else:
        for _, (H, delta_t) in enumerate(zip(Hs, delta_ts)):
            U_final = jax.scipy.linalg.expm(-1j * delta_t * (H)) @ U_final
        return U_final


# TODO: see how to make it compatible with liouvillian superoperator
def mesolve(H, jump_ops, rho0, tsave):
    """
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
    dq.set_progress_meter(False)
    if H is None:
        H = [identity(rho0.shape[0]) for _ in range(len(tsave))]
    rho0 = jnp.asarray(rho0, dtype=jnp.complex128)
    # TODO: understand why there is the dimension of the length of the hamiltonian
    # the first [-1] gets the last hamiltonian?
    return (
        mesolve_dynamiqs(
            H=H,
            jump_ops=jump_ops,
            rho0=rho0,
            tsave=tsave,
        ).final_state
    )[-1].data


# def mesolve(H, jump_ops, rho0, time_grid):
#     """
#     an optional set of collapse operators, or a Liouvillian. A Liouvillian is a
#     superoperator that accounts for hamiltonian and collapse operators.

#     Args:
#         H: List of Hamiltonians for each time interval.
#         (time-dependent Hamiltonian)
#         jump_ops: List of collapse operators.
#         rho0: Initial density matrix.
#         time_grid: List of time intervals.
#     Returns:
#         rho_final: Evolved density matrix after applying the time-dependent Hamiltonians.
#     """
#     if H is None:
#         H = [identity(rho0.shape[0]) for _ in range(len(time_grid) - 1)]
#     def RK4_step(rho, H, jump_ops, delta_t):
#         """
#         """
#         k1 = lindblad(H, jump_ops, rho)
#         k2 = lindblad(H, jump_ops, rho + 0.5 * delta_t * k1)
#         k3 = lindblad(H, jump_ops, rho + 0.5 * delta_t * k2)
#         k4 = lindblad(H, jump_ops, rho + delta_t * k3)
#         return rho + (delta_t / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

#     rho = rho0
#     for H, delta_t in zip(H, time_grid):
#         rho = RK4_step(rho, H, jump_ops, delta_t)
#     return rho
