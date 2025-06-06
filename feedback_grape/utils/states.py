"""
This module contains functions to implement some basic quantum states
"""

from functools import reduce

import jax
import jax.numpy as jnp

from feedback_grape.utils.operators import create


def basis(n, k=0):
    """
    Basis state in n-dimensional Hilbert space.
    """
    one_hot = jax.nn.one_hot(k, n, dtype=jnp.complex128)
    return one_hot.reshape(n, 1)


# This can also be implemented using coherent state as a displacement from
# the ground state, which is also a ground state
def coherent(n: int, alpha: complex) -> jnp.ndarray:
    """
    coherent state; ie: eigenstate of a lowering operator.

    Parameters
    ----------
    n : int

    alpha : float/complex
        Eigenvalue of coherent state.

    Returns
    -------
    jnp.ndarray
        Coherent state in n-dimensional Hilbert space.

    Notes
    -----
    The state `|n⟩` represents the energy eigenstate (or number state)
    of the quantum harmonic oscillator with exactly n excitations
    (or n quanta/particles).
    This is also known as the Fock state. (where if 0th index is 1 then
    ground state, 1st index is 1 then 1 energy quanta
    (or photon in a cavity), etc.)

    """
    norm_factor = jnp.exp((-1 * jnp.abs(alpha) ** 2.0) / 2)

    indices = jnp.arange(n)

    alpha_powers = jnp.power(alpha, indices)

    sqrt_factorials = jnp.sqrt(jax.scipy.special.factorial(indices))

    coeffs = alpha_powers / sqrt_factorials

    coherent_state = coeffs * norm_factor

    return coherent_state.reshape(-1, 1)


def fock(n: int, n_cav: int) -> jnp.ndarray:
    """
    Creates a Fock state |n_cav⟩ in an n-dimensional Hilbert space.
    """
    return basis(n, n_cav)


# TODO: confirm that implementation is indeed correct
# TODO: This can actually be implemented related to basis states
# TODO: test and compare with the qutip implementation
# TODO: see if can be improved
def fock_2(n: int, n_cav: int) -> jnp.ndarray:
    """
    Defines a fock state
    """
    numerator = reduce(jnp.matmul, [create(n) for _ in range(n_cav)])
    denominator = jnp.pow(jax.scipy.special.factorial(n_cav), (0.5))
    fock_state = (numerator / denominator) @ basis(n)
    return fock_state.reshape(-1, 1)
