"""
This module contains functions for generating common operators
used to generate hamiltonians and unitary transformations.

gates.py are more predefined in terms of dimensions I think?
"""

# TODO : see if we should give user ability to choose the dtype of the gates
# TODO : see if we should jit the operators

import jax.numpy as jnp


def sigmax():
    """
    Pauli X operator.
    """
    return jnp.array([[0, 1], [1, 0]])


def sigmay():
    """
    Pauli Y operator.
    """
    return jnp.array([[0, -1j], [1j, 0]])


def sigmaz():
    """
    Pauli Z operator.
    """
    return jnp.array([[1, 0], [0, -1]])


def sigmap():
    """
    Raising operator.
    """
    return jnp.array([[0, 1], [0, 0]])


def sigmam():
    """
    Lowering operator.
    """
    return jnp.array([[0, 0], [1, 0]])


# TODO : check for exact dimensions since dynamiqs and qutip support nested
# TODO : Hilbert space dimensions and so on.
def identity(dimensions, *, dtype=jnp.float32):
    """
    Identity operator.

    Args:
        dimensions (int): Dimensions of the identity operator.
        dtype (dtype): Data type of the identity operator.
    Returns:
        jnp.ndarray: Identity operator of given dimensions and dtype.
    """
    return jnp.eye(dimensions, dtype=dtype)


def _ladder(n, *, dagger: bool):
    """
    n-dimensional ladder operator
    """
    values = jnp.sqrt(jnp.arange(1, n, dtype=jnp.complex64))
    shift = -1 * dagger + 1 * (not dagger)
    return jnp.diag(values, k=shift)


def create(n):
    """
    n-dimensional creation operator
    """
    return _ladder(n, dagger=True)


def destroy(n):
    """
    n-dimensional destruction operator
    """
    return _ladder(n, dagger=False)
