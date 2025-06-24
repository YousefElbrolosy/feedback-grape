"""
This module contains functions for generating common operators
used to generate hamiltonians and unitary transformations.

gates.py are more predefined in terms of dimensions I think?
"""

# TODO : see if we should jit the operators

import jax
import jax.numpy as jnp


def sigmax(dtype=jnp.complex128):
    """
    Pauli X operator.
    """
    return jnp.array([[0, 1], [1, 0]], dtype=dtype)


def sigmay(dtype=jnp.complex128):
    """
    Pauli Y operator.
    """
    return jnp.array([[0, -1j], [1j, 0]], dtype=dtype)


def sigmaz(dtype=jnp.complex128):
    """
    Pauli Z operator.
    """
    return jnp.array([[1, 0], [0, -1]], dtype=dtype)


def sigmap(dtype=jnp.complex128):
    """
    Raising operator.
    """
    return jnp.array([[0, 1], [0, 0]], dtype=dtype)


def sigmam(dtype=jnp.complex128):
    """
    Lowering operator.
    """
    return jnp.array([[0, 0], [1, 0]], dtype=dtype)


def identity(dimensions, *, dtype=jnp.complex128):
    """
    Identity operator.

    Args:
        dimensions (int): Dimensions of the identity operator.
        dtype (dtype): Data type of the identity operator.
    Returns:
        jnp.ndarray: Identity operator of given dimensions and dtype.
    """
    return jnp.eye(dimensions, dtype=dtype)


def _ladder(
    n: int,
    *,
    dagger: bool,
    dtype: jnp.dtype = jnp.complex128,  # type: ignore
) -> jnp.ndarray:
    """
    n-dimensional ladder operator
    """
    values = jnp.sqrt(jnp.arange(1, n, dtype=dtype))
    shift = -1 * dagger + 1 * (not dagger)
    return jnp.diag(values, k=shift)


def create(n: int, dtype: jnp.dtype = jnp.complex128) -> jnp.ndarray:  # type: ignore
    """
    n-dimensional creation operator
    """
    return _ladder(n, dagger=True, dtype=dtype)


def destroy(n: int, dtype: jnp.dtype = jnp.complex128) -> jnp.ndarray:  # type: ignore
    """
    n-dimensional destruction operator
    """
    return _ladder(n, dagger=False, dtype=dtype)


def cosm(a: jnp.ndarray) -> jnp.ndarray:
    """
    Cosine of a matrix.
    """
    return (jax.scipy.linalg.expm(1j * a) + jax.scipy.linalg.expm(-1j * a)) / 2


def sinm(a: jnp.ndarray) -> jnp.ndarray:
    """
    Sine of a matrix.
    """
    return (jax.scipy.linalg.expm(1j * a) - jax.scipy.linalg.expm(-1j * a)) / (
        2j
    )
