"""
Module to create composite quantum objects via tensor product.

Definition according to Nelsen and Chuang: The tensor product is a way of
putting vector spaces together to form larger vector spaces.
This construction is crucial to understanding the quantum mechanics of multi-
particle systems.
"""

import jax.numpy as jnp

# TODO : Add proper citation for Nelsen and Chuang's book
# TODO: Check if the output would rather be n object of our creation
# TODO: make tensor work for n dimensions
# TODO: check how tensor can take only one arg


def tensor(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the tensor/Kronecker product of two quantum objects.

    Args:
        a (jnp.ndarray): First quantum object.
        b (jnp.ndarray): Second quantum object.
    Returns:
        jnp.ndarray: The resulting quantum object after applying the
        tensor product.
    """
    # Ensure inputs are reshaped properly for Kronecker product
    a = a.reshape(-1, 1) if a.ndim == 1 else a
    b = b.reshape(-1, 1) if b.ndim == 1 else b
    return jnp.kron(a, b)

def tensor_einsum(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the tensor/Kronecker product of two quantum objects.

    Args:
        a (jnp.ndarray): First quantum object.
        b (jnp.ndarray): Second quantum object.
    Returns:
        jnp.ndarray: The resulting quantum object after applying the
        tensor product.
    """
    # Check if a and b are 1D arrays
    if a.ndim == 1 and b.ndim == 1:
        return jnp.outer(a, b).reshape(a.shape[0] * b.shape[0], 1 * 1)
    # Check if a is a 1D array and b is a 2D array
    elif a.ndim == 1 and b.ndim == 2:
        return jnp.einsum("i,jk->ijk", a, b).reshape(
            a.shape[0] * b.shape[0], 1 * b.shape[1]
        )
    # Check if a is a 2D array and b is a 1D array
    elif a.ndim == 2 and b.ndim == 1:
        return jnp.einsum("ij,k->ikj", a, b).reshape(
            a.shape[0] * b.shape[0], a.shape[1] * 1
        )
    # Both a and b are 2D arrays
    else:
        return jnp.einsum("ij,kl->ikjl", a, b).reshape(
            a.shape[0] * b.shape[0], a.shape[1] * b.shape[1]
        )
