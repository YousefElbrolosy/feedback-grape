"""
Module to create composite quantum objects via tensor product.

Definition according to Nelsen and Chuang: The tensor product is a way of
putting vector spaces together to form larger vector spaces.
This construction is crucial to understanding the quantum mechanics of multi-
particle systems.
"""

import jax.numpy as jnp


def tensor(*args: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the tensor/Kronecker product of two or more quantum objects.

    Args:
        *args (jnp.ndarray): Arrays to be tensored together.
    Returns:
        jnp.ndarray: The resulting quantum object after applying the
        tensor product.
    """
    # Ensure inputs are reshaped properly for Kronecker product
    if len(args) < 2:
        return args[0]
    args = [arg.reshape(-1, 1) if arg.ndim == 1 else arg for arg in args]  # type: ignore
    result = args[0]
    for arg in args[1:]:
        result = jnp.kron(result, arg)
    return result
