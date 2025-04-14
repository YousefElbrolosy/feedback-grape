"""
This module contains functions to implement some basic quantum states
"""

import jax
import jax.numpy as jnp


def basis(n, k=0):
    """
    Basis state in n-dimensional Hilbert space
    """
    one_hot = jax.nn.one_hot(k, n, dtype=jnp.complex64)
    return one_hot.reshape(n, 1)
