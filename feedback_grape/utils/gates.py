"""
This module define some basic quantum gates and their matrix representations.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# Answer : see if we should give user ability to choose the dtype of the gates
#  --> not for such simple gates also user can always write his own gate

# JIT compilation is generally beneficial for functions that are
# computationally intensive and called repeatedly with the same
# input shapes/types. For simple operator generators like these
# (mostly returning small constant arrays), JIT will not provide
# significant speedup and may add unnecessary overhead. Only
# consider jitting functions that are performance bottlenecks
# in practice, such as those involving large matrix operations
# or repeated numerical computation.


def cnot():
    """
    Controlled NOT gate.
    """
    return jnp.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=jnp.complex128,
    )


# Answer: Check for the hadamard definition from qutip for n qubits user can
# use qutip's then use jaxify
def hadamard():
    """
    Hadamard transform operator.
    """
    return jnp.array([[1, 1], [1, -1]], dtype=jnp.complex128) / jnp.sqrt(2)
