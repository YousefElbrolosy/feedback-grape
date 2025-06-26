"""
This module define some basic quantum gates and their matrix representations.
"""

import jax.numpy as jnp


# Answer : see if we should give user ability to choose the dtype of the gates
#  --> not for such simple gates
# TODO : see if we should jit the gates
def cnot():
    """
    Controlled NOT gate.
    """
    return jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


# TODO: Check for the hadamard definition from qutip for n qubits
def hadamard():
    """
    Hadamard transform operator.
    """
    return jnp.array([[1, 1], [1, -1]]) / jnp.sqrt(2)
