"""
This module define some basic quantum gates and their matrix representations.
"""

import jax.numpy as jnp


# TODO : see if we should give user ability to choose the dtype of the gates
# TODO : see if we should jit the gates
def cnot():
    """
    Controlled NOT gate.
    """
    return jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
