import jax.numpy as jnp
import pytest
import qutip as qt

from feedback_grape.utils.operators import sigmax, sigmay
from feedback_grape.utils.tensor import tensor

# TODO: further tests


@pytest.mark.parametrize(
    "a, b",
    [
        (sigmax(), sigmay()),  # 2D by 2D
        (sigmay(), sigmax()),  # 2D by 2D
        (jnp.array([1, 0]), sigmax()),  # 1D by 2D
        (sigmax(), jnp.array([1, 0])),  # 2D by 1D
        (jnp.array([1, 0]), jnp.array([0, 1])),  # 1D by 1D
    ],
)
def test_tensor(a: jnp.ndarray, b: jnp.ndarray):
    """
    Test the tensor function.
    """
    our_implementation = tensor(a, b)
    qt_implementation = qt.tensor(qt.Qobj(a), qt.Qobj(b))
    assert (our_implementation == qt_implementation.full()).all()
