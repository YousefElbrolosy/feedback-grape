import jax
import jax.numpy as jnp
import pytest
import qutip as qt

from feedback_grape.utils.operators import sigmax, sigmay
from feedback_grape.utils.tensor import tensor

jax.config.update("jax_enable_x64", True)  # Ensure we use 64-bit precision


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


@pytest.mark.parametrize(
    "arrays",
    [
        ([sigmax(), sigmay(), sigmax()]),  # 3x 2D
        ([jnp.array([1, 0]), sigmax(), jnp.array([0, 1])]),  # 1D, 2D, 1D
        ([sigmay(), sigmay(), sigmay(), sigmay()]),  # 4x 2D
    ],
)
def test_tensor_multiple(arrays):
    """
    Test the tensor function with more than two arrays.
    """
    our_implementation = tensor(*arrays)
    qt_implementation = qt.tensor(*[qt.Qobj(a) for a in arrays])
    assert (our_implementation == qt_implementation.full()).all()


def test_tensor_identity():
    """
    Test tensor with identity matrix and vector.
    """
    identity = jnp.eye(2)
    vec = jnp.array([1, 0])
    result = tensor(identity, vec)
    expected = qt.tensor(qt.Qobj(identity), qt.Qobj(vec)).full()
    assert (result == expected).all()


def test_tensor_shape():
    """
    Test that tensor returns correct shape.
    """
    a = jnp.ones((2, 2))
    b = jnp.ones((3, 3))
    result = tensor(a, b)
    assert result.shape == (6, 6)
