# ruff: noqa N8
import jax
from feedback_grape.utils.superoperator import sprepost
from feedback_grape.utils.operators import sigmax, sigmay
import qutip as qt
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def test_sprepost():
    """
    Test the sprepost function.
    """
    # Define operators
    a = sigmax()
    b = sigmay()

    # Compute the superoperator
    S = sprepost(a, b)

    # Check the shape of the superoperator
    assert S.shape == (4, 4), "Sprepost shape mismatch"

    # Compare with QuTiP
    qt_S = qt.sprepost(qt.Qobj(a), qt.Qobj(b))
    assert jnp.allclose(S, qt_S.full(), atol=1e-10), (
        "Sprepost computation mismatch with QuTiP"
    )
