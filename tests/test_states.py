import jax
import jax.numpy as jnp
import pytest
import qutip as qt

from feedback_grape.utils.states import basis, coherent

jax.config.update("jax_enable_x64", True)


def test_basis():
    """
    Test the basis function.
    """
    result = basis(2, 0)
    expected = qt.basis(2, 0).full()
    assert (result == expected).all()

    result = basis(2, 1)
    expected = qt.basis(2, 1).full()
    assert (result == expected).all()


@pytest.mark.parametrize("n, alpha", [(2, 1), (10, 0.5), (4, 0.002)])
def test_coherent_parametrized(n, alpha):
    """
    Test the coherent function with parametrized inputs.
    """
    result = coherent(n, alpha)
    expected = (
        qt.coherent(n, alpha, method="analytic")
        .full()
        .flatten()
        .reshape(-1, 1)
    )
    print(f"result: {result}, \n expected: {expected}")
    assert jnp.allclose(result, expected, atol=1e-4), (
        "The coherent state is not close enough to qutip's."
    )
