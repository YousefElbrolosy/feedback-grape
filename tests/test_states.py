# ruff: noqa
import jax
import jax.numpy as jnp
import pytest
import qutip as qt
from feedback_grape.utils.states import basis, coherent, fock
from feedback_grape.utils.operators import jaxify

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


@pytest.mark.parametrize("n", [2, 10, 4, 29])
def test_fock(n):
    N_cav = 30
    result = fock(N_cav, n)
    expected = qt.fock(N_cav, n).full().flatten().reshape(-1, 1)
    print(f"result: {result}, \n expected: {expected}")
    assert jnp.allclose(result, expected, atol=1e-10), (
        "The fock state is not close enough to qutip's."
    )


@pytest.mark.parametrize(
    "jax_state, qt_state",
    [
        (basis(2, 0), qt.basis(2, 0)),
        (basis(2, 1), qt.basis(2, 1)),
        (coherent(4, 1.0), qt.coherent(4, 1.0, method="analytic")),
        (fock(5, 3), qt.fock(5, 3)),
    ],
)
def test_jaxify_states(jax_state, qt_state):
    """
    Test the jaxify function to ensure it converts Qobj states to JAX arrays correctly.
    """
    result = jaxify(qt_state)
    expected = jax_state

    assert jnp.allclose(result, expected, atol=1e-10), (
        f"jaxify result: {result}, expected: {expected}"
    )
    assert result.dtype == expected.dtype, (
        f"jaxify dtype mismatch: {result.dtype} != {expected.dtype}"
    )
