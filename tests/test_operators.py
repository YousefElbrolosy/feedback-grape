"""
Tests for the GRAPE package.
"""
# ruff: noqa
import jax
import jax.numpy as jnp
import pytest
import qutip as qt

from feedback_grape.utils.operators import (
    cosm,
    create,
    destroy,
    identity,
    jaxify,
    sigmam,
    sigmap,
    sigmax,
    sigmay,
    sigmaz,
    sinm,
)

jax.config.update("jax_enable_x64", True)  # Ensure we use 64-bit precision

# Check documentation for pytest for more decorators


# Here I do not want to use isclose, because those are so elementary that
# They should be exactly the same
def test_sigmax():
    """
    Test the sigmax function.
    """
    result = sigmax()
    expected = qt.sigmax().full()
    assert jnp.allclose(result, expected)


def test_sigmay():
    """
    Test the sigmay function.
    """
    result = sigmay()
    expected = qt.sigmay().full()
    assert jnp.allclose(result, expected)


def test_sigmaz():
    """
    Test the sigmaz function.
    """
    result = sigmaz()
    expected = qt.sigmaz().full()
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize(
    "dimensions, y",
    [
        (2, qt.identity(2).full()),
        (
            4,
            qt.identity(4).full(),
        ),
    ],
)
def test_identity(dimensions, y):
    """
    Test the identity function.
    """
    result = identity(dimensions)
    expected = y
    assert jnp.allclose(result, expected)


def test_sigmap():
    """
    Test the sigmap function.
    """
    result = sigmap()
    expected = qt.sigmap().full()
    assert jnp.allclose(result, expected)


def test_sigmam():
    """
    Test the sigmam function.
    """
    result = sigmam()
    expected = qt.sigmam().full()
    assert jnp.allclose(result, expected)


def test_create():
    """
    Test the create function.
    """
    result = create(4)
    expected = qt.create(4).full()
    print(f"{result}")
    print(f"{expected}")
    assert jnp.allclose(result, expected)


def test_destroy():
    """
    Test the destroy function.
    """
    result = destroy(4)
    expected = qt.destroy(4).full()
    # default is 1e-8
    assert jnp.allclose(result, expected)


def povm_measure_operator(measurement_outcome, gamma, delta):
    number_operator = create(4) @ destroy(4)
    angle = (gamma * number_operator) + delta / 2
    return jnp.where(
        measurement_outcome == 1,
        cosm(angle),
        sinm(angle),
    )


def test_cosm():
    """
    Test the cosm function.
    """
    result = cosm(povm_measure_operator(1, 1.0, 0.5))
    expected = qt.Qobj(povm_measure_operator(1, 1.0, 0.5)).cosm().full()
    print(f"result: {result}")
    print(f"expected: {expected}")
    assert jnp.allclose(result, expected), (
        f"cosm result: {result}, expected: {expected}"
    )


def test_sinm():
    """
    Test the sinm function.
    """
    result = sinm(povm_measure_operator(1, 1.0, 0.5))
    expected = qt.Qobj(povm_measure_operator(1, 1.0, 0.5)).sinm().full()
    print(f"result: {result}")
    print(f"expected: {expected}")
    assert jnp.allclose(result, expected), "Not Close enough"


@pytest.mark.parametrize(
    "func, expected, dtype",
    [
        (sigmax, qt.sigmax().full().astype(jnp.float32), jnp.float32),
        (sigmay, qt.sigmay().full().astype(jnp.complex64), jnp.complex64),
        (sigmaz, qt.sigmaz().full().astype(jnp.float32), jnp.float32),
        (sigmap, qt.sigmap().full().astype(jnp.complex64), jnp.complex64),
        (sigmam, qt.sigmam().full().astype(jnp.complex64), jnp.complex64),
    ],
)
def test_pauli_dtype(func, expected, dtype):
    """
    Test Pauli and ladder operators with different dtypes.
    """
    result = func(dtype=dtype)
    assert result.dtype == dtype
    assert jnp.allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize(
    "dimensions, dtype",
    [
        (2, jnp.float32),
        (3, jnp.complex64),
        (5, jnp.complex128),
    ],
)
def test_identity_dtype(dimensions, dtype):
    """
    Test identity with different dimensions and dtypes.
    """
    result = identity(dimensions, dtype=dtype)
    expected = jnp.eye(dimensions, dtype=dtype)
    assert result.dtype == dtype
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize(
    "n, dtype",
    [
        (2, jnp.complex64),
        (3, jnp.complex128),
        (5, jnp.complex64),
    ],
)
def test_create_destroy_dtype(n, dtype):
    """
    Test create and destroy with different dimensions and dtypes.
    """
    result_create = create(n, dtype=dtype)
    expected_create = qt.create(n).full().astype(dtype)
    assert result_create.dtype == dtype
    assert jnp.allclose(result_create, expected_create, atol=1e-6)

    result_destroy = destroy(n, dtype=dtype)
    expected_destroy = qt.destroy(n).full().astype(dtype)
    assert result_destroy.dtype == dtype
    assert jnp.allclose(result_destroy, expected_destroy, atol=1e-6)


@pytest.mark.parametrize(
    "jax_array, qt_operator",
    [
        (identity(2), qt.qeye(2)),
        (sigmax(), qt.sigmax()),
        (sigmay(), qt.sigmay()),
        (sigmaz(), qt.sigmaz()),
        (sigmap(), qt.sigmap()),
        (sigmam(), qt.sigmam()),
        (create(4), qt.create(4)),
        (destroy(4), qt.destroy(4)),
    ],
)
def test_jaxify(jax_array, qt_operator):
    """
    Test the jaxify function to ensure it converts Qobj to JAX arrays correctly.
    """
    result = jaxify(qt_operator)
    expected = jax_array
    assert jnp.allclose(result, expected), (
        f"jaxify result: {result}, expected: {expected}"
    )
    assert result.dtype == expected.dtype, (
        f"jaxify dtype mismatch: {result.dtype} != {expected.dtype}"
    )
