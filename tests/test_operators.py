"""
Tests for the GRAPE package.
"""

import jax.numpy as jnp
import pytest
import qutip as qt

from feedback_grape.utils.operators import (
    create,
    destroy,
    identity,
    sigmam,
    sigmap,
    sigmax,
    sigmay,
    sigmaz,
)

# Check documentation for pytest for more decorators
# TODO: add tests for changing dimesions and data type
# TODO: see if you can add tests related to a certain file in its directory


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
