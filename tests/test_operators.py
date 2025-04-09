"""
Tests for the GRAPE package.
"""

import jax.numpy as jnp
import pytest

from feedback_grape.utils.operators import identity, sigmax, sigmay, sigmaz

# Check documentation for pytest for more decorators
# TODO: add tests for changing dimesions
# TODO: see if you can add tests related to a certain file in its directory


def test_sigmax():
    """
    Test the sigmax function.
    """
    result = sigmax()
    expected = jnp.array([[0, 1], [1, 0]])
    assert (result == expected).all()


def test_sigmay():
    """
    Test the sigmay function.
    """
    result = sigmay()
    expected = jnp.array([[0, -1j], [1j, 0]])
    assert (result == expected).all()


def test_sigmaz():
    """
    Test the sigmaz function.
    """
    result = sigmaz()
    expected = jnp.array([[1, 0], [0, -1]])
    assert (result == expected).all()


@pytest.mark.parametrize(
    "dimensions, y",
    [
        (2, jnp.array([[1, 0], [0, 1]])),
        (
            4,
            jnp.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            ),
        ),
    ],
)
def test_identity(dimensions, y):
    """
    Test the identity function.
    """
    result = identity(dimensions)
    expected = y
    assert (result == expected).all()
