"""
Tests for the GRAPE package.
"""

import pytest
import qutip as qt

from feedback_grape.utils.operators import identity, sigmax, sigmay, sigmaz

# Check documentation for pytest for more decorators
# TODO: add tests for changing dimesions
# TODO: see if you can add tests related to a certain file in its directory


def test_sigmax():
    """
    Test the sigmax function.
    """
    result = sigmax()
    expected = qt.sigmax().full()
    assert (result == expected).all()


def test_sigmay():
    """
    Test the sigmay function.
    """
    result = sigmay()
    expected = qt.sigmay().full()
    assert (result == expected).all()


def test_sigmaz():
    """
    Test the sigmaz function.
    """
    result = sigmaz()
    expected = qt.sigmaz().full()
    assert (result == expected).all()


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
    assert (result == expected).all()
