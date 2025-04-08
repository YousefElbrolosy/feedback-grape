import jax.numpy as jnp

from feedback_grape.utils.gates import cnot


def test_cnot():
    """
    Test the CNOT gate.
    """
    # Define the CNOT gate
    cnot_test = cnot()

    # Check the shape of the CNOT gate
    assert cnot_test.shape == (4, 4)

    # Check the values of the CNOT gate
    expected_cnot = jnp.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    )
    assert (cnot_test == expected_cnot).all()
