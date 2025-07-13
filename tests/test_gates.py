import jax
import pytest
import qutip as qt
import qutip_qip.operations.gates as qip

import feedback_grape.utils.gates as fg
from feedback_grape.utils.gates import cnot, hadamard
from feedback_grape.utils.operators import jaxify

# ruff: noqa

jax.config.update("jax_enable_x64", True)  # Ensure we use 64-bit precision


def test_cnot():
    """
    Test the CNOT gate.
    """
    # Define the CNOT gate
    cnot_test = fg.cnot()

    # Check the shape of the CNOT gate
    assert cnot_test.shape == (4, 4)

    # Check the values of the CNOT gate
    expected_cnot = qt.core.gates.cnot().full()
    assert (cnot_test == expected_cnot).all()


def test_hadamard():
    """
    Test the Hadamard gate.
    """
    # Define the Hadamard gate
    hadamard_test = fg.hadamard()

    # Check the values of the Hadamard gate
    expected_hadamard = qip.hadamard_transform(1).full()
    assert (hadamard_test == expected_hadamard).all()


import jax.numpy as jnp


@pytest.mark.parametrize(
    "gate_func, qt_gate_func",
    [
        (cnot, qt.core.gates.cnot),
        (hadamard, lambda: qip.hadamard_transform(1)),
    ],
)
def test_jaxify_gates(gate_func, qt_gate_func):
    """
    Test the jaxify function to ensure it converts Qobj gates to JAX arrays correctly.
    """
    qt_gate = qt_gate_func()
    jax_gate = gate_func()
    jaxified_qt_gate = jaxify(qt_gate)

    assert jnp.allclose(jax_gate, jaxified_qt_gate, atol=1e-10), (
        f"jax_gate: {jax_gate}, jaxified_qt_gate: {jaxified_qt_gate}"
    )
    assert jax_gate.dtype == jaxified_qt_gate.dtype, (
        f"dtype mismatch: {jax_gate.dtype} != {jaxified_qt_gate.dtype}"
    )
