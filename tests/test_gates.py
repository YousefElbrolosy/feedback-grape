import qutip as qt
import qutip_qip.operations.gates as qip

import feedback_grape.utils.gates as fg


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
