# ruff: noqa N8
from feedback_grape.utils.superoperator import liouvillian, sprepost
from feedback_grape.utils.operators import sigmax, sigmay, sigmaz, sigmam
import qutip as qt
import jax.numpy as jnp


# TODO: improve coverage
def test_liouvillan():
    """
    Test the Liouvillian superoperator function.
    """

    Sm = sigmam()

    # Hamiltonian
    Del = 0.1  # Tunnelling term
    wq = 1.0  # Energy of the 2-level system.
    H0 = 0.5 * wq * sigmaz() + 0.5 * Del * sigmax()

    # Amplitude damping#
    # Damping rate:
    gamma = 0.1
    L0 = liouvillian(H0, [jnp.sqrt(gamma) * Sm])

    # Drift
    drift = L0

    # Check the shape of the Liouvillian
    assert drift.shape == (4, 4), "Liouvillian shape mismatch"

    # Check if the Liouvillian is Hermitian
    assert not jnp.allclose(drift, drift.conj().T), (
        "Liouvillian is Hermitian while it should be not"
    )

    qt_L0 = qt.liouvillian(qt.Qobj(H0), [jnp.sqrt(gamma) * qt.Qobj(Sm)])
    assert jnp.allclose(drift, qt_L0.full(), atol=1e-1), (
        "Liouvillian computation mismatch with QuTiP"
    )


def test_liouvillian_2():
    """
    Test the Liouvillian superoperator function.
    """

    # Simple Hamiltonian
    H_simple = sigmaz()

    # No collapse operators
    L_simple = liouvillian(H_simple, [])

    # Check the shape of the Liouvillian
    assert L_simple.shape == (4, 4), (
        "Liouvillian shape mismatch for simple case"
    )

    # Check if the Liouvillian is Hermitian
    assert not jnp.allclose(L_simple, L_simple.conj().T), (
        "Liouvillian is not Hermitian for simple case"
    )

    # Compare with QuTiP
    qt_L_simple = qt.liouvillian(qt.Qobj(H_simple), [])
    assert jnp.allclose(L_simple, qt_L_simple.full(), atol=1e-10), (
        "Liouvillian computation mismatch with QuTiP for simple case"
    )


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
