import qutip as qt

from feedback_grape.utils.states import basis


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
