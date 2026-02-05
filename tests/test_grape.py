"""
Tests for the GRAPE package.
"""

# ruff: noqa
import re
import jax
import jax.numpy as jnp
import pytest
import qutip as qt

from feedback_grape.utils.fidelity import fidelity
from feedback_grape.utils.fidelity import sqrtm_eig
from feedback_grape.utils.fidelity import is_positive_semi_definite
from feedback_grape.grape import optimize_pulse
from feedback_grape.utils.fidelity import ket2dm
from feedback_grape.utils.states import basis
from tests.helper_for_tests import (
    get_finals,
    get_results_for_cnot_problem,
    get_results_for_density_example,
    get_results_for_hadamard_problen,
    get_results_for_qubit_in_cavity_problem,
    get_targets_for_cnot_problem,
    get_targets_for_density_example,
    get_targets_for_hadamard_problem,
    get_targets_for_qubit_in_cavity_problem,
    get_results_for_new_dissipation_problem,
)

jax.config.update("jax_enable_x64", True)


# Check documentation for pytest for more decorators


# Testing target Operator transformations


def test_cnot(optimizer="l-bfgs", propcomp="time-efficient"):
    result_fg, result_qt = get_results_for_cnot_problem(optimizer, propcomp)
    # print("reference.evo_full_final.full()): ",reference.evo_full_final.full())
    # print("fg result: ",result)
    # assert jnp.allclose(result["final_operator"], reference.evo_full_final.full(), atol=1e-1), "The matrices are not close enough."
    print("reference.fidelity: ", result_qt.fid_err)
    print("fg result[final_fidelity]: ", result_fg.final_fidelity)
    assert jnp.allclose(
        1 - result_fg.final_fidelity, result_qt.fid_err, atol=1e-3
    ), "The fidelities are not close enough."


@pytest.mark.parametrize(
    "optimizer, propcomp",
    [
        ("adam", "memory-efficient"),
        ("l-bfgs", "memory-efficient"),
        ("adam", "time-efficient"),
        ("l-bfgs", "time-efficient"),
    ],
)
def test_hadamard(optimizer, propcomp):
    result_fg, result_qt = get_results_for_hadamard_problen(
        optimizer, propcomp
    )
    print("result_qt.fid_err: ", result_qt.fid_err)
    print("result_fg.final_fidelity: ", result_fg.final_fidelity)
    assert jnp.allclose(
        1 - result_fg.final_fidelity, result_qt.fid_err, atol=1e-3
    ), "The fidelities are not close enough."


# Testing states
@pytest.mark.parametrize(
    "optimizer, propcomp",
    [
        ("adam", "memory-efficient"),
        ("l-bfgs", "memory-efficient"),
        ("adam", "time-efficient"),
        ("l-bfgs", "time-efficient"),
    ],
)
def test_qubit_in_cavity(optimizer, propcomp):
    result_fg, result_qt = get_results_for_qubit_in_cavity_problem(
        optimizer, propcomp
    )
    print("result_qt.fid: ", 1 - result_qt.fid_err)
    print("result_fg.final_fidelity: ", result_fg.final_fidelity)
    assert jnp.allclose(
        1 - result_fg.final_fidelity, result_qt.fid_err, atol=1e-1
    ), "The fidelities are not close enough."


def test_new_dissipative_model():
    """
    Test the new dissipative model.
    """
    # This test is not parametrized because it is a specific test for the new dissipative model
    # that does not require different optimizers or propcomps.
    result_fg = get_results_for_new_dissipation_problem(
        "adam", "memory-efficient"
    )
    print("result_fg.final_fidelity: ", result_fg.final_fidelity)
    assert result_fg.final_fidelity > 0.9, (
        "The final fidelity is not high enough."
    )


@pytest.mark.parametrize(
    "optimizer, propcomp",
    [
        ("adam", "memory-efficient"),
        ("l-bfgs", "memory-efficient"),
        ("adam", "time-efficient"),
        ("l-bfgs", "time-efficient"),
    ],
)
def test_density_example(optimizer, propcomp):
    """
    Test the density matrix function.
    """
    result_fg, result_qt = get_results_for_density_example(optimizer, propcomp)
    print("result_qt.fid_err: ", 1 - result_qt.fid_err)
    print("result_fg.final_fidelity: ", result_fg.final_fidelity)
    assert jnp.allclose(
        1 - result_fg.final_fidelity, result_qt.fid_err, atol=1e-1
    ), "The fidelities are not close enough."


@pytest.mark.parametrize(
    "fid_type, target, final",
    [
        (
            "state",
            get_targets_for_qubit_in_cavity_problem("state")[0],
            get_finals(
                *get_results_for_qubit_in_cavity_problem(
                    "l-bfgs", "time-efficient"
                )
            )[0],
        ),
        (
            "unitary",
            get_targets_for_cnot_problem()[0],
            get_finals(
                *get_results_for_cnot_problem("l-bfgs", "time-efficient")
            )[0],
        ),
        (
            "density",
            get_targets_for_density_example()[0],
            get_finals(
                *get_results_for_density_example("adam", "time-efficient")
            )[0],
        ),
        (
            "unitary",
            get_targets_for_hadamard_problem()[0],
            get_finals(
                *get_results_for_hadamard_problen("adam", "time-efficient")
            )[0],
        ),
    ],
)
def test_fidelity_fn(fid_type, target, final):
    """
    This tests the fidelity function for Unitary gates, states and density matrices
    """

    # Normalize the target and final states

    fidelity_fg = fidelity(C_target=target, U_final=final, evo_type=fid_type)

    if fid_type == "state" or fid_type == "density":
        fidelity_qt = qt.fidelity(
            qt.Qobj(target).unit(), qt.Qobj(final).unit()
        )
    else:
        fidelity_qt = (
            abs((qt.Qobj(target).dag() * qt.Qobj(final)).tr())
            / target.shape[0]
        )
    print(f"qt fidelity for {fid_type}: ", fidelity_qt)
    print(f"fg fidelity: for {fid_type}", fidelity_fg)
    assert jnp.allclose(fidelity_fg, fidelity_qt, atol=1e-1), (
        "fidelities not close enough"
    )


@pytest.mark.parametrize(
    "A",
    [
        jnp.array([[2, 0], [0, 3]]),
        jnp.array([[4, 2], [2, 4]]),
        jnp.array([[1, 0], [0, 0]]),
        jnp.array([[5, 4], [4, 5]]),
        jnp.array([[3, 0], [0, 7]]),
    ],
)
def test_sqrtm_eig(A):
    """Test that verifies mathematical correctness for positive semidefinite matrices."""
    # Check Hermitian
    assert is_positive_semi_definite(A), "A is not positive semi definite"

    sqrt_A = sqrtm_eig(A)

    sqrtm_jax = jax.scipy.linalg.sqrtm(A)

    assert jnp.allclose(sqrt_A, sqrtm_jax, atol=1e-9), (
        "Square root matrices are not close enough."
    )

    reconstructed = sqrt_A @ sqrt_A
    assert jnp.allclose(reconstructed, A, atol=1e-9), (
        "Square root verification failed: sqrt(A) @ sqrt(A) != A"
    )


def test_sesolve():
    """
    Test the sesolve function from qutip.
    """
    psi_fg, psi_qt = get_targets_for_qubit_in_cavity_problem("state")
    psi_fg_2, psi_qt_2 = get_targets_for_qubit_in_cavity_problem("density")
    assert jnp.allclose(psi_fg, psi_qt.full(), atol=1e-2), (
        "The states are not close enough."
    )

    # NOTE: qt.sesolve(does not solve for density matrices)
    # Schrodinger equation evolution of a state vector or unitary matrix
    # for a given Hamiltonian.
    # that's why we use internally when state == "density" the mesolve function
    print("psi_fg_2: ", psi_fg_2)
    print("psi_qt_2: ", psi_qt_2.full())
    assert jnp.allclose(psi_fg_2, psi_qt_2.full(), atol=1e-2), (
        "The states are not close enough."
    )


def test_for_errors():
    """
    Test for errors in the GRAPE package.
    """
    import jax.numpy as jnp

    # Test invalid evo_type
    def test_invalid_evo_type():
        """Test that invalid evo_type raises ValueError."""
        H_drift = jnp.array([[0, 1], [1, 0]])
        H_control = [jnp.array([[1, 0], [0, -1]])]
        U_0 = jnp.array([[1, 0], [0, 1]])
        C_target = jnp.array([[0, 1], [1, 0]])
        
        with pytest.raises(ValueError, match="Invalid evo_type. Choose 'state' or 'density' or 'unitary'."):
            optimize_pulse(
                H_drift=H_drift,
                H_control=H_control,
                U_0=U_0,
                C_target=C_target,
                num_t_slots=10,
                total_evo_time=1.0,
                evo_type="invalid_type"
            )

    # Test None U_0
    def test_none_u0():
        """Test that None U_0 raises ValueError."""
        H_drift = jnp.array([[0, 1], [1, 0]])
        H_control = [jnp.array([[1, 0], [0, -1]])]
        C_target = jnp.array([[0, 1], [1, 0]])
        
        with pytest.raises(ValueError, match="Please provide an initial state/density matrix/unitary gate U_0."):
            optimize_pulse(
                H_drift=H_drift,
                H_control=H_control,
                U_0=None,
                C_target=C_target,
                num_t_slots=10,
                total_evo_time=1.0,
                evo_type="unitary"
            )

    # Test bra states
    def test_bra_states():
        """Test that bra states raise TypeError."""
        H_drift = jnp.array([[0, 1], [1, 0]])
        H_control = [jnp.array([[1, 0], [0, -1]])]
        # Create bra (row vector)
        U_0_bra = jnp.array([[1, 0]])  # Row vector (bra)
        C_target = jnp.array([[1], [0]])  # Column vector (ket)
        
        with pytest.raises(TypeError, match="Please provide initial and target states as kets"):
            optimize_pulse(
                H_drift=H_drift,
                H_control=H_control,
                U_0=U_0_bra,
                C_target=C_target,
                num_t_slots=10,
                total_evo_time=1.0,
                evo_type="state"
            )

    def test_target_bra_states():
        """Test that target bra states raise TypeError."""
        H_drift = jnp.array([[0, 1], [1, 0]])
        H_control = [jnp.array([[1, 0], [0, -1]])]
        U_0 = jnp.array([[1], [0]])  # Column vector (ket)
        C_target_bra = jnp.array([[1, 0]])  # Row vector (bra)
        
        with pytest.raises(TypeError, match="Please provide initial and target states as kets"):
            optimize_pulse(
                H_drift=H_drift,
                H_control=H_control,
                U_0=U_0,
                C_target=C_target_bra,
                num_t_slots=10,
                total_evo_time=1.0,
                evo_type="state"
            )

    # Test non-positive semi-definite density matrices
    def test_non_positive_semidefinite_density():
        """Test that non-positive semi-definite density matrices raise TypeError."""
        H_drift = jnp.array([[0, 1], [1, 0]])
        H_control = [jnp.array([[1, 0], [0, -1]])]
        # Non-positive semi-definite matrix (negative eigenvalue)
        U_0_bad = jnp.array([[1, 0], [0, -1]])
        C_target = jnp.array([[0.5, 0], [0, 0.5]])
        
        with pytest.raises(TypeError, match="your initial and target rhos must be positive semi-definite"):
            optimize_pulse(
                H_drift=H_drift,
                H_control=H_control,
                U_0=U_0_bad,
                C_target=C_target,
                num_t_slots=10,
                total_evo_time=1.0,
                evo_type="density"
            )

    def test_non_positive_semidefinite_target_density():
        """Test that non-positive semi-definite target density matrices raise TypeError."""
        H_drift = jnp.array([[0, 1], [1, 0]])
        H_control = [jnp.array([[1, 0], [0, -1]])]
        U_0 = jnp.array([[0.5, 0], [0, 0.5]])
        # Non-positive semi-definite matrix (negative eigenvalue)
        C_target_bad = jnp.array([[1, 0], [0, -1]])
        
        with pytest.raises(TypeError, match="If evo_type=`density` your initial and target rhos must be positive semi-definite"):
            optimize_pulse(
                H_drift=H_drift,
                H_control=H_control,
                U_0=U_0,
                C_target=C_target_bad,
                num_t_slots=10,
                total_evo_time=1.0,
                evo_type="density"
            )

    # Test collapse operators with state evolution
    def test_collapse_ops_with_state_evo():
        """Test that collapse operators with state evolution raise ValueError."""
        H_drift = jnp.array([[0, 1], [1, 0]])
        H_control = [jnp.array([[1, 0], [0, -1]])]
        U_0 = jnp.array([[1], [0]])  # Ket state
        C_target = jnp.array([[0], [1]])  # Ket state
        c_ops = [jnp.array([[0, 1], [0, 0]])]  # Collapse operator
        
        with pytest.raises(ValueError, match="You supplied collapse operators"):
            optimize_pulse(
                H_drift=H_drift,
                H_control=H_control,
                U_0=U_0,
                C_target=C_target,
                num_t_slots=10,
                total_evo_time=1.0,
                evo_type="state",
                c_ops=c_ops
            )

    def test_density_with_ket_states():
        """Test that collapse operators with ket states (even with density evo_type) raise ValueError."""
        H_drift = jnp.array([[0, 1], [1, 0]])
        H_control = [jnp.array([[1, 0], [0, -1]])]
        U_0 = basis(2)  # Ket state
        C_target = ket2dm(basis(2))  # Density matrix from ket
        c_ops = [jnp.array([[0, 1], [0, 0]])]  # Collapse operator
        
        with pytest.raises(TypeError, match=re.escape("For evo_type='density', please provide initial and target states as density matrices.")):
            optimize_pulse(
                H_drift=H_drift,
                H_control=H_control,
                U_0=U_0,
                C_target=C_target,
                num_t_slots=10,
                total_evo_time=1.0,
                evo_type="density",
                c_ops=c_ops
            )

    def test_invalid_optimizer():
        """Test that invalid optimizer raises ValueError."""
        H_drift = jnp.array([[0, 1], [1, 0]])
        H_control = [jnp.array([[1, 0], [0, -1]])]
        U_0 = jnp.array([[1, 0], [0, 1]])
        C_target = jnp.array([[0, 1], [1, 0]])
        
        with pytest.raises(ValueError, match="Optimizer invalid_optimizer not supported. Use 'adam' or 'l-bfgs'."):
            optimize_pulse(
                H_drift=H_drift,
                H_control=H_control,
                U_0=U_0,
                C_target=C_target,
                num_t_slots=10,
                total_evo_time=1.0,
                evo_type="unitary",
                optimizer="invalid_optimizer"
            )

    # Run all tests
    test_invalid_evo_type()
    test_none_u0()
    test_bra_states()
    test_target_bra_states()
    test_non_positive_semidefinite_density() 
    test_non_positive_semidefinite_target_density()
    test_collapse_ops_with_state_evo()
    test_density_with_ket_states()
    test_invalid_optimizer()