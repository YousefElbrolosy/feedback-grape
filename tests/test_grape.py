"""
Tests for the GRAPE package.
"""

# ruff: noqa
import jax.numpy as jnp
import pytest
import qutip as qt

from feedback_grape.grape import fidelity
from tests.helper_for_tests import (
    get_finals,
    get_results_for_cnot_problem,
    get_results_for_density_example,
    get_results_for_dissipation_problem,
    get_results_for_hadamard_problen,
    get_results_for_qubit_in_cavity_problem,
    get_targets_for_cnot_problem,
    get_targets_for_density_example,
    get_targets_for_dissipation_problem,
    get_targets_for_hadamard_problem,
    get_targets_for_qubit_in_cavity_problem,
)

# Check documentation for pytest for more decorators


# Testing target Operator transformations

# TODO: test more thoroughly, not just using differences in fidelity with qutip, because that may be faulty


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


@pytest.mark.parametrize(
    "optimizer, propcomp",
    [
        ("adam", "memory-efficient"),
        ("l-bfgs", "memory-efficient"),
        ("adam", "time-efficient"),
        ("l-bfgs", "time-efficient"),
    ],
)
def test_dissipative_model(optimizer, propcomp):
    result_fg, result_qt = get_results_for_dissipation_problem(
        optimizer, propcomp
    )
    print("result_qt.fid_err: ", result_qt.fid_err)
    print("result_fg.final_fidelity: ", result_fg.final_fidelity)
    assert jnp.allclose(
        1 - result_fg.final_fidelity, result_qt.fid_err, atol=1e-1
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
        (
            "superoperator",
            get_targets_for_dissipation_problem()[0],
            get_finals(
                *get_results_for_dissipation_problem("adam", "time-efficient")
            )[0],
        ),
    ],
)
def test_fidelity_fn(fid_type, target, final):
    """
    This tests the fidelity function for Unitary gates, states and density matrices
    """

    # Normalize the target and final states

    fidelity_fg = fidelity(C_target=target, U_final=final, type=fid_type)
    if fid_type == "superoperator":
        fidelity_qt = qt.tracedist(
            qt.Qobj(target).unit(), qt.Qobj(final).unit()
        )
    elif fid_type == "state" or fid_type == "density":
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

def test_sesolve():
    """
    Test the sesolve function from qutip.
    """
    psi_fg, psi_qt = get_targets_for_qubit_in_cavity_problem("state")
    assert jnp.allclose(psi_fg, psi_qt.full(), atol=1e-2), (
        "The states are not close enough."
    )
