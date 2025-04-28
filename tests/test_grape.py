"""
Tests for the GRAPE package.
"""

# ruff: noqa N8
import numpy as np
import qutip as qt
import qutip_qip.operations.gates as qip
import pytest
import jax.numpy as jnp
import jax
from feedback_grape.grape import optimize_pulse, sesolve, fidelity
from feedback_grape.utils.gates import cnot, hadamard
from feedback_grape.utils.operators import (
    identity,
    sigmax,
    sigmay,
    sigmaz,
    sigmap,
    sigmam,
    destroy,
)
from feedback_grape.utils.superoperator import liouvillian, sprepost
from feedback_grape.utils.tensor import tensor
from feedback_grape.utils.states import basis

import qutip_qtrl.pulseoptim as qtrl
from qutip.core.superoperator import (
    liouvillian as qt_liouvillian,
    sprepost as qt_sprepost,
)

# Check documentation for pytest for more decorators


# Testing target Operator transformations

# TODO: check if the parameterize synatx is correct for different optimizers and propcomps
# TODO: test more thoroughly, not just using differences in fidelity with qutip, because that may be faulty


# State to state transfer example
def get_targets_for_qubit_in_cavity_problem():
    N_cav = 10
    chi = 0.2385 * (2 * jnp.pi)
    mu_qub = 4.0
    mu_cav = 8.0
    hconj = lambda a: jnp.swapaxes(a.conj(), -1, -2)

    time_start = 0.0
    time_end = 1.0
    time_intervals_num = 5
    N_cav = 10
    t_grid = jnp.linspace(time_start, time_end, time_intervals_num + 1)
    delta_ts = t_grid[1:] - t_grid[:-1]
    fake_random_key = jax.random.key(seed=0)
    e_data = jax.random.uniform(
        fake_random_key, shape=(4, len(delta_ts)), minval=-1, maxval=1
    )
    e_qub = e_data[0] + 1j * e_data[1]
    e_cav = e_data[2] + 1j * e_data[3]

    # Using fg

    @jax.vmap
    def build_ham(e_qub, e_cav):
        """
        Build Hamiltonian for given (complex) e_qub and e_cav
        """

        a = tensor(identity(2), destroy(N_cav))
        adag = hconj(a)
        n_phot = adag @ a
        sigz = tensor(sigmaz(), identity(N_cav))
        sigp = tensor(sigmap(), identity(N_cav))
        one = tensor(identity(2), identity(N_cav))

        H0 = +(chi / 2) * n_phot @ (sigz + one)

        H_ctrl = mu_qub * sigp * e_qub + mu_cav * adag * e_cav
        H_ctrl += hconj(H_ctrl)
        # You just pass an array of the Hamiltonian matrices "Hs" corresponding to the time
        # intervals "delta_ts" (that is, "Hs" is a 3D array).
        return H0, H_ctrl

    H0, H_ctrl = build_ham(e_qub, e_cav)

    psi0 = tensor(basis(2), basis(N_cav))
    psi_fg = sesolve(H0 + H_ctrl, psi0, delta_ts)

    # Using qutip QTRL

    def build_ham_qt(e_qub, e_cav):
        a = qt.tensor(qt.identity(2), qt.destroy(N_cav))
        adag = a.dag()
        n_phot = adag * a
        sigz = qt.tensor(qt.sigmaz(), qt.identity(N_cav))
        sigp = qt.tensor(qt.sigmap(), qt.identity(N_cav))
        one = qt.tensor(qt.identity(2), qt.identity(N_cav))

        H0 = +(chi / 2) * n_phot * (sigz + one)

        H_ctrl_qub = mu_qub * sigp
        H_ctrl_cav = mu_cav * adag

        H = [
            # time independent part
            H0,
            # time dependent on e_qub (you can consider e_qub an array of different coefficients each time step to
            # represent changing Hamiltonian with time)
            [H_ctrl_qub, e_qub],
            # time dependent on e_qub.conj()
            [H_ctrl_qub.dag(), e_qub.conj()],
            # time dependent on e_cav
            [H_ctrl_cav, e_cav],
            # time dependent on e_cav.conj()
            [H_ctrl_cav.dag(), e_cav.conj()],
        ]

        return H

    psi0_qt = qt.tensor(qt.basis(2), qt.basis(N_cav))
    time_subintervals_num_qt = 100
    t_grid_qt = np.linspace(
        time_start, time_end, time_subintervals_num_qt * time_intervals_num
    )

    t_grid = np.linspace(time_start, time_end, time_intervals_num + 1)
    delta_ts = t_grid[1:] - t_grid[:-1]
    fake_random_key = jax.random.key(seed=0)
    e_data = jax.random.uniform(
        fake_random_key, shape=(4, len(delta_ts)), minval=-1, maxval=1
    )
    e_qub = e_data[0] + 1j * e_data[1]
    e_cav = e_data[2] + 1j * e_data[3]
    e_qub_qt = np.repeat(np.array(e_qub), time_subintervals_num_qt)
    e_cav_qt = np.repeat(np.array(e_cav), time_subintervals_num_qt)
    H_qt = build_ham_qt(e_qub_qt, e_cav_qt)
    psi_qt = qt.sesolve(H_qt, psi0_qt, t_grid_qt).states[-1]

    return psi_fg, psi_qt


def get_results_for_qubit_in_cavity_problem(optimizer, propcomp):
    N_cav = 10
    chi = 0.2385 * (2 * jnp.pi)
    mu_qub = 4.0
    mu_cav = 8.0
    hconj = lambda a: jnp.swapaxes(a.conj(), -1, -2)

    time_start = 0.0
    time_end = 1.0
    time_intervals_num = 5
    N_cav = 10
    t_grid = jnp.linspace(time_start, time_end, time_intervals_num + 1)
    delta_ts = t_grid[1:] - t_grid[:-1]
    fake_random_key = jax.random.key(seed=0)
    e_data = jax.random.uniform(
        fake_random_key, shape=(4, len(delta_ts)), minval=-1, maxval=1
    )
    e_qub = e_data[0] + 1j * e_data[1]
    e_cav = e_data[2] + 1j * e_data[3]

    # Using fg

    @jax.vmap
    def build_ham(e_qub, e_cav):
        """
        Build Hamiltonian for given (complex) e_qub and e_cav
        """

        a = tensor(identity(2), destroy(N_cav))
        adag = hconj(a)
        n_phot = adag @ a
        sigz = tensor(sigmaz(), identity(N_cav))
        sigp = tensor(sigmap(), identity(N_cav))
        one = tensor(identity(2), identity(N_cav))

        H0 = +(chi / 2) * n_phot @ (sigz + one)

        H_ctrl = mu_qub * sigp * e_qub + mu_cav * adag * e_cav
        H_ctrl += hconj(H_ctrl)
        # You just pass an array of the Hamiltonian matrices "Hs" corresponding to the time
        # intervals "delta_ts" (that is, "Hs" is a 3D array).
        return H0, H_ctrl

    def build_grape_format_ham():
        """
        Build Hamiltonian for given (complex) e_qub and e_cav
        """

        a = tensor(identity(2), destroy(N_cav))
        adag = hconj(a)
        n_phot = adag @ a
        sigz = tensor(sigmaz(), identity(N_cav))
        sigp = tensor(sigmap(), identity(N_cav))
        one = tensor(identity(2), identity(N_cav))

        H0 = +(chi / 2) * n_phot @ (sigz + one)
        H_ctrl_qub = mu_qub * sigp
        H_ctrl_qub_dag = hconj(H_ctrl_qub)
        H_ctrl_cav = mu_cav * adag
        H_ctrl_cav_dag = hconj(H_ctrl_cav)

        H_ctrl = [H_ctrl_qub, H_ctrl_qub_dag, H_ctrl_cav, H_ctrl_cav_dag]

        return H0, H_ctrl

    H0, H_ctrl = build_ham(e_qub, e_cav)

    # Representation for time dependent Hamiltonian
    def solve(Hs, delta_ts):
        """
        Find evolution operator for piecewise Hs on time intervals delts_ts
        """
        for i, (H, delta_t) in enumerate(zip(Hs, delta_ts)):
            U_intv = jax.scipy.linalg.expm(-1j * H * delta_t)
            U = U_intv if i == 0 else U_intv @ U
        return U

    U = solve(H0 + H_ctrl, delta_ts)
    psi0 = tensor(basis(2), basis(N_cav))
    psi = U @ psi0

    H0_grape, H_ctrl_grape = build_grape_format_ham()

    result_fg = optimize_pulse(
        H0_grape,
        H_ctrl_grape,
        psi0,
        psi,
        int(
            (time_end - time_start) / delta_ts[0]
        ),  # Ensure this is an integer
        time_end - time_start,
        max_iter=10000,
        # when you decrease convergence threshold, it is more accurate
        convergence_threshold=1e-3,
        learning_rate=1e-2,
        type="state",
        optimizer=optimizer,
        propcomp=propcomp,
    )

    # Using qutip QTRL

    def build_ham_qt(e_qub, e_cav):
        a = qt.tensor(qt.identity(2), qt.destroy(N_cav))
        adag = a.dag()
        n_phot = adag * a
        sigz = qt.tensor(qt.sigmaz(), qt.identity(N_cav))
        sigp = qt.tensor(qt.sigmap(), qt.identity(N_cav))
        one = qt.tensor(qt.identity(2), qt.identity(N_cav))

        H0 = +(chi / 2) * n_phot * (sigz + one)

        H_ctrl_qub = mu_qub * sigp
        H_ctrl_cav = mu_cav * adag

        H = [
            # time independent part
            H0,
            # time dependent on e_qub (you can consider e_qub an array of different coefficients each time step to
            # represent changing Hamiltonian with time)
            [H_ctrl_qub, e_qub],
            # time dependent on e_qub.conj()
            [H_ctrl_qub.dag(), e_qub.conj()],
            # time dependent on e_cav
            [H_ctrl_cav, e_cav],
            # time dependent on e_cav.conj()
            [H_ctrl_cav.dag(), e_cav.conj()],
        ]

        return H

    psi0_qt = qt.tensor(qt.basis(2), qt.basis(N_cav))
    time_subintervals_num_qt = 100
    t_grid_qt = np.linspace(
        time_start, time_end, time_subintervals_num_qt * time_intervals_num
    )

    t_grid = np.linspace(time_start, time_end, time_intervals_num + 1)
    delta_ts = t_grid[1:] - t_grid[:-1]
    fake_random_key = jax.random.key(seed=0)
    e_data = jax.random.uniform(
        fake_random_key, shape=(4, len(delta_ts)), minval=-1, maxval=1
    )
    e_qub = e_data[0] + 1j * e_data[1]
    e_cav = e_data[2] + 1j * e_data[3]
    e_qub_qt = np.repeat(np.array(e_qub), time_subintervals_num_qt)
    e_cav_qt = np.repeat(np.array(e_cav), time_subintervals_num_qt)
    H_qt = build_ham_qt(e_qub_qt, e_cav_qt)
    psi_qt = qt.sesolve(H_qt, psi0_qt, t_grid_qt).states[-1]

    # Extract just the control operators from H_qt[1:] (not the coefficient arrays) (but that just completely discards the time dep part!)
    # However, it is weird, because the fidelity is quite high
    ctrls = [H_part[0] for H_part in H_qt[1:]]

    result_qt = qtrl.optimize_pulse(
        H_qt[0],  # Drift Hamiltonian
        ctrls,  # Pass just the control operators
        psi0_qt,
        psi_qt,
        int(
            (time_end - time_start) / delta_ts[0]
        ),  # Ensure this is an integer
        time_end - time_start,
        max_iter=10000,
    )
    return (result_fg, result_qt)


# Target Unitary examples
def get_targets_for_cnot_problem():
    C_target_fg = cnot()
    C_target_qt = qt.core.gates.cnot()
    return C_target_fg, C_target_qt


def get_results_for_cnot_problem(optimizer, propcomp):
    """
    Test the optimize_pulse function.
    """

    ### General
    num_t_slots = 500
    total_evo_time = 2 * jnp.pi
    ##

    ###### fg
    g = 0  # Small coupling strength
    H_drift = g * (tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()))
    H_ctrl = [
        tensor(sigmax(), identity(2)),
        tensor(sigmay(), identity(2)),
        tensor(sigmaz(), identity(2)),
        tensor(identity(2), sigmax()),
        tensor(identity(2), sigmay()),
        tensor(identity(2), sigmaz()),
        tensor(sigmax(), sigmax()),
        tensor(sigmay(), sigmay()),
        tensor(sigmaz(), sigmaz()),
    ]

    U_0 = identity(4)
    C_target = cnot()

    result_fg = optimize_pulse(
        H_drift,
        H_ctrl,
        U_0,
        C_target,
        num_t_slots,
        total_evo_time,
        max_iter=500,
        learning_rate=1e-2,
        optimizer=optimizer,
        propcomp=propcomp,
    )

    ############ Qutip QTRL

    g = 0  # Small coupling strength
    H_drift_qt = g * (
        qt.tensor(qt.sigmax(), qt.sigmax())
        + qt.tensor(qt.sigmay(), qt.sigmay())
    )
    H_ctrl_qt = [
        qt.tensor(qt.sigmax(), qt.operators.identity(2)),
        qt.tensor(qt.sigmay(), qt.operators.identity(2)),
        qt.tensor(qt.sigmaz(), qt.operators.identity(2)),
        qt.tensor(qt.operators.identity(2), qt.sigmax()),
        qt.tensor(qt.operators.identity(2), qt.sigmay()),
        qt.tensor(qt.operators.identity(2), qt.sigmaz()),
        qt.tensor(qt.sigmax(), qt.sigmax()),
        qt.tensor(qt.sigmay(), qt.sigmay()),
        qt.tensor(qt.sigmaz(), qt.sigmaz()),
    ]

    U_0_qt = qt.operators.identity(4)
    # Target operator (CNOT gate)
    C_target_qt = qt.core.gates.cnot()

    result_qt = qtrl.optimize_pulse_unitary(
        H_drift_qt,
        H_ctrl_qt,
        U_0_qt,
        C_target_qt,
        num_t_slots,
        total_evo_time,
        max_iter=500,
    )

    result_qt.dims = [[2, 2], [2, 2]]
    return (result_fg, result_qt)


def get_targets_for_hadamard_problem():
    C_target_fg = hadamard()
    C_target_qt = qip.hadamard_transform(1)
    return C_target_fg, C_target_qt


def get_results_for_hadamard_problen(optimizer, propcomp):
    ###### General
    # Number of time slots
    n_ts = 10
    # Time allowed for the evolution
    evo_time = 10

    # Fidelity error target
    fid_err_targ = 1e-10
    # Maximum iterations for the optisation algorithm
    max_iter = 200
    # Maximum (elapsed) time allowed in seconds
    max_wall_time = 120
    # Minimum gradient (sum of gradients squared)
    # as this tends to 0 -> local minima has been found
    min_grad = 1e-20

    ###### Qutip QTRL
    # Drift Hamiltonian
    H_d = qt.sigmaz()
    # The (single) control Hamiltonian
    H_c = [qt.sigmax()]
    # start point for the gate evolution
    U_0 = qt.operators.identity(2)
    # Target for the gate evolution Hadamard gate
    U_targ = qip.hadamard_transform(1)

    p_type = 'SINE'
    result_qt = qtrl.optimize_pulse_unitary(
        H_d,
        H_c,
        U_0,
        U_targ,
        n_ts,
        evo_time,
        fid_err_targ=fid_err_targ,
        min_grad=min_grad,
        max_iter=max_iter,
        max_wall_time=max_wall_time,
        init_pulse_type=p_type,
        gen_stats=True,
    )

    ###### fg
    # Drift Hamiltonian
    H_d = sigmaz()
    # The (single) control Hamiltonian
    H_c = [sigmax()]
    # start point for the gate evolution
    U_0 = identity(2)
    # Target for the gate evolution Hadamard gate
    U_targ = hadamard()

    result_fg = optimize_pulse(
        H_d,
        H_c,
        U_0,
        U_targ,
        n_ts,
        evo_time,
        max_iter=max_iter,
        learning_rate=1e-2,
        optimizer=optimizer,
        propcomp=propcomp,
    )

    return (result_fg, result_qt)


# Target Superoperator examples
def get_targets_for_dissipation_problem():
    Sx = sigmax()
    Sy = sigmay()
    Sz = sigmaz()
    Sm = sigmam()
    Si = identity(2)
    had_gate = hadamard()
    # Hamiltonian
    Del = 0.1  # Tunnelling term
    wq = 1.0  # Energy of the 2-level system.
    H0 = 0.5 * wq * sigmaz() + 0.5 * Del * sigmax()

    # Amplitude damping#
    # Damping rate:
    gamma = 0.1
    L0 = liouvillian(H0, [jnp.sqrt(gamma) * Sm])

    # sigma X control
    LC_x = liouvillian(Sx)
    # sigma Y control
    LC_y = liouvillian(Sy)
    # sigma Z control
    LC_z = liouvillian(Sz)

    # Drift
    drift = L0
    # Controls - different combinations can be tried
    ctrls = [LC_z, LC_x]
    # Number of ctrls
    n_ctrls = len(ctrls)

    # start point for the map evolution
    E0 = sprepost(Si, Si)

    # target for map evolution
    C_target_fg = sprepost(had_gate, had_gate)

    ## Qutip
    Sx = qt.sigmax()
    Sy = qt.sigmay()
    Sz = qt.sigmaz()
    Sm = qt.sigmam()
    Si = qt.identity(2)
    # Hadamard gate
    had_gate = qip.hadamard_transform(1)

    # Hamiltonian
    Del = 0.1  # Tunnelling term
    wq = 1.0  # Energy of the 2-level system.
    H0 = 0.5 * wq * qt.sigmaz() + 0.5 * Del * qt.sigmax()

    # Amplitude damping#
    # Damping rate:
    gamma = 0.1
    L0 = qt_liouvillian(H0, [np.sqrt(gamma) * Sm])

    # sigma X control
    LC_x = qt_liouvillian(Sx)
    # sigma Y control
    LC_y = qt_liouvillian(Sy)
    # sigma Z control
    LC_z = qt_liouvillian(Sz)

    # Drift
    drift = L0
    # Controls - different combinations can be tried
    ctrls = [LC_z, LC_x]
    # Number of ctrls
    n_ctrls = len(ctrls)

    # start point for the map evolution
    E0 = qt_sprepost(Si, Si)

    # target for map evolution
    C_target_qt = qt_sprepost(had_gate, had_gate)

    return C_target_fg, C_target_qt


def get_results_for_dissipation_problem(optimizer, propcomp):
    # Number of time slots
    n_ts = 10
    # Time allowed for the evolution
    evo_time = 2
    Sx = sigmax()
    Sy = sigmay()
    Sz = sigmaz()
    Sm = sigmam()
    Si = identity(2)
    had_gate = hadamard()
    # Hamiltonian
    Del = 0.1  # Tunnelling term
    wq = 1.0  # Energy of the 2-level system.
    H0 = 0.5 * wq * sigmaz() + 0.5 * Del * sigmax()

    # Amplitude damping#
    # Damping rate:
    gamma = 0.1
    L0 = liouvillian(H0, [jnp.sqrt(gamma) * Sm])

    # sigma X control
    LC_x = liouvillian(Sx)
    # sigma Y control
    LC_y = liouvillian(Sy)
    # sigma Z control
    LC_z = liouvillian(Sz)

    # Drift
    drift = L0
    # Controls - different combinations can be tried
    ctrls = [LC_z, LC_x]
    # Number of ctrls
    n_ctrls = len(ctrls)

    # start point for the map evolution
    E0 = sprepost(Si, Si)

    # target for map evolution
    C_target_fg = sprepost(had_gate, had_gate)

    result_fg = optimize_pulse(
        drift,
        ctrls,
        E0,
        C_target_fg,
        n_ts,
        evo_time,
        type="superoperator",
        optimizer=optimizer,
        convergence_threshold=1e-16,
        max_iter=1000,
        learning_rate=0.1,
        propcomp=propcomp,
    )

    ## Qutip
    Sx = qt.sigmax()
    Sy = qt.sigmay()
    Sz = qt.sigmaz()
    Sm = qt.sigmam()
    Si = qt.identity(2)
    # Hadamard gate
    had_gate = qip.hadamard_transform(1)

    # Hamiltonian
    Del = 0.1  # Tunnelling term
    wq = 1.0  # Energy of the 2-level system.
    H0 = 0.5 * wq * qt.sigmaz() + 0.5 * Del * qt.sigmax()

    # Amplitude damping#
    # Damping rate:
    gamma = 0.1
    L0 = qt_liouvillian(H0, [np.sqrt(gamma) * Sm])

    # sigma X control
    LC_x = qt_liouvillian(Sx)
    # sigma Y control
    LC_y = qt_liouvillian(Sy)
    # sigma Z control
    LC_z = qt_liouvillian(Sz)

    # Drift
    drift = L0
    # Controls - different combinations can be tried
    ctrls = [LC_z, LC_x]
    # Number of ctrls
    n_ctrls = len(ctrls)

    # start point for the map evolution
    E0 = qt_sprepost(Si, Si)

    # target for map evolution
    C_target_qt = qt_sprepost(had_gate, had_gate)

    # Fidelity error target
    fid_err_targ = 1e-3
    # Maximum iterations for the optisation algorithm
    max_iter = 200
    # Maximum (elapsed) time allowed in seconds
    max_wall_time = 30
    # Minimum gradient (sum of gradients squared)
    # as this tends to 0 -> local minima has been found
    min_grad = 1e-20
    p_type = 'RND'
    result_qt = qtrl.optimize_pulse(
        drift,
        ctrls,
        E0,
        C_target_qt,
        n_ts,
        evo_time,
        fid_err_targ=fid_err_targ,
        min_grad=min_grad,
        max_iter=max_iter,
        max_wall_time=max_wall_time,
        init_pulse_type=p_type,
        gen_stats=True,
    )

    return (result_fg, result_qt)


def get_finals(result_fg, result_qt):
    """
    Get the final operators for the given results.
    """
    # Final operator for fg
    final_operator_fg = result_fg.final_operator
    # Final operator for qt
    final_operator_qt = result_qt.evo_full_final

    return final_operator_fg, final_operator_qt


@pytest.mark.parametrize(
    "optimize, propcomp",
    [
        ("adam", "memory-efficient"),
        ("l-bfgs", "memory-efficient"),
        ("adam", "time-efficient"),
        ("l-bfgs", "time-efficient"),
    ],
)
def test_cnot(optimize, propcomp):
    result_fg, result_qt = get_results_for_cnot_problem(optimize, propcomp)
    # print("reference.evo_full_final.full()): ",reference.evo_full_final.full())
    # print("fg result: ",result)
    # assert jnp.allclose(result["final_operator"], reference.evo_full_final.full(), atol=1e-1), "The matrices are not close enough."
    print("reference.fidelity: ", result_qt.fid_err)
    print("fg result[final_fidelity]: ", result_fg.final_fidelity)
    assert jnp.allclose(
        1 - result_fg.final_fidelity, result_qt.fid_err, atol=1e-3
    ), "The fidelities are not close enough."


@pytest.mark.parametrize(
    "optimize, propcomp",
    [
        ("adam", "memory-efficient"),
        ("l-bfgs", "memory-efficient"),
        ("adam", "time-efficient"),
        ("l-bfgs", "time-efficient"),
    ],
)
def test_hadamard(optimize, propcomp):
    result_fg, result_qt = get_results_for_hadamard_problen(optimize, propcomp)
    print("result_qt.fid_err: ", result_qt.fid_err)
    print("result_fg.final_fidelity: ", result_fg.final_fidelity)
    assert jnp.allclose(
        1 - result_fg.final_fidelity, result_qt.fid_err, atol=1e-3
    ), "The fidelities are not close enough."


# Testing states
@pytest.mark.parametrize(
    "optimize, propcomp",
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


# TODO:
@pytest.mark.parametrize(
    "fid_type, target, final",
    [
        (
            "state",
            get_targets_for_qubit_in_cavity_problem()[0],
            get_finals(
                *get_results_for_qubit_in_cavity_problem(
                    "adam", "time-efficient"
                )
            )[0],
        ),
        (
            "unitary",
            get_targets_for_cnot_problem()[0],
            get_finals(
                *get_results_for_cnot_problem("adam", "memory-efficient")
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
# TODO: Should really study how to obtain a good fidelity and why we need to normalize
# with qutip
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
    else:
        fidelity_qt = qt.fidelity(
            qt.Qobj(target).unit(), qt.Qobj(final).unit()
        )
    print(f"qt fidelity for {fid_type}: ", fidelity_qt)
    print(f"fg fidelity: for {fid_type}", fidelity_fg)
    assert jnp.allclose(fidelity_fg, fidelity_qt, atol=1e-1), (
        "fidelities not close enough"
    )


# TODO:
def test_sesolve():
    """
    Test the sesolve function from qutip.
    """
    psi_fg, psi_qt = get_targets_for_qubit_in_cavity_problem()
    assert jnp.allclose(psi_fg, psi_qt.full(), atol=1e-2), (
        "The states are not close enough."
    )
