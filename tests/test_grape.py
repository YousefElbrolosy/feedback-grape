"""
Tests for the GRAPE package.
"""
# ruff: noqa N8

import qutip as qt
import qutip_qip.operations.gates as qip
import jax.numpy as jnp
from feedback_grape.grape import optimize_pulse
from feedback_grape.utils.gates import cnot, hadamard
from feedback_grape.utils.operators import identity, sigmax, sigmay, sigmaz
from feedback_grape.utils.tensor import tensor
import qutip_qtrl.pulseoptim as qtrl

# Check documentation for pytest for more decorators


# TODO: should see how we can further test more thoroughly
def test_cnot():
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

    result = optimize_pulse(
        H_drift,
        H_ctrl,
        U_0,
        C_target,
        num_t_slots,
        total_evo_time,
        max_iter=500,
        learning_rate=1e-2,
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

    reference = qtrl.optimize_pulse_unitary(
        H_drift_qt,
        H_ctrl_qt,
        U_0_qt,
        C_target_qt,
        num_t_slots,
        total_evo_time,
        max_iter=500,
    )

    reference.dims = [[2, 2], [2, 2]]

    # print("reference.evo_full_final.full()): ",reference.evo_full_final.full())
    # print("fg result: ",result)
    # assert jnp.allclose(result["final_operator"], reference.evo_full_final.full(), atol=1e-1), "The matrices are not close enough."
    print("reference.fidelity: ", reference.fid_err)
    print("fg result[final_fidelity]: ", result.final_fidelity)
    assert jnp.allclose(
        1 - result.final_fidelity, reference.fid_err, atol=1e-3
    ), "The fidelities are not close enough."


def test_hadamard():
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
    )
    print("result_qt.fid_err: ", result_qt.fid_err)
    print("result_fg.final_fidelity: ", result_fg.final_fidelity)
    assert jnp.allclose(
        1 - result_fg.final_fidelity, result_qt.fid_err, atol=1e-3
    ), "The fidelities are not close enough."
