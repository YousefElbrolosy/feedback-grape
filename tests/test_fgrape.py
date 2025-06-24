# ruff: noqa
import pytest


def example_A_body():
    from feedback_grape.fgrape import optimize_pulse_with_feedback
    from feedback_grape.utils.operators import (
        sigmap,
        sigmam,
        create,
        destroy,
        identity,
    )
    from feedback_grape.utils.states import basis, fock
    from feedback_grape.utils.tensor import tensor
    import jax.numpy as jnp
    from jax.scipy.linalg import expm

    N_cav = 30

    def qubit_unitary(alpha_re, alpha_im):
        alpha = alpha_re + 1j * alpha_im
        return tensor(
            identity(N_cav),
            expm(-1j * (alpha * sigmap() + alpha.conjugate() * sigmam()) / 2),
        )

    def qubit_cavity_unitary(beta_re):
        beta = beta_re
        return expm(
            -1j
            * (
                beta * (tensor(destroy(N_cav), sigmap()))
                + beta.conjugate() * (tensor(create(N_cav), sigmam()))
            )
            / 2
        )

    time_steps = 5

    psi0 = tensor(basis(N_cav), basis(2))
    psi0 = psi0 / jnp.linalg.norm(psi0)
    psi_target = tensor(
        (fock(N_cav, 1) + fock(N_cav, 3)) / jnp.sqrt(2), basis(2)
    )
    psi_target = psi_target / jnp.linalg.norm(psi_target)

    from feedback_grape.utils.fidelity import ket2dm
    import jax

    key = jax.random.PRNGKey(42)
    # not provideing param_constraints just propagates the same initial_parameters for each time step
    qub_unitary = {
        "gate": qubit_unitary,
        "initial_params": jax.random.uniform(
            key,
            shape=(1, 2),  # 2 for gamma and delta
            minval=-jnp.pi,
            maxval=jnp.pi,
        )[0].tolist(),
        "measurement_flag": False,
        "param_constraints": [
            [-2 * jnp.pi, 2 * jnp.pi],
            [-2 * jnp.pi, 2 * jnp.pi],
        ],
    }

    qub_cav = {
        "gate": qubit_cavity_unitary,
        "initial_params": jax.random.uniform(
            key,
            shape=(1, 1),  # 2 for gamma and delta
            minval=-jnp.pi,
            maxval=jnp.pi,
        )[0].tolist(),
        "measurement_flag": False,
        "param_constraints": [[-2 * jnp.pi, 2 * jnp.pi]],
    }

    system_params = [qub_unitary, qub_cav]

    result = optimize_pulse_with_feedback(
        U_0=ket2dm(psi0),
        C_target=ket2dm(psi_target),
        system_params=system_params,
        num_time_steps=time_steps,
        max_iter=1000,
        convergence_threshold=1e-16,
        evo_type="density",
        mode="no-measurement",
        goal="fidelity",
        learning_rate=0.02,
        batch_size=10,
        eval_batch_size=2,
    )

    if result.final_fidelity > 0.99:
        return True
    else:
        print(f"Final fidelity: {result.final_fidelity}")
        return False


def example_B_body():
    # B. State purification with qubit-mediated measurement
    from feedback_grape.fgrape import optimize_pulse_with_feedback
    import jax.numpy as jnp

    # initial state is a thermal state
    n_average = 2
    N_cavity = 30
    # natural logarithm
    beta = jnp.log((1 / n_average) + 1)
    diags = jnp.exp(-beta * jnp.arange(N_cavity))
    normalized_diags = diags / jnp.sum(diags, axis=0)
    rho_cav = jnp.diag(normalized_diags)

    from feedback_grape.utils.operators import cosm, sinm
    from feedback_grape.utils.operators import create, destroy
    import jax

    def povm_measure_operator(measurement_outcome, gamma, delta):
        """
        POVM for the measurement of the cavity state.
        returns Mm ( NOT the POVM element Em = Mm_dag @ Mm ), given measurement_outcome m, gamma and delta
        """
        number_operator = create(N_cavity) @ destroy(N_cavity)
        angle = (gamma * number_operator) + delta / 2
        return jnp.where(
            measurement_outcome == 1,
            cosm(angle),
            sinm(angle),
        )

    measure = {
        "gate": povm_measure_operator,
        "initial_params": jax.random.uniform(
            key=jax.random.PRNGKey(42), shape=(1, 2), minval=0.0, maxval=jnp.pi
        )[0].tolist(),
        "measurement_flag": True,
    }

    system_params = [measure]

    result = optimize_pulse_with_feedback(
        U_0=rho_cav,
        C_target=None,
        system_params=system_params,
        num_time_steps=5,
        mode="lookup",
        goal="purity",
        max_iter=1000,
        convergence_threshold=1e-20,
        learning_rate=0.01,
        evo_type="density",
        batch_size=10,
    )
    print(result.final_purity)
    from feedback_grape.utils.purity import purity

    print("initial purity:", purity(rho=rho_cav))
    for i, state in enumerate(result.final_state):
        print(f"Purity of state {i}:", purity(rho=state))
        if purity(rho=state) > 0.9:
            return True
    return False


def example_C_body():
    # C. State preparation from a thermal state with Jaynes-Cummings controls
    from feedback_grape.fgrape import optimize_pulse_with_feedback
    from feedback_grape.utils.operators import (
        sigmap,
        sigmam,
        create,
        destroy,
        identity,
        cosm,
        sinm,
    )
    from feedback_grape.utils.states import basis, fock
    from feedback_grape.utils.tensor import tensor
    import jax.numpy as jnp
    import jax
    from jax.scipy.linalg import expm

    N_cav = 20

    def qubit_unitary(alpha_re, alpha_im):
        alpha = alpha_re + 1j * alpha_im
        return tensor(
            identity(N_cav),
            expm(-1j * (alpha * sigmap() + alpha.conjugate() * sigmam()) / 2),
        )

    def qubit_cavity_unitary(beta_re, beta_im):
        beta = beta_re + 1j * beta_im
        return expm(
            -1j
            * (
                beta * (tensor(destroy(N_cav), sigmap()))
                + beta.conjugate() * (tensor(create(N_cav), sigmam()))
            )
            / 2
        )

    from feedback_grape.utils.operators import create, destroy

    def povm_measure_operator(measurement_outcome, gamma, delta):
        """
        POVM for the measurement of the cavity state.
        returns Mm ( NOT the POVM element Em = Mm_dag @ Mm ), given measurement_outcome m, gamma and delta
        """
        number_operator = tensor(create(N_cav) @ destroy(N_cav), identity(2))
        angle = (gamma * number_operator) + delta / 2
        meas_op = jnp.where(
            measurement_outcome == 1,
            cosm(angle),
            sinm(angle),
        )
        return meas_op

    ### defining initial (thermal) state
    # initial state is a thermal state coupled to a qubit in the ground state?
    n_average = 1
    # natural logarithm
    beta = jnp.log((1 / n_average) + 1)
    diags = jnp.exp(-beta * jnp.arange(N_cav))
    normalized_diags = diags / jnp.sum(diags, axis=0)
    rho_cav = jnp.diag(normalized_diags)
    rho_cav.shape
    rho0 = tensor(rho_cav, basis(2, 0) @ basis(2, 0).conj().T)

    ### defining target state
    psi_target = tensor(
        (fock(N_cav, 1) + fock(N_cav, 2) + fock(N_cav, 3)) / jnp.sqrt(3),
        basis(2),
    )
    psi_target = psi_target / jnp.linalg.norm(psi_target)

    rho_target = psi_target @ psi_target.conj().T
    ### initialize random params
    import jax

    num_time_steps = 5
    num_of_iterations = 1000
    learning_rate = 0.05
    key = jax.random.PRNGKey(0)
    measure = {
        "gate": povm_measure_operator,
        "initial_params": jax.random.uniform(
            key,
            shape=(1, 2),  # 2 for gamma and delta
            minval=-jnp.pi,
            maxval=jnp.pi,
        )[0].tolist(),
        "measurement_flag": True,
        # "param_constraints": [[0, jnp.pi], [-2*jnp.pi, 2*jnp.pi]],
    }

    qub_unitary = {
        "gate": qubit_unitary,
        "initial_params": jax.random.uniform(
            key,
            shape=(1, 2),  # 2 for gamma and delta
            minval=-jnp.pi,
            maxval=jnp.pi,
        )[0].tolist(),
        "measurement_flag": False,
        # "param_constraints": [[-2*jnp.pi, 2*jnp.pi], [-2*jnp.pi, 2*jnp.pi]],
    }

    qub_cav = {
        "gate": qubit_cavity_unitary,
        "initial_params": jax.random.uniform(
            key,
            shape=(1, 2),  # 2 for gamma and delta
            minval=-jnp.pi,
            maxval=jnp.pi,
        )[0].tolist(),
        "measurement_flag": False,
        # "param_constraints": [[-jnp.pi, jnp.pi], [-jnp.pi, jnp.pi]],
    }

    system_params = [measure, qub_unitary, qub_cav]

    result = optimize_pulse_with_feedback(
        U_0=rho0,
        C_target=rho_target,
        system_params=system_params,
        num_time_steps=num_time_steps,
        mode="lookup",
        goal="fidelity",
        max_iter=num_of_iterations,
        convergence_threshold=1e-6,
        learning_rate=learning_rate,
        evo_type="density",
        batch_size=10,
    )
    print(result.final_fidelity)
    from feedback_grape.utils.fidelity import fidelity

    print(
        "initial fidelity:",
        fidelity(C_target=rho_target, U_final=rho0, evo_type="density"),
    )
    for i, state in enumerate(result.final_state):
        fid_val = fidelity(
            C_target=rho_target, U_final=state, evo_type="density"
        )
        print(f"fidelity of state {i}:", fid_val)
        if fid_val > 0.8:
            return True

    return False


def example_D_body():
    no_dissipation_flag = False
    dissipation_flag = False

    from feedback_grape.fgrape import optimize_pulse_with_feedback
    from feedback_grape.utils.operators import (
        sigmap,
        sigmam,
        create,
        destroy,
        identity,
        cosm,
        sinm,
    )
    from feedback_grape.utils.states import basis, fock
    from feedback_grape.utils.tensor import tensor
    import jax.numpy as jnp
    import jax
    from jax.scipy.linalg import expm

    N_cav = 30

    def qubit_unitary(alpha_re):
        alpha = alpha_re
        return tensor(
            identity(N_cav),
            expm(-1j * (alpha * sigmap() + alpha.conjugate() * sigmam()) / 2),
        )

    def qubit_cavity_unitary(beta_re):
        beta = beta_re
        return expm(
            -1j
            * (
                beta * (tensor(destroy(N_cav), sigmap()))
                + beta.conjugate() * (tensor(create(N_cav), sigmam()))
            )
            / 2
        )

    from feedback_grape.utils.operators import create, destroy

    def povm_measure_operator(measurement_outcome, gamma, delta):
        """
        POVM for the measurement of the cavity state.
        returns Mm ( NOT the POVM element Em = Mm_dag @ Mm ), given measurement_outcome m, gamma and delta
        """
        number_operator = tensor(create(N_cav) @ destroy(N_cav), identity(2))
        angle = (gamma * number_operator) + delta / 2
        meas_op = jnp.where(
            measurement_outcome == 1,
            cosm(angle),
            sinm(angle),
        )
        return meas_op

    from feedback_grape.utils.states import coherent

    alpha = 3
    psi_target = tensor(
        coherent(N_cav, alpha)
        + coherent(N_cav, -alpha)
        + coherent(N_cav, 1j * alpha)
        + coherent(N_cav, -1j * alpha),
        basis(2),
    )  # 4-legged state

    # Normalize psi_target before constructing rho_target
    psi_target = psi_target / jnp.linalg.norm(psi_target)
    rho_target = psi_target @ psi_target.conj().T

    # Here the loss directly corressponds to the -fidelity (when converging) because log(1) is 0 and

    # the algorithm is choosing params that makes the POVM generate prob = 1
    measure = {
        "gate": povm_measure_operator,
        "initial_params": [0.058, jnp.pi / 2],  # gamma and delta
        "measurement_flag": True,
        # "param_constraints": [[0, 0.5], [-1, 1]],
    }

    qub_unitary = {
        "gate": qubit_unitary,
        "initial_params": [jnp.pi / 3],
        "measurement_flag": False,
        # "param_constraints": [[0, 0.5], [-1, 1]],
    }

    qub_cav = {
        "gate": qubit_cavity_unitary,
        "initial_params": [jnp.pi / 3],
        "measurement_flag": False,
        # "param_constraints": [[0, 0.5], [-1, 1]],
    }

    system_params = [measure, qub_unitary, qub_cav]
    result = optimize_pulse_with_feedback(
        U_0=rho_target,
        C_target=rho_target,
        system_params=system_params,
        num_time_steps=1,
        mode="lookup",
        goal="fidelity",
        max_iter=1000,
        convergence_threshold=1e-16,
        learning_rate=0.02,
        evo_type="density",
        batch_size=1,
    )

    if result.final_fidelity > 0.99:
        no_dissipation_flag = True

    # Now we add dissipation

    result = optimize_pulse_with_feedback(
        U_0=rho_target,
        C_target=rho_target,
        decay={
            "decay_indices": [0],
            "c_ops": {
                "tm": [tensor(identity(N_cav), jnp.sqrt(0.15) * sigmam())],
            },
            "tsave": jnp.linspace(0, 1, 2),  # time grid for decay
            "Hamiltonian": None,
        },
        system_params=system_params,
        num_time_steps=1,
        mode="lookup",
        goal="fidelity",
        max_iter=1000,
        convergence_threshold=1e-6,
        learning_rate=0.02,
        evo_type="density",
        batch_size=1,
    )

    if result.final_fidelity < 0.99:
        dissipation_flag = True

    if no_dissipation_flag and dissipation_flag:
        return True
    else:
        print(
            f"Final fidelity without dissipation: {result.final_fidelity}, with dissipation: {result.final_fidelity}"
        )
        return False


def example_E_body():
    # E: State stabilization with SNAP gates and displacement gates
    # ruff: noqa
    from feedback_grape.fgrape import optimize_pulse_with_feedback
    from feedback_grape.utils.operators import sigmam, identity, cosm, sinm
    from feedback_grape.utils.states import coherent, basis
    from feedback_grape.utils.tensor import tensor
    import jax.numpy as jnp
    import jax

    jax.config.update("jax_enable_x64", True)
    ## Initialize states
    from feedback_grape.utils.fidelity import ket2dm

    N_cav = 30  # number of cavity modes
    N_snap = 15

    alpha = 2
    psi_target = coherent(N_cav, alpha) + coherent(N_cav, -alpha)

    # Normalize psi_target before constructing rho_target
    psi_target = psi_target / jnp.linalg.norm(psi_target)

    rho_target = ket2dm(psi_target)

    rho_target = tensor(rho_target, ket2dm(basis(2)))
    # Parity Operator
    from feedback_grape.utils.operators import create, destroy

    ## Initialize the parameterized Gates
    def displacement_gate(alpha_re, alpha_im):
        """Displacement operator for a coherent state."""
        alpha = alpha_re + 1j * alpha_im
        gate = jax.scipy.linalg.expm(
            alpha * create(N_cav) - alpha.conj() * destroy(N_cav)
        )
        return tensor(gate, identity(2))

    def displacement_gate_dag(alpha_re, alpha_im):
        """Displacement operator for a coherent state."""
        alpha = alpha_re + 1j * alpha_im
        gate = (
            jax.scipy.linalg.expm(
                alpha * create(N_cav) - alpha.conj() * destroy(N_cav)
            )
            .conj()
            .T
        )
        return tensor(gate, identity(2))

    def snap_gate(
        phase0,
        phase1,
        phase2,
        phase3,
        phase4,
        phase5,
        phase6,
        phase7,
        phase8,
        phase9,
        phase10,
        phase11,
        phase12,
        phase13,
        phase14,
    ):
        phase_list = [
            phase0,
            phase1,
            phase2,
            phase3,
            phase4,
            phase5,
            phase6,
            phase7,
            phase8,
            phase9,
            phase10,
            phase11,
            phase12,
            phase13,
            phase14,
        ]
        diags = jnp.ones(shape=(N_cav - len(phase_list)))
        exponentiated = jnp.exp(1j * jnp.array(phase_list))
        diags = jnp.concatenate((exponentiated, diags))
        return tensor(jnp.diag(diags), identity(2))

    from feedback_grape.utils.operators import create, destroy

    def povm_measure_operator(measurement_outcome, gamma, delta):
        """
        POVM for the measurement of the cavity state.
        returns Mm ( NOT the POVM element Em = Mm_dag @ Mm ), given measurement_outcome m, gamma and delta
        """
        number_operator = tensor(create(N_cav) @ destroy(N_cav), identity(2))
        angle = (gamma * number_operator) + delta / 2
        meas_op = jnp.where(
            measurement_outcome == 1,
            cosm(angle),
            sinm(angle),
        )
        return meas_op

    ## Initialize RNN of choice
    import flax.linen as nn

    # You can do whatever you want inside so long as you maintaing the hidden_size and output size shapes
    class RNN(nn.Module):
        hidden_size: int  # number of features in the hidden state
        output_size: int  # number of features in the output ( 2 in the case of gamma and beta)

        @nn.compact
        def __call__(self, measurement, hidden_state):
            """
            If your GRU has a hidden state increasing number of features in the hidden stateH means:

            - You're allowing the model to store more information across time steps

            - Each time step can represent more complex features, patterns, or dependencies

            - You're giving the GRU more representational capacity
            """
            gru_cell = nn.GRUCell(
                features=self.hidden_size,
                gate_fn=nn.sigmoid,
                activation_fn=nn.tanh,
            )
            self.make_rng('dropout')

            if measurement.ndim == 1:
                measurement = measurement.reshape(1, -1)

            new_hidden_state, _ = gru_cell(hidden_state, measurement)
            new_hidden_state = nn.Dropout(rate=0.1, deterministic=False)(
                new_hidden_state
            )
            # this returns the povm_params after linear regression through the hidden state which contains
            # the information of the previous time steps and this is optimized to output best povm_params
            # new_hidden_state = nn.Dense(features=self.hidden_size)(new_hidden_state)
            new_hidden_state = nn.Dense(
                features=self.hidden_size,
                kernel_init=nn.initializers.glorot_uniform(),
            )(new_hidden_state)
            new_hidden_state = nn.relu(new_hidden_state)
            new_hidden_state = nn.Dense(
                features=self.hidden_size,
                kernel_init=nn.initializers.glorot_uniform(),
            )(new_hidden_state)
            new_hidden_state = nn.relu(new_hidden_state)
            output = nn.Dense(
                features=self.output_size,
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.constant(0.1),
            )(new_hidden_state)
            output = nn.relu(output)
            # output = jnp.asarray(output)
            return output[0], new_hidden_state

    ### In this notebook, we decreased the convergence threshold and evaluate for num_time_steps = 2
    # Note if tsave = jnp.linspace(0, 1, 1) = [0.0] then the decay is not applied ?
    # because the first time step has the original non decayed state
    key = jax.random.PRNGKey(42)
    snap_init = jax.random.uniform(
        key, shape=(N_snap,), minval=-jnp.pi, maxval=jnp.pi
    )
    # TODO/QUESTION: In documentation, clarify that the initial_params are the params up to the
    # point where measurement occurs, compared with other modes where the initial_params
    # are the initial params for the entire system for all time steps.
    measure = {
        "gate": povm_measure_operator,
        "initial_params": jax.random.uniform(
            key,
            shape=(1, 2),  # 2 for gamma and delta
            minval=-jnp.pi,
            maxval=jnp.pi,
        )[0].tolist(),
        "measurement_flag": True,
        # "param_constraints": [[0, 0.5], [-1, 1]],
    }

    displacement = {
        "gate": displacement_gate,
        "initial_params": jax.random.uniform(
            key, shape=(1, 2), minval=-jnp.pi, maxval=jnp.pi
        )[0].tolist(),
        "measurement_flag": False,
    }

    snap = {
        "gate": snap_gate,
        "initial_params": snap_init.tolist(),
        "measurement_flag": False,
    }

    displacement_dag = {
        "gate": displacement_gate_dag,
        "initial_params": jax.random.uniform(
            key, shape=(1, 2), minval=-jnp.pi, maxval=jnp.pi
        )[0].tolist(),
        "measurement_flag": False,
    }

    system_params = [measure, displacement, snap, displacement_dag]

    result = optimize_pulse_with_feedback(
        U_0=rho_target,
        C_target=rho_target,
        decay={
            "decay_indices": [
                0,
                1,
            ],  # indices of gates before which decay occurs
            "c_ops": {
                "tm": [tensor(identity(N_cav), jnp.sqrt(0.005) * sigmam())],
                "tc": [tensor(identity(N_cav), jnp.sqrt(0.005) * sigmam())],
            },
            "tsave": jnp.linspace(0, 1, 2),  # time grid for decay
            "Hamiltonian": None,
        },
        system_params=system_params,
        num_time_steps=2,
        mode="nn",
        goal="fidelity",
        max_iter=1000,
        convergence_threshold=1e-6,
        learning_rate=0.09,
        evo_type="density",
        batch_size=16,
        rnn=RNN,
        rnn_hidden_size=30,
    )
    from feedback_grape.utils.fidelity import ket2dm

    N_cav = 30  # number of cavity modes
    N_snap = 15

    alpha = 2
    psi_target = coherent(N_cav, alpha) + coherent(N_cav, -alpha)

    # Normalize psi_target before constructing rho_target
    psi_target = psi_target / jnp.linalg.norm(psi_target)

    rho_target = ket2dm(psi_target)

    rho_target = tensor(rho_target, ket2dm(basis(2)))
    from feedback_grape.utils.fidelity import fidelity

    for i, state in enumerate(result.final_state):
        fid_val = fidelity(
            C_target=rho_target, U_final=state, evo_type="density"
        )
        print(f"fidelity of state {i}:", fid_val)
        if fid_val > 0.9:
            return True

    return False


# test normal state preparation using parameterized grape
def test_example_A():
    """
    This test tests if the max fidelity reached by example_B is above 0.99
    """
    assert example_A_body(), (
        "The max fidelity reached by example_A for batch size 10 and eval_batch_size 2 is below 0.99"
    )


@pytest.mark.slow
def test_example_B():
    assert example_B_body(), "The max purity reached by example_B is below 0.9"


@pytest.mark.slow
def test_example_C():
    assert example_C_body(), (
        "The max fidelity reached by example_C is below 0.8"
    )


@pytest.mark.slow
def test_example_D():
    """
    This test tests if the max fidelity reached by example_C is above 0.9
    """
    assert example_D_body(), (
        "The max fidelity reached by example_D is not within expected range (>0.99 for no dissipation and below 0.99 for dissipation)."
    )


@pytest.mark.slow
def test_example_E():
    assert example_E_body(), (
        "The max fidelity reached by example_E is below 0.9"
    )
