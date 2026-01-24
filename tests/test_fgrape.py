# ruff: noqa
from unittest import result
import jax
import pytest
from feedback_grape.fgrape import Gate, Decay
from feedback_grape.fgrape import optimize_pulse, Gate, Decay
from feedback_grape.utils.states import basis
from feedback_grape.utils.tensor import tensor
import re

jax.config.update("jax_enable_x64", True)


def example_A_body():
    from feedback_grape.fgrape import optimize_pulse
    from feedback_grape.utils.operators import (
        sigmap,
        sigmam,
        create,
        destroy,
        identity,
    )
    from feedback_grape.utils.states import basis, fock
    from feedback_grape.utils.tensor import tensor
    from feedback_grape.fgrape import Gate
    from feedback_grape.utils.states import coherent
    import jax.numpy as jnp
    from jax.scipy.linalg import expm
    import jax

    # -- Initialize intitial state --
    N_cav = 40
    psi0 = tensor(basis(N_cav), basis(2))
    psi0 = psi0 / jnp.linalg.norm(psi0)

    # -- Initialize Target state --
    average_photon_number = 9
    cat = (
        coherent(N_cav, 3)
        + coherent(N_cav, -3)
        + coherent(N_cav, 1j * 3)
        + coherent(N_cav, -1j * 3)
    )
    psi_target = tensor(cat, basis(2))
    psi_target = psi_target / jnp.linalg.norm(psi_target)

    # -- Initialize Parameterized Gates --
    def qubit_unitary(alphas):
        alpha_re = alphas[0]
        alpha_im = alphas[1]
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

    # -- Initialize System Parameters --
    key = jax.random.PRNGKey(42)
    qub_unitary = Gate(
        gate=qubit_unitary,
        initial_params=jax.random.uniform(
            key,
            shape=(2,),  # 2 for gamma and delta
            minval=-2 * jnp.pi,
            maxval=2 * jnp.pi,
            dtype=jnp.float64,
        ),
        measurement_flag=False,
    )

    qub_cav = Gate(
        gate=qubit_cavity_unitary,
        initial_params=jax.random.uniform(
            key,
            shape=(1,),
            minval=-2 * jnp.pi,
            maxval=2 * jnp.pi,
            dtype=jnp.float64,
        ),
        measurement_flag=False,
    )

    system_params = [qub_unitary, qub_cav]

    # -- Optimization --
    time_steps = 20
    result = optimize_pulse(
        U_0=psi0,
        C_target=psi_target,
        system_params=system_params,
        num_time_steps=time_steps,
        max_iter=3000,
        convergence_threshold=None,
        evo_type="state",
        mode="no-measurement",
        goal="fidelity",
        learning_rate=0.01,
        batch_size=10,
        eval_batch_size=1,
        progress=True,
    )

    if result.final_fidelity > 0.99:
        return True
    else:
        print(f"Final fidelity: {result.final_fidelity}")
        return False


def example_B_body():
    # B. State purification with qubit-mediated measurement
    from feedback_grape.fgrape import optimize_pulse
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

    def povm_measure_operator(measurement_outcome, params):
        """
        POVM for the measurement of the cavity state.
        returns Mm ( NOT the POVM element Em = Mm_dag @ Mm ), given measurement_outcome m, gamma and delta
        """
        gamma, delta = params
        number_operator = create(N_cavity) @ destroy(N_cavity)
        angle = (gamma * number_operator) + delta / 2
        return jnp.where(
            measurement_outcome == 1,
            cosm(angle),
            sinm(angle),
        )

    measure = Gate(
        gate=povm_measure_operator,
        initial_params=jax.random.uniform(
            key=jax.random.PRNGKey(42),
            shape=(2,),
            minval=0.0,
            maxval=jnp.pi,
            dtype=jnp.float64,
        ),
        measurement_flag=True,
    )

    system_params = [measure]

    result = optimize_pulse(
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
    # for i, state in enumerate(result.final_state):
    #     print(f"Purity of state {i}:", purity(rho=state))
    #     if purity(rho=state) > 0.9:
    #         return True
    if result.final_purity > 0.9:
        return True
    return False


def example_C_body():
    # C. State preparation from a thermal state with Jaynes-Cummings controls
    from feedback_grape.fgrape import optimize_pulse
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

    def qubit_unitary(alphas):
        alpha_re, alpha_im = alphas
        alpha = alpha_re + 1j * alpha_im
        return tensor(
            identity(N_cav),
            expm(-1j * (alpha * sigmap() + alpha.conjugate() * sigmam()) / 2),
        )

    def qubit_cavity_unitary(betas):
        beta_re, beta_im = betas
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

    def povm_measure_operator(measurement_outcome, params):
        """
        POVM for the measurement of the cavity state.
        returns Mm ( NOT the POVM element Em = Mm_dag @ Mm ), given measurement_outcome m, gamma and delta
        """
        gamma, delta = params
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
    learning_rate = 0.02
    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(42), 3)
    measure = Gate(
        gate=povm_measure_operator,
        initial_params=jax.random.uniform(
            key1,
            shape=(2,),  # 2 for gamma and delta
            minval=-2 * jnp.pi,
            maxval=2 * jnp.pi,
        ),
        measurement_flag=True,
        # param_constraints=[[0, jnp.pi], [-2*jnp.pi, 2*jnp.pi]],
    )

    qub_unitary = Gate(
        gate=qubit_unitary,
        initial_params=jax.random.uniform(
            key2,
            shape=(2,),  # 2 for gamma and delta
            minval=-2 * jnp.pi,
            maxval=2 * jnp.pi,
        ),
        measurement_flag=False,
        # param_constraints=[[-2*jnp.pi, 2*jnp.pi], [-2*jnp.pi, 2*jnp.pi]],
    )

    qub_cav = Gate(
        gate=qubit_cavity_unitary,
        initial_params=jax.random.uniform(
            key3,
            shape=(2,),  # 2 for gamma and delta
            minval=-2 * jnp.pi,
            maxval=2 * jnp.pi,
        ),
        measurement_flag=False,
        # param_constraints=[[-jnp.pi, jnp.pi], [-jnp.pi, jnp.pi]],
    )

    system_params = [measure, qub_unitary, qub_cav]

    result = optimize_pulse(
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
    print("result.final_fidelity: ", result.final_fidelity)
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
        if fid_val > 0.9:
            return True

    return False


def example_D_body():
    no_dissipation_flag = False
    dissipation_flag = False

    from feedback_grape.fgrape import optimize_pulse
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

    def povm_measure_operator(measurement_outcome, params):
        """
        POVM for the measurement of the cavity state.
        returns Mm ( NOT the POVM element Em = Mm_dag @ Mm ), given measurement_outcome m, gamma and delta
        """
        gamma, delta = params
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
    key = jax.random.PRNGKey(42)

    measure = Gate(
        gate=povm_measure_operator,
        initial_params=jax.random.uniform(
            key,
            shape=(2,),  # 2 for gamma and delta
            minval=-jnp.pi,
            maxval=jnp.pi,
        ),
        measurement_flag=True,
    )

    qub_unitary = Gate(
        gate=qubit_unitary,
        initial_params=jax.random.uniform(
            key,
            shape=(1,),  # 2 for gamma and delta
            minval=-jnp.pi,
            maxval=jnp.pi,
        ),
        measurement_flag=False,
    )

    qub_cav = Gate(
        gate=qubit_cavity_unitary,
        initial_params=jax.random.uniform(
            key,
            shape=(1,),  # 2 for gamma and delta
            minval=-jnp.pi,
            maxval=jnp.pi,
        ),
        measurement_flag=False,
    )

    system_params = [measure, qub_unitary, qub_cav]
    result = optimize_pulse(
        U_0=rho_target,
        C_target=rho_target,
        system_params=system_params,
        num_time_steps=1,
        mode="lookup",
        goal="fidelity",
        max_iter=1000,
        convergence_threshold=1e-16,
        learning_rate=0.025,
        evo_type="density",
        batch_size=1,
    )

    if result.final_fidelity > 0.95:
        no_dissipation_flag = True

    # Now we add dissipation

    decay = Decay(
        c_ops=[tensor(identity(N_cav), jnp.sqrt(0.10) * sigmam())],
    )

    system_params = [decay, measure, qub_unitary, qub_cav]
    result = optimize_pulse(
        U_0=rho_target,
        C_target=rho_target,
        system_params=system_params,
        num_time_steps=1,
        mode="lookup",
        goal="fidelity",
        max_iter=1000,
        convergence_threshold=1e-6,
        learning_rate=0.025,
        evo_type="density",
        batch_size=1,
    )

    if result.final_fidelity < 0.95 and result.final_fidelity > 0.8:
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
    from feedback_grape.fgrape import optimize_pulse
    from feedback_grape.utils.operators import sigmam, identity, cosm, sinm
    from feedback_grape.utils.states import coherent, basis
    from feedback_grape.utils.tensor import tensor
    import jax.numpy as jnp
    import jax

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
    def displacement_gate(alphas):
        """Displacement operator for a coherent state."""
        alpha_re, alpha_im = alphas
        alpha = alpha_re + 1j * alpha_im
        gate = jax.scipy.linalg.expm(
            alpha * create(N_cav) - alpha.conj() * destroy(N_cav)
        )
        return tensor(gate, identity(2))

    def displacement_gate_dag(alphas):
        """Displacement operator for a coherent state."""
        alpha_re, alpha_im = alphas
        alpha = alpha_re + 1j * alpha_im
        gate = (
            jax.scipy.linalg.expm(
                alpha * create(N_cav) - alpha.conj() * destroy(N_cav)
            )
            .conj()
            .T
        )
        return tensor(gate, identity(2))

    def snap_gate(phase_list):
        diags = jnp.ones(shape=(N_cav - len(phase_list)))
        exponentiated = jnp.exp(1j * jnp.array(phase_list))
        diags = jnp.concatenate((exponentiated, diags))
        return tensor(jnp.diag(diags), identity(2))

    from feedback_grape.utils.operators import create, destroy

    def povm_measure_operator(measurement_outcome, params):
        """
        POVM for the measurement of the cavity state.
        returns Mm ( NOT the POVM element Em = Mm_dag @ Mm ), given measurement_outcome m, gamma and delta
        """
        gamma, delta = params
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

    measure = Gate(
        gate=povm_measure_operator,
        initial_params=jax.random.uniform(
            key,
            shape=(2,),  # 2 for gamma and delta
            minval=-jnp.pi / 2,
            maxval=jnp.pi / 2,
            dtype=jnp.float64,
        ),
        measurement_flag=True,
        # param_constraints=[[0, 0.5], [-1, 1]],
    )

    displacement = Gate(
        gate=displacement_gate,
        initial_params=jax.random.uniform(
            key,
            shape=(2,),
            minval=-jnp.pi / 2,
            maxval=jnp.pi / 2,
            dtype=jnp.float64,
        ),
        measurement_flag=False,
    )

    snap = Gate(
        gate=snap_gate,
        initial_params=jax.random.uniform(
            key,
            shape=(N_snap,),
            minval=-jnp.pi / 2,
            maxval=jnp.pi / 2,
            dtype=jnp.float64,
        ),
        measurement_flag=False,
    )

    displacement_dag = Gate(
        gate=displacement_gate_dag,
        initial_params=jax.random.uniform(
            key,
            shape=(2,),
            minval=-jnp.pi / 2,
            maxval=jnp.pi / 2,
            dtype=jnp.float64,
        ),
        measurement_flag=False,
    )

    decay = Decay(c_ops=[tensor(identity(N_cav), jnp.sqrt(0.005) * sigmam())])

    system_params = [
        decay,
        measure,
        decay,
        displacement,
        snap,
        displacement_dag,
    ]

    result = optimize_pulse(
        U_0=rho_target,
        C_target=rho_target,
        system_params=system_params,
        num_time_steps=2,
        mode="nn",
        goal="fidelity",
        max_iter=1000,
        convergence_threshold=1e-6,
        learning_rate=0.01,
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
    # from feedback_grape.utils.fidelity import fidelity

    # for i, state in enumerate(result.final_state):
    #     fid_val = fidelity(
    #         C_target=rho_target, U_final=state, evo_type="density"
    #     )
    #     print(f"fidelity of state {i}:", fid_val)
    #     if fid_val > 0.9:
    #         return True
    print(f"Final fidelity: {result.final_fidelity}")
    if result.final_fidelity > 0.95:
        return True
    return False


# Test some specific features
#@pytest.mark.slow # (approx 1 min)
def test_reward_weights():
    """
    Tests a example_D with different reward weights (and less iterations for speed)
    to check that only timesteps included in the reward_weights contribute to the optimization.
    """
    from feedback_grape.fgrape import optimize_pulse
    from feedback_grape.utils.operators import (
        sigmap,
        sigmam,
        create,
        destroy,
        identity,
        cosm,
        sinm,
    )
    from feedback_grape.utils.states import basis, coherent
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

    def povm_measure_operator(measurement_outcome, params):
        """
        POVM for the measurement of the cavity state.
        returns Mm ( NOT the POVM element Em = Mm_dag @ Mm ), given measurement_outcome m, gamma and delta
        """
        gamma, delta = params
        number_operator = tensor(create(N_cav) @ destroy(N_cav), identity(2))
        angle = (gamma * number_operator) + delta / 2
        meas_op = jnp.where(
            measurement_outcome == 1,
            cosm(angle),
            sinm(angle),
        )
        return meas_op

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
    key = jax.random.PRNGKey(42)

    measure = Gate(
        gate=povm_measure_operator,
        initial_params=jax.random.uniform(
            key,
            shape=(2,),  # 2 for gamma and delta
            minval=-jnp.pi,
            maxval=jnp.pi,
        ),
        measurement_flag=True,
    )

    qub_unitary = Gate(
        gate=qubit_unitary,
        initial_params=jax.random.uniform(
            key,
            shape=(1,),  # 2 for gamma and delta
            minval=-jnp.pi,
            maxval=jnp.pi,
        ),
        measurement_flag=False,
    )

    qub_cav = Gate(
        gate=qubit_cavity_unitary,
        initial_params=jax.random.uniform(
            key,
            shape=(1,),  # 2 for gamma and delta
            minval=-jnp.pi,
            maxval=jnp.pi,
        ),
        measurement_flag=False,
    )

    system_params = [measure, qub_unitary, qub_cav]

    def wrapper(reward_weights):
        fidelities = optimize_pulse(
            U_0=rho_target,
            C_target=rho_target,
            system_params=system_params,
            num_time_steps=len(reward_weights),
            reward_weights=reward_weights,
            mode="lookup",
            goal="fidelity",
            max_iter=100,
            convergence_threshold=None,
            learning_rate=0.02,
            evo_type="density",
            batch_size=1,
            eval_batch_size=16,
            eval_time_steps=5,
        ).fidelity_each_timestep

        batch_mean = jnp.mean(jnp.array(fidelities), axis=1)

        return batch_mean
    
    result1 = wrapper(reward_weights=[1.0, 0.0])
    result2 = wrapper(reward_weights=[0.0, 1.0])
    result3 = wrapper(reward_weights=[1.0, 1.0])
    result4 = wrapper(reward_weights=[1.0])

    assert jnp.allclose(jnp.array([result1[0], result2[0], result3[0], result4[0]]), 1), "Fidelity at timestep 0 should be 1 because initial and target state are identical."
    assert jnp.all(result1[1] > 0.7) and jnp.all(result1[2:] < 0.3), "For reward_weights=[1.0, 0.0], only first timestep should be optimized."
    assert jnp.all(result2[1] < 0.2) and jnp.all(result2[2]  > 0.7) and jnp.all(result2[3:] < 0.5), "For reward_weights=[0.0, 1.0], only second timestep should be optimized."
    assert jnp.all(result3[1:3] > 0.7) and jnp.all(result3[3] < 0.5), "For reward_weights=[1.0, 1.0], two timesteps should be optimized."

    assert jnp.allclose(result1[1], result4[1]), "Unrewarded future should not affect optimization up to that timestep."
    assert not jnp.allclose(result1[2:], result4[2:]), "LUT column not contributing to loss should not be optimized."


#@pytest.mark.slow # (approx 45 seconds)
def test_eval_time_steps():
    """
        Tests example_E with eval_time_steps to check fidelity decay per timestep.
        It was observed that with the current setup, the fidelity decays between 0.95 and 0.99 per timestep.
    """
    from feedback_grape.fgrape import optimize_pulse
    from feedback_grape.utils.operators import cosm, sinm, identity
    from feedback_grape.utils.states import coherent
    import jax.numpy as jnp
    import jax

    from feedback_grape.utils.fidelity import ket2dm

    N_cav = 30  # number of cavity modes
    N_snap = 15

    alpha = 2
    psi_target = coherent(N_cav, alpha) + coherent(N_cav, -alpha)

    # Normalize psi_target before constructing rho_target
    psi_target = psi_target / jnp.linalg.norm(psi_target)

    rho_target = ket2dm(psi_target)

    from feedback_grape.utils.operators import create, destroy

    def displacement_gate(alphas):
        """Displacement operator for a coherent state."""
        alpha_re, alpha_im = alphas
        alpha = alpha_re + 1j * alpha_im
        gate = jax.scipy.linalg.expm(
            alpha * create(N_cav) - alpha.conj() * destroy(N_cav)
        )
        return gate


    def displacement_gate_dag(alphas):
        """Displacement operator for a coherent state."""
        alpha_re, alpha_im = alphas
        alpha = alpha_re + 1j * alpha_im
        gate = (
            jax.scipy.linalg.expm(
                alpha * create(N_cav) - alpha.conj() * destroy(N_cav)
            )
            .conj()
            .T
        )
        return gate

    def snap_gate(phase_list):
        diags = jnp.ones(shape=(N_cav - len(phase_list)))
        exponentiated = jnp.exp(1j * jnp.array(phase_list))
        diags = jnp.concatenate((exponentiated, diags))
        return jnp.diag(diags)

    def povm_measure_operator(measurement_outcome, params):
        """
        POVM for the measurement of the cavity state.
        returns Mm ( NOT the POVM element Em = Mm_dag @ Mm ), given measurement_outcome m, gamma and delta
        """
        gamma, delta = params
        angle = gamma * create(N_cav) @ destroy(N_cav) + delta / 2 * identity(N_cav)
        meas_op = jnp.where(
            measurement_outcome == 1,
            cosm(angle),
            sinm(angle),
        )
        return meas_op

    import flax.linen as nn

    # You can do whatever you want inside so long as you maintaing the hidden_size and output size shapes
    class RNN(nn.Module):
        hidden_size: int  # number of features in the hidden state
        output_size: int  # number of features in the output (inferred from the number of parameters) just provide those attributes to the class

        @nn.compact
        def __call__(self, measurement, hidden_state):

            if measurement.ndim == 1:
                measurement = measurement.reshape(1, -1)

            ###############
            ### Free to change whatever you want below as long as hidden layers have size self.hidden_size
            ### and output layer has size self.output_size
            ###############

            gru_cell = nn.GRUCell(
                features=self.hidden_size,
                gate_fn=nn.sigmoid,
                activation_fn=nn.tanh,
            )
            self.make_rng('dropout')

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

            ###############
            ### Do not change the return statement
            ###############

            return output[0], new_hidden_state

    from feedback_grape.fgrape import Decay, Gate

    key = jax.random.PRNGKey(42)

    measure = Gate(
        gate=povm_measure_operator,
        initial_params=jax.random.uniform(
            key,
            shape=(2,),  # 2 for gamma and delta
            minval=-jnp.pi / 2,
            maxval=jnp.pi / 2,
            dtype=jnp.float64,
        ),
        measurement_flag=True,
    )

    displacement = Gate(
        gate=displacement_gate,
        initial_params=jax.random.uniform(
            key,
            shape=(2,),
            minval=-jnp.pi / 2,
            maxval=jnp.pi / 2,
            dtype=jnp.float64,
        ),
        measurement_flag=False,
    )

    snap = Gate(
        gate=snap_gate,
        initial_params=jax.random.uniform(
            key,
            shape=(N_snap,),
            minval=-jnp.pi / 2,
            maxval=jnp.pi / 2,
            dtype=jnp.float64,
        ),
        measurement_flag=False,
    )

    displacement_dag = Gate(
        gate=displacement_gate_dag,
        initial_params=jax.random.uniform(
            key,
            shape=(2,),
            minval=-jnp.pi / 2,
            maxval=jnp.pi / 2,
            dtype=jnp.float64,
        ),
        measurement_flag=False,
    )

    decay = Decay(c_ops=[jnp.sqrt(0.005) * destroy(N_cav)])

    system_params = [decay, measure, decay, displacement, snap, displacement_dag]

    result = optimize_pulse(
        U_0=rho_target,
        C_target=rho_target,
        system_params=system_params,
        num_time_steps=2,
        reward_weights=[1.0, 1.0],
        mode="nn",
        goal="fidelity",
        max_iter=210,
        convergence_threshold=None,
        learning_rate=0.01,
        evo_type="density",
        batch_size=1,
        rnn=RNN,
        rnn_hidden_size=30,
        eval_batch_size=1,
        eval_time_steps=100,
        progress=True,
    )

    fidelities = jnp.mean(jnp.array(result.fidelity_each_timestep), axis=1)
    const = fidelities[1:] / fidelities[:-1]

    assert len(fidelities) == 101, "eval_time_steps not working as expected." # initial fidelity + 100 time steps
    assert jnp.all(const > 0.95) and jnp.all(const < 0.99), "Fidelity decay per timestep not as expected."


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


def test_for_errors():
    import jax.numpy as jnp

    # Minimal valid gate for tests
    def dummy_gate(params):
        return jnp.eye(2)

    valid_gate = Gate(
        gate=dummy_gate,
        initial_params=jnp.array([0.0]),
        measurement_flag=False,
    )

    # 1. num_time_steps <= 0
    with pytest.raises(ValueError,  match=re.escape("Time steps must be greater than 0.")):
        optimize_pulse(
            U_0=basis(2),
            C_target=basis(2),
            system_params=[valid_gate],
            num_time_steps=0,
            max_iter=1,
            convergence_threshold=None,
            learning_rate=0.01,
            evo_type="state",
        )

    # 2. Invalid evo_type
    with pytest.raises(ValueError,  match=re.escape("Invalid evo_type. Choose 'state' or 'density'.")):
        optimize_pulse(
            U_0=basis(2),
            C_target=basis(2),
            system_params=[valid_gate],
            num_time_steps=1,
            max_iter=1,
            convergence_threshold=None,
            learning_rate=0.01,
            evo_type="invalid",
        )

    # 3. U_0 is None
    with pytest.raises(ValueError,  match=re.escape("Please provide an initial state U_0.")):
        optimize_pulse(
            U_0=None,
            C_target=basis(2),
            system_params=[valid_gate],
            num_time_steps=1,
            max_iter=1,
            convergence_threshold=None,
            learning_rate=0.01,
            evo_type="state",
        )

    # 4. C_target is None and goal is fidelity
    with pytest.raises(ValueError,  match=re.escape("Please provide a target state C_target for fidelity calculation.")):
        optimize_pulse(
            U_0=basis(2),
            C_target=None,
            system_params=[valid_gate],
            num_time_steps=1,
            max_iter=1,
            convergence_threshold=None,
            learning_rate=0.01,
            evo_type="state",
            goal="fidelity",
        )

    # 5. evo_type='state' but not both kets
    with pytest.raises(TypeError,  match=re.escape("For evo_type='state', please provide initial and target states as kets (column vectors).")):
        optimize_pulse(
            U_0=jnp.eye(2),
            C_target=basis(2),
            system_params=[valid_gate],
            num_time_steps=1,
            max_iter=1,
            convergence_threshold=None,
            learning_rate=0.01,
            evo_type="state",
        )

    # 6. evo_type='density' but one is ket
    with pytest.raises(TypeError,  match=re.escape("For evo_type='density', please provide initial and target states as density matrices.")):
        optimize_pulse(
            U_0=basis(2),
            C_target=jnp.eye(2),
            system_params=[valid_gate],
            num_time_steps=1,
            max_iter=1,
            convergence_threshold=None,
            learning_rate=0.01,
            evo_type="density",
        )

    # 7. isbra(U_0) or isbra(C_target)
    # A bra is a row vector, e.g. shape (2,)
    with pytest.raises(TypeError,  match=re.escape("Please provide initial and target states as kets (column vectors) or density matrices.")):
        optimize_pulse(
            U_0=basis(2).conj().T,  # row vector
            C_target=basis(2),
            system_params=[valid_gate],
            num_time_steps=1,
            max_iter=1,
            convergence_threshold=None,
            learning_rate=0.01,
            evo_type="density",
        )

    # 8. Not positive semi-definite for density
    with pytest.raises(TypeError,  match=re.escape("If evo_type=`density` Your initial and target rhos must be positive semi-definite.")):
        optimize_pulse(
            U_0=jnp.array([[1, 2], [2, -3]]),  # not psd
            C_target=jnp.eye(2),
            system_params=[valid_gate],
            num_time_steps=1,
            max_iter=1,
            convergence_threshold=None,
            learning_rate=0.01,
            evo_type="density",
        )

    # 9. goal in ["purity", "both"] and evo_type == "state"
    with pytest.raises(ValueError,  match=re.escape("Purity is not defined for evo_type='state'. Please use evo_type='density' for purity calculation.")):
        optimize_pulse(
            U_0=basis(2),
            C_target=basis(2),
            system_params=[valid_gate],
            num_time_steps=1,
            max_iter=1,
            convergence_threshold=None,
            learning_rate=0.01,
            evo_type="state",
            goal="purity",
        )

    # 10. Decay with state evo_type
    decay = Decay(c_ops=[jnp.eye(2)])
    with pytest.raises(ValueError,  match=re.escape("Decay requires a density matrix representation of your inital and target states because, the solver uses Lindblad equation to evolve the system with dissipation. \n"
            "Please provide U_0 and U_target as density matrices perhaps using `utils.fidelity.ket2dm` and use evo_type='density'.")):
        optimize_pulse(
            U_0=basis(2),
            C_target=basis(2),
            system_params=[decay, valid_gate],
            num_time_steps=1,
            max_iter=1,
            convergence_threshold=None,
            learning_rate=0.01,
            evo_type="state",
        )
    
    # 11. param_constraints wrong length
    def dummy_gate(params):
        return jnp.eye(2)

    valid_gate_1 = Gate(
        gate=dummy_gate,
        initial_params=jnp.array([0.0, 0.0]),
        measurement_flag=False,
        param_constraints=[0, 1, 2],  # wrong length        
    )

    # param_constraints should be 2 * num_of_params (upper and lower for each param)
    with pytest.raises(TypeError, match=re.escape("Please provide upper and lower constraints for each variable in each gate, or don't provide `param_constraints` to use the default.")):
        optimize_pulse(
            U_0=basis(2),
            C_target=basis(2),
            system_params=[valid_gate_1],
            num_time_steps=1,
            max_iter=1,
            convergence_threshold=None,
            learning_rate=0.01,
            evo_type="state",
            mode="no-measurement",
        )

    # 12. no-measurement mode but measurement_flag True
    valid_gate_meas = Gate(
        gate=dummy_gate,
        initial_params=jnp.array([0.0]),
        measurement_flag=True,
    )
    with pytest.raises(ValueError, match=re.escape("The Positive operator valued measure gate you supplied must have at least two arguments. "
                        "The first argument is the measurement outcome (1, or -1) and the second argument is the list "
                        "of optimizable parameters for the measurement gate.")):
        optimize_pulse(
            U_0=basis(2),
            C_target=basis(2),
            system_params=[valid_gate_meas],
            num_time_steps=1,
            max_iter=1,
            convergence_threshold=None,
            learning_rate=0.01,
            evo_type="state",
            mode="no-measurement",
        )
    
    def dummy_meas_gate(meas_outcome, params):
        return jnp.eye(2)
    
    valid_gate_meas_2 = Gate(
        gate=dummy_meas_gate,
        initial_params=jnp.array([0.0]),
        measurement_flag=True,
    )
    with pytest.raises(ValueError, match=re.escape("You set a measurement flag to true, but no-measurement mode is used. Please set mode to 'nn' or 'lookup'.")):
        optimize_pulse(
            U_0=basis(2),
            C_target=basis(2),
            system_params=[valid_gate_meas_2],
            num_time_steps=1,
            max_iter=1,
            convergence_threshold=None,
            learning_rate=0.01,
            evo_type="state",
            mode="no-measurement",
        )
    
    valid_gate = Gate(
        gate=dummy_gate,
        initial_params=jnp.array([0.0, 0.0]),
        measurement_flag=False,
    )

    # 13. nn/lookup mode but no measurement operator
    with pytest.raises(ValueError, match=re.escape("For modes 'nn' and 'lookup', you must provide at least one measurement operator in your system_params. ")):
        optimize_pulse(
            U_0=basis(2),
            C_target=basis(2),
            system_params=[valid_gate],
            num_time_steps=1,
            max_iter=1,
            convergence_threshold=None,
            learning_rate=0.01,
            evo_type="state",
            mode="nn",
        )
    with pytest.raises(ValueError, match=re.escape("For modes 'nn' and 'lookup', you must provide at least one measurement operator in your system_params. ")):
        optimize_pulse(
            U_0=basis(2),
            C_target=basis(2),
            system_params=[valid_gate],
            num_time_steps=1,
            max_iter=1,
            convergence_threshold=None,
            learning_rate=0.01,
            evo_type="state",
            mode="lookup",
        )

    # 14. Invalid mode
    with pytest.raises(ValueError, match=re.escape("Invalid mode. Choose 'nn' or 'lookup' or 'no-measurement'.")):
        optimize_pulse(
            U_0=basis(2),
            C_target=basis(2),
            system_params=[valid_gate_meas_2],
            num_time_steps=1,
            max_iter=1,
            convergence_threshold=None,
            learning_rate=0.01,
            evo_type="state",
            mode="invalid_mode",
        )



    def dummy_gate(params):
        return jnp.eye(2)

    # Two gates, but only one has param_constraints
    gate1 = Gate(
        gate=dummy_gate,
        initial_params=jnp.array([0.0]),
        measurement_flag=False,
        param_constraints=[[0, 1]],  # valid for gate1
    )
    gate2 = Gate(
        gate=dummy_gate,
        initial_params=jnp.array([0.0]),
        measurement_flag=False,
        # No param_constraints for gate2
    )

    # 15. If you provide parameter constraints for some gates, you need to provide them for all gates.
    with pytest.raises(TypeError, match=re.escape("If you provide parameter constraints for some gates, you need to provide them for all gates.")):
        optimize_pulse(
            U_0=basis(2),
            C_target=basis(2),
            system_params=[gate1, gate2],
            num_time_steps=1,
            max_iter=1,
            convergence_threshold=None,
            learning_rate=0.01,
            evo_type="state",
            mode="no-measurement",
        )