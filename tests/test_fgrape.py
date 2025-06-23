# examples B,C,E are not tested because they take a lot of time to run
# ruff: noqa
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
    psi_target = tensor((fock(N_cav, 1) + fock(N_cav, 3)) / jnp.sqrt(2), basis(2))
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
        type="density",
        mode="no-measurement",
        goal="fidelity",
        learning_rate=0.02,
        batch_size=10,
        eval_batch_size=2,
    )

    if (result.final_fidelity > 0.99):
        return True
    else:
        print(f"Final fidelity: {result.final_fidelity}")
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
        type="density",
        batch_size=1,
    )

    if (result.final_fidelity > 0.99):
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
        type="density",
        batch_size=1,
    )

    if (result.final_fidelity < 0.99):
        dissipation_flag = True
    

    if no_dissipation_flag and dissipation_flag:
        return True
    else:
        print(
            f"Final fidelity without dissipation: {result.final_fidelity}, with dissipation: {result.final_fidelity}"
        )
        return False


# test normal state preparation using parameterized grape
def test_example_A():
    """
    This test tests if the max fidelity reached by example_B is above 0.99
    """
    assert example_A_body(), (
        "The max fidelity reached by example_A for batch size 10 and eval_batch_size 2 is below 0.99"
    )


def test_example_D():
    """
    This test tests if the max fidelity reached by example_C is above 0.9
    """
    assert example_D_body(), (
        "The max fidelity reached by example_D is not within expected range (>0.99 for no dissipation and below 0.99 for dissipation)."
    )
