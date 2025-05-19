from feedback_grape.grape import fidelity


# TODO + NOTE: Currently L-bfgs with the same learning rate donverges at a local minimum of 0.5
# For kitten state you would need to account for avg photon number
def test_parameterized_grape():
    """
    Test the parameterized grape function.
    """
    # ruff: noqa
    from feedback_grape.grape_paramaterized import optimize_pulse_parameterized
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

    N_cav = 4

    def qubit_unitary(alpha):
        return expm(
            -1j
            * (
                alpha * tensor(identity(N_cav), sigmap())
                + alpha.conjugate() * tensor(identity(N_cav), sigmam())
            )
            / 2
        )

    def qubit_cavity_unitary(beta):
        return expm(
            -1j
            * (
                beta
                * (
                    tensor(destroy(N_cav), identity(2))
                    @ tensor(identity(N_cav), sigmap())
                )
                + beta.conjugate()
                * (
                    tensor(create(N_cav), identity(2))
                    @ tensor(identity(N_cav), sigmam())
                )
            )
            / 2
        )

    alpha = 0.1 + 0.1j
    beta = 0.1 + 0.1j
    Uq = qubit_unitary(alpha)
    Uqc = qubit_cavity_unitary(beta)
    print(
        "Uq unitary check:",
        jnp.allclose(Uq.conj().T @ Uq, jnp.eye(Uq.shape[0]), atol=1e-7),
    )
    print(
        "Uqc unitary check:",
        jnp.allclose(Uqc.conj().T @ Uqc, jnp.eye(Uqc.shape[0]), atol=1e-7),
    )
    time_steps = 3  # corressponds to maximal excitation number of an arbitrary Fock State Superposition
    psi0 = tensor(basis(N_cav), basis(2))
    psi_target = tensor((fock(N_cav, 1) + fock(N_cav, 3)), basis(2))
    num_gates = len(
        [qubit_unitary, qubit_cavity_unitary]
    )  # Number of parameterized gates
    initial_parameters = jnp.full(
        (time_steps, num_gates), 0.1, dtype=jnp.float64
    )
    result = optimize_pulse_parameterized(
        U_0=psi0,
        C_target=psi_target,
        parameterized_gates=[qubit_unitary, qubit_cavity_unitary],
        initial_parameters=initial_parameters,
        num_time_steps=time_steps,
        optimizer="adam",
        max_iter=1000,
        convergence_threshold=1e-6,
        type="state",
        propcomp="time-efficient",
        learning_rate=0.3,
    )
    print("Result:", result)
    print(
        fidelity(
            U_final=result.final_operator, C_target=psi_target, type="state"
        )
    )
    assert (
        fidelity(
            C_target=psi_target,
            U_final=result.final_operator,
            type="state",
        )
        > 0.99
    ), "Final state does not match target state"
