import jax
import jax.numpy as jnp

from feedback_grape.utils.fidelity import isbra, isket
from .fgrape_helpers import clip_params
# ruff: noqa N8

jax.config.update("jax_enable_x64", True)


def _probability_of_a_measurement_outcome_given_a_certain_state(
    rho_cav,
    measurement_outcome,
    M_plus,
    M_minus,
    evo_type,
):
    """
    Calculate the probability of a measurement outcome given a quantum state.

    Args:
        rho_cav: Density matrix of the cavity
        measurement_outcome: The measurement outcome
        M_plus: The measurement operator for outcome +1
        M_minus: The measurement operator for outcome -1
        evo_type: Evolution type, either 'state' or 'density_matrix'

    Returns:
        Probability of the measurement outcome
    """
    Mm = jnp.where(
        measurement_outcome == 1,
        M_plus,
        M_minus,
    )

    if evo_type == "state":
        if not isket(rho_cav):
            raise TypeError(
                "rho_cav must be a ket (column vector) for evo_type 'state'."
            )
        # Answer: without jnp.real the result is complex and jnp.grad can no longer do it, but are we then losing information? --> No because probability is real
        # and if the math is correct this quality should be real or have very very small imaginary part because its the probability (the .real is essential for jnp.grad to work)
        # just to remove the +0j
        numerator = Mm @ rho_cav
        prob = jnp.real(jnp.vdot(numerator, numerator))
    elif evo_type == "density":
        if (
            isket(rho_cav)
            or isbra(rho_cav)
            or rho_cav.ndim != 2
            or rho_cav.shape[0] != rho_cav.shape[1]
        ):
            raise TypeError(
                "rho_cav must be a density matrix for evo_type 'density'."
            )
        
        prob = jnp.real(jnp.trace(Mm.conj().T @ Mm @ rho_cav))

        # 2x faster because it only evaluates diagonal elements
        # of second matrix multiplication before taking trace,
        # hence saving one full matrix multiplication.
        # Numerical issues when executing on GPU.
        # prob = jnp.real(jnp.vdot(Mm, Mm @ rho_cav))
    else:
        raise ValueError(f"Invalid evo_type: {evo_type}.")

    return prob


def _post_measurement_state(
    rho_cav,
    measurement_outcome,
    M_plus,
    M_minus,
    prob_plus,
    evo_type,
):
    """
    Returns the state after the measurement

    Args:
        rho_cav: Density matrix of the cavity
        measurement_outcome: The measurement outcome
        M_plus: The measurement operator for outcome +1
        M_minus: The measurement operator for outcome -1
        evo_type: Evolution type, either 'state' or 'density_matrix'

    Returns:
        Post-measurement state
    """
    Mm_op = jnp.where(
        measurement_outcome == 1,
        M_plus,
        M_minus,
    )
    prob = jnp.where(
        measurement_outcome == 1,
        prob_plus,
        1 - prob_plus,
    )

    if evo_type == "state":
        if not isket(rho_cav):
            raise TypeError(
                "rho_cav must be a ket (column vector) for evo_type 'state'."
            )
        numerator = Mm_op @ rho_cav
        prob = jnp.real(jnp.vdot(numerator, numerator))
    elif evo_type == "density":
        if (
            isket(rho_cav)
            or isbra(rho_cav)
            or rho_cav.ndim != 2
            or rho_cav.shape[0] != rho_cav.shape[1]
        ):
            raise TypeError(
                "rho_cav must be a density matrix for evo_type 'density'."
            )
        numerator = Mm_op @ rho_cav @ Mm_op.conj().T
        prob = jnp.real(jnp.trace(numerator))
    else:
        raise ValueError(f"Invalid evo_type: {evo_type}.")

    prob = jnp.maximum(prob, 1e-10)

    return numerator / prob


def povm(
    rho_cav,
    povm_measure_operator,  # type: ignore
    initial_povm_params,
    gate_param_constraints,
    rng_key,
    evo_type,
):
    """
    Perform a POVM measurement on the given state. Gets called when user provides measurement_flag=True in one of the Gate NamedTuples.

    Args:
        rho_cav (jnp.ndarray): The density matrix of the cavity.
        povm_measure_operator (callable): The POVM measurement operator.
            - It should take a measurement outcome and list of parameters as input.
            - The measurement outcome options are either 1 or -1
        initial_povm_params (list): Initial parameters for the POVM measurement operator.
        gate_param_constraints (list): Constraints for the gate parameters.
        rng_key (jax.random.PRNGKey): Random number generator key for stochastic operations.
        evo_type (str): Evolution type, either 'state' or 'density_matrix'.

    Returns:
        tuple: A tuple containing the post-measurement state, the measurement result, and the log probability of the measurement outcome.
    """

    initial_povm_params = clip_params(
        initial_povm_params, gate_param_constraints
    )

    M_plus = povm_measure_operator(1, *[initial_povm_params])
    M_minus = povm_measure_operator(-1, *[initial_povm_params])

    prob_plus = _probability_of_a_measurement_outcome_given_a_certain_state(
        rho_cav, 1, M_plus, M_minus, evo_type
    )
    random_value = jax.random.uniform(rng_key, shape=())
    measurement_outcome = jnp.where(random_value < prob_plus, 1, -1)
    rho_meas = _post_measurement_state(
        rho_cav,
        measurement_outcome,
        M_plus,
        M_minus,
        prob_plus,
        evo_type,
    )
    prob = jnp.where(
        measurement_outcome == 1,
        prob_plus,
        1 - prob_plus,
    )
    # QUESTION: If prob is 0 though then the log prob is -inf ( and 1e-10 will be a very huge number)
    # ANSWER: No, it will be log(1e-10) = -23 which should be okay.
    log_prob = jnp.log(jnp.maximum(prob, 1e-10))
    return rho_meas, measurement_outcome, log_prob
