import jax
import jax.numpy as jnp
# ruff: noqa N8


# TODO: see if you need to handle the case where the input is a state rather than a density matrix
def _probability_of_a_measurement_outcome_given_a_certain_state(
    rho_cav, measurement_outcome, povm_measure_operator, initial_params
):
    """
    Calculate the probability of a measurement outcome given a quantum state.

    Args:
        rho_cav: Density matrix of the cavity
        measurement_outcome: The measurement outcome
        povm_measure_operator: The POVM measurement operator
        params: Parameters for the POVM operator

    Returns:
        Probability of the measurement outcome
    """
    Em = povm_measure_operator(
        measurement_outcome, *initial_params
    ).conj().T @ povm_measure_operator(measurement_outcome, *initial_params)
    # QUESTION: would jnp.real be useful here?
    return jnp.real(jnp.trace(Em @ rho_cav))


# TODO: should generalize if input is a state rather than a density matrix
def _post_measurement_state(
    rho_cav, measurement_outcome, povm_measure_operator, initial_params
):
    """
    Returns the state after the measurement

    Args:
        rho_cav: Density matrix of the cavity
        measurement_outcome: The measurement outcome
        povm_measure_operator: The POVM measurement operator
        params: Parameters for the POVM operator

    Returns:
        Post-measurement state
    """
    Mm_op = povm_measure_operator(measurement_outcome, *initial_params)
    prob = _probability_of_a_measurement_outcome_given_a_certain_state(
        rho_cav,
        measurement_outcome,
        povm_measure_operator,
        initial_params,
    )
    # QUESTION: should we use jnp.clip here?
    prob = jnp.maximum(prob, 1e-10)
    # is denominator sqrted or not?
    return Mm_op @ rho_cav @ Mm_op.conj().T / prob


# TODO: need to handle the case where the rho_cav is not a density matrix
def povm(
    rho_cav,
    povm_measure_operator,  # type: ignore
    initial_povm_params,
    rng_key,
):
    """
    Perform a POVM measurement on the given state.

    Args:
        rho_cav (jnp.ndarray): The density matrix of the cavity.
        povm_measure_operator (callable): The POVM measurement operator.
        initial_povm_params (list): Initial parameters for the POVM measurement operator.

    Returns:
        tuple: A tuple containing the post-measurement state, the measurement result, and the log probability of the measurement outcome.
    """
    # TODO: this should be generalized to all possible measurement outcomes
    prob_plus = _probability_of_a_measurement_outcome_given_a_certain_state(
        rho_cav, 1, povm_measure_operator, initial_povm_params
    )
    random_value = jax.random.uniform(rng_key, shape=())
    measurement = jnp.where(random_value < prob_plus, 1, -1)
    rho_meas = _post_measurement_state(
        rho_cav, measurement, povm_measure_operator, initial_povm_params
    )
    # TODO: handle if there are more than 2 possibilities
    prob = jnp.where(
        measurement == 1,
        prob_plus,
        1 - prob_plus,
    )
    jax.debug.print("Measurement outcome: {}, Probability: {}, prob_plus: {}", measurement, prob, prob_plus)
    # QUESTION: If prob is 0 though then the log prob is -inf ( and 1e-10 will be a very huge number)
    log_prob = jnp.log(jnp.maximum(prob, 1e-10))
    return rho_meas, measurement, log_prob
