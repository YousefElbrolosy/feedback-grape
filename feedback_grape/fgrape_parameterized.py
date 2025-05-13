import jax.numpy as jnp
from feedback_grape.grape import result
# ruff: noqa N8


def _probability_of_a_measurement_outcome_given_a_certain_state(
    rho_cav, measurement_outcome, povm_measure_operator, **kwargs
):
    Em = povm_measure_operator(
        measurement_outcome, **kwargs
    ).conj().T @ povm_measure_operator(measurement_outcome, **kwargs)
    return jnp.trace(Em @ rho_cav)


# TODO: should generalize if input is a state rather than a density matrix
def _post_measurement_state(
    rho_cav, measurement_outcome, povm_measure_operator, initial_params
):
    """
    returns the state after the measurement
    """
    return (
        povm_measure_operator(measurement_outcome, **initial_params)
        @ rho_cav
        @ povm_measure_operator(measurement_outcome, **initial_params).conj().T
        / jnp.sqrt(
            _probability_of_a_measurement_outcome_given_a_certain_state(
                rho_cav,
                measurement_outcome,
                povm_measure_operator,
                **initial_params,
            )
        )
    )


def povm(
    rho_cav: jnp.ndarray,
    measurement_outcome: int,
    povm_measure_operator: callable,
    initial_params: jnp.ndarray,
) -> tuple[jnp.ndarray, int, float]:
    """
    Perform a POVM measurement on the given state.

    Args:
        rho_cav (jnp.ndarray): The density matrix of the cavity.
        measurement_outcome (int): The measurement outcome.
        povm_measure_operator (callable): The POVM measurement operator.
        initial_params (jnp.ndarray): Initial parameters for the POVM measurement operator.

    Returns:
        tuple: A tuple containing the post-measurement state, the measurement result, and the log probability of the measurement outcome.
    """
    if measurement_outcome == 0:
        measurement = -1
    else:
        measurement = 1

    rho_meas = _post_measurement_state(
        rho_cav, measurement_outcome, povm_measure_operator, **initial_params
    )
    prob = _probability_of_a_measurement_outcome_given_a_certain_state(
        rho_cav, measurement_outcome, povm_measure_operator, **initial_params
    )

    random = jnp.random.uniform(0, 1)

    if random < prob:
        measurement = measurement_outcome
    else:
        measurement = -1 * measurement_outcome

    log_prob = jnp.log(jnp.abs(prob))

    return rho_meas, measurement, log_prob


def purity(*, rho, type="density"):
    """
    Computes the purity of a density matrix.

    Args:
        rho: Density matrix.
        type: Type of density matrix ("density" or "superoperator").
    Returns:
        purity: Purity value.
    """
    if type == "density":
        return jnp.trace(rho @ rho)
    elif type == "superoperator":
        pass  # TODO: implement superoperator purity if such a thing exists
    else:
        raise ValueError("Invalid type. Choose 'density' or 'superoperator'.")


def _calculate_time_step(
    rho_cav,
    initial_fake_measurement_outcome,
    povm_measure_operator,
    initial_params,
):
    """
    Calculate the time step for the given density matrix and POVM measurement operator.
    """
    return povm(
        rho_cav,
        initial_fake_measurement_outcome,
        povm_measure_operator,
        **initial_params,
    )


def calculate_trajectory(
    rho_cav, time_steps, povm_measure_operator, initial_params
):
    rho_meas = rho_cav
    new_params = initial_params
    # would the initial be something fake in random be better? QUESTION, TODO
    # initial_fake_measurement_outcome = jnp.random.choice([-1, 1])
    # initial_ fake measurement
    measurement = 1
    for _ in range(time_steps - 1):
        rho_meas, measurement, log_prob = _calculate_time_step(
            rho_meas, measurement, povm_measure_operator, new_params
        )
        # put measurement to RNN
        # TODO: implement RNN
        updated_params = _get_new_params_from_rnn(measurement)
        # get updated params
        new_params = updated_params
    # for the last time step
    return _calculate_time_step(rho_meas, 1, povm_measure_operator, new_params)


def optimize_pulse_parameterized(
    U_0: jnp.ndarray,
    C_target: jnp.ndarray,
    parameterized_gates: list[callable],  # type: ignore
    povm_measure_operator: callable,  # type: ignore
    initial_params: jnp.ndarray,
    goal: str,  # purity, fidelity, both
    mode: str,  # nn, lookup
    num_time_steps: int,
    optimizer: str,  # adam, l-bfgs
    max_iter: int,
    convergence_threshold: float,
    learning_rate: float,
    type: str,  # unitary, state, density, superoperator (used now mainly for fidelity calculation)
    propcomp: str = "time-efficient",  # time-efficient, memory-efficient
) -> result | None:
    """
    Optimizes pulse parameters for quantum systems based on the specified configuration.

    Args:
        U_0: Initial state or /unitary/density/super operator.
        C_target: Target state or /unitary/density/super operator.
        parameterized_gates (list[callable]): A list of parameterized gate functions to be optimized.
        povm_measure_operator (callable): The POVM measurement operator Mm.
        initial_params (jnp.ndarray): Initial parameters for the parameterized gates.
        goal (str): The optimization goal, which can be 'purity', 'fidelity', or 'both'.
        mode (str): The mode of operation, either 'nn' (neural network) or 'lookup' (lookup table).
        num_time_steps (int): The number of time steps for the optimization process.
        optimizer (str): The optimization algorithm to use, such as 'adam' or 'l-bfgs'.
        max_iter (int): The maximum number of iterations for the optimization process.
        convergence_threshold (float): The threshold for convergence to determine when to stop optimization.
        learning_rate (float): The learning rate for the optimization algorithm.
        type (str): The type of quantum system representation, such as 'unitary', 'state', 'density', or 'superoperator'.
                    This is primarily used for fidelity calculation.
        propcomp (str): The method for propagator computation, either 'time-efficient' or 'memory-efficient'.
                        This determines how the forward evolution is computed.
    Returns:
        result: Dictionary containing optimized pulse and convergence data.
    """
    if num_time_steps == 0:
        raise ValueError("Time steps must be greater than 0.")
    if goal == "purity":
        if povm_measure_operator is None:
            raise ValueError(
                "POVM measurement operator is required for purity optimization."
            )

        def _purity(params):
            """
            Computes the purity of the cavity state after a measurement.
            """
            rho_meas, _, _ = calculate_trajectory(
                rho_cav=U_0,
                time_steps=num_time_steps,
                povm_measure_operator=povm_measure_operator,
                initial_params=params,
            )
            return purity(rho_meas, type="density")

        if mode == "nn":
            # TODO: Construct NN
            network_variables = None # Placeholder for the neural network parameters
            if isinstance(optimizer, tuple):
                optimizer = optimizer[0]
            if optimizer.upper() == "L-BFGS":
                optimized_parameters, final_fidelity, iter_idx = (
                    _optimize_L_BFGS_feedback(
                        _purity,
                        initial_params,
                        max_iter,
                        convergence_threshold,
                        network_variables=network_variables,
                    )
                )
            elif optimizer.upper() == "ADAM":
                optimized_parameters, final_fidelity, iter_idx = (
                    _optimize_adam_feedback(
                        _purity,
                        initial_params,
                        max_iter,
                        learning_rate,
                        convergence_threshold,
                        network_variables=network_variables,
                    )
                )
            else:
                raise ValueError(
                    "Invalid optimizer. Choose 'adam' or 'l-bfgs'."
                )
        elif mode == "lookup":
            pass

    elif goal == "fidelity":
        pass  # TODO: implement fidelity optimization
    elif goal == "both":
        if povm_measure_operator is None:
            raise ValueError(
                "POVM measurement operator is required for purity optimization."
            )
        pass  # TODO: implement both optimization
    else:
        raise ValueError(
            "Invalid goal. Choose 'purity', 'fidelity', or 'both'."
        )


def _optimize_L_BFGS_feedback():
    """
    Optimizes the network parameters using the L-BFGS algorithm.
    """
    pass  # TODO: implement L-BFGS optimization


def _optimize_adam_feedback():
    """
    Optimizes the network parameters using the Adam algorithm.
    """
    pass  # TODO: implement Adam optimization



### Important:

#     if goal == "fidelity":
#         final_fidelity = compute_fidelity(rho, rho_target)
#         if just_last_fidelity:
#             R = final_fidelity
#         else:
#             R = fidelity

#     if goal == "purity":
#         final_purity = tf.abs(tf.linalg.trace(rho ** 2))
#         R = final_purity

#     if goal == "both":
#         final_fidelity = compute_fidelity(rho, rho_target)
#         final_purity = tf.abs(tf.linalg.trace(rho ** 2))
#         R = final_fidelity + final_purity

#     loss1 = tf.reduce_mean(-R)
#     if not feedback:
#         loss = loss1
#     if feedback and with_logP_term:
#         loss2 = tf.reduce_mean(log_probs * tf.stop_gradient(-R))
#         loss = loss1 + loss2
# # Define parameters to optimize
# if mode == "network":
#     params = network.trainable_variables
# if mode == "lookup":
#     params = [F]
# if mode == "none":
#     params = [controls_array]
# if feedback:
#     params += [F_1_0]

# grads = tape.gradient(loss, params)

# # Add noise to the gradients
# # t = tf.cast(epoch, grads[0].dtype )
# # variance = 0.01 / ((1 + t) ** 0.55)
# # grads = [ grad + tf.random.normal(grad.shape, mean=0.0, stddev=tf.math.sqrt(variance), dtype=grads[0].dtype) for grad in grads ]
# # epoch = epoch + 1

# if goal == "fidelity":
#     return grads, -loss1 / max_steps, final_fidelity, rho
# if goal == "purity":
#     return grads, loss, final_purity, rho
# if goal == "both":
#     return grads, loss, final_fidelity, final_purity, rho