import jax
import jax.numpy as jnp
from typing import List, NamedTuple
from feedback_grape.utils.optimizers import (
    _optimize_adam_feedback,
    _optimize_L_BFGS,
)
from feedback_grape.utils.fidelity import fidelity
from feedback_grape.utils.purity import purity
from feedback_grape.utils.povm import povm
from feedback_grape.fgrape_helpers import (
    prepare_parameters_from_dict,
    construct_ragged_row,
    extract_from_lut,
    reshape_params,
    apply_gate,
    RNN,
)

# TODO: see if I should replace with pmap for feedback-grape gpu version (may also be a different package)
# ruff: noqa N8
jax.config.update("jax_enable_x64", True)


class FgResult(NamedTuple):
    """
    result class to store the results of the optimization process.
    """

    optimized_rnn_parameters: jnp.ndarray
    """
    Optimized control amplitudes.
    """
    iterations: int
    """
    Number of iterations taken for optimization.
    """
    final_state: jnp.ndarray
    """
    Final operator after applying the optimized control amplitudes.
    """
    returned_params: List[List]
    """
    Array of finalsPOVM parameters for each time step.
    """
    final_purity: float | None
    """
    Final purity of the optimized control.
    """
    final_fidelity: float | None
    """
    Final fidelity of the optimized control.
    """


def _calculate_time_step(
    *,
    rho_cav,
    parameterized_gates,
    measurement_indices,
    initial_params,
    param_shapes,
    rnn_model=None,
    rnn_params=None,
    rnn_state=None,
    lut=None,
    measurement_history=None,
    type,
    rng_key,
):
    """
    Calculate the time step for the optimization process.

    Args:
        rho_cav: Density matrix of the cavity.
        povm_measure_operator: POVM measurement operator.
        initial_povm_params: Initial parameters for the POVM measurement operator.

    Returns:

    """
    rho_final = rho_cav
    total_log_prob = 0.0
    applied_params = []
    # TODO: IMP - See which is the more correct, should new params be propagated
    # directly within the same time step
    # or new parameters are together within the same time step

    if lut is not None:
        extracted_lut_params = initial_params

        # Apply each gate in sequence
        for i, gate in enumerate(parameterized_gates):
            # TODO: handle more carefully when there are multiple measurements
            if i in measurement_indices:
                rho_final, measurement, log_prob = povm(
                    rho_final, gate, extracted_lut_params[i], rng_key
                )
                measurement_history.append(measurement)
                applied_params.append(extracted_lut_params[i])
                extracted_lut_params = extract_from_lut(
                    lut, measurement_history
                )
                extracted_lut_params = reshape_params(
                    param_shapes, extracted_lut_params
                )
                total_log_prob += log_prob
            else:
                rho_final = apply_gate(
                    rho_final, gate, extracted_lut_params[i], type
                )
                applied_params.append(extracted_lut_params[i])

        return (
            rho_final,
            total_log_prob,
            extracted_lut_params,
            applied_params,
            measurement_history,
        )
    else:
        updated_params = initial_params
        new_hidden_state = rnn_state
        # Apply each gate in sequence
        for i, gate in enumerate(parameterized_gates):
            # TODO: handle more carefully when there are multiple measurements
            if i in measurement_indices:
                rho_final, measurement, log_prob = povm(
                    rho_final, gate, updated_params[i], rng_key
                )
                applied_params.append(updated_params[i])
                updated_params, new_hidden_state = rnn_model.apply(
                    rnn_params, jnp.array([measurement]), new_hidden_state
                )

                updated_params = reshape_params(param_shapes, updated_params)
                total_log_prob += log_prob
            else:
                rho_final = apply_gate(
                    rho_final, gate, updated_params[i], type
                )
                applied_params.append(updated_params[i])

        return (
            rho_final,
            total_log_prob,
            updated_params,
            applied_params,
            new_hidden_state,
        )

    # # Apply each gate in sequence
    # for i, gate in enumerate(parameterized_gates):
    #     gate_params = initial_params[i]
    #     # TODO: handle more carefully when there are multiple measurements
    #     if i in measurement_indices:
    #         rho_final, measurement, log_prob = povm(
    #             rho_final, gate, gate_params
    #         )
    #         updated_params, new_hidden_state = rnn_model.apply(
    #             rnn_params, jnp.array([measurement]), rnn_state
    #         )
    #         reshaped_rnn_params = reshape_params(param_shapes, updated_params)
    #         total_log_prob += log_prob
    #     else:
    #         rho_final = apply_gate(rho_final, gate, gate_params, type)

    # return rho_final, total_log_prob, reshaped_rnn_params, new_hidden_state


def calculate_trajectory(
    *,
    rho_cav,
    parameterized_gates,
    measurement_indices,
    initial_params,
    param_shapes,
    time_steps,
    rnn_model=None,
    rnn_params=None,
    rnn_state=None,
    lut=None,
    type,
    batch_size,
    rng_key,
):
    """
    Calculate a complete quantum trajectory with feedback.

    Args:
        rho_cav: Initial density matrix of the cavity.
        parameterized_gates: List of parameterized gates.
        measurement_indices: Indices of gates used for measurements.
        initial_params: Initial parameters for all gates.
        param_shapes: List of shapes for each gate's parameters.
        time_steps: Number of time steps within a trajectory.
        rnn_model: RNN model for feedback.
        rnn_params: Parameters of the RNN model.
        rnn_state: Initial state of the RNN model.
        type: Type of quantum system representation (e.g., "density").

    Returns:
        Final state, log probability, array of POVM parameters
    """
    # TODO + QUESTION: in the paper, it says one should average the reward over all possible measurement outcomes
    # How can one do that? Is this where batching comes into play? Should one do this averaging for log_prob as well?
    # Initialize batched rho_final for batch_size trajectories using jnp.repeat
    rho_final_batched = jnp.repeat(
        jnp.expand_dims(rho_cav, 0), batch_size, axis=0
    )

    # Split rng_key into batch_size keys for independent trajectories
    rng_keys = jax.random.split(rng_key, batch_size)

    def _calculate_single_trajectory(
        rho_cav,
        parameterized_gates,
        measurement_indices,
        initial_params,
        param_shapes,
        rnn_model,
        rnn_params,
        hidden_state,
        lut,
        type,
        rng_key,
    ):
        time_step_keys = jax.random.split(rng_key, time_steps)
        resulting_params = []
        rho_final = rho_cav
        total_log_prob = 0.0
        new_params = initial_params
        if lut is not None:
            measurement_history = []
            for i in range(time_steps):
                (
                    rho_final,
                    total_log_prob,
                    new_params,
                    applied_params,
                    measurement_history,
                ) = _calculate_time_step(
                    rho_cav=rho_final,
                    parameterized_gates=parameterized_gates,
                    measurement_indices=measurement_indices,
                    initial_params=new_params,
                    param_shapes=param_shapes,
                    lut=lut,
                    measurement_history=measurement_history,
                    type=type,
                    rng_key=time_step_keys[i],
                )
                # Thus, during - Refer to Eq(3) in fgrape paper
                # the individual time-evolution trajectory, this term may
                # be easily accumulated step by step, since the conditional
                # probabilities are known (these are just the POVM mea-
                # surement probabilities)
                total_log_prob += total_log_prob

                resulting_params.append(applied_params)

        else:
            new_hidden_state = hidden_state
            for i in range(time_steps):
                (
                    rho_final,
                    log_prob,
                    new_params,
                    applied_params,
                    new_hidden_state,
                ) = _calculate_time_step(
                    rho_cav=rho_final,
                    parameterized_gates=parameterized_gates,
                    measurement_indices=measurement_indices,
                    initial_params=new_params,
                    param_shapes=param_shapes,
                    rnn_model=rnn_model,
                    rnn_params=rnn_params,
                    rnn_state=new_hidden_state,
                    type=type,
                    rng_key=time_step_keys[i],
                )
                # Thus, during - Refer to Eq(3) in fgrape paper
                # the individual time-evolution trajectory, this term may
                # be easily accumulated step by step, since the conditional
                # probabilities are known (these are just the POVM mea-
                # surement probabilities)
                total_log_prob += log_prob

                resulting_params.append(applied_params)

        return rho_final, total_log_prob, resulting_params

    # Use jax.vmap to vectorize the trajectory calculation for batch_size
    batched_trajectory_fn = jax.vmap(
        _calculate_single_trajectory,
        in_axes=(
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            0,
        ),
    )
    return batched_trajectory_fn(
        rho_final_batched,
        parameterized_gates,
        measurement_indices,
        initial_params,
        param_shapes,
        rnn_model,
        rnn_params,
        rnn_state,
        lut,
        type,
        rng_keys,
    )


def optimize_pulse_with_feedback(
    U_0: jnp.ndarray,
    C_target: jnp.ndarray,
    parameterized_gates: list[callable],  # type: ignore
    measurement_indices: list[int],
    initial_params: list,
    goal: str,  # purity, fidelity, both
    mode: str,  # nn, lookup
    num_time_steps: int,
    optimizer: str,  # adam, l-bfgs
    max_iter: int,
    convergence_threshold: float,
    learning_rate: float,
    type: str,  # unitary, state, density, superoperator (used now mainly for fidelity calculation)
    batch_size: int,
    RNN: callable = RNN,
) -> FgResult:
    """
    Optimizes pulse parameters for quantum systems based on the specified configuration.

    Args:
        U_0: Initial state or /unitary/density/super operator.
        C_target: Target state or /unitary/density/super operator.
        parameterized_gates (list[callable]): A list of parameterized gate functions to be optimized.
        measurement_indices (list[int]): Indices of the parameterized gates that are used for measurements.
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
        RNN (callable): The RNN model to use for the optimization process. Defaults to a predefined RNN class. Only used if mode is 'nn'.
    Returns:
        result: Dictionary containing optimized pulse and convergence data.
    """
    if num_time_steps <= 0:
        raise ValueError("Time steps must be greater than 0.")

    # Convert dictionary parameters to list[list] structure
    flat_params, param_shapes = prepare_parameters_from_dict(initial_params)
    # Calculate total number of parameters
    num_of_params = len(jax.tree_util.tree_leaves(initial_params))
    trainable_params = None

    parent_rng_key = jax.random.PRNGKey(0)

    if mode == "nn":
        hidden_size = 32
        output_size = num_of_params

        rnn_model = RNN(hidden_size=hidden_size, output_size=output_size)
        h_initial_state = jnp.zeros((1, hidden_size))

        dummy_input = jnp.zeros((1, 1))  # Dummy input for RNN initialization
        trainable_params = rnn_model.init(
            parent_rng_key, dummy_input, h_initial_state
        )
    # TODO: see why this doesn't improve performance
    elif mode == "lookup":
        h_initial_state = None
        rnn_model = None
        # step 1: initialize the parameters
        num_of_columns = num_of_params
        num_of_sub_lists = num_time_steps - 1
        F = []
        # TODO: check if including initial params to be optimized is correct
        # F.append([flat_params])
        # construct ragged lookup table
        for i in range(1, num_of_sub_lists + 1):
            F.append(
                construct_ragged_row(
                    num_of_rows=2**i,
                    num_of_columns=num_of_columns,
                    param_shapes=param_shapes,
                )
            )
        # TODO: Padd the lookup table with zeros
        min_num_of_rows = 2 ** len(F)
        for i in range(len(F)):
            if len(F[i]) < min_num_of_rows:
                zeros_arrays = [
                    jnp.zeros((num_of_columns,), dtype=jnp.float64)
                    for _ in range(min_num_of_rows - len(F[i]))
                ]
                F[i] = F[i] + zeros_arrays
        trainable_params = F
    else:
        raise ValueError("Invalid mode. Choose 'nn' or 'lookup'.")

    # TODO: see if we need the log prob term
    # TODO: see if we need to implement stochastic sampling instead
    # QUESTION: should we add an accumilate log-term boolean here that decides whether we add
    # the log prob or not? ( like in porroti's implementation )?
    def loss_fn(trainable_params, rng_key):
        """
        Loss function for the optimization process.
        This function calculates the loss based on the specified goal (purity, fidelity, or both).
        Args:
            rnn_params: Parameters of the RNN model or lookup table.
            rng_key: Random key for stochastic operations.
        Returns:
            Loss value to be minimized.
        """

        if mode == "nn":
            # reseting hidden state at end of every trajectory ( does not really change the purity tho)
            h_initial_state = jnp.zeros((1, hidden_size))
            rnn_params = trainable_params
            lookup_table_params = None
        else:
            h_initial_state = None
            rnn_params = None
            rnn_model = None
            lookup_table_params = trainable_params

        rho_final, log_prob, _ = calculate_trajectory(
            rho_cav=U_0,
            parameterized_gates=parameterized_gates,
            measurement_indices=measurement_indices,
            initial_params=flat_params,
            param_shapes=param_shapes,
            time_steps=num_time_steps,
            rnn_model=rnn_model,
            rnn_params=rnn_params,
            rnn_state=h_initial_state,
            lut=lookup_table_params,
            type=type,
            batch_size=batch_size,
            rng_key=rng_key,
        )
        if goal == "purity":
            purity_values = jax.vmap(purity)(rho=rho_final)
            loss1 = jnp.mean(-purity_values)
            loss2 = jnp.mean(log_prob * jax.lax.stop_gradient(-purity_values))

        elif goal == "fidelity":
            if C_target == None:
                raise ValueError(
                    "C_target must be provided for fidelity calculation."
                )
            fidelity_value = jax.vmap(
                lambda rf: fidelity(C_target=C_target, U_final=rf, type=type)
            )(rho_final)
            loss1 = jnp.mean(-fidelity_value)
            loss2 = jnp.mean(log_prob * jax.lax.stop_gradient(-fidelity_value))

        elif goal == "both":
            fidelity_value = jax.vmap(
                lambda rf: fidelity(C_target=C_target, U_final=rf, type=type)
            )(rho_final)
            purity_values = jax.vmap(purity)(rho=rho_final)
            loss1 = jnp.mean(-(fidelity_value + purity_values))
            loss2 = jnp.mean(
                log_prob
                * jax.lax.stop_gradient(-(fidelity_value + purity_values))
            )

        return loss1 + loss2

    key, sub_key = jax.random.split(parent_rng_key)

    best_model_params, iter_idx = train(
        optimizer=optimizer,
        loss_fn=loss_fn,
        trainable_params=trainable_params,
        max_iter=max_iter,
        learning_rate=learning_rate,
        convergence_threshold=convergence_threshold,
        prng_key=key,
    )

    result = evaluate(
        U_0=U_0,
        C_target=C_target,
        parameterized_gates=parameterized_gates,
        measurement_indices=measurement_indices,
        flat_params=flat_params,
        param_shapes=param_shapes,
        best_model_params=best_model_params,
        mode=mode,
        num_time_steps=num_time_steps,
        type=type,
        batch_size=batch_size,
        prng_key=sub_key,
        h_initial_state=h_initial_state,
        rnn_model=rnn_model,
        goal=goal,
        num_iterations=iter_idx,
    )

    return result


def train(
    optimizer: str,  # adam, l-bfgs
    loss_fn,
    trainable_params,
    prng_key,
    max_iter: int,
    learning_rate: float = 0.01,
    convergence_threshold: float = 1e-6,
):
    """
    Train the model using the specified optimizer.
    """
    # Optimization
    # set up optimizer and training state
    if optimizer.upper() == "ADAM":
        best_model_params, iter_idx = _optimize_adam_feedback(
            loss_fn,
            trainable_params,
            max_iter,
            learning_rate,
            convergence_threshold,
            prng_key,
        )
    elif optimizer.upper() == "L-BFGS":
        # TODO: implement L-BFGS for feedback version
        raise NotImplementedError(
            "L-BFGS optimizer is not implemented for feedback version yet."
        )
    else:
        raise ValueError("Invalid optimizer. Choose 'adam' or 'l-bfgs'.")
    return best_model_params, iter_idx


def evaluate(
    U_0,
    C_target,
    parameterized_gates,
    measurement_indices,
    flat_params,
    param_shapes,
    best_model_params,
    mode,
    num_time_steps,
    type,
    batch_size,
    prng_key,
    h_initial_state,
    goal,
    rnn_model,
    num_iterations,
):
    if mode == "nn":
        rho_final, _, returned_params = calculate_trajectory(
            rho_cav=U_0,
            parameterized_gates=parameterized_gates,
            measurement_indices=measurement_indices,
            initial_params=flat_params,
            param_shapes=param_shapes,
            time_steps=num_time_steps,
            rnn_model=rnn_model,
            rnn_params=best_model_params,
            rnn_state=h_initial_state,
            type=type,
            batch_size=batch_size,
            rng_key=prng_key,
        )
    elif mode == "lookup":
        rho_final, _, returned_params = calculate_trajectory(
            rho_cav=U_0,
            parameterized_gates=parameterized_gates,
            measurement_indices=measurement_indices,
            initial_params=flat_params,
            param_shapes=param_shapes,
            time_steps=num_time_steps,
            lut=best_model_params,
            type=type,
            batch_size=batch_size,
            rng_key=prng_key,
        )
    else:
        raise ValueError("Invalid mode. Choose 'nn' or 'lookup'.")

    final_fidelity = None
    final_purity = None

    if goal in ["fidelity", "both"]:
        final_fidelity = jnp.mean(
            jax.vmap(
                lambda rf: fidelity(C_target=C_target, U_final=rf, type=type)
            )(rho_final)
        )

    if goal in ["purity", "both"]:
        final_purity = jnp.mean(jax.vmap(purity)(rho=rho_final))

    if goal not in ["purity", "fidelity", "both"]:
        raise ValueError(
            "Invalid goal. Choose 'purity', 'fidelity', or 'both'."
        )

    return FgResult(
        optimized_rnn_parameters=best_model_params,
        final_purity=final_purity,
        final_fidelity=final_fidelity,
        iterations=num_iterations,
        final_state=rho_final,
        returned_params=returned_params,
    )
