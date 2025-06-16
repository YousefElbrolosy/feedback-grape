import jax
import jax.numpy as jnp
from typing import List, NamedTuple
from feedback_grape.utils.optimizers import _optimize_adam_feedback
from feedback_grape.utils.solver import mesolve
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

"""
NOTE: If you want to optimize complex prameters, you need to divide your complex parameter into two real 
parts and then internaly in your defined function unitaries you need to combine them back to complex numbers.
"""


class FgResult(NamedTuple):
    """
    result class to store the results of the optimization process.
    """

    optimized_trainable_parameters: jnp.ndarray
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
    returned_params: List[jnp.ndarray]
    """
    Array of finalsPOVM parameters for each time step.
    """
    final_purity: jnp.ndarray | None
    """
    Final purity of the optimized control.
    """
    final_fidelity: jnp.ndarray | None
    """
    Final fidelity of the optimized control.
    """


class decay(NamedTuple):
    """
    decay class to store the decay parameters.
    """

    c_ops: List[jnp.ndarray]
    """
    Collapse operators for the decay process.
    """
    decay_indices: List[int]
    """
    Indices of the gates that are used for decay.
    """
    tsave: List[float]
    """
    Time grid for the decay process.
    """
    Hamiltonian: jnp.ndarray | None = None
    """
    Hamiltonian for the decay process, if applicable.
    """


def _calculate_time_step(
    *,
    rho_cav,
    parameterized_gates,
    measurement_indices,
    initial_params,
    param_shapes,
    decay,
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
    # TODO: throw error based on content of decay indices here
    # if not (decay_indices is None or decay_indices == []):

    # TODO: IMP - See which is the more correct, should new params be propagated
    # directly within the same time step
    # or new parameters are together within the same time step
    key = rng_key
    if decay is not None:
        res, _ = prepare_parameters_from_dict(decay['c_ops'])

    if rnn_model is None and lut is None:
        extracted_params = initial_params
        # Apply each gate in sequence
        for i, gate in enumerate(parameterized_gates):
            # TODO: see what would happen if this is a state --> because it will still output rho
            if decay is not None:
                if i in decay['decay_indices']:
                    if len(res) == 0:
                        raise ValueError(
                            "Decay indices provided, but no corressponding collapse operators found in decay parameters."
                        )
                    rho_final = mesolve(
                        H=decay['Hamiltonian'],
                        jump_ops=res.pop(0),
                        rho0=rho_final,
                        tsave=decay['tsave'],
                    )
            rho_final = apply_gate(
                rho_final, gate, extracted_params[i], type
            )
            applied_params.append(extracted_params[i])
        return (
            rho_final,
            total_log_prob,
            None,
            applied_params,
            None,
        ) 
    elif lut is not None:
        extracted_lut_params = initial_params

        # Apply each gate in sequence
        for i, gate in enumerate(parameterized_gates):
            # TODO: see what would happen if this is a state --> because it will still output rho
            if decay is not None:
                if i in decay['decay_indices']:
                    if len(res) == 0:
                        raise ValueError(
                            "Decay indices provided, but no corressponding collapse operators found in decay parameters."
                        )
                    rho_final = mesolve(
                        H=decay['Hamiltonian'],
                        jump_ops=res.pop(0),
                        rho0=rho_final,
                        tsave=decay['tsave'],
                    )
            key, _ = jax.random.split(key)
            if i in measurement_indices:
                rho_final, measurement, log_prob = povm(
                    rho_final, gate, extracted_lut_params[i], key
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
            # TODO: see what would happen if this is a state --> because it will still output rho
            if decay is not None:
                if i in decay['decay_indices']:
                    if len(res) == 0:
                        raise ValueError(
                            "Decay indices provided, but no corressponding collapse operators found in decay parameters."
                        )
                    rho_final = mesolve(
                        H=decay['Hamiltonian'],
                        jump_ops=res.pop(0),
                        rho0=rho_final,
                        tsave=decay['tsave'],
                    )

            key, subkey = jax.random.split(key)
            if i in measurement_indices:
                rho_final, measurement, log_prob = povm(
                    rho_final, gate, updated_params[i], key
                )
                applied_params.append(updated_params[i])
                updated_params, new_hidden_state = rnn_model.apply(
                    rnn_params,
                    jnp.array([measurement]),
                    new_hidden_state,
                    rngs={'dropout': subkey},
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


def calculate_trajectory(
    *,
    rho_cav,
    parameterized_gates,
    measurement_indices,
    initial_params,
    param_shapes,
    time_steps,
    decay=None,
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
        decay: Decay parameters, if applicable.
        initial_params: Initial parameters for all gates.
        param_shapes: List of shapes for each gate's parameters.
        time_steps: Number of time steps within a trajectory.
        rnn_model: rnn model for feedback.
        rnn_params: Parameters of the rnn model.
        rnn_state: Initial state of the rnn model.
        type: Type of quantum system representation (e.g., "density").

    Returns:
        Final state, log probability, array of POVM parameters
    """

    # Split rng_key into batch_size keys for independent trajectories
    rng_keys = jax.random.split(rng_key, batch_size)

    def _calculate_single_trajectory(
        rng_key,
    ):
        time_step_keys = jax.random.split(rng_key, time_steps)
        resulting_params = []
        rho_final = rho_cav
        total_log_prob = 0.0
        new_params = initial_params
        if rnn_model is None and lut is None:
            for i in range(time_steps):
                (
                    rho_final,
                    _,
                    _,
                    applied_params,
                    _,
                ) = _calculate_time_step(
                    rho_cav=rho_final,
                    parameterized_gates=parameterized_gates,
                    measurement_indices=measurement_indices,
                    decay=decay,
                    initial_params=new_params[i],
                    param_shapes=param_shapes,
                    type=type,
                    rng_key=time_step_keys[i],
                )

                resulting_params.append(applied_params)
        elif lut is not None:
            measurement_history: list[int] = []
            for i in range(time_steps):
                (
                    rho_final,
                    log_prob,
                    new_params,
                    applied_params,
                    measurement_history,
                ) = _calculate_time_step(
                    rho_cav=rho_final,
                    parameterized_gates=parameterized_gates,
                    measurement_indices=measurement_indices,
                    decay=decay,
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
                total_log_prob += log_prob

                resulting_params.append(applied_params)

        else:
            new_hidden_state = rnn_state
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
                    decay=decay,
                    initial_params=new_params,
                    param_shapes=param_shapes,
                    rnn_model=rnn_model,
                    rnn_params=rnn_params,
                    rnn_state=new_hidden_state,
                    type=type,
                    rng_key=time_step_keys[i],
                )

                total_log_prob += log_prob

                resulting_params.append(applied_params)

        return rho_final, total_log_prob, resulting_params

    # Use jax.vmap to vectorize the trajectory calculation for batch_size
    return jax.vmap(
        _calculate_single_trajectory,
    )(rng_keys)


def optimize_pulse_with_feedback(
    U_0: jnp.ndarray,
    C_target: jnp.ndarray,
    parameterized_gates: list[callable],  # type: ignore
    initial_params: dict[str, list[float | complex]],
    num_time_steps: int,
    max_iter: int,
    convergence_threshold: float,
    learning_rate: float,
    type: str,  # unitary, state, density, liouvillian (used now mainly for fidelity calculation)
    goal: str = "fidelity",  # purity, fidelity, both
    batch_size: int = 1,
    eval_batch_size: int = 10,
    mode: str = "lookup",  # nn, lookup
    lookup_min_init_value: float = 0.0,
    lookup_max_init_value: float = jnp.pi,
    measurement_indices: list[int] = [],
    decay: decay | None = None,
    rnn: callable = RNN,  # type: ignore
    rnn_hidden_size: int = 30,
) -> FgResult:
    """
    Optimizes pulse parameters for quantum systems based on the specified configuration using ADAM.

    Args:
        U_0: Initial state or /unitary/density/super operator.
        C_target: Target state or /unitary/density/super operator.
        parameterized_gates (list[callable]): A list of parameterized gate functions to be optimized.
        initial_params (jnp.ndarray): Initial parameters for the parameterized gates.
        num_time_steps (int): The number of time steps for the optimization process.
        max_iter (int): The maximum number of iterations for the optimization process.
        convergence_threshold (float): The threshold for convergence to determine when to stop optimization.
        learning_rate (float): The learning rate for the optimization algorithm.
        type (str): The type of quantum system representation, such as 'unitary', 'state', 'density', or 'liouvillian'.
                    This is primarily used for fidelity calculation.
        goal (str): The optimization goal, which can be 'purity', 'fidelity', or 'both'.
        batch_size (int): The number of trajectories to process in parallel.
        mode (str): The mode of operation, either 'nn' (neural network) or 'lookup' (lookup table).
        measurement_indices (list[int]): Indices of the parameterized gates that are used for measurements.
        decay (decay | None): Decay parameters, if applicable. If None, no decay is applied.
        rnn (callable): The rnn model to use for the optimization process. Defaults to a predefined rnn class. Only used if mode is 'nn'.
        rnn_hidden_size (int): The hidden size of the rnn model. Only used if mode is 'nn'. (output size is inferred from the number of parameters)
    Returns:
        result: Dictionary containing optimized pulse and convergence data.
    """
    if num_time_steps <= 0:
        raise ValueError("Time steps must be greater than 0.")


    parent_rng_key = jax.random.PRNGKey(0)
    key, sub_key = jax.random.split(parent_rng_key)

    trainable_params = None
    param_shapes = None

    if mode == "no-measurement":
        # If no feedback is used, we can just use the initial parameters
        h_initial_state = None
        rnn_model = None
        trainable_params = initial_params
        if not (measurement_indices == [] or measurement_indices is None):
            raise ValueError(
                "You provided a measurement indices, but no feedback is used. Please set mode to 'nn' or 'lookup'."
            )
    else:
        # Convert dictionary parameters to list[list] structure
        flat_params, param_shapes = prepare_parameters_from_dict(initial_params)
        print("parameter shapes:", param_shapes)
        # Calculate total number of parameters
        num_of_params = len(jax.tree_util.tree_leaves(initial_params))
        print("Number of parameters:", num_of_params)
        if mode == "nn":
            hidden_size = rnn_hidden_size
            output_size = num_of_params

            rnn_model = rnn(hidden_size=hidden_size, output_size=output_size)  # type: ignore

            # TODO: should we use some better initialization for the rnn?
            h_initial_state = jnp.zeros((1, hidden_size))

            # TODO: should this be .zeros? our input is only 1 or -1
            dummy_input = jnp.zeros((1, 1))  # Dummy input for rnn initialization
            trainable_params = {
                'rnn_params': rnn_model.init(
                    parent_rng_key, dummy_input, h_initial_state
                ),
                'initial_params': flat_params,
            }
            # trainable_params = rnn_model.init(
            #     parent_rng_key, dummy_input, h_initial_state
            # )
        elif mode == "lookup":
            h_initial_state = None
            rnn_model = None
            # step 1: initialize the parameters
            num_of_columns = num_of_params
            num_of_sub_lists = len(measurement_indices) * num_time_steps
            F = []
            # construct ragged lookup table
            row_key = sub_key
            for i in range(1, num_of_sub_lists + 1):
                row_key, _ = jax.random.split(row_key)
                F.append(
                    construct_ragged_row(
                        num_of_rows=2**i,
                        num_of_columns=num_of_columns,
                        minval=lookup_min_init_value,
                        maxval=lookup_max_init_value,
                        rng_key=row_key,
                    )
                )
            # step 2: pad the arrays to have the same number of rows
            min_num_of_rows = 2 ** len(F)
            for i in range(len(F)):
                if len(F[i]) < min_num_of_rows:
                    zeros_arrays = [
                        jnp.zeros((num_of_columns,), dtype=jnp.float64)
                        for _ in range(min_num_of_rows - len(F[i]))
                    ]
                    F[i] = F[i] + zeros_arrays
            trainable_params = {'lookup_table': F, 'initial_params': flat_params}
        else:
            raise ValueError("Invalid mode. Choose 'nn' or 'lookup' or 'no-measurement'.")

    # TODO: see if we need the log prob term
    # TODO: see if we need to implement stochastic sampling instead
    # QUESTION: should we add an accumilate log-term boolean here that decides whether we add
    # the log prob or not? ( like in porroti's implementation )?
    def loss_fn(trainable_params, rng_key):
        """
        Loss function for the optimization process.
        This function calculates the loss based on the specified goal (purity, fidelity, or both).
        Args:
            rnn_params: Parameters of the rnn model or lookup table.
            rng_key: Random key for stochastic operations.
        Returns:
            Loss value to be minimized.
        """


        if mode == "no-measurement":
            h_initial_state = None
            rnn_params = None
            lookup_table_params = None
            initial_params_opt = trainable_params
        elif mode == "nn":
            # reseting hidden state at end of every trajectory ( does not really change the purity tho)
            h_initial_state = jnp.zeros((1, hidden_size))
            rnn_params = trainable_params['rnn_params']
            initial_params_opt = trainable_params['initial_params']
            lookup_table_params = None
        elif mode == "lookup":
            h_initial_state = None
            rnn_params = None
            lookup_table_params = trainable_params['lookup_table']
            initial_params_opt = trainable_params['initial_params']

        rho_final, log_prob, _ = calculate_trajectory(
            rho_cav=U_0,
            parameterized_gates=parameterized_gates,
            measurement_indices=measurement_indices,
            decay=decay,
            initial_params=initial_params_opt,
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

    train_key, eval_key = jax.random.split(key)

    best_model_params, iter_idx = train(
        loss_fn=loss_fn,
        trainable_params=trainable_params,
        max_iter=max_iter,
        learning_rate=learning_rate,
        convergence_threshold=convergence_threshold,
        prng_key=train_key,
    )

    result = evaluate(
        U_0=U_0,
        C_target=C_target,
        parameterized_gates=parameterized_gates,
        measurement_indices=measurement_indices,
        decay=decay,
        param_shapes=param_shapes,
        best_model_params=best_model_params,
        mode=mode,
        num_time_steps=num_time_steps,
        type=type,
        eval_batch_size=eval_batch_size,
        prng_key=eval_key,
        h_initial_state=h_initial_state,
        rnn_model=rnn_model,
        goal=goal,
        num_iterations=iter_idx,
    )

    return result


def train(
    loss_fn,
    trainable_params,
    prng_key,
    max_iter,
    learning_rate,
    convergence_threshold,
):
    """
    Train the model using the specified optimizer.
    """
    # Optimization
    # set up optimizer and training state
    best_model_params, iter_idx = _optimize_adam_feedback(
        loss_fn,
        trainable_params,
        max_iter,
        learning_rate,
        convergence_threshold,
        prng_key,
    )

    # Due to the complex parameter l-bfgs is very slow and leads to bad results so is omitted

    return best_model_params, iter_idx


def evaluate(
    U_0,
    C_target,
    parameterized_gates,
    measurement_indices,
    decay,
    param_shapes,
    best_model_params,
    mode,
    num_time_steps,
    type,
    eval_batch_size,
    prng_key,
    h_initial_state,
    goal,
    rnn_model,
    num_iterations,
):
    if mode == "no-measurement":
        rho_final, _, returned_params = calculate_trajectory(
            rho_cav=U_0,
            parameterized_gates=parameterized_gates,
            measurement_indices=measurement_indices,
            decay=decay,
            initial_params=best_model_params,
            param_shapes=param_shapes,
            time_steps=num_time_steps,
            type=type,
            batch_size=eval_batch_size,
            rng_key=prng_key,
        )
    elif mode == "nn":
        rho_final, _, returned_params = calculate_trajectory(
            rho_cav=U_0,
            parameterized_gates=parameterized_gates,
            measurement_indices=measurement_indices,
            decay=decay,
            initial_params=best_model_params['initial_params'],
            param_shapes=param_shapes,
            time_steps=num_time_steps,
            rnn_model=rnn_model,
            rnn_params=best_model_params['rnn_params'],
            rnn_state=h_initial_state,
            type=type,
            batch_size=eval_batch_size,
            rng_key=prng_key,
        )
    elif mode == "lookup":
        rho_final, _, returned_params = calculate_trajectory(
            rho_cav=U_0,
            parameterized_gates=parameterized_gates,
            measurement_indices=measurement_indices,
            decay=decay,
            initial_params=best_model_params['initial_params'],
            param_shapes=param_shapes,
            time_steps=num_time_steps,
            lut=best_model_params['lookup_table'],
            type=type,
            batch_size=eval_batch_size,
            rng_key=prng_key,
        )
    else:
        raise ValueError("Invalid mode. Choose 'nn' or 'lookup' or 'no-measurement'.")

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
        optimized_trainable_parameters=best_model_params,
        final_purity=final_purity,
        final_fidelity=final_fidelity,
        iterations=num_iterations,
        final_state=rho_final,
        returned_params=returned_params,
    )
