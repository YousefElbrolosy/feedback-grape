"""
GRadient Ascent Pulse Engineering (GRAPE) with feedback.
"""

import jax
from enum import Enum
import jax.numpy as jnp
from .utils.solver import mesolve
from typing import List, NamedTuple
from .utils.optimizers import optimize_adam_feedback
from .utils.fidelity import (
    isbra,
    isket,
    fidelity,
    is_positive_semi_definite,
)
from .utils.purity import purity
from .utils.povm import povm
from .utils.fgrape_helpers import (
    get_trainable_parameters_for_no_meas,
    prepare_parameters_from_dict,
    convert_system_params,
    construct_ragged_row,
    extract_from_lut,
    reshape_params,
    apply_gate,
    RNN,
)

# Answer: see if I should replace with pmap for feedback-grape gpu version (may also be a different package)
# Answer: No, both do different things, pmap is for parallelizing over multiple devices, while vmap is for vectorizing over a single device.
# Answer: Pmap should be used by the user if he has a slurm script that runs grape on multiple devices.
# ruff: noqa N8
jax.config.update("jax_enable_x64", True)

"""
NOTE: If you want to optimize complex prameters, you need to divide your complex parameter into two real 
parts and then internaly in your defined function unitaries you need to combine them back to complex numbers.
"""

# TODO: Make sure that a global variable isn't being changed in the process
# Just like what happened with the c_ops.copy

# Answer: see if there is a way to enforce 64 bit over whole repository --> Added to top of each file


# TODO: Would be useful in the documentation to explain to the user the shapes of the outputs
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


class _DEFAULTS(Enum):
    BATCH_SIZE = 1
    EVAL_BATCH_SIZE = 10
    MODE = "lookup"
    RNN = RNN
    RNN_HIDDEN_SIZE = 30
    GOAL = "fidelity"
    DECAY = None
    PROGRESS = False


class Gate(NamedTuple):
    """
    Gate class to store the parameters for each gate.
    """

    gate: callable  # type: ignore
    """
    Function that applies the gate to the state.
    """
    initial_params: list[float]
    """
    Initial parameters for the gate.
    """
    measurement_flag: bool
    """
    Flag indicating if the gate is used for measurement.
    """
    param_constraints: list[float] | None = None
    # TODO: IMPORTANT Is that what florian wanted?
    """
    This constraints the initialization of the parameters to be within the specified range.
    This also constraints the parameters that gets applied to the gates by clipping to your specified range using a 
    sigmoid function.
    """


class Decay(NamedTuple):
    """
    Decay class to store the parameters for each decay.
    """

    c_ops: list[jnp.ndarray]
    """
    Collapse operators for the decay.
    """


def _calculate_time_step(
    *,
    rho_cav,
    parameterized_gates,
    measurement_indices,
    initial_params,
    param_shapes,
    param_constraints,
    c_ops,
    decay_indices,
    rnn_model=None,
    rnn_params=None,
    rnn_state=None,
    lut=None,
    measurement_history=None,
    evo_type,
    time_step_key,
):
    """
    Calculate the time step for the optimization process.
    """
    rho_final = rho_cav
    total_log_prob = 0.0
    applied_params = []

    key = time_step_key

    jump_operators = c_ops.copy()

    decay_count_so_far = 0

    if rnn_model is None and lut is None:
        extracted_params = initial_params
        # Apply each gate in sequence
        for i in range(len(parameterized_gates) + len(decay_indices)):
            # Answer: see what would happen if this is a state --> because it will still output rho
            # Answer: states are now automatically converted to density matrices
            if i in decay_indices:
                decay_count_so_far += 1
                if len(jump_operators) == 0:
                    raise ValueError(
                        "No Corressponding collapse operators for this time step."
                    )
                rho_final = mesolve(
                    jump_ops=jump_operators.pop(0),
                    rho0=rho_final,
                )
            else:
                rho_final = apply_gate(
                    rho_final,
                    parameterized_gates[i - decay_count_so_far],
                    extracted_params[i - decay_count_so_far],
                    evo_type,
                    gate_param_constraints=param_constraints[
                        i - decay_count_so_far
                    ]
                    if param_constraints != []
                    else [],
                )
                applied_params.append(extracted_params[i - decay_count_so_far])
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
        for i in range(len(parameterized_gates) + len(decay_indices)):
            # Answer: see what would happen if this is a state --> because it will still output rho
            # Answer: states are now automatically converted to density matrices
            key, subkey = jax.random.split(key)
            if i in decay_indices:
                decay_count_so_far += 1
                if len(jump_operators) == 0:
                    raise ValueError(
                        "No Corressponding collapse operators for this time step."
                    )
                rho_final = mesolve(
                    jump_ops=jump_operators.pop(0),
                    rho0=rho_final,
                )
            elif i in measurement_indices:
                rho_final, measurement, log_prob = povm(
                    rho_final,
                    parameterized_gates[i - decay_count_so_far],
                    extracted_lut_params[i - decay_count_so_far],
                    gate_param_constraints=param_constraints[
                        i - decay_count_so_far
                    ]
                    if param_constraints != []
                    else [],
                    rng_key=subkey,
                    evo_type=evo_type,
                )
                measurement_history.append(measurement)
                applied_params.append(
                    extracted_lut_params[i - decay_count_so_far]
                )
                extracted_lut_params = extract_from_lut(
                    lut, measurement_history
                )
                extracted_lut_params = reshape_params(
                    param_shapes, extracted_lut_params
                )
                total_log_prob += log_prob
            else:
                rho_final = apply_gate(
                    rho_final,
                    parameterized_gates[i - decay_count_so_far],
                    extracted_lut_params[i - decay_count_so_far],
                    evo_type,
                    gate_param_constraints=param_constraints[
                        i - decay_count_so_far
                    ]
                    if param_constraints != []
                    else [],
                )
                applied_params.append(
                    extracted_lut_params[i - decay_count_so_far]
                )

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
        for i in range(len(parameterized_gates) + len(decay_indices)):
            # Answer: see what would happen if this is a state --> because it will still output rho
            # Answer: states are now automatically converted to density matrices
            key, subkey = jax.random.split(key)
            meas_key, dropout_key = jax.random.split(subkey)
            if i in decay_indices:
                decay_count_so_far += 1
                if len(jump_operators) == 0:
                    raise ValueError(
                        "No Corressponding collapse operators for this time step."
                    )
                rho_final = mesolve(
                    jump_ops=jump_operators.pop(0),
                    rho0=rho_final,
                )
            elif i in measurement_indices:
                rho_final, measurement, log_prob = povm(
                    rho_final,
                    parameterized_gates[i - decay_count_so_far],
                    updated_params[i - decay_count_so_far],
                    gate_param_constraints=param_constraints[
                        i - decay_count_so_far
                    ]
                    if param_constraints != []
                    else [],
                    rng_key=meas_key,
                    evo_type=evo_type,
                )
                applied_params.append(updated_params[i - decay_count_so_far])
                updated_params, new_hidden_state = rnn_model.apply(
                    rnn_params,
                    jnp.array([measurement]),
                    new_hidden_state,
                    rngs={'dropout': dropout_key},
                )

                updated_params = reshape_params(param_shapes, updated_params)
                total_log_prob += log_prob
            else:
                rho_final = apply_gate(
                    rho_final,
                    parameterized_gates[i - decay_count_so_far],
                    updated_params[i - decay_count_so_far],
                    evo_type,
                    gate_param_constraints=param_constraints[
                        i - decay_count_so_far
                    ]
                    if param_constraints != []
                    else [],
                )
                applied_params.append(updated_params[i - decay_count_so_far])

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
    param_constraints,
    c_ops,
    decay_indices,
    time_steps,
    rnn_model=None,
    rnn_params=None,
    rnn_state=None,
    lut=None,
    evo_type,
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
        rnn_model: rnn model for feedback.
        rnn_params: Parameters of the rnn model.
        rnn_state: Initial state of the rnn model.
        evo_type: Type of quantum system representation (e.g., "density").

    Returns:
        Final state, log probability, array of POVM parameters
    """

    # Split rng_key into batch_size keys for independent trajectories
    batch_keys = jax.random.split(rng_key, batch_size)

    def _calculate_single_trajectory(
        batch_key,
    ):
        time_step_keys = jax.random.split(batch_key, time_steps)
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
                    param_constraints=param_constraints,
                    c_ops=c_ops,
                    decay_indices=decay_indices,
                    initial_params=new_params[i],
                    param_shapes=param_shapes,
                    evo_type=evo_type,
                    time_step_key=time_step_keys[i],
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
                    param_constraints=param_constraints,
                    c_ops=c_ops,
                    decay_indices=decay_indices,
                    initial_params=new_params,
                    param_shapes=param_shapes,
                    lut=lut,
                    measurement_history=measurement_history,
                    evo_type=evo_type,
                    time_step_key=time_step_keys[i],
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
                    param_constraints=param_constraints,
                    c_ops=c_ops,
                    decay_indices=decay_indices,
                    initial_params=new_params,
                    param_shapes=param_shapes,
                    rnn_model=rnn_model,
                    rnn_params=rnn_params,
                    rnn_state=new_hidden_state,
                    evo_type=evo_type,
                    time_step_key=time_step_keys[i],
                )

                total_log_prob += log_prob

                resulting_params.append(applied_params)

        return rho_final, total_log_prob, resulting_params

    # Use jax.vmap to vectorize the trajectory calculation for batch_size
    return jax.vmap(
        _calculate_single_trajectory,
    )(batch_keys)


def optimize_pulse(
    U_0: jnp.ndarray,
    C_target: jnp.ndarray,
    system_params: list[Gate],
    num_time_steps: int,
    max_iter: int,
    convergence_threshold: float,
    learning_rate: float,
    evo_type: str,  # state, density (used now mainly for fidelity calculation)
    goal: str = _DEFAULTS.GOAL.value,  # purity, fidelity, both
    batch_size: int = _DEFAULTS.BATCH_SIZE.value,
    eval_batch_size: int = _DEFAULTS.EVAL_BATCH_SIZE.value,
    mode: str = _DEFAULTS.MODE.value,  # nn, lookup
    rnn: callable = _DEFAULTS.RNN.value,  # type: ignore
    rnn_hidden_size: int = _DEFAULTS.RNN_HIDDEN_SIZE.value,
    progress: bool = _DEFAULTS.PROGRESS.value,
) -> FgResult:
    """
    Optimizes pulse parameters for quantum systems based on the specified configuration using ADAM.

    Args:
        U_0: Initial state or density matrix.
        C_target: Target state or density matrix.
        system_params: List of Gate objects containing gate functions, initial parameters, measurement flags, and parameter constraints.
        num_time_steps (int): The number of time steps for the optimization process.
        max_iter (int): The maximum number of iterations for the optimization process.
        convergence_threshold (float): The threshold for convergence to determine when to stop optimization provide None to enforce max iterations.
        learning_rate (float): The learning rate for the optimization algorithm.
        evo_type (str): The evo_type of quantum system representation, such as 'state', 'density'.
        goal (str): The optimization goal, which can be `purity`, `fidelity`, or `both` \n
            - (default: fidelity)
        batch_size (int): The number of trajectories to process in parallel \n
            - (default: 1)
        eval_batch_size (int): The number of trajectories to process in parallel during evaluation \n
            - (default: 10)
        mode (str): The mode of operation, either 'nn' (neural network) or 'lookup' (lookup table) \n
            - (default: lookup)
        rnn (callable): The rnn model to use for the optimization process. Defaults to a predefined rnn class. Only used if mode is 'nn'. \n
            - (default: RNN)
        rnn_hidden_size (int): The hidden size of the rnn model. Only used if mode is 'nn'. (output size is inferred from the number of parameters) \n
            - (default: 30)
        progress: Whether to show progress (cost every 10 iterations) during optimization. (for debugging purposes). This may significantly slow down the optimization process \n
            - (default: False).
    Returns:
        result: Dictionary containing optimized pulse and convergence data.
    """
    if convergence_threshold == None:
        early_stop = False
    else:
        early_stop = True
    if num_time_steps <= 0:
        raise ValueError("Time steps must be greater than 0.")

    if evo_type not in ["state", "density"]:
        raise ValueError("Invalid evo_type. Choose 'state' or 'density'.")

    if U_0 is None:
        raise ValueError("Please provide an initial state U_0.")

    if C_target is None and goal in ["fidelity", "both"]:
        raise ValueError(
            "Please provide a target state C_target for fidelity calculation."
        )

    if isbra(U_0) or isbra(C_target):
        raise TypeError(
            "Please provide initial and target states as kets (column vectors) or density matrices."
        )
    
    if evo_type == "state" and not (isket(U_0) and isket(C_target)):
        raise TypeError(
            "For evo_type='state', please provide initial and target states as kets (column vectors)."
        )

    if evo_type == "density" and (isket(U_0) or isket(C_target)):
        raise TypeError(
            "For evo_type='density', please provide initial and target states as density matrices."
        )


    if (
        evo_type == "density"
        and (
            not is_positive_semi_definite(U_0)
            or not is_positive_semi_definite(C_target)
        )
    ):
        raise TypeError(
            'If evo_type=`density` Your initial and target rhos must be positive semi-definite.'
        )

    if goal in ["purity", "both"] and evo_type == "state":
        raise ValueError(
            "Purity is not defined for evo_type='state'. Please use evo_type='density' for purity calculation."
        )

    (
        initial_params,
        parameterized_gates,
        measurement_indices,
        param_constraints,
        c_ops,
        decay_indices,
    ) = convert_system_params(system_params)

    if (
        evo_type == "state" or (isket(U_0) or isket(C_target))
    ) and decay_indices != []:
        raise ValueError(
            "Decay requires a density matrix representation of your inital and target states because, the solver uses Lindblad equation to evolve the system with dissipation. \n"
            "Please provide U_0 and U_target as density matrices perhaps using `utils.fidelity.ket2dm` and use evo_type='density'."
        )

    parent_rng_key = jax.random.PRNGKey(0)
    train_eval_key, sub_key, rnn_key = jax.random.split(parent_rng_key, 3)
    row_key, no_meas_key = jax.random.split(sub_key)
    trainable_params = None
    param_shapes = None
    num_of_params = len(jax.tree_util.tree_leaves(initial_params))

    if param_constraints != []:
        if (
            len(jax.tree_util.tree_leaves(param_constraints))
            != num_of_params * 2
        ):
            raise TypeError(
                "Please provide upper and lower constraints for each variable in each gate, or don't provide `param_constraints` to use the default."
            )

    if mode == "no-measurement":
        # If no feedback is used, we can just use the initial parameters
        h_initial_state = None
        rnn_model = None
        trainable_params = get_trainable_parameters_for_no_meas(
            initial_params, param_constraints, num_time_steps, no_meas_key
        )
        if not (measurement_indices == [] or measurement_indices is None):
            raise ValueError(
                "You set a measurement flag to true, but no-measurement mode is used. Please set mode to 'nn' or 'lookup'."
            )
    else:
        if measurement_indices == [] or measurement_indices is None:
            raise ValueError(
                "For modes 'nn' and 'lookup', you must provide at least one measurement operator in your system_params. "
            )
        # Convert dictionary parameters to list[list] structure
        flat_params, param_shapes = prepare_parameters_from_dict(
            initial_params
        )

        # Calculate total number of parameters
        if mode == "nn":
            hidden_size = rnn_hidden_size
            output_size = num_of_params

            rnn_model = rnn(hidden_size=hidden_size, output_size=output_size)  # type: ignore

            h_initial_state = jnp.zeros((1, hidden_size))

            # Answer: should this be .zeros? our input is only 1 or -1? Does not matter this is only for initialization parameters,
            # and I tried .ones it did not differ
            dummy_input = jnp.zeros(
                (1, 1)
            )  # Dummy input for rnn initialization
            trainable_params = {
                'rnn_params': rnn_model.init(
                    rnn_key, dummy_input, h_initial_state
                ),
                'initial_params': flat_params,
            }

        elif mode == "lookup":
            h_initial_state = None
            rnn_model = None
            # step 1: initialize the parameters
            num_of_columns = num_of_params
            num_of_sub_lists = len(measurement_indices) * num_time_steps
            F = []
            param_constraints_reshaped = jnp.array(param_constraints).reshape(
                -1, 2
            )

            # construct ragged lookup table
            for i in range(1, num_of_sub_lists + 1):
                row_key, row_sub_key = jax.random.split(row_key)
                F.append(
                    construct_ragged_row(
                        num_of_rows=2**i,
                        num_of_columns=num_of_columns,
                        param_constraints=param_constraints_reshaped,
                        init_flat_params=flat_params,
                        rng_key=row_sub_key,
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
            trainable_params = {
                'lookup_table': F,
                'initial_params': flat_params,
            }
        else:
            raise ValueError(
                "Invalid mode. Choose 'nn' or 'lookup' or 'no-measurement'."
            )

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
            # jax.debug.print("trainable params: {} \n", trainable_params)
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
            param_constraints=param_constraints,
            c_ops=c_ops,
            decay_indices=decay_indices,
            initial_params=initial_params_opt,
            param_shapes=param_shapes,
            time_steps=num_time_steps,
            rnn_model=rnn_model,
            rnn_params=rnn_params,
            rnn_state=h_initial_state,
            lut=lookup_table_params,
            evo_type=evo_type,
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
                lambda rf: fidelity(
                    C_target=C_target, U_final=rf, evo_type=evo_type
                )
            )(rho_final)
            loss1 = jnp.mean(-fidelity_value)
            loss2 = jnp.mean(log_prob * jax.lax.stop_gradient(-fidelity_value))

        elif goal == "both":
            fidelity_value = jax.vmap(
                lambda rf: fidelity(
                    C_target=C_target, U_final=rf, evo_type=evo_type
                )
            )(rho_final)
            purity_values = jax.vmap(purity)(rho=rho_final)
            loss1 = jnp.mean(-(fidelity_value + purity_values))
            loss2 = jnp.mean(
                log_prob
                * jax.lax.stop_gradient(-(fidelity_value + purity_values))
            )

        return loss1 + loss2

    train_key, eval_key = jax.random.split(train_eval_key)

    best_model_params, iter_idx = _train(
        loss_fn=loss_fn,
        trainable_params=trainable_params,
        max_iter=max_iter,
        learning_rate=learning_rate,
        convergence_threshold=convergence_threshold,
        prng_key=train_key,
        progress=progress,
        early_stop=early_stop,
    )

    result = _evaluate(
        U_0=U_0,
        C_target=C_target,
        parameterized_gates=parameterized_gates,
        measurement_indices=measurement_indices,
        param_constraints=param_constraints,
        c_ops=c_ops,
        decay_indices=decay_indices,
        param_shapes=param_shapes,
        best_model_params=best_model_params,
        mode=mode,
        num_time_steps=num_time_steps,
        evo_type=evo_type,
        eval_batch_size=eval_batch_size,
        prng_key=eval_key,
        h_initial_state=h_initial_state,
        rnn_model=rnn_model,
        goal=goal,
        num_iterations=iter_idx,
    )

    return result


def _train(
    loss_fn,
    trainable_params,
    prng_key,
    max_iter,
    learning_rate,
    convergence_threshold,
    progress,
    early_stop,
):
    """
    Train the model using the specified optimizer.
    """
    # Optimization
    # set up optimizer and training state
    best_model_params, iter_idx = optimize_adam_feedback(
        loss_fn,
        trainable_params,
        max_iter,
        learning_rate,
        convergence_threshold,
        prng_key,
        progress,
        early_stop,
    )

    # Due to the complex parameter l-bfgs is very slow and leads to bad results so is omitted

    return best_model_params, iter_idx


def _evaluate(
    U_0,
    C_target,
    parameterized_gates,
    measurement_indices,
    param_shapes,
    param_constraints,
    c_ops,
    decay_indices,
    best_model_params,
    mode,
    num_time_steps,
    evo_type,
    eval_batch_size,
    prng_key,
    h_initial_state,
    goal,
    rnn_model,
    num_iterations,
):
    """
    Evaluate the model using the best parameters found during training.
    """
    if mode == "no-measurement":
        rho_final, _, returned_params = calculate_trajectory(
            rho_cav=U_0,
            parameterized_gates=parameterized_gates,
            measurement_indices=measurement_indices,
            param_constraints=param_constraints,
            c_ops=c_ops,
            decay_indices=decay_indices,
            initial_params=best_model_params,
            param_shapes=param_shapes,
            time_steps=num_time_steps,
            evo_type=evo_type,
            batch_size=eval_batch_size,
            rng_key=prng_key,
        )
    elif mode == "nn":
        rho_final, _, returned_params = calculate_trajectory(
            rho_cav=U_0,
            parameterized_gates=parameterized_gates,
            measurement_indices=measurement_indices,
            param_constraints=param_constraints,
            c_ops=c_ops,
            decay_indices=decay_indices,
            initial_params=best_model_params['initial_params'],
            param_shapes=param_shapes,
            time_steps=num_time_steps,
            rnn_model=rnn_model,
            rnn_params=best_model_params['rnn_params'],
            rnn_state=h_initial_state,
            evo_type=evo_type,
            batch_size=eval_batch_size,
            rng_key=prng_key,
        )
    elif mode == "lookup":
        rho_final, _, returned_params = calculate_trajectory(
            rho_cav=U_0,
            parameterized_gates=parameterized_gates,
            measurement_indices=measurement_indices,
            param_constraints=param_constraints,
            c_ops=c_ops,
            decay_indices=decay_indices,
            initial_params=best_model_params['initial_params'],
            param_shapes=param_shapes,
            time_steps=num_time_steps,
            lut=best_model_params['lookup_table'],
            evo_type=evo_type,
            batch_size=eval_batch_size,
            rng_key=prng_key,
        )
    else:
        raise ValueError(
            "Invalid mode. Choose 'nn' or 'lookup' or 'no-measurement'."
        )

    final_fidelity = None
    final_purity = None

    if goal in ["fidelity", "both"]:
        final_fidelity = jnp.mean(
            jax.vmap(
                lambda rf: fidelity(
                    C_target=C_target, U_final=rf, evo_type=evo_type
                )
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
