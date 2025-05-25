from typing import List, NamedTuple
import jax.numpy as jnp
import numpy as np
from feedback_grape.grape import _optimize_adam, _optimize_L_BFGS
import jax
import flax.linen as nn
from feedback_grape.utils.fidelity import fidelity
from feedback_grape.utils.purity import purity
from feedback_grape.utils.povm import povm

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
    arr_of_povm_params: List[List]
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


def apply_gate(rho_cav, gate, params, type):
    """
    Apply a gate to the given state, with measurement if needed.

    Args:
        rho_cav: Density matrix of the cavity.
        gate: The gate function to apply.
        params: Parameters for the gate.

    Returns:
        tuple: Updated state, measurement result (or None), log probability (or 0.0).
    """
    # For non-measurement gates, apply the gate without measurement
    operator = gate(*params)
    if type == "density":
        rho_meas = operator @ rho_cav @ operator.conj().T
    else:
        rho_meas = operator @ rho_cav
    return rho_meas


def convert_to_index(measurement_history):
    # Convert measurement history from [1, -1, ...] to [0, 1, ...] and then to an integer index
    binary_history = jnp.where(jnp.array(measurement_history) == 1, 0, 1)
    # Convert binary list to integer index (e.g., [0,1] -> 1)
    reversed_binary = binary_history[::-1]
    int_index = sum(
        (2**i) * reversed_binary[i] for i in range(len(reversed_binary))
    )
    return int_index

@jax.jit
def extract_from_lut(lut, measurement_history):
    """
    Extract parameters from the lookup table based on the measurement history.

    Args:
        lut: Lookup table for parameters.
        measurement_history: History of measurements.
        time_step: Current time step.

    Returns:
        Extracted parameters.
    """
    sub_array_idx = len(measurement_history) - 1
    sub_array_param_idx = convert_to_index(measurement_history)
    return jnp.array(lut)[sub_array_idx][sub_array_param_idx]


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
                    rho_final, gate, extracted_lut_params[i]
                )
                measurement_history.append(measurement)
                extracted_lut_params = extract_from_lut(
                    lut, measurement_history
                )
                extracted_lut_params = reshape_params(
                    param_shapes, extracted_lut_params
                )
                total_log_prob += log_prob
            else:
                rho_final = apply_gate(rho_final, gate, extracted_lut_params[i], type)

        return (
            rho_final,
            total_log_prob,
            extracted_lut_params,
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
                    rho_final, gate, updated_params[i]
                )
                updated_params, new_hidden_state = rnn_model.apply(
                    rnn_params, jnp.array([measurement]), new_hidden_state
                )
                updated_params = reshape_params(param_shapes, updated_params)
                total_log_prob += log_prob
            else:
                rho_final = apply_gate(
                    rho_final, gate, updated_params[i], type
                )

        return rho_final, total_log_prob, updated_params, new_hidden_state

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


def reshape_params(param_shapes, rnn_flattened_params):
    """
    Reshape the parameters for the gates.
    """
    # Reshape the flattened parameters from RNN output according
    # to each gate corressponding params
    reshaped_params = []
    param_idx = 0
    for shape in param_shapes:
        num_params = int(np.prod(shape))
        # rnn outputs a flat list, this takes each and assigns according to the shape
        gate_params = rnn_flattened_params[
            param_idx : param_idx + num_params
        ].reshape(shape)
        reshaped_params.append(gate_params)
        param_idx += num_params

    new_params = reshaped_params
    return new_params


def _calculate_trajectory(
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
    rho_final = rho_cav
    # TODO: fix arr_povm_params behavior now that we use eager initialization
    arr_of_povm_params = [initial_params]
    new_params = initial_params
    new_hidden_state = rnn_state
    measurement_history = []
    total_log_prob = 0.0

    if lut is not None:
        for i in range(time_steps):
            rho_final, log_prob, new_params, measurement_history = (
                _calculate_time_step(
                    rho_cav=rho_final,
                    parameterized_gates=parameterized_gates,
                    measurement_indices=measurement_indices,
                    initial_params=new_params,
                    param_shapes=param_shapes,
                    lut=lut,
                    measurement_history=measurement_history,
                    type=type,
                )
            )
            # Thus, during - Refer to Eq(3) in fgrape paper
            # the individual time-evolution trajectory, this term may
            # be easily accumulated step by step, since the conditional
            # probabilities are known (these are just the POVM mea-
            # surement probabilities)
            total_log_prob += log_prob

            if i < time_steps - 1:
                arr_of_povm_params.append(new_params)
        return rho_final, total_log_prob, arr_of_povm_params

    else:
        for i in range(time_steps):
            rho_final, log_prob, new_params, new_hidden_state = (
                _calculate_time_step(
                    rho_cav=rho_final,
                    parameterized_gates=parameterized_gates,
                    measurement_indices=measurement_indices,
                    initial_params=new_params,
                    param_shapes=param_shapes,
                    rnn_model=rnn_model,
                    rnn_params=rnn_params,
                    rnn_state=new_hidden_state,
                    type=type,
                )
            )
            # Thus, during - Refer to Eq(3) in fgrape paper
            # the individual time-evolution trajectory, this term may
            # be easily accumulated step by step, since the conditional
            # probabilities are known (these are just the POVM mea-
            # surement probabilities)
            total_log_prob += log_prob

            if i < time_steps - 1:
                arr_of_povm_params.append(new_params)
        return rho_final, total_log_prob, arr_of_povm_params


# TODO: figure out why using jnp.array leads to lower fidelity
def prepare_parameters_from_dict(params_dict):
    """
    Convert a nested dictionary of parameters to a flat list and record shapes.

    Args:
        params_dict: Nested dictionary of parameters.

    Returns:
        tuple: Flattened parameters list and list of shapes.
    """
    res = []
    shapes = []
    for value in params_dict.values():
        flat_params = jax.tree_util.tree_leaves(value)
        res.append(jnp.array(flat_params))
        shapes.append(jnp.array(flat_params).shape[0])
    return res, shapes


def construct_ragged_row(num_of_rows, num_of_columns, param_shapes):
    res = []
    for i in range(num_of_rows):
        flattened = jax.random.uniform(
            jax.random.PRNGKey(0 + i),
            shape=(num_of_columns,),
            minval=-jnp.pi,
            maxval=jnp.pi,
        )
        res.append(flattened)
    return res


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
    propcomp: str = "time-efficient",  # time-efficient, memory-efficient
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
        propcomp (str): The method for propagator computation, either 'time-efficient' or 'memory-efficient'.
                        This determines how the forward evolution is computed.
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

    if mode == "nn":
        hidden_size = 32
        batch_size = 1
        output_size = num_of_params

        rnn_model = RNN(hidden_size=hidden_size, output_size=output_size)
        h_initial_state = jnp.zeros((batch_size, hidden_size))

        dummy_input = jnp.zeros((1, 1))  # Dummy input for RNN initialization
        trainable_params = rnn_model.init(
            jax.random.PRNGKey(0), dummy_input, h_initial_state
        )
        if goal == "purity":

            def loss_fn(rnn_params):
                """
                Loss function for purity optimization.
                Returns negative purity (we want to minimize this).
                """

                # reseting hidden state at end of every trajectory ( does not really change the purity tho)
                h_initial_state = jnp.zeros((batch_size, hidden_size))

                rho_final, log_prob, _ = _calculate_trajectory(
                    rho_cav=U_0,
                    parameterized_gates=parameterized_gates,
                    measurement_indices=measurement_indices,
                    initial_params=flat_params,
                    param_shapes=param_shapes,
                    time_steps=num_time_steps,
                    rnn_model=rnn_model,
                    rnn_params=rnn_params,
                    rnn_state=h_initial_state,
                    type=type,
                )
                # TODO: see if we need the log prob term
                # TODO: see if we need to implement stochastic sampling instead
                # QUESTION: should we add an accumilate log-term boolean here that decides whether we add
                # the log prob or not? ( like in porroti's implementation )?
                purity_value = purity(rho=rho_final)
                loss1 = -purity_value
                loss2 = log_prob * jax.lax.stop_gradient(loss1)
                return loss1 + loss2

        elif goal == "fidelity":

            def loss_fn(rnn_params):
                """
                Loss function for purity optimization.
                Returns negative purity (we want to minimize this).
                """

                # reseting hidden state at end of every trajectory ( does not really change the purity tho)
                h_initial_state = jnp.zeros((batch_size, hidden_size))

                rho_final, log_prob, _ = _calculate_trajectory(
                    rho_cav=U_0,
                    parameterized_gates=parameterized_gates,
                    measurement_indices=measurement_indices,
                    initial_params=flat_params,
                    param_shapes=param_shapes,
                    time_steps=num_time_steps,
                    rnn_model=rnn_model,
                    rnn_params=rnn_params,
                    rnn_state=h_initial_state,
                    type=type,
                )
                fidelity_value = fidelity(
                    C_target=C_target, U_final=rho_final, type=type
                )
                loss1 = -fidelity_value
                loss2 = log_prob * jax.lax.stop_gradient(loss1)
                return loss1 + loss2

        elif goal == "both":

            def loss_fn(rnn_params):
                """
                Loss function for purity optimization.
                Returns negative purity (we want to minimize this).
                """

                # reseting hidden state at end of every trajectory ( does not really change the purity tho)
                h_initial_state = jnp.zeros((batch_size, hidden_size))

                rho_final, log_prob, _ = _calculate_trajectory(
                    rho_cav=U_0,
                    parameterized_gates=parameterized_gates,
                    measurement_indices=measurement_indices,
                    initial_params=flat_params,
                    param_shapes=param_shapes,
                    time_steps=num_time_steps,
                    rnn_model=rnn_model,
                    rnn_params=rnn_params,
                    rnn_state=h_initial_state,
                    type=type,
                )
                fidelity_value = fidelity(
                    C_target=C_target, U_final=rho_final, type=type
                )
                purity_value = purity(rho=rho_final)
                loss1 = -(fidelity_value + purity_value)
                loss2 = log_prob * jax.lax.stop_gradient(loss1)
                return loss1 + loss2

        else:
            raise ValueError(
                "Invalid goal. Choose 'purity', 'fidelity', or 'both'."
            )

    elif mode == "lookup":
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
                        jnp.zeros((num_of_columns,), dtype=jnp.float32)
                    for _ in range(min_num_of_rows - len(F[i]))
                ]
                F[i] = F[i] + zeros_arrays
        trainable_params = F
        if goal == "purity":

            def loss_fn(lookup_table_params):
                """
                loss function
                """
                rho_final, log_prob, _ = _calculate_trajectory(
                    rho_cav=U_0,
                    parameterized_gates=parameterized_gates,
                    measurement_indices=measurement_indices,
                    initial_params=flat_params,
                    param_shapes=param_shapes,
                    time_steps=num_time_steps,
                    lut=lookup_table_params,
                    type=type,
                )
                purity_value = purity(rho=rho_final)
                loss1 = -purity_value
                loss2 = log_prob * jax.lax.stop_gradient(loss1)
                return loss1 + loss2

        elif goal == "fidelity":

            def loss_fn(lookup_table_params):
                """
                loss function
                """
                rho_final, log_prob, _ = _calculate_trajectory(
                    rho_cav=U_0,
                    parameterized_gates=parameterized_gates,
                    measurement_indices=measurement_indices,
                    initial_params=flat_params,
                    param_shapes=param_shapes,
                    time_steps=num_time_steps,
                    lut=lookup_table_params,
                    type=type,
                )
                fidelity_value = fidelity(
                    C_target=C_target, U_final=rho_final, type=type
                )
                loss1 = -fidelity_value
                loss2 = log_prob * jax.lax.stop_gradient(loss1)
                return loss1 + loss2
        elif goal == "both":

            def loss_fn(lookup_table_params):
                """
                loss function
                """
                rho_final, log_prob, _ = _calculate_trajectory(
                    rho_cav=U_0,
                    parameterized_gates=parameterized_gates,
                    measurement_indices=measurement_indices,
                    initial_params=flat_params,
                    param_shapes=param_shapes,
                    time_steps=num_time_steps,
                    lut=lookup_table_params,
                    type=type,
                )
                fidelity_value = fidelity(
                    C_target=C_target, U_final=rho_final, type=type
                )
                purity_value = purity(rho=rho_final)
                loss1 = -(fidelity_value + purity_value)
                loss2 = log_prob * jax.lax.stop_gradient(loss1)
                return loss1 + loss2
        else:
            raise ValueError(
                "Invalid goal. Choose 'purity', 'fidelity', or 'both'."
            )

    else:
        raise ValueError("Invalid mode. Choose 'nn' or 'lookup'.")

    # Optimization
    # set up optimizer and training state
    if optimizer.upper() == "ADAM":
        best_model_params, iter_idx = _optimize_adam(
            loss_fn,
            trainable_params,
            max_iter,
            learning_rate,
            convergence_threshold,
        )

    elif optimizer.upper() == "L-BFGS":
        best_model_params, iter_idx = _optimize_L_BFGS(
            loss_fn,
            trainable_params,
            max_iter,
            learning_rate,
            convergence_threshold,
        )
    else:
        raise ValueError("Invalid optimizer. Choose 'adam' or 'l-bfgs'.")
    final_fidelity = None
    final_purity = None
    # Calculate final state and purity
    if mode == "nn":
        rho_final, _, arr_of_povm_params = _calculate_trajectory(
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
        )
    elif mode == "lookup":
        rho_final, _, arr_of_povm_params = _calculate_trajectory(
            rho_cav=U_0,
            parameterized_gates=parameterized_gates,
            measurement_indices=measurement_indices,
            initial_params=flat_params,
            param_shapes=param_shapes,
            time_steps=num_time_steps,
            lut=best_model_params,
            type=type,
        )
    else:
        raise ValueError("Invalid mode. Choose 'nn' or 'lookup'.")
    if goal == "fidelity":
        final_fidelity = fidelity(
            C_target=C_target,
            U_final=rho_final,
            type=type,
        )

    elif goal == "purity":
        final_purity = purity(rho=rho_final)

    elif goal == "both":
        final_fidelity = fidelity(
            C_target=C_target,
            U_final=rho_final,
            type=type,
        )
        final_purity = purity(rho=rho_final)
    else:
        raise ValueError(
            "Invalid goal. Choose 'purity', 'fidelity', or 'both'."
        )

    # Answer: know why even, does it have array types in the all but first entries?
    # --> Because the first entry is the initial params, given by user, but the rest are
    # are generated by the RNN which outputs a jnp array, and cannot change them in the middle
    # because this leads to doing actions on tracer values
    # Ensure every param in arr_of_povm_params is a list
    arr_of_povm_params = [
        [p if isinstance(p, list) else p.tolist() for p in params]
        for params in arr_of_povm_params
    ]

    return FgResult(
        optimized_rnn_parameters=best_model_params,
        final_purity=final_purity,
        final_fidelity=final_fidelity,
        iterations=iter_idx,
        final_state=rho_final,
        arr_of_povm_params=arr_of_povm_params,
    )


# RNN
class GRUCell(nn.Module):
    features: int

    @nn.compact
    def __call__(self, hidden_state, x_input):
        r"""
        The mathematical definition of the cell is as follows

        .. math::

            \begin{array}{ll}
            r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
            z = \sigma(W_{iz} x + b_{iz} + W_{hz} h) + b_{hz} \\
            h~ = \tanh(W_{ih~} x + b_{i~} + r * (W_{h~} h + b_{h~})) \\
            h_new = (1 - z) * h~ + z * h \\
            \end{array}
        """
        # Dense is just a linear layer w x + b ( and it does this for input and for the hidden state)
        # r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        r = nn.sigmoid(
            nn.Dense(features=self.features, name='reset_gate')(
                jnp.concatenate([x_input, hidden_state], axis=-1)
            )
        )
        # z = \sigma(W_{iz} x + b_{iz} + W_{hz} h) + b_{hz} \\
        z = nn.sigmoid(
            nn.Dense(features=self.features, name='update_gate')(
                jnp.concatenate([x_input, hidden_state], axis=-1)
            )
        )
        # n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h_telda = nn.tanh(
            nn.Dense(features=self.features, name='candidate_gate')(
                jnp.concatenate([x_input, r * hidden_state], axis=-1)
            )
        )
        # note that this h_new, just tells us how much we should update the hidden state according to the update gate to the new candidate
        h_new = (1 - z) * h_telda + z * hidden_state
        return h_new, h_new


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
        gru_cell = GRUCell(features=self.hidden_size)

        if measurement.ndim == 1:
            measurement = measurement.reshape(1, -1)
        new_hidden_state, _ = gru_cell(hidden_state, measurement)
        # this returns the povm_params after linear regression through the hidden state which contains
        # the information of the previous time steps and this is optimized to output best povm_params
        output = nn.Dense(features=self.output_size)(new_hidden_state)
        # output = jnp.asarray(output)
        return output[0], new_hidden_state
