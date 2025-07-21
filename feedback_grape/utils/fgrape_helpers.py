import jax
import numpy as np
import flax.linen as nn
import jax.numpy as jnp
# ruff: noqa N8

jax.config.update("jax_enable_x64", True)


# Answer: add in docs an example of how they can construct their own `Network to use it.`
# --> the example E nn is suitable enough to show how to use it
# Answer: make all these functions private? or just not include them in the docs? --> just not include them in the docs
class RNN(nn.Module):
    hidden_size: int  # number of features in the hidden state
    output_size: int  # number of features in the output ( 2 in the case of gamma and beta)

    @nn.compact
    def __call__(self, measurement, hidden_state):
        gru_cell = nn.GRUCell(features=self.hidden_size)

        if measurement.ndim == 1:
            measurement = measurement.reshape(1, -1)
        new_hidden_state, _ = gru_cell(hidden_state, measurement)
        # this returns the params after linear regression through the hidden state which contains
        # the information of the previous time steps and this is optimized to output best params
        # new_hidden_state = nn.Dense(features=self.hidden_size)(new_hidden_state)
        output = nn.Dense(
            features=self.output_size,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.constant(jnp.pi),
        )(new_hidden_state)
        output = nn.relu(output)
        # output = jnp.asarray(output)
        return output[0], new_hidden_state


def clip_params(params, gate_param_constraints):
    """
    Clip the parameters to be within the specified constraints. if the parameters are within the bounds, they remain unchanged.
    If they are outside the bounds, they are mapped to the bounds using a sigmoid function.

    Args:
        params: Parameters to be clipped.
        param_constraints: List of tuples specifying (min, max) for each parameter.

    Returns:
        Clipped parameters.
    """
    if gate_param_constraints == []:
        return params

    mapped_params = []
    for i, param in enumerate(params):
        min_val, max_val = gate_param_constraints[i]
        within_bounds = (param >= min_val) & (param <= max_val)

        # If within bounds, keep original; otherwise apply sigmoid mapping
        sigmoid_mapped = min_val + (max_val - min_val) * jax.nn.sigmoid(param)
        mapped_param = jnp.where(within_bounds, param, sigmoid_mapped)
        mapped_params.append(mapped_param)

    return jnp.array(mapped_params)


def apply_gate(rho_cav, gate, params, evo_type, gate_param_constraints):
    """
    Apply a gate to the given state. This also clips the parameters
    to be within the specified constraints specified by the user.

    Args:
        rho_cav: Density matrix of the cavity.
        gate: The gate function to apply.
        params: Parameters for the gate.
        evo_type: Evolution type, either "density" or "state".
        gate_param_constraints: Constraints for the parameters.

    Returns:
        tuple: Updated state.
    """
    # For non-measurement gates, apply the gate without measurement
    params = clip_params(params, gate_param_constraints)
    operator = gate(*[params])
    if evo_type == "density":
        rho_meas = operator @ rho_cav @ operator.conj().T
    else:
        rho_meas = operator @ rho_cav
    return rho_meas


def convert_to_index(measurement_history):
    """

    Convert measurement history from [1, -1, ...] to [0, 1, ...] and then to an integer index

    Args:
        measurement_history: List of measurements, where 1 indicates a positive measurement and -1 indicates
                             a negative measurement.
    Returns:
        int: Integer index representing the measurement history for accessing the lut.

    """
    binary_history = jnp.where(jnp.array(measurement_history) == 1, 0, 1)
    # Convert binary list to integer index (e.g., [0,1] -> 1)
    reversed_binary = binary_history[::-1]
    int_index = jnp.sum(
        (2 ** jnp.arange(len(reversed_binary))) * reversed_binary
    )
    return int_index


def extract_from_lut(lut, measurement_history):
    """
    Extract parameters from the lookup table based on the measurement history.

    Args:
        lut: Lookup table for parameters.
        measurement_history: History of measurements.

    Returns:
        Extracted parameters.
    """
    sub_array_idx = len(measurement_history) - 1
    sub_array_param_idx = convert_to_index(measurement_history)
    return jnp.array(lut)[sub_array_idx][sub_array_param_idx]


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
        res.append(jnp.array(flat_params, dtype=jnp.float64))
        shapes.append(jnp.array(flat_params).shape[0])
    return res, shapes


def construct_ragged_row(
    num_of_rows, num_of_columns, param_constraints, init_flat_params, rng_key
):
    """
    Construct a ragged row of parameters for the gates in the lookup table.

    Args:
        num_of_rows: Number of rows in this array which would be a ragged row in the lut before padding.
        num_of_columns: Number of columns of the array (the total number of parameters of the system).
        param_constraints: List of tuples specifying (min, max) for each parameter. If not specfied == [] and then
            the initial flat parameters are used for all rows.
        init_flat_params: Initial flat parameters for the gates.
        rng_key: JAX random key for random parameter initialization.

    Returns:
        One ragged row of the lookup table with the specified number of rows and columns.

    """
    res = []
    if len(param_constraints) == 0:
        for i in range(num_of_rows):
            flattened = jnp.concatenate([arr for arr in init_flat_params])
            res.append(flattened)
        return res
    else:
        for i in range(num_of_rows):
            row = []
            for j in range(num_of_columns):
                rng_key, subkey = jax.random.split(rng_key)
                val = jax.random.uniform(
                    subkey,
                    shape=(),
                    minval=param_constraints[j][0],
                    maxval=param_constraints[j][1],
                )
                row.append(val)
            res.append(jnp.array(row))
        return res


def convert_system_params(system_params):
    """
    Convert system_params format to (initial_params, parameterized_gates, measurement_indices, param_constraints, c_ops, decay_indices) format.

    Args:
        system_params: List of NamedTuples. Either Gate or Decay NamedTuples.

    Returns:
        tuple:
            - initial_params: dict mapping gate names/types to parameter lists
            - parameterized_gates: list of gate functions
            - measurement_indices: list of indices where measurement gates appear
            - param_constraints: list of parameter constraints for each gate
            - c_ops: list of collapse operators for decay gates
            - decay_indices: list of indices where decay gates appear
    """
    initial_params = {}
    parameterized_gates = []
    measurement_indices = []
    param_constraints = []
    c_ops = []
    decay_indices = []

    for i, gate_config in enumerate(system_params):
        if hasattr(gate_config, "c_ops"):
            c_ops.append(gate_config.c_ops)
            decay_indices.append(i)
        else:
            gate_func = gate_config.gate
            if isinstance(gate_config.initial_params, jnp.ndarray):
                # If initial_params is a numpy array, convert it to a list
                params = gate_config.initial_params.tolist()
            else:
                params = gate_config.initial_params
            is_measurement = gate_config.measurement_flag

            # Add gate to parameterized_gates list
            parameterized_gates.append(gate_func)

            # If this is a measurement gate, add its index
            if is_measurement:
                if gate_func.__code__.co_argcount < 2:
                    raise ValueError(
                        "The Positive operator valued measure gate you supplied must have at least two arguments. "
                        "The first argument is the measurement outcome (1, or -1) and the second argument is the list "
                        "of optimizable parameters for the measurement gate."
                    )
                measurement_indices.append(i)

            param_name = f"gate_{i}"

            initial_params[param_name] = params

            # Add parameter constraints if provided
            if gate_config.param_constraints is not None:
                param_constraints.append(gate_config.param_constraints)

            if len(param_constraints) > 0 and (
                len(param_constraints) != len(parameterized_gates)
            ):
                raise TypeError(
                    "If you provide parameter constraints for some gates, you need to provide them for all gates."
                )

    return (
        initial_params,
        parameterized_gates,
        measurement_indices,
        param_constraints,
        c_ops,
        decay_indices,
    )


def get_trainable_parameters_for_no_meas(
    initial_parameters, param_constraints, num_time_steps, rng_key
):
    """

    This function prepares the trainable parameters for the case for which no measurement is
    performed. Meaning this is just for normal gate-parameterized GRAPE optimization.

    User enters the initial parameters and if they want to constrain the parameters
    to be within a certain range, they can specify the `param_constraints` argument.

    if that is not provided the initial parameters are used for all time steps.

    Args:
        initial_parameters: Initial parameters for the gates.
        param_constraints: List of tuples specifying (min, max) for each parameter.
        num_time_steps: Number of time steps for the optimization. (used to infer the dimension of the trainable parameters)
        rng_key: JAX random key for random parameter initialization between specified bounds if param_constraints is provided.

    Returns:
        List of trainable parameters for each time step.


    """
    trainable_params = []
    flat_params, _ = prepare_parameters_from_dict(initial_parameters)
    trainable_params.append(flat_params)
    for i in range(num_time_steps - 1):
        gate_params_list = []
        if param_constraints != []:
            for gate_constraints in param_constraints:
                sampled_params = []
                for var_bounds in gate_constraints:
                    rng_key, subkey = jax.random.split(rng_key)
                    var = jax.random.uniform(
                        subkey,
                        shape=(),
                        minval=var_bounds[0],
                        maxval=var_bounds[1],
                    )
                    sampled_params.append(var)
                gate_params_list.append(jnp.array(sampled_params))
            trainable_params.append(gate_params_list)
        else:
            # if no parameter constraints are provided, we just use the initial parameters
            # for all time steps as initial parameters
            trainable_params.append(flat_params)

    return trainable_params
