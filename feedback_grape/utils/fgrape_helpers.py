import jax
import numpy as np
from enum import Enum
import flax.linen as nn
import jax.numpy as jnp
# ruff: noqa N8

jax.config.update("jax_enable_x64", True)


# Answer: add in docs an example of how they can construct their own `Network to use it.`
# --> the example E nn is suitable enough to show how to use it
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
        gru_cell = nn.GRUCell(features=self.hidden_size)

        if measurement.ndim == 1:
            measurement = measurement.reshape(1, -1)
        new_hidden_state, _ = gru_cell(hidden_state, measurement)
        # this returns the povm_params after linear regression through the hidden state which contains
        # the information of the previous time steps and this is optimized to output best povm_params
        # new_hidden_state = nn.Dense(features=self.hidden_size)(new_hidden_state)
        output = nn.Dense(
            features=self.output_size,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.constant(jnp.pi),
        )(new_hidden_state)
        output = nn.relu(output)
        # output = jnp.asarray(output)
        return output[0], new_hidden_state


class DEFAULTS(Enum):
    BATCH_SIZE = 1
    EVAL_BATCH_SIZE = 10
    MODE = "lookup"
    RNN = RNN
    RNN_HIDDEN_SIZE = 30
    GOAL = "fidelity"
    DECAY = None


def clip_params(params, gate_param_constraints):
    """
    Clip the parameters to be within the specified constraints.

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
        param_constraints: Constraints for the parameters.
        gate_index: Index of the gate in the system.

    Returns:
        tuple: Updated state, measurement result (or None), log probability (or 0.0).
    """
    # For non-measurement gates, apply the gate without measurement
    params = clip_params(params, gate_param_constraints)
    operator = gate(*params)
    if evo_type == "density":
        rho_meas = operator @ rho_cav @ operator.conj().T
    else:
        rho_meas = operator @ rho_cav
    return rho_meas


def convert_to_index(measurement_history):
    # Convert measurement history from [1, -1, ...] to [0, 1, ...] and then to an integer index
    binary_history = jnp.where(jnp.array(measurement_history) == 1, 0, 1)
    # Convert binary list to integer index (e.g., [0,1] -> 1)
    reversed_binary = binary_history[::-1]
    int_index = jnp.sum(
        (2 ** jnp.arange(len(reversed_binary))) * reversed_binary
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
        res.append(jnp.array(flat_params))
        shapes.append(jnp.array(flat_params).shape[0])
    return res, shapes


def construct_ragged_row(
    num_of_rows, num_of_columns, param_constraints, init_flat_params, rng_key
):
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
    Convert system_params format to (initial_params, parameterized_gates, measurement_indices) format.

    Args:
        system_params: List of dictionaries, each containing:
            - "gate": gate function
            - "initial_params": list of parameters
            - "measurement_flag": boolean indicating if this is a measurement gate
            - "param_constraints": optional list of parameter constraints (min, max) for each gate

    Returns:
        tuple: (initial_params, parameterized_gates, measurement_indices)
            - initial_params: dict mapping gate names/types to parameter lists
            - parameterized_gates: list of gate functions
            - measurement_indices: list of indices where measurement gates appear
            - param_constraints: list of parameter constraints for each gate
    """
    initial_params = {}
    parameterized_gates = []
    measurement_indices = []
    param_constraints = []

    for i, gate_config in enumerate(system_params):
        gate_func = gate_config["gate"]
        params = gate_config["initial_params"]
        is_measurement = gate_config["measurement_flag"]

        # Add gate to parameterized_gates list
        parameterized_gates.append(gate_func)

        # If this is a measurement gate, add its index
        if is_measurement:
            measurement_indices.append(i)

        param_name = f"gate_{i}"

        initial_params[param_name] = params

        # Add parameter constraints if provided
        if "param_constraints" in gate_config:
            param_constraints.append(
                gate_config.get("param_constraints", None)
            )

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
    )


def get_trainable_parameters_for_no_meas(
    initial_parameters, param_constraints, num_time_steps, rng_key
):
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
        else:  # TODO: explain those differences in the docs
            # if no parameter constraints are provided, we just use the initial parameters
            # for all time steps as initial parameters
            trainable_params.append(flat_params)

    return trainable_params
