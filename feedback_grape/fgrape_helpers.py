import jax
import numpy as np
import flax.linen as nn
import jax.numpy as jnp
# ruff: noqa N8


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


def construct_ragged_row(num_of_rows, num_of_columns, minval, maxval, rng_key):
    res = []
    for i in range(num_of_rows):
        flattened = jax.random.uniform(
            rng_key,
            shape=(num_of_columns,),
            minval=minval,
            maxval=maxval,
            dtype=jnp.float64,
        )
        res.append(flattened)
    return res


def convert_system_params(system_params):
    """
    Convert system_params format to (initial_params, parameterized_gates, measurement_indices) format.
    
    Args:
        system_params: List of dictionaries, each containing:
            - "gate": gate function
            - "initial_params": list of parameters
            - "measurement_flag": boolean indicating if this is a measurement gate
    
    Returns:
        tuple: (initial_params, parameterized_gates, measurement_indices)
            - initial_params: dict mapping gate names/types to parameter lists
            - parameterized_gates: list of gate functions
            - measurement_indices: list of indices where measurement gates appear
    """
    initial_params = {}
    parameterized_gates = []
    measurement_indices = []
    param_constrains = []
    
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
        if "param_constrains" in gate_config:
            param_constrains.append(gate_config.get("param_constrains", None))

        if len(param_constrains) != len(parameterized_gates):
            raise ValueError("If you provide parameter constraints for some gates, you need to provide them for all gates.")
    
    return initial_params, parameterized_gates, measurement_indices, param_constrains


def get_trainable_parameters(initial_parameters, param_constrains, num_time_steps, rng_key):
    trainable_params = []
    flat_params ,_ = prepare_parameters_from_dict(initial_parameters)
    trainable_params.append(flat_params)
    for i in range(num_time_steps - 1):
        gate_params_list = []
        if param_constrains != []:
            for gate_params, gate_constraints in zip(flat_params, param_constrains):
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
        else: # TODO: mention in the docs that if no constraints are given, the parameters are sampled uniformly from -2pi to 2pi
            for gate_params in flat_params:
                sampled_params = []
                for _ in range(gate_params.shape[0]):
                    rng_key, subkey = jax.random.split(rng_key)
                    var = jax.random.uniform(
                        subkey,
                        shape=(),
                        minval=-2 *jnp.pi,
                        maxval=2 * jnp.pi,
                    )
                    sampled_params.append(var)
                gate_params_list.append(jnp.array(sampled_params))
        trainable_params.append(gate_params_list)



    return trainable_params

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

