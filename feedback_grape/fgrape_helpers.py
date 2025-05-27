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
