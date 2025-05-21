from typing import NamedTuple, List, Dict, Any, Tuple
import jax.numpy as jnp
from feedback_grape.grape import _optimize_adam, _optimize_L_BFGS
import jax
import flax.linen as nn
import numpy as np
# ruff: noqa N8
# TODO: need to fix problem where dict leaves are reversed

class FgResultPurity(NamedTuple):
    """
    result class to store the results of the optimization process.
    """

    optimized_rnn_parameters: jnp.ndarray
    """
    Optimized control amplitudes.
    """
    final_purity: float
    """
    Final fidelity of the optimized control.
    """
    iterations: int
    """
    Number of iterations taken for optimization.
    """
    final_state: jnp.ndarray
    """
    Final operator after applying the optimized control amplitudes.
    """
    arr_of_povm_params: List[jnp.ndarray]


def _probability_of_a_measurement_outcome_given_a_certain_state(
    rho_cav, measurement_outcome, povm_measure_operator, params
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
        measurement_outcome, params
    ).conj().T @ povm_measure_operator(measurement_outcome, params)
    return jnp.real(jnp.trace(Em @ rho_cav))


def _post_measurement_state(
    rho_cav, measurement_outcome, povm_measure_operator, params
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
    Mm_op = povm_measure_operator(measurement_outcome, params)
    prob = _probability_of_a_measurement_outcome_given_a_certain_state(
        rho_cav,
        measurement_outcome,
        povm_measure_operator,
        params,
    )
    prob = jnp.maximum(prob, 1e-10)
    return Mm_op @ rho_cav @ Mm_op.conj().T / prob


def povm(
    rho_cav: jnp.ndarray,
    povm_measure_operator: callable,
    params: jnp.ndarray,
) -> tuple[jnp.ndarray, int, float]:
    """
    Perform a POVM measurement on the given state.

    Args:
        rho_cav (jnp.ndarray): The density matrix of the cavity.
        povm_measure_operator (callable): The POVM measurement operator.
        params (jnp.ndarray): Parameters for the POVM measurement operator.

    Returns:
        tuple: A tuple containing the post-measurement state, the measurement result, and the log probability of the measurement outcome.
    """
    prob_plus = _probability_of_a_measurement_outcome_given_a_certain_state(
        rho_cav, 1, povm_measure_operator, params
    )
    random_value = jax.random.uniform(jax.random.PRNGKey(0), shape=())
    measurement = jnp.where(random_value < prob_plus, 1, -1)
    rho_meas = _post_measurement_state(
        rho_cav, measurement, povm_measure_operator, params
    )
    prob = jnp.where(
        measurement == 1,
        prob_plus,
        1 - prob_plus,
    )
    log_prob = jnp.log(jnp.maximum(prob, 1e-10))
    return rho_meas, measurement, log_prob


def purity(*, rho):
    """
    Computes the purity of a density matrix.

    Args:
        rho: Density matrix.
    Returns:
        purity: Purity value.
    """
    return jnp.real(jnp.trace(rho @ rho))


def apply_gate(rho_cav, gate, params, gate_idx, measurement_indices):
    """
    Apply a gate to the given state, with measurement if needed.

    Args:
        rho_cav: Density matrix of the cavity.
        gate: The gate function to apply.
        params: Parameters for the gate.
        gate_idx: Index of the gate in the list of gates.
        measurement_indices: Indices of gates that should perform measurements.

    Returns:
        tuple: Updated state, measurement result (or None), log probability (or 0.0).
    """
    # Check if this gate should perform a measurement
    if gate_idx in measurement_indices:
        rho_meas, measurement, log_prob = povm(rho_cav, gate, params)
        return rho_meas, measurement, log_prob
    
    # For non-measurement gates, apply the gate without measurement
    # (This would need to be implemented based on your gate semantics)
    # For now, assuming gates return an operator that acts on the state:
    operator = gate(params)
    rho_meas = operator @ rho_cav @ operator.conj().T
    return rho_meas, None, 0.0


def _calculate_time_step(
    *,
    rho_cav,
    parameterized_gates,
    measurement_indices,
    all_params,
    rnn_model,
    rnn_params,
    rnn_state,
):
    """
    Calculate the time step for the optimization process.

    Args:
        rho_cav: Density matrix of the cavity.
        parameterized_gates: List of parameterized gates.
        measurement_indices: Indices of gates used for measurements.
        all_params: Flattened array of all gate parameters.
        rnn_model: RNN model for feedback.
        rnn_params: Parameters of the RNN model.
        rnn_state: State of the RNN model.

    Returns:
        tuple: Updated state, log probability, updated parameters, new RNN state.
    """
    # Split the flattened parameters into chunks for each gate
    param_lists = [all_params[i] for i in range(len(all_params))]
    
    rho_final = rho_cav
    total_log_prob = 0.0
    measurement_results = []
    
    # Apply each gate in sequence
    for i, gate in enumerate(parameterized_gates):
        gate_params = param_lists[i]
        rho_final, measurement, log_prob = apply_gate(
            rho_final, gate, gate_params, i, measurement_indices
        )
        total_log_prob += log_prob
        if measurement is not None:
            measurement_results.append(measurement)
    
    # If no measurements were made, use a dummy value
    if not measurement_results:
        measurement_results = [0]
    
    # Get updated parameters from RNN based on measurements
    measurement_array = jnp.array(measurement_results)
    updated_params, new_hidden_state = rnn_model.apply(
        rnn_params, measurement_array, rnn_state
    )
    
    # The RNN now outputs all parameters in a flat array
    # We'll reshape them during calculate_trajectory
    
    return rho_final, total_log_prob, updated_params, new_hidden_state


def calculate_trajectory(
    *,
    rho_cav,
    parameterized_gates,
    measurement_indices,
    initial_params,
    time_steps,
    rnn_model,
    rnn_params,
    rnn_state,
    param_shapes,
):
    """
    Calculate a complete quantum trajectory with feedback.

    Args:
        rho_cav: Initial density matrix of the cavity.
        parameterized_gates: List of parameterized gates.
        measurement_indices: Indices of gates used for measurements.
        initial_params: Initial parameters for all gates.
        time_steps: Number of time steps within a trajectory.
        rnn_model: RNN model for feedback.
        rnn_params: Parameters of the RNN model.
        rnn_state: Initial state of the RNN model.
        param_shapes: List of shapes for each gate's parameters.

    Returns:
        Final state, log probability, array of parameter histories
    """
    rho_final = rho_cav
    all_params_history = [initial_params[0]]
    current_params = initial_params
    current_hidden_state = rnn_state
    total_log_prob = 0.0
    
    for i in range(time_steps):
        rho_final, log_prob, new_params, current_hidden_state = _calculate_time_step(
            rho_cav=rho_final,
            parameterized_gates=parameterized_gates,
            measurement_indices=measurement_indices,
            all_params=current_params,
            rnn_model=rnn_model,
            rnn_params=rnn_params,
            rnn_state=current_hidden_state,
        )
        
        total_log_prob += log_prob


        # Reshape the flattened parameters from RNN output according
        # to each gate corressponding params
        reshaped_params = []
        param_idx = 0
        for shape in param_shapes:
            num_params = int(np.prod(shape))
            # rnn outputs a flat list, this takes each and assigns according to the shape
            gate_params = new_params[param_idx:param_idx + num_params].reshape(shape)
            reshaped_params.append(gate_params)
            param_idx += num_params
            
        current_params = reshaped_params
        
        if i < time_steps - 1:
            all_params_history.extend(current_params)
    
    return rho_final, total_log_prob, all_params_history

def prepare_parameters_from_dict(params_dict):
    """
    Convert a nested dictionary of parameters to a flat list and record shapes.
    
    Args:
        params_dict: Nested dictionary of parameters.
        
    Returns:
        tuple: Flattened parameters list and list of shapes.
    """
    flat_params = []
    param_shapes = []
    
    def process_dict(d):
        result = []
        for key, value in d.items(): 
            if isinstance(value, dict):
                result.extend(process_dict(value))
            else:
                result.append(value)
        return result
    
    # Process each top-level gate
    for gate_name, gate_params in params_dict.items():
        if isinstance(gate_params, dict):
            # Extract parameters for this gate
            gate_flat_params = process_dict(gate_params)
        else:
            # If already a flat array
            gate_flat_params = gate_params
        if not (isinstance(gate_flat_params, list)):
            param_shapes.append(1)
        else:
            flat_params.append(gate_flat_params)
            param_shapes.append(len(gate_flat_params))
    
    return flat_params, param_shapes


def optimize_pulse_with_feedback(
    U_0: jnp.ndarray,
    C_target: jnp.ndarray,
    parameterized_gates: List[callable],
    measurement_indices: List[int],
    initial_params: Dict[str, Any],
    goal: str,  # purity, fidelity, both
    mode: str,  # nn, lookup
    num_time_steps: int,
    optimizer: str,  # adam, l-bfgs
    max_iter: int,
    convergence_threshold: float,
    learning_rate: float,
    type: str,  # unitary, state, density, superoperator
    propcomp: str = "time-efficient",
) -> FgResultPurity:
    """
    Optimizes pulse parameters for quantum systems based on the specified configuration.

    Args:
        U_0: Initial state or /unitary/density/super operator.
        C_target: Target state or /unitary/density/super operator.
        parameterized_gates: A list of parameterized gate functions to be optimized.
        measurement_indices: Indices of the parameterized gates that are used for measurements.
        initial_params: Initial parameters for the parameterized gates.
        goal: The optimization goal ('purity', 'fidelity', or 'both').
        mode: The mode of operation ('nn' for neural network or 'lookup' for lookup table).
        num_time_steps: The number of time steps for the optimization process.
        optimizer: The optimization algorithm to use ('adam' or 'l-bfgs').
        max_iter: The maximum number of iterations for the optimization process.
        convergence_threshold: The threshold for convergence.
        learning_rate: The learning rate for the optimization algorithm.
        type: The type of quantum system representation.
        propcomp: The method for propagator computation.
        
    Returns:
        result: FgResultPurity containing optimized pulse and convergence data.
    """
    if num_time_steps <= 0:
        raise ValueError("Time steps must be greater than 0.")
    
    # Convert dictionary parameters to flat structure
    flat_params, param_shapes = prepare_parameters_from_dict(initial_params)
    print("Flat params:", flat_params)
    print("Param shapes:", param_shapes)
    # Calculate total number of parameters
    total_params = len(jax.tree_util.tree_leaves(initial_params))
    
    if mode == "nn":
        hidden_size = 32
        batch_size = 1
        output_size = total_params  # RNN outputs all parameters for all gates
        
        rnn_model = RNN(hidden_size=hidden_size, output_size=output_size)
        h_initial_state = jnp.zeros((batch_size, hidden_size))
        
        dummy_input = jnp.zeros((1, 1))  # Dummy input for RNN initialization
        rnn_params = rnn_model.init(
            jax.random.PRNGKey(0), dummy_input, h_initial_state
        )
        
        if goal == "purity":
            def loss_fn(rnn_params):
                """
                Loss function for purity optimization.
                Returns negative purity (we want to minimize this).
                """
                h_initial_state = jnp.zeros((batch_size, hidden_size))
                
                updated_rnn_params = rnn_params
                rho_final, log_prob, _ = calculate_trajectory(
                    rho_cav=U_0,
                    parameterized_gates=parameterized_gates,
                    measurement_indices=measurement_indices,
                    initial_params=flat_params,
                    time_steps=num_time_steps,
                    rnn_model=rnn_model,
                    rnn_params=updated_rnn_params,
                    rnn_state=h_initial_state,
                    param_shapes=param_shapes,
                )
                
                purity_value = purity(rho=rho_final)
                loss1 = -purity_value
                loss2 = log_prob * jax.lax.stop_gradient(-purity_value)
                return loss1 + loss2
            
            # Set up optimizer and training state
            if optimizer.upper() == "ADAM":
                best_model_params, iter_idx = _optimize_adam(
                    loss_fn,
                    rnn_params,
                    max_iter,
                    learning_rate,
                    convergence_threshold,
                )
            
            elif optimizer.upper() == "L-BFGS":
                best_model_params, iter_idx = _optimize_L_BFGS(
                    loss_fn,
                    rnn_params,
                    max_iter,
                    learning_rate,
                    convergence_threshold,
                )
            else:
                raise ValueError(
                    "Invalid optimizer. Choose 'adam' or 'l-bfgs'."
                )
                
            # Calculate final state and purity
            rho_final, _, arr_of_povm_params = calculate_trajectory(
                rho_cav=U_0,
                parameterized_gates=parameterized_gates,
                measurement_indices=measurement_indices,
                initial_params=flat_params,
                time_steps=num_time_steps,
                rnn_model=rnn_model,
                rnn_params=best_model_params,
                rnn_state=h_initial_state,
                param_shapes=param_shapes,
            )
            final_purity = purity(rho=rho_final)
            
            return FgResultPurity(
                optimized_rnn_parameters=best_model_params,
                final_purity=final_purity,
                iterations=iter_idx,
                final_state=rho_final,
                arr_of_povm_params=arr_of_povm_params,
            )
            
        elif goal == "fidelity":
            # TODO: Implement fidelity optimization
            raise NotImplementedError(
                "Fidelity optimization not implemented yet."
            )
            
        elif goal == "both":
            # TODO: Implement combined optimization
            raise NotImplementedError(
                "Combined optimization not implemented yet."
            )
            
        else:
            raise ValueError(
                "Invalid goal. Choose 'purity', 'fidelity', or 'both'."
            )
            
    elif mode == "lookup":
        # TODO: Implement look-up table approach
        raise NotImplementedError(
            "Look-up table approach not implemented yet."
        )
        
    else:
        raise ValueError("Invalid mode. Choose 'nn' or 'lookup'.")


# RNN implementation
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
        # Dense is just a linear layer w x + b
        r = nn.sigmoid(
            nn.Dense(features=self.features, name='reset_gate')(
                jnp.concatenate([x_input, hidden_state], axis=-1)
            )
        )
        
        z = nn.sigmoid(
            nn.Dense(features=self.features, name='update_gate')(
                jnp.concatenate([x_input, hidden_state], axis=-1)
            )
        )
        
        h_telda = nn.tanh(
            nn.Dense(features=self.features, name='candidate_gate')(
                jnp.concatenate([x_input, r * hidden_state], axis=-1)
            )
        )
        
        h_new = (1 - z) * h_telda + z * hidden_state
        return h_new, h_new


class RNN(nn.Module):
    hidden_size: int  # number of features in the hidden state
    output_size: int  # number of features in the output

    @nn.compact
    def __call__(self, measurement, hidden_state):
        """
        Creates a GRU-based RNN that processes measurements and outputs parameters.
        """
        gru_cell = GRUCell(features=self.hidden_size)

        if measurement.ndim == 1:
            measurement = measurement.reshape(1, -1)
        
        new_hidden_state, _ = gru_cell(hidden_state, measurement)
        
        # Output all gate parameters at once
        output = nn.Dense(features=self.output_size)(new_hidden_state)
        
        return output[0], new_hidden_state
