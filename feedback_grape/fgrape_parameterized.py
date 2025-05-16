from typing import NamedTuple
import jax.numpy as jnp
import optax
from feedback_grape.utils.optimizers import _optimize_adam, _optimize_L_BFGS
import jax
import flax.linen as nn
import optax.tree_utils as otu  # type: ignore

# ruff: noqa N8


class fg_result_purity(NamedTuple):
    """
    result class to store the results of the optimization process.
    """

    optimized_parameters: jnp.ndarray
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
    arr_of_povm_params: jnp.ndarray


def _probability_of_a_measurement_outcome_given_a_certain_state(
    rho_cav, measurement_outcome, povm_measure_operator, initial_params
):
    """
    Calculate the probability of a measurement outcome given a quantum state.

    Args:
        rho_cav: Density matrix of the cavity
        measurement_outcome: The measurement outcome
        povm_measure_operator: The POVM measurement operator
        kwargs: Additional parameters for the POVM operator

    Returns:
        Probability of the measurement outcome
    """
    Em = povm_measure_operator(
        measurement_outcome, **initial_params
    ).conj().T @ povm_measure_operator(measurement_outcome, **initial_params)
    # would jnp.real be useful here?
    return jnp.real(jnp.trace(Em @ rho_cav))


# TODO: should generalize if input is a state rather than a density matrix
def _post_measurement_state(
    rho_cav, measurement_outcome, povm_measure_operator, initial_params
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
    Mm_op = povm_measure_operator(measurement_outcome, **initial_params)
    prob = _probability_of_a_measurement_outcome_given_a_certain_state(
        rho_cav,
        measurement_outcome,
        povm_measure_operator,
        initial_params,
    )

    prob = jnp.maximum(prob, 1e-10)
    # is denominator sqrted or not?
    return Mm_op @ rho_cav @ Mm_op.conj().T / prob


class GRUCellImplementation(nn.Module):
    features: int

    @nn.compact
    def __call__(self, h, x):
        """
        Args:
            h: hidden state [batch_size, features] (carry)
            x: input [batch_size, input_size] (input of current time step)
        Returns:
            New hidden state [batch_size, features]
        """
        h_stack = jnp.concatenate([x, h], axis=-1)
        # Reset gate Decides how much of the past state to forget
        r = nn.sigmoid(
            nn.Dense(features=self.features, name='reset_gate')(h_stack)
        )

        # Update gate decides what info to throw away and what to keep
        z = nn.sigmoid(
            nn.Dense(features=self.features, name='update_gate')(h_stack)
        )
        # Candidate hidden state
        h_reset = jnp.concatenate([x, r * h], axis=-1)
        h_tilde = jnp.tanh(
            nn.Dense(features=self.features, name='candidate')(h_reset)
        )
        new_h = (1 - z) * h + z * h_tilde
        return new_h, new_h


class FeedbackRNN(nn.Module):
    """
    Recurrent Neural Network (RNN) for feedback control.
    """

    hidden_size: int
    output_size: int

    @nn.compact
    def __call__(self, measurements, reset=False):
        # Initialize or get hidden state
        # state is the collection name, h is name of the variable, and this is the non-trainable variable
        h = self.variable(
            'state',
            'h',
            lambda: jnp.zeros((measurements.shape[0], self.hidden_size)),
        )
        if reset:
            h.value = jnp.zeros((measurements.shape[0], self.hidden_size))

        # Process each measurement
        gru_cell = nn.GRUCell(features=self.hidden_size)

        # Expand dimensions if needed for single sample
        if measurements.ndim == 1:
            measurements = measurements.reshape(1, -1)

        # Update hidden state with the measurement
        h.value, _ = gru_cell(h.value, measurements)

        # Output layer
        output = nn.Dense(features=self.output_size)(h.value)
        return output


def povm(
    rho_cav: jnp.ndarray,
    povm_measure_operator: callable,
    initial_params: jnp.ndarray,
    key: jnp.ndarray,
) -> tuple[jnp.ndarray, int, float]:
    """
    Perform a POVM measurement on the given state.

    Args:
        rho_cav (jnp.ndarray): The density matrix of the cavity.
        measurement_outcome (int): The measurement outcome.
        povm_measure_operator (callable): The POVM measurement operator.
        initial_params (jnp.ndarray): Initial parameters for the POVM measurement operator.
        key (jnp.ndarray): Random key for stochastic operations.

    Returns:
        tuple: A tuple containing the post-measurement state, the measurement result, and the log probability of the measurement outcome.
    """

    prob_plus = _probability_of_a_measurement_outcome_given_a_certain_state(
        rho_cav, 1, povm_measure_operator, initial_params
    )
    prob_minus = _probability_of_a_measurement_outcome_given_a_certain_state(
        rho_cav, -1, povm_measure_operator, initial_params
    )

    prob_sum = prob_plus + prob_minus
    prob_plus = prob_plus / prob_sum
    prob_minus = prob_minus / prob_sum

    key, subkey = jax.random.split(key)
    random_value = jax.random.uniform(subkey, shape=())
    measurement = jnp.where(random_value < prob_plus, 1, -1)

    rho_meas = _post_measurement_state(
        rho_cav, measurement, povm_measure_operator, initial_params
    )
    prob = jax.lax.cond(
        measurement == 1,
        lambda _: prob_plus,
        lambda _: prob_minus,
        operand=None,
    )
    log_prob = jnp.log(jnp.maximum(prob, 1e-10))
    log_prob = jnp.clip(log_prob, -10, 10)
    return rho_meas, measurement, log_prob, key


def _get_new_params_from_rnn(rnn_model, params, measurement, rnn_state, key):
    """
    Get new parameters from the RNN model based on measurement outcomes.

    Args:
        rnn_model: The RNN model
        params: Current model parameters
        measurement: The measurement outcome
        rnn_state: Current RNN state
        key: JAX random key

    Returns:
        New parameters and updated RNN state
    """
    variables = {'params': params, 'state': rnn_state}
    measurement_input = jnp.array([measurement])
    output, new_variables = rnn_model.apply(
        variables, measurement_input, mutable=['state']
    )

    gamma = output[0, 0]
    delta = output[0, 1]  # No constraints on delta

    updated_params = {'gamma': gamma, 'delta': delta}

    return updated_params, new_variables['state'], key


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
        # do we need to do jnp.real?
        return jnp.real(jnp.trace(rho @ rho))
    elif type == "superoperator":
        pass  # TODO: implement superoperator purity if such a thing exists
    else:
        raise ValueError("Invalid type. Choose 'density' or 'superoperator'.")


# TODO: understand state management
def _calculate_time_step(
    rho_cav,
    povm_measure_operator,
    params,
    rnn_model=None,
    rnn_params=None,
    rnn_state=None,
    key=None,
):
    """
    Calculate the time step for the given density matrix and POVM measurement operator.

    Args:
        rho_cav: Density matrix of the cavity
        current_measurement_outcome: Current measurement outcome
        povm_measure_operator: POVM measurement operator
        params: Parameters for the POVM operator
        rnn_model: RNN model for feedback (optional)
        rnn_params: RNN parameters (optional)
        rnn_state: RNN state (optional)
        key: JAX random key

    Returns:
        Updated state, measurement outcome, log probability, updated params, updated RNN state, and key
    """
    # Perform measurement
    rho_meas, measurement, log_prob, key = povm(
        rho_cav, povm_measure_operator, params, key
    )

    # If using an RNN for feedback
    updated_params = params
    if rnn_model is not None and rnn_params is not None:
        updated_params, rnn_state, key = _get_new_params_from_rnn(
            rnn_model, rnn_params, measurement, rnn_state, key
        )

    return rho_meas, measurement, log_prob, updated_params, rnn_state, key


# TODO: should one accumilate the log probabilities?
# TODO: understand state management
def calculate_trajectory(
    rho_cav,
    time_steps,
    povm_measure_operator,
    initial_params,
    rnn_model=None,
    rnn_params=None,
    key=None,
):
    """
    Calculate a complete quantum trajectory with feedback.

    Args:
        rho_cav: Initial density matrix
        time_steps: Number of time steps
        povm_measure_operator: POVM measurement operator
        initial_params: Initial parameters for the POVM
        rnn_model: RNN model for feedback (optional)
        rnn_params: RNN parameters (optional)
        key: JAX random key

    Returns:
        Final state, final measurement, log probability
    """
    rho_meas = rho_cav
    current_params = initial_params
    total_log_prob = 0.0
    arr_of_povm_params = jnp.zeros((time_steps, len(initial_params)))
    # Initialize RNN state if using RNN
    rnn_state = None
    if rnn_model is not None:
        # Create initial state
        batch_size = 1
        hidden_size = rnn_model.hidden_size
        rnn_state = {'h': jnp.zeros((batch_size, hidden_size))}

    # Run through all time steps
    for i in range(time_steps):
        rho_meas, measurement, log_prob, current_params, rnn_state, key = (
            _calculate_time_step(
                rho_meas,
                povm_measure_operator,
                current_params,
                rnn_model,
                rnn_params,
                rnn_state,
                key,
            )
        )
        total_log_prob += log_prob
        gamma_delta_dict = {
            'gamma': current_params['gamma'],
            'delta': current_params['delta'],
        }
        # print("current_povm_params: ", gamma_delta_dict)
        arr_of_povm_params = arr_of_povm_params.at[i].set(
            jnp.array([current_params['gamma'], current_params['delta']])
        )

    return rho_meas, measurement, total_log_prob, arr_of_povm_params


def optimize_pulse_with_feedback(
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
) -> fg_result_purity | None:
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
    if num_time_steps <= 0:
        raise ValueError("Time steps must be greater than 0.")

    key = jax.random.PRNGKey(0)
    if mode == "nn":
        # print("input shape: ", initial_params.shape)
        hidden_size = 30
        output_size = initial_params.shape[0]

        rnn_model = FeedbackRNN(
            hidden_size=hidden_size, output_size=output_size
        )

        # Initialize RNN parameters
        key, subkey = jax.random.split(key)
        dummy_input = jnp.zeros((1, 1))  # Dummy input for RNN initialization
        rnn_params = rnn_model.init(subkey, dummy_input)['params']

        if goal == "purity":

            def loss_fn(params):
                """
                Loss function for purity optimization.
                Returns negative purity (we want to minimize this).
                """
                updated_rnn_params = params
                povm_params = {
                    'gamma': initial_params[0][0],
                    'delta': initial_params[0][1],
                }
                key_local = jax.random.PRNGKey(0)
                rho_final, _, log_prob, _ = calculate_trajectory(
                    rho_cav=U_0,
                    time_steps=num_time_steps,
                    povm_measure_operator=povm_measure_operator,
                    initial_params=povm_params,
                    rnn_model=rnn_model,
                    rnn_params=updated_rnn_params,
                    key=key_local,
                )
                # Calculate purity
                purity_value = purity(rho=rho_final, type=type)

                # Return negative purity since we're minimizing
                loss1 = -purity_value
                loss2 = log_prob * jax.lax.stop_gradient(-purity_value)

                return loss1 + loss2

            # set up optimizer and training state
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
            povm_params = {
                'gamma': initial_params[0][0],
                'delta': initial_params[0][1],
            }

            rho_meas_best, _, _, arr_of_povm_params = calculate_trajectory(
                rho_cav=U_0,
                time_steps=num_time_steps,
                povm_measure_operator=povm_measure_operator,
                initial_params=povm_params,
                rnn_model=rnn_model,
                rnn_params=best_model_params,
                key=jax.random.PRNGKey(0),
            )
            best_purity = purity(
                rho=rho_meas_best, type=type
            )
            final_result = fg_result_purity(
                optimized_parameters=best_model_params,
                final_purity=best_purity,
                iterations=iter_idx,
                final_state=rho_meas_best,
                arr_of_povm_params=arr_of_povm_params,
            )

            return final_result
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
