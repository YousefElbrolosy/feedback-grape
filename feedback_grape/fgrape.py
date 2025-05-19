from typing import NamedTuple
import jax.numpy as jnp
from feedback_grape.grape import _optimize_adam, _optimize_L_BFGS
import jax
import flax.linen as nn
# ruff: noqa N8


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
        measurement_outcome, initial_params
    ).conj().T @ povm_measure_operator(measurement_outcome, initial_params)
    # QUESTION: would jnp.real be useful here?
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
    Mm_op = povm_measure_operator(measurement_outcome, initial_params)
    prob = _probability_of_a_measurement_outcome_given_a_certain_state(
        rho_cav,
        measurement_outcome,
        povm_measure_operator,
        initial_params,
    )
    # QUESTION: should we use jnp.clip here?
    prob = jnp.maximum(prob, 1e-10)
    # is denominator sqrted or not?
    return Mm_op @ rho_cav @ Mm_op.conj().T / prob


# TODO: need to handle the case where the
def povm(
    rho_cav: jnp.ndarray,
    povm_measure_operator: callable,
    initial_povm_params: jnp.ndarray,
) -> tuple[jnp.ndarray, int, float]:
    """
    Perform a POVM measurement on the given state.

    Args:
        rho_cav (jnp.ndarray): The density matrix of the cavity.
        measurement_outcome (int): The measurement outcome.
        povm_measure_operator (callable): The POVM measurement operator.
        initial_povm_params (jnp.ndarray): Initial parameters for the POVM measurement operator.
        key (jnp.ndarray): Random key for stochastic operations.

    Returns:
        tuple: A tuple containing the post-measurement state, the measurement result, and the log probability of the measurement outcome.
    """
    # TODO: this should be generalized to all possible measurement outcomes
    prob_plus = _probability_of_a_measurement_outcome_given_a_certain_state(
        rho_cav, 1, povm_measure_operator, initial_povm_params
    )
    random_value = jax.random.uniform(jax.random.PRNGKey(0), shape=())
    measurement = jnp.where(random_value < prob_plus, 1, -1)
    rho_meas = _post_measurement_state(
        rho_cav, measurement, povm_measure_operator, initial_povm_params
    )
    prob = jnp.where(
        measurement == 1,
        prob_plus,
        1 - prob_plus,
    )
    # QUESTION: If prob is 0 though then the log prob is -inf ( and 1e-10 will be a very huge number)
    log_prob = jnp.log(jnp.maximum(prob, 1e-10))
    return rho_meas, measurement, log_prob


# TODO + QUESTION: ask pavlo if purity of a superoperator is something important
def purity(*, rho):
    """
    Computes the purity of a density matrix.

    Args:
        rho: Density matrix.
        type: Type of density matrix ("density" or "superoperator").
    Returns:
        purity: Purity value.
    """
    return jnp.real(jnp.trace(rho @ rho))


def _calculate_time_step(
    *,
    rho_cav,
    povm_measure_operator,
    initial_povm_params,
    rnn_model,
    rnn_params,
    rnn_state,
):
    """
    Calculate the time step for the optimization process.

    Args:
        rho_cav: Density matrix of the cavity.
        povm_measure_operator: POVM measurement operator.
        initial_povm_params: Initial parameters for the POVM measurement operator.

    Returns:

    """

    rho_meas, measurement, log_prob = povm(
        rho_cav, povm_measure_operator, initial_povm_params
    )

    # TODO: feed the measurement outcome to the RNN and get the new params and the new hidden state
    updated_params, new_hidden_state = rnn_model.apply(
        rnn_params, jnp.array([measurement]), rnn_state
    )
    return rho_meas, log_prob, updated_params, new_hidden_state


def calculate_trajectory(
    *,
    rho_cav,
    povm_measure_operator,
    initial_povm_params,
    time_steps,
    rnn_model,
    rnn_params,
    rnn_state,
):
    """
    Calculate a complete quantum trajectory with feedback.

    Args:
        rho_cav: Initial density matrix
        povm_measure_operator: POVM measurement operator
        initial_povm_params: Initial parameters for the POVM
        time_steps: Number of time steps

    Returns:
        Final state, log probability, array of POVM parameters
    """
    # TODO + QUESTION: in the paper, it says one should average the reward over all possible measurement outcomes
    # How can one do that? Is this where batching comes into play? Should one do this averaging for log_prob as well?
    rho_final = rho_cav
    arr_of_povm_params = [initial_povm_params]
    new_params = initial_povm_params
    new_hidden_state = rnn_state
    total_log_prob = 0.0
    for i in range(time_steps):
        rho_final, log_prob, new_params, new_hidden_state = (
            _calculate_time_step(
                rho_cav=rho_final,
                povm_measure_operator=povm_measure_operator,
                initial_povm_params=new_params,
                rnn_model=rnn_model,
                rnn_params=rnn_params,
                rnn_state=new_hidden_state,
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
) -> FgResultPurity:
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

    if mode == "nn":
        hidden_size = 32
        batch_size = 1
        output_size = initial_params.shape[0]

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
                updated_rnn_params = rnn_params
                povm_params = initial_params
                rho_final, log_prob, _ = calculate_trajectory(
                    rho_cav=U_0,
                    povm_measure_operator=povm_measure_operator,
                    initial_povm_params=povm_params,
                    time_steps=num_time_steps,
                    rnn_model=rnn_model,
                    rnn_params=updated_rnn_params,
                    rnn_state=h_initial_state,
                )
                # TODO: see if we need the log prob term
                # TODO: see if we need to implement stochastic sampling instead
                # QUESTION: should we add an accumilate log-term boolean here that decides whether we add
                # the log prob or not? ( like in porroti's implementation )?
                purity_value = purity(rho=rho_final)
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
            # Calculate final state and purity
            rho_final, _, arr_of_povm_params = calculate_trajectory(
                rho_cav=U_0,
                povm_measure_operator=povm_measure_operator,
                initial_povm_params=initial_params,
                time_steps=num_time_steps,
                rnn_model=rnn_model,
                rnn_params=best_model_params,
                rnn_state=h_initial_state,
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
        return output, new_hidden_state
