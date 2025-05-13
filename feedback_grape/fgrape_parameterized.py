import jax.numpy as jnp
from feedback_grape.grape import result
# ruff: noqa N8

def _probability_of_a_measurement_outcome_given_a_certain_state(
    rho_cav, measurement_outcome, povm_measure_operator, **kwargs
):
    Em = povm_measure_operator(
        measurement_outcome, **kwargs
    ).conj().T @ povm_measure_operator(measurement_outcome, **kwargs)
    return jnp.trace(Em @ rho_cav)

# TODO: should generalize if input is a state rather than a density matrix
def _post_measurement_state(
    rho_cav, measurement_outcome, povm_measure_operator, **kwargs
):
    """
    returns the state after the measurement
    """
    return (
        povm_measure_operator(measurement_outcome, **kwargs)
        @ rho_cav
        @ povm_measure_operator(measurement_outcome, **kwargs).conj().T
        / jnp.sqrt(
            _probability_of_a_measurement_outcome_given_a_certain_state(
                rho_cav, measurement_outcome, **kwargs
            )
        )
    )


def povm(rho_cav:jnp.ndarray, measurement_outcome: int, povm_measure_operator: callable, **kwargs) -> tuple[jnp.ndarray, int, float]:
    """
    Perform a POVM measurement on the given state.

    Args:
        rho_cav (jnp.ndarray): The density matrix of the cavity.
        measurement_outcome (int): The measurement outcome.
        povm_measure_operator (callable): The POVM measurement operator.
        **kwargs: Additional arguments for the POVM operator.

    Returns:
        tuple: A tuple containing the post-measurement state, the measurement result, and the log probability of the measurement outcome.
    """
    if(measurement_outcome == 0):
        measurement = -1
    else:
        measurement = 1


    rho_meas = _post_measurement_state(
        rho_cav, measurement_outcome, povm_measure_operator, **kwargs
    )
    prob = _probability_of_a_measurement_outcome_given_a_certain_state(
        rho_cav, measurement_outcome, povm_measure_operator, **kwargs
    )

    random = jnp.random.uniform(0, 1)

    if random < prob:
        measurement = measurement_outcome
    else:
        measurement = -1 * measurement_outcome


    log_prob = jnp.log(jnp.abs(prob))


    return rho_meas, measurement, log_prob
    

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
        return jnp.trace(rho @ rho)
    elif type == "superoperator":
        pass  # TODO: implement superoperator purity if such a thing exists
    else:
        raise ValueError("Invalid type. Choose 'density' or 'superoperator'.")
    
def optimize_pulse_parameterized(
    U_0: jnp.ndarray,
    C_target: jnp.ndarray,
    parameterized_gates: list[callable],  # type: ignore
    povm_measure_operator: callable, # type: ignore
    initial_parameters: jnp.ndarray,
    goal: str,  # purity, fidelity, both
    mode: str,  # nn, lookup
    num_time_steps: int,
    optimizer: str,  # adam, l-bfgs
    max_iter: int,
    convergence_threshold: float,
    learning_rate: float,
    type: str,  # unitary, state, density, superoperator (used now mainly for fidelity calculation)
    propcomp: str = "time-efficient",  # time-efficient, memory-efficient
) -> result | None:
    """
    Optimizes pulse parameters for quantum systems based on the specified configuration.

    Args:
        U_0: Initial state or /unitary/density/super operator.
        C_target: Target state or /unitary/density/super operator.
        parameterized_gates (list[callable]): A list of parameterized gate functions to be optimized.
        povm_measure_operator (callable): The POVM measurement operator Mm. 
        initial_parameters (jnp.ndarray): Initial parameters for the parameterized gates.
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
    if goal == "purity":
        pass
    elif goal == "fidelity":
        pass # TODO: implement fidelity optimization
    elif goal == "both":
        pass # TODO: implement both optimization
    else:
        raise ValueError("Invalid goal. Choose 'purity', 'fidelity', or 'both'.")


