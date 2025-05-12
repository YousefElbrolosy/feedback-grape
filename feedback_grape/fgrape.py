# ruff: noqa N8
from feedback_grape.grape import fidelity, result, _optimize_adam, _optimize_L_BFGS
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def _calculate_sequence_unitary(parameterized_gates, parameters, time_step):
    size = parameterized_gates[0](parameters[time_step][0]).shape[0]
    combined_unitary = jnp.eye(size)
    for gate, parameter in zip(parameterized_gates, parameters[time_step]):
        combined_unitary = combined_unitary @ gate(parameter)
    return combined_unitary

# either this way or the parameter vectors are repeated for each time step
def _calculate_trajectory(U_0, parameters, time_steps, parameterized_gates, type):
    if type == "density":
        rho_t = U_0
        for t in range(time_steps):
            combined_unitary = _calculate_sequence_unitary(parameterized_gates, parameters, t)
            rho_t = combined_unitary @ rho_t @ combined_unitary.conj().T
            # if feedback is done, then it happens here (after each time step) based on measurement outcome
        return rho_t
    else:
        U_t = U_0
        for t in range(time_steps):
            combined_unitary = _calculate_sequence_unitary(parameterized_gates, parameters, t)
            U_t = combined_unitary @ U_t
            # if feedback is done, then it happens here (after each time step) based on measurement outcome
        return U_t

# QUESTION: should non-parameterized hava an option for feedback as well? -- I think yes
# QUESTION: should initial parameters be provided by the user?
# NOTE: Here a deliberate choice is made that the user should provide
# initial parameters, since he/she might have a better guess than random initialization
def optimize_pulse_parameterized(
    U_0: jnp.ndarray,
    C_target: jnp.ndarray,
    feedback: bool,  # True, False
    parameterized_gates: list[callable],
    initial_parameters: jnp.ndarray,
    num_time_steps: int,
    mode: str,  # nn, lookup
    goal: str,  # purity, fidelity, both
    optimizer: str,  # adam, l-bfgs
    max_iter: int,
    convergence_threshold: float,
    learning_rate: float,
    type: str,  # unitary, state, density, superoperator (used now mainly for fidelity calculation)
    propcomp: str = "time-efficient",  # time-efficient, memory-efficient
) -> result:
    """
    Optimizes pulse parameters for quantum systems based on the specified configuration.

    Args:
        U_0: Initial state or /unitary/density/super operator.
        C_target: Target state or /unitary/density/super operator.
        feedback (bool): Indicates whether feedback is enabled (True) or disabled (False).
        parameterized_gates (list[callable]): A list of parameterized gate functions to be optimized.
        initial_parameters (jnp.ndarray): Initial parameters for the parameterized gates.
        num_time_steps (int): The number of time steps for the optimization process.
        mode (str): The mode of operation, either 'nn' (neural network) or 'lookup' (lookup table).
        goal (str): The optimization goal, which can be 'purity', 'fidelity', or 'both'.
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
    if goal == "fidelity":

        def _fidelity(initial_parameters):
            # Compute the forward evolution using the parameterized gates
            if not feedback:
                U_final = _calculate_trajectory(
                    U_0, initial_parameters, num_time_steps, parameterized_gates, type
                )
            return fidelity(
                C_target=C_target,
                U_final=U_final,
                type=type,
            )

        if isinstance(optimizer, tuple):
            optimizer = optimizer[0]
        if optimizer.upper() == "L-BFGS":
            optimized_parameters, final_fidelity, iter_idx = _optimize_L_BFGS(
                _fidelity,
                initial_parameters,
                max_iter,
                convergence_threshold,
            )
        else:
            optimized_parameters, final_fidelity, iter_idx = _optimize_adam(
                _fidelity,
                initial_parameters,
                max_iter,
                learning_rate,
                convergence_threshold,
            )

        if not feedback:
            U_final = _calculate_trajectory(
                    U_0, optimized_parameters, num_time_steps, parameterized_gates, type
            )

            final_res = result(
                optimized_parameters,
                final_fidelity,
                iter_idx,
                U_final,
            )

        return final_res
    else:
        pass  # TODO: implement this part

