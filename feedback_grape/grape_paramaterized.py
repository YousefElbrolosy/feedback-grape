# ruff: noqa N8
from feedback_grape.utils.fidelity import fidelity
from feedback_grape.grape import result
from feedback_grape.utils.optimizers import (
    _optimize_adam,
    _optimize_L_BFGS,
)
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# TODO: CLEANUP AND CHECK AGAIN For Correct unitary dynamics

# Answer: Check if the handling of complex numbers is correct, I know l-bfgs outputs error
# --> No, it wasn't it should be handled at the interface level by user dividing it into
# two real parts


# here the time_step is important, because we do an intialization to all
# parameters in all time steps, since we don't have feedback from one
# time step to the next, and then all paramaters are updated
# by the optimization loop at once
def _calculate_sequence_unitary(parameterized_gates, parameters):
    size = parameterized_gates[0](*parameters[0]).shape[0]
    combined_unitary = jnp.eye(size)
    for i, gate in enumerate(parameterized_gates):
        param = parameters[i]
        gate_unitary = gate(*param)
        combined_unitary = gate_unitary @ combined_unitary

    return combined_unitary


def _compute_time_step_time_efficient(
    U_0, parameterized_gates, parameters, type
):
    combined_unitary = _calculate_sequence_unitary(
        parameterized_gates, parameters
    )
    if type == "density":
        rho_t = combined_unitary @ U_0 @ combined_unitary.conj().T
        return rho_t
    else:
        U_t = combined_unitary @ U_0
        return U_t


# either this way or the parameter vectors are repeated for each time step
def _compute_time_step_memory_efficient(
    U_0, parameterized_gates, parameters, type
):
    U_final = U_0
    if type == "density":
        for gate, gate_parameters in zip(parameterized_gates, parameters):
            gate_unitary = gate(*gate_parameters)
            U_final = gate_unitary @ U_final @ gate_unitary.conj().T
    else:
        for gate, gate_parameters in zip(parameterized_gates, parameters):
            # jax.debug.print("gate_parameters: {}" , gate_parameters)
            gate_unitary = gate(*gate_parameters)
            U_final = gate_unitary @ U_final
    return U_final


# either this way or the parameter vectors are repeated for each time step
def calculate_trajectory(
    U_0, parameters, time_steps, parameterized_gates, propcomp, type
):
    U_t = U_0
    if propcomp == "time-efficient":
        for t in range(time_steps):
            U_t = _compute_time_step_time_efficient(
                U_t, parameterized_gates, parameters[t], type
            )
    elif propcomp == "memory-efficient":
        for t in range(time_steps):
            U_t = _compute_time_step_memory_efficient(
                U_t, parameterized_gates, parameters[t], type
            )
    else:
        raise ValueError(
            f"propcomp {propcomp} not supported. Use 'time-efficient' or 'memory-efficient'."
        )
    return U_t


# QUESTION: should non-parameterized hava an option for feedback as well? -- I think yes
# QUESTION: should initial parameters be provided by the user?
# NOTE: Here a deliberate choice is made that the user should provide
# initial parameters, since he/she might have a better guess than random initialization
def optimize_pulse_parameterized(
    U_0: jnp.ndarray,
    C_target: jnp.ndarray,
    parameterized_gates: list[callable],  # type: ignore
    initial_parameters: jnp.ndarray,
    num_time_steps: int,
    optimizer: str,  # adam, l-bfgs
    max_iter: int,
    convergence_threshold: float,
    learning_rate: float,
    type: str,  # unitary, state, density, liouvillian (used now mainly for fidelity calculation)
    propcomp: str = "time-efficient",  # time-efficient, memory-efficient
) -> result | None:
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
        type (str): The type of quantum system representation, such as 'unitary', 'state', 'density', or 'liouvillian'.
                    This is primarily used for fidelity calculation.
        propcomp (str): The method for propagator computation, either 'time-efficient' or 'memory-efficient'.
                        This determines how the forward evolution is computed.
    Returns:
        result: Dictionary containing optimized pulse and convergence data.
    """

    def _loss(initial_parameters):
        # Compute the forward evolution using the parameterized gates

        U_final = calculate_trajectory(
            U_0,
            initial_parameters,
            num_time_steps,
            parameterized_gates,
            propcomp,
            type,
        )

        return -1 * fidelity(
            C_target=C_target,
            U_final=U_final,
            type=type,
        )

    optimized_parameters, iter_idx = train(
        _loss,
        initial_parameters,
        optimizer,
        max_iter,
        convergence_threshold,
        learning_rate,
    )

    final_res = evaluate(
        U_0,
        C_target,
        optimized_parameters,
        num_time_steps,
        parameterized_gates,
        type,
        iter_idx,
        propcomp,
    )

    return final_res


def evaluate(
    U_0,
    C_target,
    optimized_parameters,
    num_time_steps,
    parameterized_gates,
    type,
    iter_idx,
    propcomp,
):
    U_final = calculate_trajectory(
        U_0,
        optimized_parameters,
        num_time_steps,
        parameterized_gates,
        propcomp,
        type,
    )

    final_fidelity = fidelity(
        C_target=C_target,
        U_final=U_final,
        type=type,
    )
    # TODO: check if iter_idx outputs the correct number of iterations
    final_res = result(
        optimized_parameters,
        final_fidelity,
        iter_idx,
        U_final,
    )

    return final_res


def train(
    _loss,
    initial_parameters,
    optimizer,
    max_iter,
    convergence_threshold,
    learning_rate,
):
    if isinstance(optimizer, tuple):
        optimizer = optimizer[0]
    if optimizer.upper() == "L-BFGS":
        optimized_parameters, iter_idx = _optimize_L_BFGS(
            _loss,
            initial_parameters,
            max_iter,
            convergence_threshold,
            learning_rate,
        )
    elif optimizer.upper() == "ADAM":
        optimized_parameters, iter_idx = _optimize_adam(
            _loss,
            initial_parameters,
            max_iter,
            learning_rate,
            convergence_threshold,
        )
    else:
        raise ValueError(
            f"Optimizer {optimizer} not supported. Use 'adam' or 'l-bfgs'."
        )
    return optimized_parameters, iter_idx
