"""
Gradient Ascent Pulse Engineering (GRAPE)
"""

# ruff: noqa N8
import jax
import optax  # type: ignore
import optax.tree_utils as otu  # type: ignore
from typing import NamedTuple
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial

jax.config.update("jax_enable_x64", True)
# Implemented adam/L-BFGS optimizers


class result(NamedTuple):
    """
    result class to store the results of the optimization process.
    """

    control_amplitudes: jnp.ndarray
    """
    Optimized control amplitudes.
    """
    final_fidelity: float
    """
    Final fidelity of the optimized control.
    """
    iterations: int
    """
    Number of iterations taken for optimization.
    """
    final_operator: jnp.ndarray
    """
    Final operator after applying the optimized control amplitudes.
    """


# TODO: for next 3 functions related to propagator computation
# TODO: Confirm if for a Lioviliian (superoperator) the same evolution technique
# TODO: can be used or should it be different?
def _compute_propagators(
    H_drift,
    H_control_array,
    delta_t,
    control_amplitudes,
):
    """
    Compute propagators for each time step according to Equation (4).
    Args:
        H_drift: Drift Hamiltonian.
        H_control_array: Array of control Hamiltonians.
        delta_t: Time step for evolution.
        control_amplitudes: Control amplitudes for each time slot.
    Returns:
        propagators: Array of propagators for each time step.
    """
    num_t_slots = control_amplitudes.shape[0]

    # Compute each Uj according to Equation
    def compute_propagator_j(j):
        # Calculate total Hamiltonian for time step j
        H_0 = H_drift
        H_control = 0
        for k in range(len(H_control_array)):
            H_control += control_amplitudes[j, k] * H_control_array[k]

        H_total = H_0 + H_control

        U_j = jax.scipy.linalg.expm(-1j * delta_t * H_total)
        return U_j

    # Create an array of propagators
    propagators = jax.vmap(compute_propagator_j)(jnp.arange(num_t_slots))
    return propagators


def _compute_forward_evolution_time_efficient(propagators, U_0, type):
    """
    Compute the forward evolution states (ρⱼ) according to the paper's definition.
    ρⱼ = Uⱼ···U₁ρ₀U₁†···Uⱼ†

    Args:
        propagators: List of propagators for each time step.
        U_0: Initial operator.
    Returns:
        U_final: final operator after evolution.
    """

    if type == "density":
        rho_final = U_0
        for U_j in propagators:
            # Forward evolution
            # Use below if density operator is used
            rho_final = U_j @ rho_final @ U_j.conj().T
        return rho_final

    else:
        U_final = U_0

        for U_j in propagators:
            # Forward evolution
            # Use below if density operator is used
            # rho_final = U_j @ rho_final @ U_j.conj().T
            U_final = U_j @ U_final

        return U_final


def _compute_forward_evolution_memory_efficient(
    H_drift, H_control_array, delta_t, control_amplitudes, U_0, type
):
    """
    Computes the forward evolution using a memory-efficient method.
    Where we donot precompute all propagators, but rather compute them
    on the fly.

    Args:
        H_drift: Drift Hamiltonian.
        H_control_array: Array of control Hamiltonians.
        delta_t: Time step for evolution.
        control_amplitudes: Control amplitudes for each time slot.
        U_0: Initial operator.

    Returns:
        U_final: Final operator after evolution.
    """

    num_t_slots = control_amplitudes.shape[0]

    if type == "density":
        rho_final = U_0
        for j in range(num_t_slots):
            # Calculate total Hamiltonian for time step j
            H_0 = H_drift
            H_control = 0
            for k in range(len(H_control_array)):
                H_control += control_amplitudes[j, k] * H_control_array[k]

            # Compute U_j and immediately update rho_final to discard U_j from memory
            rho_final = (
                jax.scipy.linalg.expm(-1j * delta_t * (H_0 + H_control))
                @ rho_final
                @ jax.scipy.linalg.expm(-1j * delta_t * (H_0 + H_control))
                .conj()
                .T
            )

        return rho_final

    else:
        U_final = U_0

        for j in range(num_t_slots):
            # Calculate total Hamiltonian for time step j
            H_0 = H_drift
            H_control = 0
            for k in range(len(H_control_array)):
                H_control += control_amplitudes[j, k] * H_control_array[k]

            # Compute U_j and immediately update U_final to discard U_j from memory
            U_final = (
                jax.scipy.linalg.expm(-1j * delta_t * (H_0 + H_control))
                @ U_final
            )

        return U_final


# TODO: Why is this controlled by an amplitude
# NOTE: try different seeds for random initialization and choose the best fidelity
def _init_control_amplitudes(num_t_slots, num_controls):
    """
    Initialize control amplitudes for the optimization process.
    Args:
        num_t_slots: Number of time slots.
        num_controls: Number of control Hamiltonians.
    Returns:
        init_control_amplitudes: Initialized control amplitudes.
    """
    # Random initialization
    # Here, you can't initialize with zeros, as it will lead to zero gradients
    # and no updates. Instead, use a small random value.
    key = jax.random.PRNGKey(42)
    return jax.random.uniform(
        key,
        (num_t_slots, num_controls),
        minval=-(2 * jnp.pi * 0.05),
        maxval=(2 * jnp.pi * 0.05),
    )


def _optimize_adam(
    _fidelity,
    control_amplitudes,
    max_iter,
    learning_rate,
    convergence_threshold,
):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(control_amplitudes)
    fidelities = []

    @jax.jit
    def step(params, state):
        loss = -_fidelity(params)  # Minimize -_fidelity
        grads = jax.grad(lambda x: -_fidelity(x))(params)
        updates, new_state = optimizer.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, -loss

    params = control_amplitudes
    # setting it to -1 in the beginning in case the max_iter is 0
    iter_idx = -1
    for iter_idx in range(max_iter):
        params, opt_state, current_fidelity = step(params, opt_state)
        fidelities.append(current_fidelity)

        if (
            iter_idx > 0
            and abs(fidelities[-1] - fidelities[-2]) < convergence_threshold
        ):
            break

    final_fidelity = _fidelity(params)
    # final_fidelity = fidelities[-1]
    return params, final_fidelity, iter_idx + 1


# TODO: L_bfgs ouputs error when params are complex amplitudes
def _optimize_L_BFGS(
    _fidelity,
    control_amplitudes,
    max_iter,
    convergence_threshold,
):
    """
    Uses L-BFGS to optimize the control amplitudes.
    Args:
        _fidelity: Function to compute fidelity.
        control_amplitudes: Initial control amplitudes.
        max_iter: Maximum number of iterations.
        convergence_threshold: Convergence threshold for optimization.
    Returns:
        control_amplitudes: Optimized control amplitudes.
        fidelities: List of fidelity values during optimization.
    """

    def neg_fidelity(params, **kwargs):
        return -_fidelity(params, **kwargs)

    opt = optax.lbfgs()

    value_and_grad_fn = optax.value_and_grad_from_state(neg_fidelity)

    def step(carry):
        control_amplitudes, state, iter_idx = carry
        value, grad = value_and_grad_fn(control_amplitudes, state=state)
        updates, state = opt.update(
            grad,
            state,
            control_amplitudes,
            value=value,
            grad=grad,
            value_fn=neg_fidelity,
        )
        control_amplitudes = optax.apply_updates(control_amplitudes, updates)
        return control_amplitudes, state, iter_idx + 1

    def continuing_criterion(carry):
        _, state, _ = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_l2_norm(grad)
        return ((iter_num == 0) & (max_iter != 0)) | (iter_num < max_iter) & (
            err >= convergence_threshold
        )

    init_carry = (control_amplitudes, opt.init(control_amplitudes), 0)
    final_params, _, final_iter_idx = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    final_fidelity = _fidelity(final_params)
    return final_params, final_fidelity, final_iter_idx


def _isket(a: jnp.ndarray) -> bool:
    """
    Check if the input is a ket (column vector).
    Args:
        A: Input array.
    Returns:
        bool: True if A is a ket, False otherwise.
    """
    if not isinstance(a, jnp.ndarray):
        return False

    # Check shape - a ket should be a column vector (n x 1)
    shape = a.shape
    if len(shape) != 2 or shape[1] != 1:
        return False

    return True


def _isbra(a: jnp.ndarray) -> bool:
    """
    Check if the input is a bra (row vector).
    Args:
        A: Input array.
    Returns:
        bool: True if A is a bra, False otherwise.
    """
    if not isinstance(a, jnp.ndarray):
        return False

    # Check shape - a bra should be a row vector (1 x n)
    shape = a.shape
    if len(shape) != 2 or shape[0] != 1:
        return False

    return True


def _ket2dm(a: jnp.ndarray) -> jnp.ndarray:
    """
    Convert a ket to a density matrix.
    Args:
        a: Input ket (column vector).
    Returns:
        dm: Density matrix corresponding to the input ket.
    """
    return jnp.outer(a, a.conj())


def _state_density_fidelity(A, B):
    """
    Inspired by qutip's implementation
    Calculates the fidelity (pseudo-metric) between two density matrices.

    Notes
    -----
    Uses the definition from Nielsen & Chuang, "Quantum Computation and Quantum
    Information". It is the square root of the fidelity defined in
    R. Jozsa, Journal of Modern Optics, 41:12, 2315 (1994), used in
    :func:`qutip.core.metrics.process_fidelity`.

    Parameters
    ----------
    A : qobj
        Density matrix or state vector.
    B : qobj
        Density matrix or state vector with same dimensions as A.

    Returns
    -------
    fid : float
        Fidelity pseudo-metric between A and B.

    """
    if _isket(A) or _isbra(A):
        if _isket(B) or _isbra(B):
            A = A / jnp.linalg.norm(A)
            B = B / jnp.linalg.norm(B)
            # The fidelity for pure states reduces to the modulus of their
            # inner product.
            return jnp.abs(jnp.vdot(A, B)) ** 2
        # Take advantage of the fact that the density operator for A
        # is a projector to avoid a sqrtm call.
        A = A / jnp.linalg.norm(A)
        sqrtmA = _ket2dm(A)
    else:
        if _isket(B) or _isbra(B):
            # Swap the order so that we can take a more numerically
            # stable square root of B.
            return _state_density_fidelity(B, A)
        # If we made it here, both A and B are operators, so
        # we have to take the sqrtm of one of them.
        A = A / jnp.linalg.trace(A)
        B = B / jnp.linalg.trace(B)
        sqrtmA = jax.scipy.linalg.sqrtm(A)

    if sqrtmA.shape != B.shape:
        raise TypeError('Density matrices do not have same dimensions.')

    # We don't actually need the whole matrix here, just the trace
    # of its square root, so let's just get its eigenenergies instead.
    # We also truncate negative eigenvalues to avoid nan propagation;
    # even for positive semidefinite matrices, small negative eigenvalues
    # can be reported.
    eig_vals = jnp.linalg.eigvals(sqrtmA @ B @ sqrtmA)
    return jnp.real(jnp.sum(jnp.sqrt(eig_vals)))


def fidelity(*, C_target, U_final, type="unitary"):
    """
    Computes the fidelity of the final state/operator/density matrix/superoperator
    with respect to the target state/operator/density matrix/superoperator.

    For calculating the fidelity of superoperators, the tracediff method is used.
    The fidelity is calculated as:
    - For unitary: ``Tr(C_target^† U_final) / dim``
    - For state: ``|<C_target|U_final>|^2`` where ``C_target`` and ``U_final`` are normalized
    - For density: ``|<C_target|U_final>|^2`` where ``C_target`` and ``U_final`` are normalized
    - For superoperator: ``1 - (0.5 * Tr(|C_target - U_final|)) / C_target.dim``

    Args:
        C_target: Target operator.
        U_final: Final operator after evolution.
        type: Type of fidelity calculation ("unitary", "state", "density", or "superoperator (using tracediff method)")
    Returns:
        fidelity: Fidelity value.
    """
    if type == "superoperator":
        # TRACEDIFF fidelity: 1 - 0.5*Tr(|C_target - U_final|)
        # Where |A| is the matrix absolute value (element-wise)
        diff = C_target - U_final
        # Alternative approach: use the trace of the absolute value directly
        trace_diff = 0.5 * jnp.abs(jnp.trace(diff))
        return 1.0 - trace_diff / C_target.shape[0]
    elif type == "unitary":
        # TODO: check accuracy of this, do we really need vector conjugate or .dot will simply work?
        norm_C_target = C_target / jnp.linalg.norm(C_target)
        norm_U_final = U_final / jnp.linalg.norm(U_final)

        overlap = jnp.vdot(norm_C_target, norm_U_final)
    elif type == "density" or type == "state":
        # normalization occurs in the _state_density_fidelity function
        return _state_density_fidelity(
            C_target,
            U_final,
        )
    else:
        raise ValueError(
            "Invalid type. Choose 'unitary', 'state', 'density', 'superoperator'."
        )
    return jnp.abs(overlap) ** 2


# TODO: hyperparameter search space for finding best set of hyper paramters (Bayesian optimization)
# TODO: see if we need to implement purity functionality for normal grape as well
def optimize_pulse(
    H_drift: jnp.ndarray,
    H_control: list[jnp.ndarray],
    U_0: jnp.ndarray,
    C_target: jnp.ndarray,
    num_t_slots: int,
    total_evo_time: float,
    max_iter: int = 1000,
    convergence_threshold: float = 1e-6,
    learning_rate: float = 0.01,
    type: str = "unitary",
    optimizer: str = "adam",
    propcomp: str = "time-efficient",
) -> result:
    """
    Uses GRAPE to optimize a pulse.

    Args:
        H_drift: Drift Hamiltonian.
        H_control: List of Control Hamiltonians.
        U_0: Initial state or /unitary/density/super operator.
        C_target: Target state or /unitary/density/super operator.
        num_t_slots: Number of time slots.
        total_evo_time: Total evolution time.
        max_iter: Maximum number of iterations.
        convergence_threshold: Convergence threshold for _fidelity change.
        learning_rate: Learning rate for gradient ascent.
        type: Type of fidelity calculation ("unitary" or "state" or "density" or "superoperator").
            When to use each type:
            - "unitary": For unitary evolution.
            - "state": For state evolution.
            - "density": For density matrix evolution.
            - "superoperator": For superoperator evolution (using tracediff method).
        optimizer: Optimizer to use ("adam" or "L-BFGS").
        propcomp: Propagator computation method ("time-efficient" or "memory-efficient").
    Returns:
        result: Dictionary containing optimized pulse and convergence data.
    """
    # Step 1: Initialize control amplitudes
    control_amplitudes = _init_control_amplitudes(num_t_slots, len(H_control))
    delta_t = total_evo_time / num_t_slots

    # Convert H_control to array for easier manipulation
    H_control_array = jnp.array(H_control)

    # Step 2: Gradient ascent loop

    def _fidelity(control_amplitudes):
        if propcomp == "time-efficient":
            propagators = _compute_propagators(
                H_drift, H_control_array, delta_t, control_amplitudes
            )
            U_final = _compute_forward_evolution_time_efficient(
                propagators, U_0, type
            )
        else:
            U_final = _compute_forward_evolution_memory_efficient(
                H_drift,
                H_control_array,
                delta_t,
                control_amplitudes,
                U_0,
                type,
            )
        return fidelity(
            C_target=C_target,
            U_final=U_final,
            type=type,
        )

    if isinstance(optimizer, tuple):
        optimizer = optimizer[0]
    if optimizer.upper() == "L-BFGS":
        control_amplitudes, final_fidelity, iter_idx = _optimize_L_BFGS(
            _fidelity,
            control_amplitudes,
            max_iter,
            convergence_threshold,
        )
    elif optimizer.upper() == "ADAM":
        control_amplitudes, final_fidelity, iter_idx = _optimize_adam(
            _fidelity,
            control_amplitudes,
            max_iter,
            learning_rate,
            convergence_threshold,
        )
    else:
        raise ValueError(
            f"Optimizer {optimizer} not supported. Use 'adam' or 'l-bfgs'."
        )

    if propcomp == "time-efficient":
        propagators = _compute_propagators(
            H_drift, H_control_array, delta_t, control_amplitudes
        )
        rho_final = _compute_forward_evolution_time_efficient(
            propagators, U_0, type
        )
    else:
        rho_final = _compute_forward_evolution_memory_efficient(
            H_drift, H_control_array, delta_t, control_amplitudes, U_0, type
        )

    final_res = result(
        control_amplitudes,
        final_fidelity,
        iter_idx,
        rho_final,
    )

    return final_res


def plot_control_amplitudes(times, final_amps, labels):
    """
    Plot control amplitudes with fixed y-axis scale highlighting each control
    amplitude with respect to the other in its respective plot.

    Args:
        times: Time points for the x-axis.
        final_amps: Control amplitudes to plot.
        labels: Labels for each control amplitude.
    """

    num_controls = final_amps.shape[1]

    # y_max = 0.1  # Fixed y-axis scale
    # y_min = -0.1

    for i in range(num_controls):
        fig, ax = plt.subplots(figsize=(8, 3))

        for j in range(num_controls):
            color = (
                'black' if i == j else 'gray'
            )  # Highlight the current control
            alpha = 1.0 if i == j else 0.1
            ax.plot(
                times,
                final_amps[:, j],
                label=labels[j],
                color=color,
                alpha=alpha,
            )
        ax.set_title(f"Control Fields Highlighting: {labels[i]}")
        ax.set_xlabel("Time")
        ax.set_ylabel(labels[i])
        # ax.set_ylim(y_min, y_max)  # Set fixed y-axis limits
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()
