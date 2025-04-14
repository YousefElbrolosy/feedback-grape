"""
Gradient Ascent Pulse Engineering (GRAPE)
"""

# ruff: noqa N8
import jax
import optax  # type: ignore
from typing import NamedTuple
import jax.numpy as jnp
import matplotlib.pyplot as plt

# TODO: Implement this with Pavlo's Cavity + Qubit coupled in dispersive regime
# TODO: remove side effects
# TODO: implement optimizer same as qutip_qtrl fmin_lbfgs or sth


class result(NamedTuple):
    control_amplitudes: jnp.ndarray
    final_fidelity: float
    fidelity_history: jnp.ndarray
    iterations: int
    final_operator: jnp.ndarray


def sesolve(Hs, delta_ts):
    """
    Find evolution operator for piecewise Hs on time intervals delts_ts
    """
    for i, (H, delta_t) in enumerate(zip(Hs, delta_ts)):
        U_intv = jax.scipy.linalg.expm(-1j * delta_t * H)
        U = U_intv if i == 0 else U_intv @ U
    return U

def _compute_propagators(
    H_drift, H_control_array, delta_t, control_amplitudes, time_dep=False, delta_ts=None,
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
        if(not time_dep):
            U_j = jax.scipy.linalg.expm(-1j * delta_t * H_total)
        else:
            U_j = sesolve(H_total, delta_ts)
        return U_j

    # Create an array of propagators
    propagators = jax.vmap(compute_propagator_j)(jnp.arange(num_t_slots))
    return propagators


def _compute_forward_evolution(propagators, U_0):
    """
    Compute the forward evolution states (ρⱼ) according to the paper's definition.
    ρⱼ = Uⱼ···U₁ρ₀U₁†···Uⱼ†

    Args:
        propagators: List of propagators for each time step.
        U_0: Initial density operator.
    Returns:
        rho_j: List of density operators for each time step j.
    """

    U_final = U_0
    for U_j in propagators:
        # Forward evolution
        # Use below if density operator is used
        # rho_final = U_j @ rho_final @ U_j.conj().T
        U_final = U_j @ U_final

    return U_final


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
    # and no updates. Instead, use a small random value. (perhaps because of adam, but
    # TODO: use FMIN_L_BFGS_B instead of adam)
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
    for iter_idx in range(max_iter):
        params, opt_state, current_fidelity = step(params, opt_state)
        fidelities.append(current_fidelity)

        if (
            iter_idx > 0
            and abs(fidelities[-1] - fidelities[-2]) < convergence_threshold
        ):
            print(f"Converged after {iter_idx} iterations.")
            break

        if iter_idx % 10 == 0:
            print(f"Iteration {iter_idx}, _fidelity: {current_fidelity}")

    return params, fidelities, iter_idx


# for unitary evolution (not using density operator)
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
    time_dep=False, 
    delta_ts=None,
) -> result:
    """
    Uses GRAPE to optimize a pulse.

    Args:
        H_drift: Drift Hamiltonian.
        H_control: List of Control Hamiltonians.
        U_0: Initial density operator.
        C_target: Target operator.
        num_t_slots: Number of time slots.
        total_evo_time: Total evolution time.
        max_iter: Maximum number of iterations.
        convergence_threshold: Convergence threshold for _fidelity change.
        learning_rate: Learning rate for gradient ascent.
    Returns:
        result: Dictionary containing optimized pulse and convergence data.
    """
    # Step 1: Initialize control amplitudes
    control_amplitudes = _init_control_amplitudes(num_t_slots, len(H_control))
    delta_t = total_evo_time / num_t_slots

    # Convert H_control to array for easier manipulation
    H_control_array = jnp.array(H_control)

    def _fidelity(control_amplitudes):
        propagators = _compute_propagators(
            H_drift, H_control_array, delta_t, control_amplitudes, time_dep, delta_ts
        )
        U_final = _compute_forward_evolution(propagators, U_0)
        overlap = (
            jnp.trace(jnp.matmul(C_target.conj().T, U_final))
            / C_target.shape[0]
        )
        # just getting the real part of the overlap
        return jnp.abs(overlap) ** 2

    # Step 2: Gradient ascent loop
    control_amplitudes, fidelities, iter_idx = _optimize_adam(
        _fidelity,
        control_amplitudes,
        max_iter,
        learning_rate,
        convergence_threshold,
    )

    propagators = _compute_propagators(
        H_drift, H_control_array, delta_t, control_amplitudes, time_dep, delta_ts
    )
    rho_final = _compute_forward_evolution(propagators, U_0)

    final_res = result(
        control_amplitudes,
        fidelities[-1],
        jnp.array(fidelities),
        iter_idx + 1,
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

    y_max = 0.1  # Fixed y-axis scale
    y_min = -0.1

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
        ax.set_ylim(y_min, y_max)  # Set fixed y-axis limits
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()
