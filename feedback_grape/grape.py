"""
GRadient Ascent Pulse Engineering (GRAPE)
"""

# ruff: noqa N8
from enum import Enum
import jax
from typing import NamedTuple
import jax.numpy as jnp
import matplotlib.pyplot as plt
from .utils.optimizers import (
    optimize_adam,
    optimize_L_BFGS,
)
from .utils.fidelity import fidelity, is_positive_semi_definite, isbra, isket
from .utils.solver import mesolve, sesolve

jax.config.update("jax_enable_x64", True)


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


class _DEFAULTS(Enum):
    ctrl_amp_lower_bound = -2 * jnp.pi * 0.05
    ctrl_amp_upper_bound = 2 * jnp.pi * 0.05
    C_OPS = []  # type: ignore
    MAX_ITER = 1000
    CONVERGENCE_THRESHOLD = 1e-6
    LEARNING_RATE = 0.01
    OPTIMIZER = "adam"
    PROPCOMP = "time-efficient"
    PROGRESS = False


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


def _compute_forward_evolution_time_efficient(
    H_drift, H_control_array, delta_t, control_amplitudes, U_0, evo_type
):
    """
    Compute the forward evolution states (ρⱼ) according to the paper's definition.
    ρⱼ = Uⱼ···U₁ρ₀U₁†···Uⱼ†

    Args:
        propagators: List of propagators for each time step.
        U_0: Initial operator.
    Returns:
        U_final: final operator after evolution.
    """
    propagators = _compute_propagators(
        H_drift, H_control_array, delta_t, control_amplitudes
    )
    if evo_type == "density":
        rho_final = U_0
        for U_j in propagators:
            rho_final = U_j @ rho_final @ U_j.conj().T
        return rho_final

    else:
        U_final = U_0

        for U_j in propagators:
            U_final = U_j @ U_final

        return U_final


def _compute_forward_evolution_memory_efficient(
    H_drift, H_control_array, delta_t, control_amplitudes, U_0, evo_type
):
    """
    Computes the forward evolution using a memory-efficient method,
    leveraging the sesolve function to evolve the state/operator.

    Args:
        H_drift: Drift Hamiltonian.
        H_control_array: Array of control Hamiltonians.
        delta_t: Time step for evolution.
        control_amplitudes: Control amplitudes for each time slot.
        U_0: Initial operator.

    Returns:
        U_final: Final operator after evolution.
    """
    Hs, delta_ts = build_parameterized_hamiltonian(
        control_amplitudes, H_drift, H_control_array, delta_t
    )
    return sesolve(Hs, U_0, delta_ts, evo_type=evo_type)


def build_parameterized_hamiltonian(
    control_amplitudes, H_drift, H_control_array, delta_t
):
    num_t_slots = control_amplitudes.shape[0]
    # Build the list of Hamiltonians for each time slot
    Hs = []
    for j in range(num_t_slots):
        H_total = H_drift
        for k in range(len(H_control_array)):
            H_total += control_amplitudes[j, k] * H_control_array[k]
        Hs.append(H_total)
    delta_ts = [delta_t] * num_t_slots
    return Hs, delta_ts


def _init_control_amplitudes(
    num_t_slots, num_controls, ctrl_amp_lower_bound, ctrl_amp_upper_bound
):
    """
    Initialize control amplitudes for the optimization process.
    Args:
        num_t_slots: Number of time slots.
        num_controls: Number of control Hamiltonians.
        ctrl_amp_lower_bound: Lower bound for control amplitudes.
        ctrl_amp_upper_bound: Upper bound for control amplitudes.
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
        minval=ctrl_amp_lower_bound,
        maxval=ctrl_amp_upper_bound,
    )


def optimize_pulse(
    H_drift: jnp.ndarray,
    H_control: list[jnp.ndarray],
    U_0: jnp.ndarray,
    C_target: jnp.ndarray,
    num_t_slots: int,
    total_evo_time: float,
    evo_type: str,
    ctrl_amp_lower_bound: float = _DEFAULTS.ctrl_amp_lower_bound.value,
    ctrl_amp_upper_bound: float = _DEFAULTS.ctrl_amp_upper_bound.value,
    c_ops: list[jnp.ndarray] = _DEFAULTS.C_OPS.value,
    max_iter: int = _DEFAULTS.MAX_ITER.value,
    convergence_threshold: float = _DEFAULTS.CONVERGENCE_THRESHOLD.value,
    learning_rate: float = _DEFAULTS.LEARNING_RATE.value,
    optimizer: str = _DEFAULTS.OPTIMIZER.value,
    propcomp: str = _DEFAULTS.PROPCOMP.value,
    progress: bool = _DEFAULTS.PROGRESS.value,
) -> result:
    """
    Uses GRAPE to optimize a pulse.

    Args:
        H_drift: Drift Hamiltonian.
        H_control: List of Control Hamiltonians.
        U_0: Initial state or unitary/density matrix.
        C_target: Target state or unitary/density matrix.
        num_t_slots: Number of time slots.
        total_evo_time: Total evolution time.
        evo_type: Type of fidelity and evolution calculation ("unitary" or "state" or "density"). \n
            Options:
                - "unitary": For unitary evolution.
                - "state": For state evolution.
                - "density": For density matrix evolution.
        ctrl_amp_lower_bound: Lower bound for control amplitudes initialization \n
            - (default: -2 * jnp.pi * 0.05).
        ctrl_amp_upper_bound: Upper bound for control amplitudes initialization \n
            - (default: 2 * jnp.pi * 0.05).
        c_ops: List of collapse operators (optional, used for dissipative evolution) \n
            - (default: []).
        max_iter: Maximum number of iterations \n
            - (default: 1000).
        convergence_threshold: Convergence threshold provide None to enforce max iterations \n
            - (default: 1e-6).
        learning_rate: Learning rate for gradient ascent \n
            - (default: 0.01)
        optimizer: Optimizer to use ("adam" or "L-BFGS") \n
            - (default: "adam")
        propcomp: Propagator computation method ("time-efficient" or "memory-efficient") \n
            - (default: "time-efficient").
        progress: Whether to show progress (cost every 10 iterations) during optimization. (for debugging purposes) This may significantly slow down the optimization process \n
            - (default: False).
    Returns:
        result: Dictionary containing optimized pulse and convergence data.
    """
    if convergence_threshold == None:
        early_stop = False
    else:
        early_stop = True

    if evo_type not in ["state", "density", "unitary"]:
        raise ValueError(
            "Invalid evo_type. Choose 'state' or 'density' or 'unitary'."
        )

    if U_0 is None:
        raise ValueError(
            "Please provide an initial state/density matrix/unitary gate U_0."
        )

    if isbra(U_0) or isbra(C_target):
        raise TypeError(
            "Please provide initial and target states as kets (column vectors) or density matrices or unitary matrices."
        )
    
    if evo_type == "state" and not (isket(U_0) and isket(C_target)):
        raise TypeError(
            "For evo_type='state', please provide initial and target states as kets (column vectors)."
        )
    
    if evo_type == "density" and (isket(U_0) or isket(C_target)):
        raise TypeError(
            "For evo_type='density', please provide initial and target states as density matrices."
        )

    if (
        (not is_positive_semi_definite(U_0)
        or not is_positive_semi_definite(C_target))
        and evo_type == "density"
    ):
        raise TypeError(
            'If evo_type=`density` your initial and target rhos must be positive semi-definite.'
        )

    if (
        evo_type == "state" or (isket(U_0) or isket(C_target))
    ) and c_ops != []:
        raise ValueError(
            "You supplied collapse operators (c_ops) for dissipation, but your evo_type is state or one of your initial and target are kets. "
            "Dissipation requires a density matrix representation of your inital and target states because the solver uses Lindblad equation to evolve the system with dissipation."
            "Please provide U_0 and U_target as density matrices perhaps using `utils.fidelity.ket2dm` and use evo_type='density'."
        )
    # Step 1: Initialize control amplitudes
    control_amplitudes = _init_control_amplitudes(
        num_t_slots, len(H_control), ctrl_amp_lower_bound, ctrl_amp_upper_bound
    )
    delta_t = total_evo_time / num_t_slots

    # Convert H_control to array for easier manipulation
    H_control_array = jnp.array(H_control)

    # Step 2: Gradient ascent loop

    def _loss(control_amplitudes):
        if evo_type == "density" and c_ops != []:
            Hs, _ = build_parameterized_hamiltonian(
                control_amplitudes, H_drift, H_control_array, delta_t
            )
            tsave = jnp.linspace(0, total_evo_time, num_t_slots)
            U_final = mesolve(H=Hs, jump_ops=c_ops, rho0=U_0, tsave=tsave)
        else:
            if propcomp == "time-efficient":
                U_final = _compute_forward_evolution_time_efficient(
                    H_drift,
                    H_control_array,
                    delta_t,
                    control_amplitudes,
                    U_0,
                    evo_type,
                )
            else:
                U_final = _compute_forward_evolution_memory_efficient(
                    H_drift,
                    H_control_array,
                    delta_t,
                    control_amplitudes,
                    U_0,
                    evo_type,
                )

        return -1 * fidelity(
            C_target=C_target,
            U_final=U_final,
            evo_type=evo_type,
        )

    control_amplitudes, iter_idx = train(
        _loss,
        control_amplitudes,
        max_iter,
        convergence_threshold,
        learning_rate,
        optimizer,
        progress,
        early_stop,
    )

    final_res = evaluate(
        H_drift=H_drift,
        H_control_array=H_control_array,
        U_0=U_0,
        C_target=C_target,
        c_ops=c_ops,
        control_amplitudes=control_amplitudes,
        delta_t=delta_t,
        iter_idx=iter_idx,
        evo_type=evo_type,
        propcomp=propcomp,
        total_evo_time=total_evo_time,
        num_t_slots=num_t_slots,
    )

    return final_res


def evaluate(
    H_drift,
    H_control_array,
    U_0,
    C_target,
    c_ops,
    control_amplitudes,
    delta_t,
    iter_idx,
    evo_type,
    propcomp,
    total_evo_time,
    num_t_slots,
):
    if evo_type == "density" and c_ops != []:
        Hs, _ = build_parameterized_hamiltonian(
            control_amplitudes, H_drift, H_control_array, delta_t
        )
        tsave = jnp.linspace(0, total_evo_time, num_t_slots)
        rho_final = mesolve(H=Hs, jump_ops=c_ops, rho0=U_0, tsave=tsave)
    else:
        if propcomp == "time-efficient":
            rho_final = _compute_forward_evolution_time_efficient(
                H_drift,
                H_control_array,
                delta_t,
                control_amplitudes,
                U_0,
                evo_type,
            )
        elif propcomp == "memory-efficient":
            rho_final = _compute_forward_evolution_memory_efficient(
                H_drift,
                H_control_array,
                delta_t,
                control_amplitudes,
                U_0,
                evo_type,
            )
        else:
            raise ValueError(
                f"Propagator computation method {propcomp} not supported. Use 'time-efficient' or 'memory-efficient'."
            )

    final_fidelity = fidelity(
        C_target=C_target,
        U_final=rho_final,
        evo_type=evo_type,
    )
    final_res = result(
        control_amplitudes,
        final_fidelity,
        iter_idx,
        rho_final,
    )

    return final_res


def train(
    _loss,
    control_amplitudes,
    max_iter,
    convergence_threshold,
    learning_rate,
    optimizer,
    progress,
    early_stop,
):
    if isinstance(optimizer, tuple):
        optimizer = optimizer[0]
    if optimizer.upper() == "L-BFGS":
        control_amplitudes, iter_idx = optimize_L_BFGS(
            _loss,
            control_amplitudes,
            max_iter,
            convergence_threshold,
            learning_rate,
            progress,
            early_stop,
        )
    elif optimizer.upper() == "ADAM":
        control_amplitudes, iter_idx = optimize_adam(
            _loss,
            control_amplitudes,
            max_iter,
            learning_rate,
            convergence_threshold,
            progress,
            early_stop,
        )
    else:
        raise ValueError(
            f"Optimizer {optimizer} not supported. Use 'adam' or 'l-bfgs'."
        )
    return control_amplitudes, iter_idx


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
