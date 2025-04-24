## MAIN.py with time_dep example

import jax
import qutip as qt
import numpy as np
import jax.numpy as jnp
import qutip_qtrl.pulseoptim as qtrl
from feedback_grape.grape import optimize_pulse
from feedback_grape.utils.gates import *
from feedback_grape.utils.operators import *
from feedback_grape.utils.tensor import tensor
from feedback_grape.utils.states import basis
# ruff: noqa

N_cav = 10
chi = 0.2385 * (2 * jnp.pi)
mu_qub = 4.0
mu_cav = 8.0
hconj = lambda a: jnp.swapaxes(a.conj(), -1, -2)

# Problem 1
@jax.vmap
def build_ham(e_qub, e_cav):
    """
    Build Hamiltonian for given (complex) e_qub and e_cav
    """

    a = tensor(identity(2), destroy(N_cav))
    adag = hconj(a)
    n_phot = adag @ a
    sigz = tensor(sigmaz(), identity(N_cav))
    sigp = tensor(sigmap(), identity(N_cav))
    one = tensor(identity(2), identity(N_cav))

    H0 = +(chi / 2) * n_phot @ (sigz + one)

    H_ctrl = mu_qub * sigp * e_qub + mu_cav * adag * e_cav
    H_ctrl += hconj(H_ctrl)
    # You just pass an array of the Hamiltonian matrices "Hs" corresponding to the time
    # intervals "delta_ts" (that is, "Hs" is a 3D array).
    return H0, H_ctrl


def build_grape_format_ham():
    """
    Build Hamiltonian for given (complex) e_qub and e_cav
    """

    a = tensor(identity(2), destroy(N_cav))
    adag = hconj(a)
    n_phot = adag @ a
    sigz = tensor(sigmaz(), identity(N_cav))
    sigp = tensor(sigmap(), identity(N_cav))
    one = tensor(identity(2), identity(N_cav))

    H0 = +(chi / 2) * n_phot @ (sigz + one)
    H_ctrl_qub = mu_qub * sigp
    H_ctrl_cav = mu_cav * adag
    H_ctrl_dag = hconj(H_ctrl_qub + H_ctrl_cav)

    H_ctrl = [H_ctrl_qub, H_ctrl_cav, H_ctrl_dag]

    return H0, H_ctrl

# Problem 2
def test_time_dep():
    time_start = 0.0
    time_end = 1.0
    time_intervals_num = 5
    N_cav = 10
    t_grid = jnp.linspace(time_start, time_end, time_intervals_num + 1)
    delta_ts = t_grid[1:] - t_grid[:-1]
    fake_random_key = jax.random.key(seed=0)
    e_data = jax.random.uniform(
        fake_random_key, shape=(4, len(delta_ts)), minval=-1, maxval=1
    )
    e_qub = e_data[0] + 1j * e_data[1]
    e_cav = e_data[2] + 1j * e_data[3]
    H0, H_ctrl = build_ham(e_qub, e_cav)

    # Representation for time dependent Hamiltonian
    def solve(Hs, delta_ts):
        """
        Find evolution operator for piecewise Hs on time intervals delts_ts
        """
        for i, (H, delta_t) in enumerate(zip(Hs, delta_ts)):
            U_intv = jax.scipy.linalg.expm(-1j * H * delta_t)
            U = U_intv if i == 0 else U_intv @ U
        return U

    U = solve(H0 + H_ctrl, delta_ts)
    psi0 = tensor(basis(2), basis(N_cav))
    psi = U @ psi0

    H0_grape, H_ctrl_grape = build_grape_format_ham()
    num_t_slots = int((time_end - time_start) / delta_ts[0])
    total_evo_time = time_end - time_start
    max_iter=10000
    convergence_threshold=1e-9
    learning_rate=1e-2
    type_req="state"
    optimizer="l-bfgs"
    return H0_grape, H_ctrl_grape, psi0, psi, num_t_slots, total_evo_time, max_iter, convergence_threshold, learning_rate, type_req, optimizer


def test_time_indep():
    # ruff: noqa

    """
    Gradient Ascent Pulse Engineering (GRAPE)
    """

    # Example usage
    g = 0  # Small coupling strength
    H_drift = g * (tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()))
    H_ctrl = [
        tensor(sigmax(), identity(2)),
        tensor(sigmay(), identity(2)),
        tensor(sigmaz(), identity(2)),
        tensor(identity(2), sigmax()),
        tensor(identity(2), sigmay()),
        tensor(identity(2), sigmaz()),
        tensor(sigmax(), sigmax()),
        tensor(sigmay(), sigmay()),
        tensor(sigmaz(), sigmaz()),
    ]

    U_0 = identity(4)
    # Target operator (CNOT gate)
    C_target = cnot()

    num_t_slots = 500
    total_evo_time = 2 * jnp.pi

    max_iter = 100
    convergence_threshold = 1e-9
    learning_rate = 1e-2
    type_req = "state"
    optimizer = "l-bfgs"
    return H_drift, H_ctrl, U_0, C_target, num_t_slots, total_evo_time, max_iter, convergence_threshold, learning_rate, type_req, optimizer





def simple_vectorized_wrapper():
    # Get inputs from test_time_dep
    (
        H0_grape,
        H_ctrl_grape,
        psi0,
        psi,
        num_t_slots,
        total_evo_time,
        max_iter,
        convergence_threshold,
        learning_rate,
        type_req,
        optimizer,
    ) = test_time_dep()
    batch_size = 2
    H_drift_batched = jnp.stack([H0_grape] * batch_size)  # Shape: (2, dim, dim)
    H_control_batched = [jnp.stack([h] * batch_size) for h in H_ctrl_grape]  # List of (2, dim, dim)
    U_0_batched = jnp.stack([psi0] * batch_size)  # Shape: (2, dim)
    C_target_batched = jnp.stack([psi] * batch_size)  # Shape: (2, dim)
    # Define vectorized optimize_pulse
    vectorized_optimize = jax.vmap(
        lambda H_d, H_c, U_0, C_t: optimize_pulse(
            H_d,
            H_c,
            U_0,
            C_t,
            num_t_slots,
            total_evo_time,
            max_iter,
            convergence_threshold,
            learning_rate,
            type_req,
            optimizer,
        ),
        in_axes=(0, 0, 0, 0),  # Batch H_drift, U_0, C_target; H_control is a list
    )

    # Run vectorized optimization
    results = vectorized_optimize(
        H_drift_batched, H_control_batched, U_0_batched, C_target_batched
    )

    # Extract results
    for i in range(batch_size):
        print(f"Instance {i+1}:")
        print(f"  Final Fidelity: {results.final_fidelity[i]}")
        print(f"  Iterations: {results.iterations[i]}")
        print(f"  Control Amplitudes Shape: {results.control_amplitudes[i].shape}")
        print(f"  Final Operator Shape: {results.final_operator[i].shape}")


if __name__ == "__main__":
    simple_vectorized_wrapper()
    
