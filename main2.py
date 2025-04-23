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
    max_iter=10000,
    convergence_threshold=1e-9,
    learning_rate=1e-2,
    type_req="state",
    optimizer="l-bfgs",
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

    # Run optimization
    result = optimize_pulse(
        H_drift,
        H_ctrl,
        U_0,
        C_target,
        num_t_slots,
        total_evo_time,
        max_iter=100,
        learning_rate=1e-2,
        optimizer="l-bfgs",
    )
    max_iter = 100
    convergence_threshold = 1e-9
    learning_rate = 1e-2
    type_req = "state"
    optimizer = "l-bfgs"
    return H_drift, H_ctrl, U_0, C_target, num_t_slots, total_evo_time, max_iter, convergence_threshold, learning_rate, type_req, optimizer



if __name__ == "__main__":
    # testing parralelizing gpus
    H0_1, H_ctrl_1, psi0_1, psi_1, num_t_slots_1, total_evo_time_1, max_iter_1, convergence_threshold_1, learning_rate_1, type_req_1, optimizer_1 = test_time_dep()
    H0_2, H_ctrl_2, psi0_2, psi_2, num_t_slots_2, total_evo_time_2, max_iter_2, convergence_threshold_2, learning_rate_2, type_req_2, optimizer_2 = test_time_indep()

    res = jax.pmap(
        optimize_pulse
    )(
        H_drift=jnp.array([H0_1, H0_2]),
        H_control=jnp.array([H_ctrl_1, H_ctrl_2]),
        U_0=jnp.array([psi0_1, psi0_2]),
        C_target=jnp.array([psi_1, psi_2]),
        num_t_slots=jnp.array([num_t_slots_1, num_t_slots_2]),
        total_evo_time=jnp.array([total_evo_time_1, total_evo_time_2]),
        max_iter=jnp.array([max_iter_1, max_iter_2]),
        convergence_threshold=jnp.array([convergence_threshold_1, convergence_threshold_2]),
        learning_rate=jnp.array([learning_rate_1, learning_rate_2]),
        type=jnp.array([type_req_1, type_req_2]),
        optimizer=jnp.array([optimizer_1, optimizer_2]),
    )

    for i, r in enumerate(res):
        print(f"Result {i}:")
        print("  Final fidelity: ", r.final_fidelity)
        print("  Iterations: ", r.iterations)
    
