

## MAIN.py with time_dep example

import jax
import jax.numpy as jnp
from feedback_grape.grape import optimize_pulse
from feedback_grape.utils.gates import *
from feedback_grape.utils.operators import *
from feedback_grape.utils.tensor import tensor
from feedback_grape.utils.states import basis

# ruff: noqa

@jax.vmap
def build_ham(e_qub, e_cav):
    """
    Build Hamiltonian for given (complex) e_qub and e_cav
    """
    # Constants in Hamiltonian
    chi = 0.2385 * (2 * jnp.pi)
    mu_qub = 4.0
    mu_cav = 8.0
    N_cav = 10
    hconj = lambda a: jnp.swapaxes(a.conj(), -1, -2)
    a = tensor(identity(2), destroy(N_cav))
    adag = hconj(a)
    n_phot = adag @ a
    sigz = tensor(sigmaz(), identity(N_cav))
    sigp = tensor(sigmap(), identity(N_cav))
    one = tensor(identity(2), identity(N_cav))

    H0 = +(chi / 2) * n_phot @ (sigz + one)

    H_ctrl = mu_qub * sigp * e_qub + mu_cav * adag * e_cav
    H_ctrl += hconj(H_ctrl)

    return H0, H_ctrl


def test_time_dep():
    time_start = 0.0
    time_end = 1.0
    time_intervals_num = 5
    N_cav = 10

    # Representation for time dependent Hamiltonian
    def solve(Hs, delta_ts):
        """
        Find evolution operator for piecewise Hs on time intervals delts_ts
        """
        for i, (H, delta_t) in enumerate(zip(Hs, delta_ts)):
            U_intv = jax.scipy.linalg.expm(-1j * H * delta_t)
            U = U_intv if i == 0 else U_intv @ U
        return U

    t_grid = jnp.linspace(time_start, time_end, time_intervals_num + 1)
    # why is delta_ts a 1D array?
    delta_ts = t_grid[1:] - t_grid[:-1]
    fake_random_key = jax.random.key(seed=0)
    e_data = jax.random.uniform(
        fake_random_key, shape=(4, len(delta_ts)), minval=-1, maxval=1
    )
    e_qub = e_data[0] + 1j * e_data[1]
    e_cav = e_data[2] + 1j * e_data[3]
    H0, H_ctrl = build_ham(e_qub, e_cav)
    psi0 = tensor(basis(2), basis(N_cav))
    U = solve(H0 + H_ctrl, delta_ts)
    psi = U @ psi0
    delta_t = 0.2  # Convert the first value of delta_ts to a Python scalar
    result = optimize_pulse(
        H0,
        H_ctrl,
        psi0,
        psi,
        int((time_end - time_start) / delta_t),  # Ensure this is an integer
        time_end - time_start,
        max_iter=10000,
        convergence_threshold=1e-10,
        learning_rate=1e-3,
        time_dep=True,
        delta_ts=delta_ts,
        type="state",
    )
    # print(result.final_fidelity)
    return result


def test_cnot():
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
        max_iter=500,
        learning_rate=1e-2,
    )

    print("final_fidelity: ", result.final_fidelity)
    print("U_f \n", result.final_operator)


if __name__ == "__main__":
    # Example usage
    result = test_time_dep()
    print(result.control_amplitudes)
    # test_cnot()






