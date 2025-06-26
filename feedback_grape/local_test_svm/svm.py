# ruff: noqa
import feedback_grape.grape as fg
import jax.numpy as jnp
from feedback_grape.utils.operators import *
from feedback_grape.utils.states import basis, coherent
from feedback_grape.utils.tensor import tensor
import time

T = 1  # microsecond
num_of_intervals = 100
N = 30  # dimension of hilbert space
alpha = 1.5
# Phase for the interference
phi = jnp.pi
hconj = lambda a: jnp.swapaxes(a.conj(), -1, -2)
chi = 0.2385 * (2 * jnp.pi)
mu_qub = 4.0
mu_cav = 8.0
psi0 = tensor(basis(2), basis(N))
cat_target_state = coherent(N, alpha) + jnp.exp(-1j * phi) * coherent(
    N, -alpha
)
psi_target = tensor(basis(2), cat_target_state)


# Using Jaynes-Cummings model for qubit + cavity
def build_grape_format_ham():
    """
    Build Hamiltonian for given (complex) e_qub and e_cav
    """

    a = tensor(identity(2), destroy(N))
    adag = tensor(identity(2), create(N))
    n_phot = adag @ a
    sigz = tensor(sigmaz(), identity(N))
    sigp = tensor(sigmap(), identity(N))
    one = tensor(identity(2), identity(N))

    H0 = +(chi / 2) * n_phot @ (sigz + one)
    H_ctrl_qub = mu_qub * sigp
    H_ctrl_qub_dag = hconj(H_ctrl_qub)
    H_ctrl_cav = mu_cav * adag
    H_ctrl_cav_dag = hconj(H_ctrl_cav)

    H_ctrl = [H_ctrl_qub, H_ctrl_qub_dag, H_ctrl_cav, H_ctrl_cav_dag]

    return H0, H_ctrl


def test_mem():
    start_time = time.time()
    # Outputs Fidelity of 0.9799029117042408 but in like 30 minutes
    H0, H_ctrl = build_grape_format_ham()
    res_fg = fg.optimize_pulse(
        H0,
        H_ctrl,
        psi0,
        psi_target,
        num_t_slots=num_of_intervals,
        total_evo_time=T,
        evo_type="state",
        optimizer="l-bfgs",
        propcomp="memory-efficient",
    )
    end_time = time.time()

    print(f"Execution time: {end_time - start_time} seconds")
    print(res_fg.final_fidelity)
    print(res_fg.iterations)


# Execution time: 1000.4430592060089 seconds
# 0.9947991741064574
# 1000
# =======================================
# Execution time: 100.0827054977417 seconds
# 0.9940626257082056
# 1000


def test_time():
    start_time = time.time()
    # Outputs Fidelity of 0.9799029117042408 but in like 30 minutes
    H0, H_ctrl = build_grape_format_ham()
    res_fg = fg.optimize_pulse(
        H0,
        H_ctrl,
        psi0,
        psi_target,
        num_t_slots=num_of_intervals,
        total_evo_time=T,
        evo_type="state",
        optimizer="l-bfgs",
        propcomp="time-efficient",
    )
    end_time = time.time()

    print(f"Execution time: {end_time - start_time} seconds")
    print(res_fg.final_fidelity)
    print(res_fg.iterations)


if __name__ == "__main__":
    test_mem()
    print("=======================================")
    test_time()
