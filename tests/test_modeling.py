# comparing the new modeling API with the old one and seeing if they indeed match
# ruff: noqa
from feedback_grape.utils.modeling import QubitCavity
import jax.numpy as jnp
from jax.scipy.linalg import expm
from feedback_grape.utils.operators import (
    create,
    destroy,
    identity,
    sigmam,
    sigmap,
    sigmaz,
)
from feedback_grape.utils.tensor import tensor
import jax

jax.config.update("jax_enable_x64", True)


def test_hamiltonian_forming():
    N = 30
    chi = 0.2385 * (2 * jnp.pi)
    mu_qub = 4.0
    mu_cav = 8.0
    hconj = lambda a: jnp.swapaxes(a.conj(), -1, -2)

    # Old method
    # Using Jaynes-Cummings model for qubit + cavity
    def build_grape_format_ham_old():
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

    # new method
    hs = QubitCavity(1, N)
    q = hs.qubits[0]
    c = hs.cavities[0]

    def build_grape_format_ham_new():
        """
        Build Hamiltonian for given (complex) e_qub and e_cav
        """
        n_phot = c.adag @ c.a

        H0 = (chi / 2) * n_phot @ (q.sigmaz + q.identity)
        H_ctrl_qub = mu_qub * q.sigmap
        H_ctrl_qub_dag = hconj(H_ctrl_qub)
        H_ctrl_cav = mu_cav * c.adag
        H_ctrl_cav_dag = hconj(H_ctrl_cav)

        H_ctrl = [H_ctrl_qub, H_ctrl_qub_dag, H_ctrl_cav, H_ctrl_cav_dag]

        return H0, H_ctrl

    H0_old, H_ctrl_old = build_grape_format_ham_old()
    H0_new, H_ctrl_new = build_grape_format_ham_new()

    assert jnp.allclose(H0_old, H0_new), "H0 does not match"
    assert len(H_ctrl_old) == len(H_ctrl_new), (
        "Number of control operators does not match"
    )
    for old_op, new_op in zip(H_ctrl_old, H_ctrl_new):
        assert jnp.allclose(old_op, new_op), "Control operators do not match"


def test_unitary_gate_formation():
    N_cav = 40
    # New Method
    hs = QubitCavity(1, N_cav)
    q = hs.qubits[0]

    def qubit_unitary_new(alphas):
        alpha = alphas[0] + 1j * alphas[1]
        U = expm(-1j * (alpha * q.sigmap + alpha.conjugate() * q.sigmam) / 2)
        return U

    def qubit_unitary_old(alphas):
        alpha_re = alphas[0]
        alpha_im = alphas[1]
        alpha = alpha_re + 1j * alpha_im
        return tensor(
            expm(-1j * (alpha * sigmap() + alpha.conjugate() * sigmam()) / 2),
            identity(N_cav),
        )

    assert jnp.allclose(
        qubit_unitary_new([0.5, 0.5]), qubit_unitary_old([0.5, 0.5])
    ), "Unitary gate formation does not match"


def test_qubit_cavity_unitary_gate_formation():
    N_cav = 40
    # New Method
    hs = QubitCavity(1, N_cav)
    q = hs.qubits[0]
    c = hs.cavities[0]

    def qubit_cavity_unitary(betas):
        beta = betas
        H_int = beta * (c.a @ q.sigmap) + beta.conjugate() * (
            c.adag @ q.sigmam
        )
        return expm(-1j * H_int / 2)

    def qubit_cavity_unitary_old(beta_re):
        beta = beta_re
        return expm(
            -1j
            * (
                beta * (tensor(sigmap(), destroy(N_cav)))
                + beta.conjugate() * (tensor(sigmam(), create(N_cav)))
            )
            / 2
        )

    assert jnp.allclose(
        qubit_cavity_unitary(jnp.array([0.5])),
        qubit_cavity_unitary_old(jnp.array([0.5])),
    ), "Qubit-cavity unitary gate formation does not match"
