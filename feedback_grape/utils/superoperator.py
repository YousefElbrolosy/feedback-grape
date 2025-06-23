import jax.numpy as jnp

# ruff: noqa N8


# TODO: heavily test and inspect
def liouvillian(H, c_ops=None):
    """
    Create the Liouvillian superoperator for a given Hamiltonian and optional collapse operators.
    Based on the Lindblad master equation:
    dρ/dt = -i/ħ[H,ρ] + Σ_j [2L_j ρ L_j† - {L_j†L_j, ρ}]

    Args:
        H: Hamiltonian operator or a pre-computed Liouvillian term
        c_ops: List of collapse operators (optional)

    Returns:
        L: Liouvillian superoperator in matrix form
    """
    # If H is already a Liouvillian term, return it directly
    if (
        hasattr(H, 'shape')
        and len(H.shape) == 2
        and H.shape[0] == H.shape[1] ** 2
    ):
        return H

    # Get the dimension of the Hilbert space
    n = H.shape[0]

    # Construct the commutator term: -i[H,ρ]
    # In superoperator form: -i(H⊗I - I⊗H^T)ρ
    L = -1j * (jnp.kron(jnp.eye(n), H) - jnp.kron(H.conj().T, jnp.eye(n)))

    # Add dissipative terms if collapse operators are provided
    if c_ops is not None:
        for c in c_ops:
            c_dag = c.conj().T
            c_dag_c = c_dag @ c

            # Term 2LρL†: corresponds to 2(L⊗L*)
            # Note: the factor of 2 is included here
            dissipator_term1 = 2 * jnp.kron(c.conj(), c)

            # Term -{L†L,ρ} = -(L†L)ρ - ρ(L†L)
            # In superoperator form: -(L†L⊗I + I⊗(L†L)^T)
            dissipator_term2 = -(
                jnp.kron(jnp.eye(n), c_dag_c)
                + jnp.kron(c_dag_c.conj(), jnp.eye(n))
            )

            # Add the dissipator terms to the Liouvillian
            L += dissipator_term1 + dissipator_term2

    return L


# # TODO: test
# def lindblad(H, c_ops=None, rho=None):
#     """
#     Create the Liouvillian superoperator for a given Hamiltonian and optional collapse operators.
#     Based on the Lindblad master equation:
#     dρ/dt = -i/ħ[H,ρ] + Σ_j [2L_j ρ L_j† - {L_j†L_j, ρ}]

#     Args:
#         H: Hamiltonian operator or a pre-computed Liouvillian term
#         c_ops: List of collapse operators (optional)

#     Returns:
#         L: Liouvillian superoperator in matrix form
#     """
#     # Get the dimension of the Hilbert space
#     if rho is None:
#         n = H.shape[0]
#         rho = jnp.eye(n)

#     # Construct the commutator term: -i[H,ρ]
#     drho_dt = -1j * (jnp.matmul(H, rho) - jnp.matmul(rho, H))

#     # Add dissipative terms if collapse operators are provided
#     if c_ops is not None:
#         for c in c_ops:
#             c_dag = c.conj().T
#             c_dag_c = c_dag @ c

#             # Term 2LρL†: corresponds to 2(L⊗L*)
#             # Note: the factor of 2 is included here
#             dissipator_term1 = 2 * (c @ rho @ c_dag)

#             # Term -{L†L,ρ} = -(L†L)ρ - ρ(L†L)
#             # In superoperator form: -(L†L⊗I + I⊗(L†L)^T)
#             dissipator_term2 = -(
#                 jnp.matmul(c_dag_c, rho) + jnp.matmul(rho, c_dag_c)
#             )

#             # Add the dissipator terms to the Liouvillian
#             drho_dt += dissipator_term1 + dissipator_term2

#     return drho_dt


def sprepost(a, b):
    """
    Create a superoperator that represents the action a * rho * b.dagger()

    Args:
        a: Left operator
        b: Right operator

    Returns:
        E: Superoperator that maps rho -> a * rho * b.dagger()
    """
    return jnp.kron(b.conj(), a)
