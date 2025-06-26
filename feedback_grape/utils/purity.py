import jax.numpy as jnp


def purity(*, rho):
    """
    Computes the purity of a density matrix.

    Args:
        rho: Density matrix.
    Returns:
        purity: Purity value.
    """
    return jnp.real(jnp.trace(rho @ rho))
