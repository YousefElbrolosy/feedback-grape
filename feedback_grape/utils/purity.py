import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def purity(*, rho):
    """
    Computes the purity of a density matrix.

    Args:
        rho: Density matrix.
    Returns:
        purity: Purity value.
    """
    return jnp.real(jnp.trace(rho @ rho))
