import jax.numpy as jnp


# TODO + QUESTION: ask pavlo if purity of a superoperator is something
# important
def purity(*, rho):
    """
    Computes the purity of a density matrix.

    Args:
        rho: Density matrix.
        type: Type of density matrix ("density" or "superoperator").
    Returns:
        purity: Purity value.
    """
    return jnp.real(jnp.trace(rho @ rho))
