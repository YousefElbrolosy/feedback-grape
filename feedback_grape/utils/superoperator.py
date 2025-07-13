import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
# ruff: noqa N8


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
