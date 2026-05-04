import jax
import jax.numpy as jnp
import jax.scipy as jsp


def tensor(a, b):
    c = jnp.tensordot(a, b, axes=0)
    res_shape = (a.shape[0] * b.shape[0], a.shape[1] * b.shape[1])
    res = jnp.transpose(c, axes=(0, 2, 1, 3)).reshape(res_shape)
    return res


def hconj(a):
    return jnp.swapaxes(a.conj(), -1, -2)


def identity(N):
    return jnp.eye(N, dtype=complex)


def liouv_relax_channel(c):
    """
    Represent a jump operator in Liouvillian form
        according to Eq. (192) in Ref. [1]
        
    Input
        c: jnp.ndarray(shape=[N, N]) -- jump operator
    Output
        jnp.ndarray(shape=[N**2, N**2]) -- Liouvillian repr of c
        
    [1] J. A. Gyamfi, Eur. J. Phys. 41 063002 (2020).
    """
    l = tensor(c, c.conj())
    l -= 1 / 2 * tensor(hconj(c) @ c, identity(len(c)))
    l -= 1 / 2 * tensor(identity(len(c)), (hconj(c) @ c).T)
    return l


def liouv_relax_exp(cs):
    """
    Evaluate matrix exponent of full relaxation Liouvillian
        for a list of jump operators
        according to Eq. (180) in Ref. [1]
        
    Input
        cs: list[jnp.ndarray(shape=[N, N])] -- list of jump operators
    Output
        jnp.ndarray(shape=[N**2, N**2]) -- matrix exp of full Liouvillian for cs
        
    [1] J. A. Gyamfi, Eur. J. Phys. 41 063002 (2020).
    """
    l = sum(liouv_relax_channel(c) for c in cs)
    lexp = jsp.linalg.expm(l)
    return lexp


def idle_with_relax(rho, lexp):
    """
    Evolve density matrix under relaxation given by Liouvillian
        according to Eq. (192) in Ref. [1]
        
    Input
        rho: jnp.ndarray(shape=[N, N]) -- initial density matrix
        lexp: jnp.ndarray(shape=[N**2, N**2]) -- matrix exp of Liouvillian
    Output
        jnp.ndarray(shape=[N, N]) -- final density matrix
        
    [1] J. A. Gyamfi, Eur. J. Phys. 41 063002 (2020).
    """
    rho_liouv = rho.reshape(-1, 1)
    rho_liouv = lexp @ rho_liouv
    rho = rho_liouv.reshape(rho.shape)
    return rho