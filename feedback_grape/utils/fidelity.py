import jax
import jax.numpy as jnp


# ruff: noqa N8
def _isket(a: jnp.ndarray) -> bool:
    """
    Check if the input is a ket (column vector).
    Args:
        A: Input array.
    Returns:
        bool: True if A is a ket, False otherwise.
    """
    if not isinstance(a, jnp.ndarray):
        return False

    # Check shape - a ket should be a column vector (n x 1)
    shape = a.shape
    if len(shape) != 2 or shape[1] != 1:
        return False

    return True


def _isbra(a: jnp.ndarray) -> bool:
    """
    Check if the input is a bra (row vector).
    Args:
        A: Input array.
    Returns:
        bool: True if A is a bra, False otherwise.
    """
    if not isinstance(a, jnp.ndarray):
        return False

    # Check shape - a bra should be a row vector (1 x n)
    shape = a.shape
    if len(shape) != 2 or shape[0] != 1:
        return False

    return True


def _ket2dm(a: jnp.ndarray) -> jnp.ndarray:
    """
    Convert a ket to a density matrix.
    Args:
        a: Input ket (column vector).
    Returns:
        dm: Density matrix corresponding to the input ket.
    """
    return jnp.outer(a, a.conj())


def _state_density_fidelity(A, B):
    """
    Inspired by qutip's implementation
    Calculates the fidelity (pseudo-metric) between two density matrices.

    Notes
    -----
    Uses the definition from Nielsen & Chuang, "Quantum Computation and Quantum
    Information". It is the square root of the fidelity defined in
    R. Jozsa, Journal of Modern Optics, 41:12, 2315 (1994), used in
    :func:`qutip.core.metrics.process_fidelity`.

    Parameters
    ----------
    A : qobj
        Density matrix or state vector.
    B : qobj
        Density matrix or state vector with same dimensions as A.

    Returns
    -------
    fid : float
        Fidelity pseudo-metric between A and B.

    """
    if _isket(A) or _isbra(A):
        if _isket(B) or _isbra(B):
            A = A / jnp.linalg.norm(A)
            B = B / jnp.linalg.norm(B)
            # The fidelity for pure states reduces to the modulus of their
            # inner product.
            return jnp.abs(jnp.vdot(A, B)) ** 2
        # Take advantage of the fact that the density operator for A
        # is a projector to avoid a sqrtm call.
        A = A / jnp.linalg.norm(A)
        sqrtmA = _ket2dm(A)
    else:
        if _isket(B) or _isbra(B):
            # Swap the order so that we can take a more numerically
            # stable square root of B.
            return _state_density_fidelity(B, A)
        # If we made it here, both A and B are operators, so
        # we have to take the sqrtm of one of them.
        A = A / jnp.linalg.trace(A)
        B = B / jnp.linalg.trace(B)
        sqrtmA = jax.scipy.linalg.sqrtm(A)

    if sqrtmA.shape != B.shape:
        raise TypeError('Density matrices do not have same dimensions.')

    # We don't actually need the whole matrix here, just the trace
    # of its square root, so let's just get its eigenenergies instead.
    # We also truncate negative eigenvalues to avoid nan propagation;
    # even for positive semidefinite matrices, small negative eigenvalues
    # can be reported. This REALLY HAPPENED!! In example c
    eig_vals = jnp.linalg.eigvals(sqrtmA @ B @ sqrtmA)
    eig_vals_non_neg = jnp.where(eig_vals > 0, eig_vals, 0)
    return jnp.real(jnp.sum(jnp.sqrt(eig_vals_non_neg)))


def fidelity(*, C_target, U_final, type="unitary"):
    """
    Computes the fidelity of the final state/operator/density matrix/superoperator
    with respect to the target state/operator/density matrix/superoperator.

    For calculating the fidelity of superoperators, the tracediff method is used.
    The fidelity is calculated as:
    - For unitary: ``Tr(C_target^† U_final) / dim``
    - For state: ``|<C_target|U_final>|^2`` where ``C_target`` and ``U_final`` are normalized
    - For density: ``|<C_target|U_final>|^2`` where ``C_target`` and ``U_final`` are normalized
    - For superoperator: ``1 - (0.5 * Tr(|C_target - U_final|)) / C_target.dim``

    Args:
        C_target: Target operator.
        U_final: Final operator after evolution.
        type: Type of fidelity calculation ("unitary", "state", "density", or "superoperator (using tracediff method)")
    Returns:
        fidelity: Fidelity value.
    """
    if type == "superoperator":
        # TRACEDIFF fidelity: 1 - 0.5*Tr(|C_target - U_final|)
        # Where |A| is the matrix absolute value (element-wise)
        diff = C_target - U_final
        # Alternative approach: use the trace of the absolute value directly
        trace_diff = 0.5 * jnp.abs(jnp.trace(diff))
        return 1.0 - trace_diff / C_target.shape[0]
    elif type == "unitary":
        # TODO: check accuracy of this, do we really need vector conjugate or .dot will simply work?
        norm_C_target = C_target / jnp.linalg.norm(C_target)
        norm_U_final = U_final / jnp.linalg.norm(U_final)

        overlap = jnp.vdot(norm_C_target, norm_U_final)
    elif type == "density" or type == "state":
        # normalization occurs in the _state_density_fidelity function
        return _state_density_fidelity(
            C_target,
            U_final,
        )
    else:
        raise ValueError(
            "Invalid type. Choose 'unitary', 'state', 'density', 'superoperator'."
        )
    return jnp.abs(overlap) ** 2
