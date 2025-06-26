import jax.numpy as jnp


# ruff: noqa N8
def isket(a: jnp.ndarray) -> bool:
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


def isbra(a: jnp.ndarray) -> bool:
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


def ket2dm(a: jnp.ndarray) -> jnp.ndarray:
    """
    Convert a ket to a density matrix.
    Args:
        a: Input ket (column vector).
    Returns:
        dm: Density matrix corresponding to the input ket.
    """
    if not isket(a):
        raise TypeError("Input must be a ket (column vector).")
    return jnp.outer(a, a.conj())


# Only works for hermitian matrices
def sqrtm_eig(A):
    """GPU-friendly matrix square root using eigendecomposition."""
    eigenvals, eigenvecs = jnp.linalg.eigh(A)  # eigh for Hermitian matrices
    # Clamp negative eigenvalues to avoid numerical issues
    # this may actually be more accurate than using jax.scipy.linalg.sqrtm
    # since it can return complex numbers for negative eigenvalues
    # and we want to avoid that in the square root.
    eigenvals = jnp.where(eigenvals > 0, eigenvals, 0)
    sqrt_eigenvals = jnp.sqrt(eigenvals)
    return eigenvecs @ jnp.diag(sqrt_eigenvals) @ eigenvecs.conj().T


def is_positive_semi_definite(A, tol=1e-15):
    """
    Check if a matrix is positive semi-definite.

    Parameters
    ----------
    A : jnp.ndarray
        The matrix to check.
    tol : float, optional
        Tolerance for numerical stability, default is 1e-8.

    Returns
    -------
    bool
        True if A is positive semi-definite, False otherwise.
    """
    # Check if the matrix is Hermitian
    if not is_hermitian(A, tol):
        return False

    # Check if all eigenvalues are non-negative
    eigenvalues = jnp.linalg.eigvalsh(A)
    return jnp.all(eigenvalues >= -tol)


def is_hermitian(A, tol=1e-8):
    return jnp.allclose(A, A.conj().T, atol=tol)


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
    if isket(A) or isbra(A):
        if isket(B) or isbra(B):
            A = A / jnp.linalg.norm(A)
            B = B / jnp.linalg.norm(B)
            # The fidelity for pure states reduces to the modulus of their
            # inner product.
            return jnp.vdot(A, B)
        # Take advantage of the fact that the density operator for A
        # is a projector to avoid a sqrtm call.
        A = A / jnp.linalg.norm(A)
        sqrtmA = ket2dm(A)
    else:
        if isket(B) or isbra(B):
            # Swap the order so that we can take a more numerically
            # stable square root of B.
            return _state_density_fidelity(B, A)
        # If we made it here, both A and B are operators, so
        # we have to take the sqrtm of one of them.
        A = A / jnp.linalg.trace(A)
        B = B / jnp.linalg.trace(B)

        sqrtmA = sqrtm_eig(A)
        # sqrtmA = jax.scipy.linalg.sqrtm(A)

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


# TODO: add to docs: For the Same initial and target, the fidelity of density and state may differ slightly due to the ways in which they are computed.
def fidelity(*, C_target, U_final, evo_type="unitary"):
    """
    Computes the fidelity of the final state/operator/density matrix/liouvillian
    with respect to the target state/operator/density matrix/liouvillian.

    For calculating the fidelity of liouvillians, the tracediff method is used.
    The fidelity is calculated as:
    - For unitary: ``Tr(C_target^† U_final) / dim``
    - For state: ``|<C_target|U_final>|^2`` where ``C_target`` and ``U_final`` are normalized
    - For density: ``|<C_target|U_final>|^2`` where ``C_target`` and ``U_final`` are normalized
    - For liouvillian: ``1 - (0.5 * Tr(|C_target - U_final|)) / C_target.dim``

    Args:
        C_target: Target operator.
        U_final: Final operator after evolution.
        evo_type: Type of fidelity calculation ("unitary", "state", "density", or "liouvillian (using tracediff method)")
    Returns:
        fidelity: Fidelity value.
    """
    if evo_type == "liouvillian":
        # TRACEDIFF fidelity: 1 - 0.5*Tr(|C_target - U_final|)
        # Where |A| is the matrix absolute value (element-wise)
        diff = C_target - U_final
        # Alternative approach: use the trace of the absolute value directly
        trace_diff = 0.5 * jnp.abs(jnp.trace(diff))
        return 1.0 - trace_diff / C_target.shape[0]
    elif evo_type == "unitary":
        # Answer: check accuracy of this, do we really need vector conjugate or .dot will simply work? --> no vdot is essential because we need the first term conjugated
        norm_C_target = C_target / jnp.linalg.norm(C_target)
        norm_U_final = U_final / jnp.linalg.norm(U_final)
        # equivalent to Tr(C_target^† U_final)
        # overlap = jnp.trace(norm_C_target.conj().T @ norm_U_final)
        overlap = jnp.vdot(norm_C_target, norm_U_final)
    elif evo_type == "density" or evo_type == "state":
        # normalization occurs in the _state_density_fidelity function
        overlap = _state_density_fidelity(
            C_target,
            U_final,
        )
    else:
        raise ValueError(
            "Invalid evo_type. Choose 'unitary', 'state', 'density', 'liouvillian'."
        )
    return jnp.abs(overlap) ** 2
