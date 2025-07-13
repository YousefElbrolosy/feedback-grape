import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


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


def fidelity(*, C_target, U_final, evo_type="unitary"):
    r"""
    Computes the fidelity between the final and target state/operator/density matrix.

    Parameters
    ----------
    C_target : jnp.ndarray
        Target operator, state, or density matrix.
    U_final : jnp.ndarray
        Final operator, state, or density matrix after evolution.
    evo_type : str, optional
        Type of fidelity calculation. Must be one of:

        - "unitary": Uses normalized overlap for operators.
        - "state": Uses normalized overlap for state vectors.
        - "density": Uses Uhlmann fidelity for density matrices.

    Returns
    -------
    fidelity : float
        Fidelity value in [0, 1].

    Notes
    -----
    - For ``evo_type="unitary"`` or ``"state"``, fidelity is the squared magnitude of the normalized overlap:
        - :math:`|\langle C_\mathrm{target} | U_\mathrm{final} \rangle|^2`

    - For ``evo_type="density"``, fidelity is computed using the Uhlmann formula:
        - :math:`\left(\mathrm{Tr}\left[\sqrt{\sqrt{\rho}\, \sigma\, \sqrt{\rho}}\right]\right)^2` where :math:`\rho` and :math:`\sigma` are density matrices.

    - Note that, for the same initial and target states/density matrices, the fidelity of state and density may differ slightly due to the computation method.

    """
    if evo_type == "unitary":
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
            "Invalid evo_type. Choose 'unitary', 'state', 'density'."
        )
    return jnp.abs(overlap) ** 2
