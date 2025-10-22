# ruff: noqa
import sys, os
sys.path.append(os.path.abspath("./../feedback-grape"))
sys.path.append(os.path.abspath("./../"))

# ruff: noqa
from feedback_grape.fgrape import Decay, Gate # type: ignore
from feedback_grape.utils.states import basis # type: ignore
from feedback_grape.utils.tensor import tensor # type: ignore
from feedback_grape.utils.operators import sigmap, sigmam, identity # type: ignore
import jax.numpy as jnp
import jax
from feedback_grape.utils.fidelity import ket2dm # type: ignore
from jax.scipy.linalg import expm

jax.config.update("jax_enable_x64", True)

# Helper
def embed(op: jnp.ndarray, j: int, n: int) -> jnp.ndarray:
    """
    Embed a single-qubit operator `op` into an `n`-qubit Hilbert space at position `j`.

    Parameters:
    op (np.ndarray): The single-qubit operator to embed.
    j (int): The position (0-indexed) to embed the operator.
    n (int): The total number of qubits.

    Returns:
    np.ndarray: The embedded operator in the n-qubit Hilbert space.
    """
    if j < 0 or j >= n:
        raise ValueError("Index j must be in the range [0, n-1].")
    if op.shape != (2, 2):
        raise ValueError("Operator op must be a 2x2 matrix.")
    

    ops = [identity(2)] * n
    ops[j] = op
    return tensor(*ops)

# All the operators we need
def generate_traceless_hermitian(params, dim):
    assert len(params) == dim**2 - 1, "Number of real parameters must be dim^2 - 1 for an NxN traceless Hermitian matrix."
    
    # Read the first (dim**2 - dim) / 2 as the real parts of the upper triangle
    real_parts = jnp.array(params[: (dim**2 - dim) // 2])

    # Read the next (dim**2 - dim) / 2 as the imaginary parts of the upper triangle
    imag_parts = jnp.array(params[(dim**2 - dim) // 2 : - (dim - 1)])

    # Read the last (dim - 1) as the diagonal elements, set the last diagonal element to ensure tracelessness
    trace = sum(params[- (dim - 1):])
    diag_parts = jnp.append(params[- (dim - 1):], jnp.array([-trace]))

    # Construct the Hermitian matrix
    triag_parts = real_parts + 1j * imag_parts

    return jnp.array([
        [
            diag_parts[i] if i == j else
            triag_parts[(i * (i - 1)) // 2 + j - i - 1] if i < j else
            jnp.conj(triag_parts[(j * (j - 1)) // 2 + i - j - 1])
            for j in range(dim)
        ] for i in range(dim)
    ])
generate_traceless_hermitian = jax.jit(generate_traceless_hermitian, static_argnames=['dim'])

def generate_hermitian(params, dim):
    assert len(params) == dim**2, "Number of real parameters must be dim^2 for an NxN Hermitian matrix."
    
    # Generate traceless hermitanian from first dim^2 - 1 parameters and read last parameter as trace
    return generate_traceless_hermitian(params[:-1], dim) + jnp.eye(dim) * params[-1] / dim
generate_hermitian = jax.jit(generate_hermitian, static_argnames=['dim'])

def generate_unitary(params, dim):
    assert len(params) == dim**2, "Number of real parameters must be dim^2 for an NxN unitary matrix."

    H = generate_hermitian(params, dim)
    return jax.scipy.linalg.expm(-1j * H)
generate_unitary = jax.jit(generate_unitary, static_argnames=['dim'])

def generate_special_unitary(params, dim):
    assert len(params) == dim**2 - 1, "Number of real parameters must be dim^2 - 1 for an NxN special unitary matrix."
    
    H = generate_traceless_hermitian(params, dim)
    return jax.scipy.linalg.expm(-1j * H)
generate_special_unitary = jax.jit(generate_special_unitary, static_argnames=['dim'])

def partial_trace(rho, sys_A_dim, sys_B_dim):
    """ Compute the partial trace over system A of a density matrix rho = rho_AB.
        sys_A_dim: Dimension of system A
        sys_B_dim: Dimension of system B
    """
    dim_A = sys_A_dim
    dim_B = sys_B_dim
    #assert rho.shape == (dim_A * dim_B, dim_A * dim_B), "Input density matrix has incorrect dimensions."

    rho_B = jnp.zeros((dim_B, dim_B), dtype=rho.dtype)

    def loop_body(i, rho_B):
        return rho_B + jax.lax.dynamic_slice(rho, (i*dim_B, i*dim_B), (dim_B, dim_B))
    
    rho_B = jax.lax.fori_loop(0, dim_A, loop_body, rho_B) # Compiler friendly loop

    return rho_B
partial_trace = jax.jit(partial_trace, static_argnames=['sys_A_dim', 'sys_B_dim'])

def generate_povm1(measurement_outcome, params):
    """ 
        Generate a 2-outcome POVM elements M_0 and M_1 for a qubit system.
        This function should parametrize all such POVMs up to unitary equivalence, i.e., M_i -> U M_i for some unitary U.
        I.e it parametrizes all pairs (M_0, M_1) such that M_0 M_0† + M_1 M_1† = I.

        measurement_outcome: 0 or 1, indicating which POVM element to generate.
        params: list of 4 real parameters [phi, theta, alpha, beta].

        when measurement_outcome == 1:
            M_1 = S D S†
        when measurement_outcome == -1:
            M_0 = S (I - D) S†

        phi, theta parametrize the unitary S, and alpha, beta parametrize the eigenvalues of M_1.
    """
    phi, theta, alpha, beta = params
    S = jnp.array(
        [[jnp.cos(phi),                   -jnp.sin(phi)*jnp.exp(-1j*theta)],
         [jnp.sin(phi)*jnp.exp(1j*theta),  jnp.cos(phi)                  ]]
    )
    s1 = jnp.sin(alpha)**2
    s2 = jnp.sin(beta)**2
    D_0 = jnp.array(
        [[s1, 0],
         [0,  s2]]
    )
    D_1 = jnp.array(
        [[(1 - s1*s1)**0.5, 0],
         [0, (1 - s2*s2)**0.5]]
    )

    return jnp.where(measurement_outcome == 1,
        tensor(identity(2), S @ D_0 @ S.conj().T),
        tensor(identity(2), S @ D_1 @ S.conj().T)
    )

def generate_povm2(measurement_outcome, params, dim):
    """ 
        Generate a 2-outcome POVM elements M_0 and M_1 for a system with Hilbert space dimension dim.
        This function should parametrize all such POVMs up to unitary equivalence, i.e., M_i -> U M_i for some unitary U.
        I.e it parametrizes all pairs (M_0, M_1) such that M_0 M_0† + M_1 M_1† = I.

        measurement_outcome: 0 or 1, indicating which POVM element to generate.
        params: list of dim^2 real parameters.

        when measurement_outcome == 1:
            M_1 = S D S†
        when measurement_outcome == -1:
            M_0 = S (I - D) S†

        where S is a unitary parametrized by dim^2 parameters, and D is a diagonal matrix with eigenvalues parametrized by dim parameters.
    """
    S = generate_unitary(params, dim=dim) # All parameters for unitary

    d_vec = jnp.astype(jnp.sin( params[dim*(dim-1):dim*dim] ) ** 2, jnp.complex128) # Last #dim parameters for eigenvalues
    #d_vec = 1e-6 + (1 - 2e-6) * d_vec # Avoid exactly 0 or 1 eigenvalues

    return jnp.where(measurement_outcome == 1,
        S @ jnp.diag(d_vec) @ S.conj().T,
        S @ jnp.diag(jnp.sqrt(1 - d_vec**2)) @ S.conj().T
    )
generate_povm2 = jax.jit(generate_povm2, static_argnames=['dim'])

def initialize_chain_of_zeros(rho, n, N_chains):
    """ Initialize double chain of qubits where the first pair is in state rho and the rest in |0><0|. """
    dim = 2**(N_chains*(n - 1))
    rho_zero = jnp.zeros((dim, dim), dtype=rho.dtype)
    rho_zero = rho_zero.at[-1, -1].set(1.0)

    return tensor(rho, rho_zero)
initialize_chain_of_zeros = jax.jit(initialize_chain_of_zeros, static_argnames=['n', 'N_chains'])

def transport_unitary(frac, n, N_chains):
    M = n*N_chains # Number of qubits in total

    if n == 2:
        tau = jnp.pi/2 # fixed time for each transport unitary (cf. paper)
        J = jnp.ones((n - 1,)) # uniform coupling strengths
    elif n >= 3:
        tau = jnp.pi/2 # fixed time for each transport unitary (cf. paper)
        J = jnp.array([(j*(n-j))**0.5 for j in range(1,n)])
    else:
        raise NotImplementedError("Transport unitary is only implemented for chains of length n>=2.")

    H_I = jnp.zeros((2**M, 2**M), dtype=jnp.complex128)
    for s in range(N_chains):
        for j,J_j in enumerate(J):
            idx = N_chains*j + s
            H_I = H_I + J_j*embed(sigmap(), idx, M)@embed(sigmam(), idx + N_chains, M)

    # For n >= 3:
    # Need another phase of i^(n-1) for basis vectors with one spin up. (c.f. eq 15 in [1])
    # Hence we add a term with eigenvalue -1 for all those vectors to Hamiltonian.
    # When evolving for time tau = pi/2, this adds a exp(-1j*pi/2*(-1)) = i.
    # We multiply by ((n - 1) % 4), to get the desired i^(n-1) term.
    # For n = 2, we just need a sign flip.
    H_phase = jnp.zeros((2**M, 2**M), dtype=jnp.complex128)
    for s in range(N_chains):
        for j in range(n):
            idx = N_chains*j + s
            H_phase = H_phase + embed(jnp.array([[1,0],[0,0]]), idx, M)

    if n == 2:
        H_phase = - H_phase
    else:
        H_phase = - H_phase * (n - 1)


    H_I = H_I + H_I.conj().T
    H_I = H_I + H_phase
    return expm(-1j*frac*tau*H_I)

# Functions which initialize gates
def init_first_gate(n, N_chains):
    initial_gate = Gate(
        gate=lambda rho, _: initialize_chain_of_zeros(rho, n=n, N_chains=N_chains),
        initial_params = jnp.array([]),
        measurement_flag = False,
        quantum_channel_flag = True
    )

    return initial_gate

def init_T_half_gate(n, N_chains):
    T_half = transport_unitary(0.5, n, N_chains)
    
    T_gate = Gate(
        gate=lambda _: T_half,
        initial_params = jnp.array([]),
        measurement_flag = False
    )

    return T_gate

def init_decay_gate(n, N_chains, gamma):
    decay_gate = Decay(
        c_ops = [sum([gamma * embed(sigmam(), idx, N_chains*n) for idx in range(N_chains*n)])], # dissipation on all qubits
    )

    return decay_gate

def init_ptrace_gate(n, N_chains):
    base_dim = 2**N_chains
    ptrace_gate = Gate(
        gate=lambda rho, _: partial_trace(rho, sys_A_dim=base_dim**(n-1), sys_B_dim=base_dim),
        initial_params = jnp.array([]),
        measurement_flag = False,
        quantum_channel_flag = True
    )

    return ptrace_gate

def init_povm_gate(key, N_chains):
    base_dim = 2**N_chains
    N_povm_params = base_dim**2
    povm_gate = Gate(
        gate=lambda msmt, params: generate_povm2(msmt, params, dim=base_dim),
        initial_params = jax.random.uniform(key, (N_povm_params,), minval=0.0, maxval=2*jnp.pi),
        measurement_flag = True
    )

    return povm_gate

def init_unitary_gate(key, N_chains):
    base_dim = 2**N_chains
    N_unitary_params = base_dim**2
    U_gate = Gate(
        gate=lambda params: generate_unitary(params, dim=base_dim),
        initial_params = jax.random.uniform(key, (N_unitary_params,), minval=0.0, maxval=1.0),
        measurement_flag = False
    )

    return U_gate

# Functions which initialize gate combinations for the protocols
def init_simple_protocol(n, N_chains, gamma):
    first_gate = init_first_gate(n, N_chains)
    T_half_gate = init_T_half_gate(n, N_chains)
    ptrace_gate = init_ptrace_gate(n, N_chains)
    decay_gate = init_decay_gate(n, N_chains, gamma)

    return [first_gate, T_half_gate, decay_gate, T_half_gate, ptrace_gate]

def init_grape_protocol(key, n, N_chains, gamma):
    subkey1, subkey2 = jax.random.split(key, 2)

    first_gate = init_first_gate(n, N_chains)
    T_half_gate = init_T_half_gate(n, N_chains)
    ptrace_gate = init_ptrace_gate(n, N_chains)
    decay_gate = init_decay_gate(n, N_chains, gamma)
    U_gate = init_unitary_gate(subkey1, N_chains)

    return [first_gate, T_half_gate, decay_gate, T_half_gate, ptrace_gate, U_gate]

def init_fgrape_protocol(key, n, N_chains, gamma):
    subkey1, subkey2 = jax.random.split(key, 2)

    first_gate = init_first_gate(n, N_chains)
    T_half_gate = init_T_half_gate(n, N_chains)
    ptrace_gate = init_ptrace_gate(n, N_chains)
    decay_gate = init_decay_gate(n, N_chains, gamma)
    povm_gate = init_povm_gate(subkey1, N_chains)
    U_gate = init_unitary_gate(subkey2, N_chains)

    return [first_gate, T_half_gate, decay_gate, T_half_gate, ptrace_gate, povm_gate, U_gate]

# Function to generate random initial states
def generate_random_state(key, N_chains):
    """ Generate a pair of up or down states with equal probability. """
    random_value = jax.random.uniform(key, minval=0.0, maxval=1.0)

    psi_one  = basis(2, 0)
    psi_zero = basis(2, 1)
    psi = jnp.where(random_value < 0.5, psi_one, psi_zero)

    return ket2dm(tensor(*([psi]*N_chains)))
generate_random_state = jax.jit(generate_random_state, static_argnames=['N_chains'])

# Function to generate initial states
def generate_all_states(N_chains):
    psi_one  = basis(2, 0)
    psi_zero = basis(2, 1)

    return [ket2dm(tensor(*([psi_zero]*N_chains))), ket2dm(tensor(*([psi_one]*N_chains)))]

# Tests for the implementations
def test_implementations():
    # Test unitary and special unitary generators
    for i in range(10):
        key = jax.random.PRNGKey(i)
        key, subkey1, subkey2 = jax.random.split(key, 3)

        params = jax.random.uniform(subkey1, (16,), minval=0.0, maxval=2*jnp.pi)

        U = generate_unitary(params, 4)
        SU = generate_special_unitary(params[:-1], 4)

        assert jnp.allclose(U @ U.conj().T, jnp.eye(4)), "Unitary condition failed"
        assert jnp.allclose(SU @ SU.conj().T, jnp.eye(4)), "Special Unitary condition failed"
        assert jnp.isclose(jnp.linalg.det(SU), 1.0), "Determinant condition for Special Unitary failed"

    # Test that partial trace works correctly
    for i in range(10):
        key = jax.random.PRNGKey(i)
        key, subkey1, subkey2 = jax.random.split(key, 3)

        rho_A = jax.random.normal(subkey1, (4,4)) + 1j * jax.random.normal(subkey1, (4,4))
        rho_A = rho_A @ rho_A.conj().T
        rho_A = rho_A / jnp.trace(rho_A)

        rho_B = jax.random.normal(subkey2, (4,4)) + 1j * jax.random.normal(subkey2, (4,4))
        rho_B = rho_B @ rho_B.conj().T
        rho_B = rho_B / jnp.trace(rho_B)

        rho_AB = tensor(rho_A, rho_B)

        traced_rho_B = partial_trace(rho_AB, 4, 4)

        assert jnp.allclose(traced_rho_B, rho_B), "Partial trace did not return correct result"

    # Test povm generator
    for i in range(10):
        key = jax.random.PRNGKey(i)
        key, subkey = jax.random.split(key, 2)

        for f,N_params in [(generate_povm1, 4), (lambda msmt, params: generate_povm2(msmt, params, 4), 16)]:
            params = jax.random.uniform(subkey, (N_params,), minval=0.0, maxval=2*jnp.pi)

            M_0 = f(-1, params)
            M_1 = f(1, params)

            assert jnp.allclose(M_0 @ M_0.conj().T + M_1 @ M_1.conj().T, jnp.eye(4)), "POVM elements do not sum to identity"
            assert jnp.allclose(M_0, M_0.conj().T), "POVM element M_0 is not Hermitian"
            assert jnp.allclose(M_1, M_1.conj().T), "POVM element M_1 is not Hermitian"
            assert jnp.all(jnp.linalg.eigvals(M_0) >= 0), "POVM element M_0 is not positive semidefinite"
            assert jnp.all(jnp.linalg.eigvals(M_1) >= 0), "POVM element M_1 is not positive semidefinite"

    # Test transport unitary
    for n in [2,3,4]:
        for N_chains in [1,2,3]:
            if n*N_chains > 8:
                continue # Skip too large systems for testing

            U_arr = [
                transport_unitary(1.0, n, N_chains), # single full transport
                transport_unitary(0.5, n, N_chains)@transport_unitary(0.5, n, N_chains), # two half-transports
            ]

            psi_0 = tensor(*[basis(2, 1)]*(n*N_chains)) # all spins down
            psi_1 = tensor(*[basis(2, 0)]*(N_chains) + [basis(2, 1)]*((n-1)*N_chains)) # first spins up, rest down
            psi_2 = tensor(*[basis(2, 1)]*((n-1)*N_chains) + [basis(2, 0)]*(N_chains)) # last spins up, rest down

            for U in U_arr:
                assert jnp.allclose(U @ U.conj().T, jnp.eye(2**(n*N_chains))), "Transport unitary condition failed"
                assert jnp.allclose(U @ psi_0, psi_0), "Transport unitary did not preserve ground state"
                assert jnp.allclose(U @ psi_1, psi_2), "Transport unitary did not transport first spin to last position correctly"