import jax

from feedback_grape.utils.gates import cnot, hadamard
from feedback_grape.utils.operators import (
    create,
    destroy,
    identity,
    sigmam,
    sigmap,
    sigmax,
    sigmay,
    sigmaz,
)
from feedback_grape.utils.tensor import tensor

jax.config.update("jax_enable_x64", True)


def embed(op, idx, dims):
    """
    Embed operator `op` at position `idx` in total Hilbert space of `dims`.
    """

    ops = [op if i == idx else identity(d) for i, d in enumerate(dims)]
    return tensor(*ops)


class Qubit:
    def __init__(self, index, dims):
        self.index = index
        self.dims = (
            dims  # full Hilbert space dims (including qubits and cavities)
        )

    @property
    def identity(self):
        return embed(identity(2), self.index, self.dims)

    # Pauli operators
    @property
    def sigmax(self):
        return embed(sigmax(), self.index, self.dims)

    @property
    def sigmay(self):
        return embed(sigmay(), self.index, self.dims)

    @property
    def sigmaz(self):
        return embed(sigmaz(), self.index, self.dims)

    @property
    def sigmap(self):
        return embed(sigmap(), self.index, self.dims)

    @property
    def sigmam(self):
        return embed(sigmam(), self.index, self.dims)

    # gates
    @property
    def cnot(self):
        return embed(cnot(), self.index, self.dims)

    @property
    def hadamard(self):
        return embed(hadamard(), self.index, self.dims)


class Cavity:
    def __init__(self, index, dim, dims):
        self.index = index
        self.dim = dim
        self.dims = dims

    @property
    def identity(self):
        return embed(identity(self.dim), self.index, self.dims)

    @property
    def a(self):
        return embed(destroy(self.dim), self.index, self.dims)

    @property
    def adag(self):
        return embed(create(self.dim), self.index, self.dims)


class QubitCavity:
    def __init__(self, num_qubits: int, *cavity_dims: int):
        """

        Qubit comes first in the Hilbert space, followed by cavities.

        Args:
            num_qubits (int): Number of qubits.
            *cavity_dims (int): Dimensions of each cavity.
        """
        self.num_qubits = num_qubits
        self.num_cavities = len(cavity_dims)

        # Full dimension list: qubits first, then cavities
        dims = [2] * num_qubits + list(cavity_dims)

        # Build qubits
        self.qubits = [Qubit(i, dims) for i in range(num_qubits)]

        # Build cavities
        self.cavities = [
            Cavity(num_qubits + i, cavity_dims[i], dims)
            for i in range(self.num_cavities)
        ]
