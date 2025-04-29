from utils.operators import (
    create,
    destroy,
    identity,
    sigmap,
    sigmax,
    sigmay,
    sigmaz,
)
from utils.tensor import tensor


class _Qubit:
    sigmax = None
    sigmay = None
    sigmaz = None
    identity = None
    dim_left = None
    dim_right = None

    def __init__(self, dim_left, dim_right):
        """
        Qubit constructor. Initialixes the Pauli operators and identity
        operator for a qubit.

        Args:
            dim_left (int): Dimension of the left side.
            dim_right (int): Dimension of the right side.
        """
        self.dim_left = dim_left
        self.dim_right = dim_right
        self.sigmax = tensor(
            identity(2**dim_left), sigmax(), identity(2**dim_right)
        )
        self.sigmay = tensor(
            identity(2**dim_left), sigmay(), identity(2**dim_right)
        )
        self.sigmaz = tensor(
            identity(2**dim_left), sigmaz(), identity(2**dim_right)
        )
        self.sigmap = tensor(
            identity(2**dim_left), sigmap(), identity(2**dim_right)
        )
        self.identity = tensor(
            identity(2**dim_left), identity(2), identity(2**dim_right)
        )


class _QubitRegister:
    qubits = None
    num_of_qubits = None

    def __init__(self, num_of_qubits, qubits):
        """
        QubitRegister constructor.
        Args:
            num_of_qubits (int): Number of qubits.
            qubits (list): List of Qubit objects.
        """
        self.num_of_qubits = num_of_qubits
        self.qubits = qubits

    def __getitem__(self, index):
        return self.qubits[index]


class Cavity:
    destroy = None
    create = None
    dim = None

    def __init__(self, dim: int):
        """
        Cavity constructor.

        Args:
            dim (int): Dimension of the cavity.
        """
        self.dim = dim
        self.destroy = destroy(dim)
        self.create = create(dim)


class HilbertSpace:
    qubit_register = None
    cavities = None
    num_of_cavities = None

    def __init__(self, num_of_qubits: int, *cavities: Cavity):
        """
        HilbertSpace constructor.

        Args:
            num_of_qubits (int): Number of qubits.
            *cavities (Cavity): Cavity objects.
        """
        self.qubit_register = _QubitRegister(
            num_of_qubits,
            [_Qubit(i, num_of_qubits - i - 1) for i in range(num_of_qubits)],
        )
        self.cavities = cavities
        self.num_of_cavities = len(cavities)


if __name__ == "__main__":
    # Example usage
    g = 0.1  # Coupling strength
    hilbert_space = HilbertSpace(2, Cavity(4), Cavity(4))
    q_reg = hilbert_space.qubit_register
    cav1 = hilbert_space.cavities[0]
    H = g * cav1.destroy * q_reg[1].identity
    print("H: using the interface: \n", H)

    ##############################
