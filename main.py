import jax.numpy as jnp

from feedback_grape.grape import optimize_pulse
from feedback_grape.utils.gates import cnot
from feedback_grape.utils.operators import identity, sigmax, sigmay, sigmaz
from feedback_grape.utils.tensor import tensor

if __name__ == "__main__":
    # Example usage
    g = 0  # Small coupling strength
    H_drift = g * (tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()))
    H_ctrl = [
        tensor(sigmax(), identity(2)),
        tensor(sigmay(), identity(2)),
        tensor(sigmaz(), identity(2)),
        tensor(identity(2), sigmax()),
        tensor(identity(2), sigmay()),
        tensor(identity(2), sigmaz()),
        tensor(sigmax(), sigmax()),
        tensor(sigmay(), sigmay()),
        tensor(sigmaz(), sigmaz()),
    ]

    U_0 = identity(4)
    # Target operator (CNOT gate)
    C_target = cnot()

    num_t_slots = 500
    total_evo_time = 2 * jnp.pi

    # Run optimization
    result = optimize_pulse(
        H_drift,
        H_ctrl,
        U_0,
        C_target,
        num_t_slots,
        total_evo_time,
        max_iter=500,
        learning_rate=1e-2,
    )

    print("final_fidelity: ", result.final_fidelity)
    print("U_f \n", result.final_operator)
