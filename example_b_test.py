# B. State purification with qubit-mediated measurement
# ruff: noqa
from feedback_grape.fgrape_parameterized import optimize_pulse_with_feedback
import jax.numpy as jnp
from feedback_grape.utils.operators import cosm, sinm
from feedback_grape.utils.operators import create, destroy
import jax

# initial state is a thermal state
n_average = 2
N_cavity = 30
# natural logarithm
beta = jnp.log((1 / n_average) + 1)
diags = jnp.exp(-beta * jnp.arange(N_cavity))
normalized_diags = diags / jnp.sum(diags, axis=0)
rho_cav = jnp.diag(normalized_diags)


# TODO: try to do it as a matrix exponentiation (cos of operator)
# try to see if off diagonal is not 0
def povm_measure_operator(measurement_outcome, gamma, delta):
    """
    POVM for the measurement of the cavity state.
    returns Mm ( NOT the POVM element Em = Mm_dag @ Mm ), given measurement_outcome m, gamma and delta
    """
    number_operator = create(N_cavity) @ destroy(N_cavity)
    angle = (gamma * number_operator) + delta / 2
    return jnp.where(
        measurement_outcome == 1,
        cosm(angle),
        sinm(angle),
    )


# TODO: Have a default NN and then give user the ability to supply a model or a function
if __name__ == "__main__":
    result = optimize_pulse_with_feedback(
        U_0=rho_cav,
        C_target=None,
        parameterized_gates=[],
        povm_measure_operator=povm_measure_operator,
        initial_params=jnp.array([[20.0, -10.0]]).reshape((2, 1)),
        num_time_steps=5,
        mode="nn",
        goal="purity",
        optimizer="l-bfgs",
        max_iter=1000,
        convergence_threshold=1e-6,
        learning_rate=0.01,
        type="density",
    )

    from feedback_grape.fgrape_parameterized import purity

    print("initial purity:", purity(rho=rho_cav, type="density"))
    print("Final purity:", purity(rho=result.final_state, type="density"))
    print("result ", result)
    print("povm_params: ", result.arr_of_povm_params)
    # Extract POVM parameters from the result
    povm_params = result.arr_of_povm_params
    # Initialize the state
    current_state = rho_cav

    # Apply POVM operators for 5 time steps
    for t in range(5):
        gamma, delta = povm_params[t]
        variables = {"gamma": gamma, "delta": delta}
        povm_operator = povm_measure_operator(
            -1, **variables
        )  # Example with measurement outcome -1
        current_state = povm_operator @ current_state @ povm_operator.T
        current_state /= jnp.trace(current_state)  # Normalize the state

    # Calculate the resulting purity
    final_purity = purity(rho=current_state, type="density")
    print("Resulting purity after 5 time steps:", final_purity)
    ### Check stash for replacement of dict implementation
