import jax
import optax  # type: ignore
import optax.tree_utils as otu  # type: ignore
# ruff: noqa N8


# only difference is that this one uses kayes for each time step
def _optimize_adam_feedback(
    loss_fn,
    control_amplitudes,
    max_iter,
    learning_rate,
    convergence_threshold,
    key,
    progress,
    early_stop,
):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(control_amplitudes)
    losses = []

    @jax.jit
    def step(params, state, key):
        loss = loss_fn(params, key)  # Minimize -loss_fn
        grads = jax.grad(lambda x, k: loss_fn(x, k))(params, key)
        updates, new_state = optimizer.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        new_key, _ = jax.random.split(key)
        return new_params, new_state, loss, new_key

    params = control_amplitudes
    # setting it to -1 in the beginning in case the max_iter is 0
    iter_idx = -1
    for iter_idx in range(max_iter):
        params, opt_state, loss, key = step(params, opt_state, key)
        losses.append(loss)
        if early_stop:
            if (
                iter_idx > 0
                and abs(losses[-1] - losses[-2]) < convergence_threshold
            ):
                break

        if progress:
            if iter_idx % 10 == 0:
                print(f"Iteration {iter_idx}, Loss: {loss:.6f}")

    return params, iter_idx + 1


# TODO: Throw a warning if the user uses complex parameters with L-BFGS or Adam, since they are not optimized for complex numbers
def _optimize_adam(
    loss_fn,
    control_amplitudes,
    max_iter,
    learning_rate,
    convergence_threshold,
    progress,
    early_stop,
):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(control_amplitudes)
    losses = []

    @jax.jit
    def step(params, state):
        loss = loss_fn(params)
        grads = jax.grad(lambda x: loss_fn(x))(params)
        updates, new_state = optimizer.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss

    params = control_amplitudes
    # setting it to -1 in the beginning in case the max_iter is 0
    iter_idx = -1
    for iter_idx in range(max_iter):
        params, opt_state, loss = step(params, opt_state)

        losses.append(loss)
        if early_stop:
            if (
                iter_idx > 0
                and abs(losses[-1] - losses[-2]) < convergence_threshold
            ):
                break
        if progress:
            if iter_idx % 10 == 0:
                print(f"Iteration {iter_idx}, Loss: {loss:.6f}")

    return params, iter_idx + 1


# Answer: L_bfgs ouputs error when params are complex amplitudes --> yeah both won't work with complex parameters
# user needs to use two real parameters per complex number and then in his function convert them to complex
def _optimize_L_BFGS(
    loss_fn,
    control_amplitudes,
    max_iter,
    convergence_threshold,
    learning_rate,
    progress,
    early_stop,
):
    """
    Uses L-BFGS to optimize the control amplitudes.
    Args:
        loss_fn: loss function to optimize.
        control_amplitudes: Initial control amplitudes.
        max_iter: Maximum number of iterations.
        convergence_threshold: Convergence threshold for optimization.
    Returns:
        control_amplitudes: Optimized control amplitudes.
        final_iter_idx: Number of iterations in the optimization.
    """

    opt = optax.lbfgs(learning_rate)

    value_and_grad_fn = optax.value_and_grad_from_state(loss_fn)

    @jax.jit
    def step(carry):
        control_amplitudes, state, iter_idx = carry
        value, grad = value_and_grad_fn(control_amplitudes, state=state)
        updates, state = opt.update(
            grad,
            state,
            control_amplitudes,
            value=value,
            grad=grad,
            value_fn=loss_fn,
        )
        control_amplitudes = optax.apply_updates(control_amplitudes, updates)
        jax.lax.cond(
            jax.numpy.logical_and(progress, iter_idx % 10 == 0),
            lambda: jax.debug.print("Iteration {iter_idx}, Loss: {value:.6f}", iter_idx=iter_idx, value=value),
            lambda: None
        )
        return control_amplitudes, state, iter_idx + 1

    def continuing_criterion(carry):
        _, state, _ = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_l2_norm(grad)
        import jax.numpy as jnp

        return jnp.logical_or(
            jnp.logical_and(iter_num == 0, max_iter != 0),
            jnp.logical_and(
                iter_num < max_iter,
                jnp.logical_or(
                    err >= convergence_threshold, jnp.logical_not(early_stop)
                ),
            ),
        )

    init_carry = (control_amplitudes, opt.init(control_amplitudes), 0)
    final_params, _, final_iter_idx = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_params, final_iter_idx
