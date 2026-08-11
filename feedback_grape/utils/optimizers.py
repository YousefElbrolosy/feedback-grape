import warnings

import jax
import jax.numpy as jnp
import optax  # type: ignore
import optax.tree_utils as otu  # type: ignore
from time import time
# ruff: noqa N8

jax.config.update("jax_enable_x64", True)


# only difference is that this one uses kayes for each time step
def optimize_adam_feedback(
    loss_fn,
    control_amplitudes,
    max_iter,
    learning_rate,
    convergence_threshold,
    key,
    progress,
    early_stop,
    window=50,
    patience=3,
):
    """

    Uses Adam optimizer to optimize the control amplitudes.

    The reward is a stochastic estimate (it is averaged over ``batch_size``
    sampled trajectories), so consecutive rewards differ by an amount that does
    not decay as the optimization converges. Convergence is therefore measured
    by comparing the mean reward over the last ``window`` iterations against the
    mean over the ``window`` iterations before those. Averaging suppresses the
    sampling noise, and keeping the two windows disjoint stops the comparison
    from collapsing back into a single-sample difference.

    Args:
        loss_fn: loss function to optimize.
        control_amplitudes: Initial control amplitudes.
        max_iter: Maximum number of iterations.
        learning_rate: Learning rate for the optimizer.
        convergence_threshold: Smallest improvement in mean reward, over
            ``window`` iterations, that is still considered progress. Must be
            set above the noise floor of the reward (roughly
            ``std(reward) * sqrt(2 / window)`` once the reward has plateaued) or
            it can never be reached.
        key: JAX random key for stochastic operations (so that each iteration has is different).
        progress: If True, prints the progress of the optimization.
        early_stop: If True, stops the optimization once the reward has stopped
            improving by more than ``convergence_threshold``.
        window: Number of iterations averaged into each of the two compared
            windows. Larger values suppress more noise but delay the earliest
            possible stop, which is ``2 * window`` iterations.
        patience: Number of consecutive comparisons that must fall below
            ``convergence_threshold`` before stopping. Guards against a single
            comparison landing low by chance.
    Returns:
        control_amplitudes: Optimized control amplitudes.
        final_iter_idx: Number of iterations in the optimization.
        reward_history: Reward (e.g. mean fidelity/purity, in [0, 1]) at each
            iteration, as returned by the auxiliary output of ``loss_fn``.
    """
    if window < 2:
        raise ValueError("window must be at least 2.")
    if patience < 1:
        raise ValueError("patience must be at least 1.")
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(control_amplitudes)
    losses = []
    rewards = []

    @jax.jit
    def step(params, state, key):
        (loss, reward), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, key
        )
        updates, new_state = optimizer.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        new_key, _ = jax.random.split(key)
        return new_params, new_state, loss, reward, new_key

    params = control_amplitudes
    # setting it to -1 in the beginning in case the max_iter is 0
    iter_idx = -1
    # number of consecutive window comparisons that showed no real improvement
    stall_count = 0
    converged = False
    nan_detected = False
    for iter_idx in range(max_iter):
        new_params, new_opt_state, loss, reward, key = step(
            params, opt_state, key
        )
        losses.append(loss)
        rewards.append(reward)

        nan_detected = any(
            [
                jnp.any(jnp.isnan(p))
                for p in jax.tree_util.tree_leaves(new_params)
            ]
        )
        if nan_detected:
            print(
                f"Warning: NaN values detected in updated parameters at iteration {iter_idx}. Stopping optimization."
            )
            print(
                f"Info: NaN values may occur due to high learning rates or POVM elements with zero eigenvalues."
            )
            break
        else:
            params = new_params
            opt_state = new_opt_state

        if progress:
            if iter_idx == 0:
                start_time = time()  # Start clock after first iteration which initializes compiled functions
            if iter_idx % 10 == 0 and iter_idx > 0:
                print(
                    f"Iteration {iter_idx}, Reward: {reward:.6f}, Loss: {loss:.6f}, T={int(time() - start_time)}s, eta={int((max_iter - (iter_idx - 1)) / (iter_idx + 1) * (time() - start_time))}s"
                )

        # Compared windows are disjoint: overlapping ones share all but two
        # samples, which reduces the comparison to a single-sample difference
        # and reintroduces the noise the averaging is meant to remove.
        if early_stop and len(rewards) >= 2 * window:
            recent = jnp.mean(jnp.asarray(rewards[-window:]))
            prior = jnp.mean(jnp.asarray(rewards[-2 * window : -window]))
            # Signed, so that a collapse in reward is not read as convergence.
            if recent - prior < convergence_threshold:
                stall_count += 1
            else:
                stall_count = 0
            if stall_count >= patience:
                converged = True
                break

    if early_stop and not converged and not nan_detected:
        _warn_if_threshold_unreachable(rewards, convergence_threshold, window)

    return params, iter_idx + 1, jnp.array(rewards)


def _warn_if_threshold_unreachable(rewards, convergence_threshold, window):
    """
    Warn when early stopping was asked for but could not have happened.

    A ``convergence_threshold`` far below the spread of the reward is never
    reached, so the optimization silently runs to ``max_iter``. Comparing it
    against the standard deviation of the window difference makes that visible.
    """
    if len(rewards) < 2 * window:
        return
    tail = jnp.asarray(rewards[-2 * window :])
    noise_floor = float(jnp.std(tail)) * (2.0 / window) ** 0.5
    if convergence_threshold < 0.1 * noise_floor:
        warnings.warn(
            f"early_stop was requested but never triggered: "
            f"convergence_threshold={convergence_threshold:g} is far below the "
            f"observed reward noise floor (~{noise_floor:.2g}). Consider a "
            f"threshold above that value, or increase batch_size or window to "
            f"lower the noise floor.",
            stacklevel=3,
        )


def optimize_adam(
    loss_fn,
    control_amplitudes,
    max_iter,
    learning_rate,
    convergence_threshold,
    progress,
    early_stop,
):
    """

    Uses Adam optimizer to optimize the control amplitudes.
    No stachasticity is used between iterations.

    Args:
        loss_fn: loss function to optimize.
        control_amplitudes: Initial control amplitudes.
        max_iter: Maximum number of iterations.
        learning_rate: Learning rate for the optimizer.
        convergence_threshold: Convergence threshold for optimization.
        progress: If True, prints the progress of the optimization.
        early_stop: If True, stops the optimization if the loss does not change significantly (if convergence threshold is reached).
    Returns:
        control_amplitudes: Optimized control amplitudes.
        final_iter_idx: Number of iterations in the optimization.
        reward_history: Reward (e.g. fidelity, in [0, 1]) at each iteration, as
            returned by the auxiliary output of ``loss_fn``.
    """
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(control_amplitudes)
    losses = []
    rewards = []

    @jax.jit
    def step(params, state):
        (loss, reward), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params
        )
        updates, new_state = optimizer.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss, reward

    params = control_amplitudes
    # setting it to -1 in the beginning in case the max_iter is 0
    iter_idx = -1
    for iter_idx in range(max_iter):
        params, opt_state, loss, reward = step(params, opt_state)

        losses.append(loss)
        rewards.append(reward)
        if early_stop:
            if (
                iter_idx > 0
                and abs(losses[-1] - losses[-2]) < convergence_threshold
            ):
                break

        if progress:
            if iter_idx == 0:
                start_time = time()  # Start clock after first iteration which initializes compiled functions
            if iter_idx % 10 == 0 and iter_idx > 0:
                print(
                    f"Iteration {iter_idx}, Reward: {reward:.6f}, Loss: {loss:.6f}, T={int(start_time - time())}s, eta={int((max_iter - (iter_idx - 1)) / (iter_idx + 1) * (start_time - time()))}s"
                )

    return params, iter_idx + 1, jax.numpy.array(rewards)


# Answer: L_bfgs ouputs error when params are complex amplitudes --> yeah both won't work with complex parameters
# user needs to use two real parameters per complex number and then in his function convert them to complex
def optimize_L_BFGS(
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
        learning_rate: Learning rate for the optimizer.
        progress: If True, prints the progress of the optimization (for debugging - significantly slows optimization).
        early_stop: If True, stops the optimization if the loss does not change significantly (if convergence threshold is reached).
    Returns:
        control_amplitudes: Optimized control amplitudes.
        final_iter_idx: Number of iterations in the optimization.
        reward_history: Reward (fidelity, in [0, 1]) at each iteration. The
            L-BFGS line search requires a scalar-valued ``loss_fn`` (optax does
            not support ``has_aux`` here), so the reward is recovered from the
            loss as ``-loss`` (loss = ``-fidelity``).
    """

    opt = optax.lbfgs(learning_rate)

    # optax's lbfgs line search calls ``value_fn`` internally and expects a
    # scalar, and ``value_and_grad_from_state`` does not support ``has_aux``.
    # ``loss_fn`` returns ``(loss, reward)`` for consistency with the other
    # optimizers, so we expose a scalar-only wrapper to the optimizer.
    def scalar_loss(params):
        out = loss_fn(params)
        return out[0] if isinstance(out, tuple) else out

    value_and_grad_fn = optax.value_and_grad_from_state(scalar_loss)

    @jax.jit
    def step(carry):
        control_amplitudes, state, iter_idx, rewards = carry
        value, grad = value_and_grad_fn(control_amplitudes, state=state)
        updates, state = opt.update(
            grad,
            state,
            control_amplitudes,
            value=value,
            grad=grad,
            value_fn=scalar_loss,
        )
        control_amplitudes = optax.apply_updates(control_amplitudes, updates)
        # reward == -loss for GRAPE (loss = -fidelity); out-of-bounds writes
        # (iter_idx >= max_iter) are dropped by JAX, so this is safe.
        reward = -value
        rewards = rewards.at[iter_idx].set(reward)
        jax.lax.cond(
            jax.numpy.logical_and(progress, iter_idx % 10 == 0),
            lambda: jax.debug.print(
                "Iteration {iter_idx}, Reward: {reward:.6f}, Loss: {value:.6f}",
                iter_idx=iter_idx,
                reward=reward,
                value=value,
            ),
            lambda: None,
        )
        return control_amplitudes, state, iter_idx + 1, rewards

    def continuing_criterion(carry):
        _, state, _, _ = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_l2_norm(grad)

        return jnp.logical_or(
            jnp.logical_and(iter_num == 0, max_iter != 0),
            jnp.logical_and(
                iter_num < max_iter,
                jnp.logical_or(
                    err >= convergence_threshold, jnp.logical_not(early_stop)
                ),
            ),
        )

    rewards_init = jax.numpy.full((max_iter,), jax.numpy.nan)
    init_carry = (
        control_amplitudes,
        opt.init(control_amplitudes),
        0,
        rewards_init,
    )
    final_params, _, final_iter_idx, reward_history = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_params, final_iter_idx, reward_history
