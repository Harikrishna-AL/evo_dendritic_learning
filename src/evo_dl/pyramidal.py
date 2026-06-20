"""Core pyramidal layer primitives.

The layer is intentionally functional: evolution owns the binary topology `M`,
gradient descent owns the continuous weights `W`, and both are passed explicitly.
"""

from __future__ import annotations

from collections.abc import Callable

import flax.struct
import jax
import jax.numpy as jnp
from jax import Array


@flax.struct.dataclass
class NeuronState:
    """State for one pyramidal layer topology.

    Attributes:
        W: Continuous synaptic weights with shape
            `(num_branches, input_dim, output_dim)`.
        M: Binary routing mask with the same shape as `W`. It is treated as
            non-differentiable structural state by the training/evolution loops.
    """

    W: Array
    M: Array


def assert_binary_mask(M: Array) -> None:
    """Raise if `M` contains values other than 0 or 1.

    This is a host-side validation helper for initialization, tests, and debug
    paths. Do not call it inside `jax.jit`-compiled functions.
    """

    mask_values = jax.device_get(M)
    if not ((mask_values == 0) | (mask_values == 1)).all():
        raise ValueError("M must be a strictly binary mask containing only 0 or 1.")


def init_neuron_state(
    key: Array,
    *,
    num_branches: int,
    input_dim: int,
    output_dim: int,
    mask_density: float = 0.5,
    weight_scale: float = 0.02,
) -> NeuronState:
    """Initialize one pyramidal layer state."""

    if not 0.0 <= mask_density <= 1.0:
        raise ValueError("mask_density must be in [0, 1].")
    if num_branches <= 0 or input_dim <= 0 or output_dim <= 0:
        raise ValueError("num_branches, input_dim, and output_dim must be positive.")

    weight_key, mask_key = jax.random.split(key)
    shape = (num_branches, input_dim, output_dim)
    W = weight_scale * jax.random.normal(weight_key, shape)
    M = jax.random.bernoulli(mask_key, p=mask_density, shape=shape).astype(W.dtype)
    return NeuronState(W=W, M=M)


def pyramidal_forward(
    state: NeuronState,
    X: Array,
    branch_nonlinearity: Callable[[Array], Array] = jax.nn.tanh,
) -> Array:
    """Evaluate one masked pyramidal layer.

    Implements `sum_b branch_nonlinearity(sum_i(M[b, i, o] * W[b, i, o] * X[i]))`.

    Args:
        state: `NeuronState` for one topology.
        X: Input vector with shape `(input_dim,)` or batch with shape
            `(batch_size, input_dim)`.
        branch_nonlinearity: Non-linearity applied independently to every
            branch/output activation before branch aggregation.

    Returns:
        Output vector `(output_dim,)` for a single input or matrix
        `(batch_size, output_dim)` for batched input.
    """

    masked_weights = jax.lax.stop_gradient(state.M) * state.W

    if X.ndim == 1:
        branch_activations = jnp.einsum("i,bio->bo", X, masked_weights)
    elif X.ndim == 2:
        branch_activations = jnp.einsum("ni,bio->nbo", X, masked_weights)
    else:
        raise ValueError("X must have shape (input_dim,) or (batch_size, input_dim).")

    return branch_nonlinearity(branch_activations).sum(axis=-2)


def population_forward(
    states: NeuronState,
    X: Array,
    branch_nonlinearity: Callable[[Array], Array] = jax.nn.tanh,
) -> Array:
    """Evaluate a population of topologies in parallel with `jax.vmap`.

    Args:
        states: Batched state with `W` and `M` shaped
            `(population_size, num_branches, input_dim, output_dim)`.
        X: Shared input vector or batch evaluated against every population member.
        branch_nonlinearity: Branch non-linearity.

    Returns:
        Shape `(population_size, output_dim)` for one input vector, or
        `(population_size, batch_size, output_dim)` for batched inputs.
    """

    return jax.vmap(
        lambda one_state: pyramidal_forward(one_state, X, branch_nonlinearity)
    )(states)
