import jax
import jax.numpy as jnp
import pytest

from evo_dl.pyramidal import (
    NeuronState,
    assert_binary_mask,
    init_neuron_state,
    population_forward,
    pyramidal_forward,
)


def test_init_neuron_state_shapes_and_binary_mask():
    state = init_neuron_state(
        jax.random.PRNGKey(0),
        num_branches=3,
        input_dim=4,
        output_dim=2,
        mask_density=0.5,
    )

    assert state.W.shape == (3, 4, 2)
    assert state.M.shape == (3, 4, 2)
    assert_binary_mask(state.M)


def test_forward_applies_mask_before_branch_sum():
    state = NeuronState(
        W=jnp.array(
            [
                [[1.0], [2.0], [3.0]],
                [[10.0], [20.0], [30.0]],
            ]
        ),
        M=jnp.array(
            [
                [[1.0], [0.0], [1.0]],
                [[0.0], [1.0], [0.0]],
            ]
        ),
    )
    x = jnp.array([2.0, 3.0, 5.0])

    out = pyramidal_forward(state, x, branch_nonlinearity=lambda z: z)

    assert out.shape == (1,)
    assert jnp.allclose(out, jnp.array([77.0]))


def test_forward_supports_batched_inputs():
    state = NeuronState(
        W=jnp.ones((2, 3, 4)),
        M=jnp.ones((2, 3, 4)),
    )
    x = jnp.ones((5, 3))

    out = pyramidal_forward(state, x, branch_nonlinearity=lambda z: z)

    assert out.shape == (5, 4)
    assert jnp.allclose(out, 6.0)


def test_population_forward_vectorizes_masks():
    W = jnp.ones((2, 2, 3, 1))
    M = jnp.array(
        [
            [
                [[1.0], [0.0], [0.0]],
                [[0.0], [1.0], [0.0]],
            ],
            [
                [[0.0], [1.0], [0.0]],
                [[0.0], [0.0], [1.0]],
            ],
        ]
    )
    states = NeuronState(W=W, M=M)
    x = jnp.array([2.0, 3.0, 5.0])

    out = population_forward(states, x, branch_nonlinearity=lambda z: z)

    assert out.shape == (2, 1)
    assert jnp.allclose(out, jnp.array([[5.0], [8.0]]))


def test_population_forward_supports_batched_inputs():
    states = NeuronState(
        W=jnp.ones((4, 2, 3, 1)),
        M=jnp.ones((4, 2, 3, 1)),
    )
    x = jnp.ones((5, 3))

    out = population_forward(states, x, branch_nonlinearity=lambda z: z)

    assert out.shape == (4, 5, 1)
    assert jnp.allclose(out, 6.0)


def test_assert_binary_mask_rejects_non_binary_values():
    with pytest.raises(ValueError, match="strictly binary"):
        assert_binary_mask(jnp.array([[0.0, 0.5, 1.0]]))
