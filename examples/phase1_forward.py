import jax
import jax.numpy as jnp

from evo_dl.pyramidal import NeuronState, init_neuron_state, population_forward


def main() -> None:
    key = jax.random.PRNGKey(42)
    population_size = 8
    num_branches = 4
    input_dim = 16
    output_dim = 3

    keys = jax.random.split(key, population_size)
    states = jax.vmap(
        lambda k: init_neuron_state(
            k,
            num_branches=num_branches,
            input_dim=input_dim,
            output_dim=output_dim,
            mask_density=0.25,
        )
    )(keys)

    x = jnp.ones((32, input_dim))
    y = population_forward(states, x)
    print(y.shape)


if __name__ == "__main__":
    main()
