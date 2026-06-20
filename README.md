# evo-dl

JAX implementation scaffold for a hybrid neuro-evolutionary pyramidal model.

## Phase 1: Pyramidal Layer

The initial implementation provides:

- `NeuronState`: differentiable weights `W` and binary routing mask `M`
- `init_neuron_state`: random state initializer
- `pyramidal_forward`: masked branch-local forward pass
- `population_forward`: `jax.vmap` wrapper for evaluating many masks in parallel

Install locally with:

```bash
python3 -m pip install -e ".[dev]"
```

Run tests with:

```bash
pytest
```
