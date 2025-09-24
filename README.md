# GradNet

GradNet provides differentiable parameterizations of graph adjacency matrices with explicit budget and structure constraints. It pairs these parameterizations with ODE solvers and a lightweight PyTorch Lightning training loop so you can prototype network optimization problems quickly.

<p align="center">
  <img src="docs/source/_static/rewiring_net.gif"
       alt="A random network rewires itself with GradNet to optimize synchronization in the Kuramoto model."
       width="300" />
</p>

## Highlights
- Learn dense or sparse adjacency updates with norm, sign, and symmetry constraints.
- Projected parameterizations that stay differentiable and GPU friendly.
- Torchdiffeq-backed integration utilities for graph-driven dynamical systems.
- Minimal Lightning trainer that wraps custom loss functions in just a few lines.

## Installation
Install the released package from PyPI:

```bash
pip install gradnet
```

To work off the latest sources instead, clone the repository and install in editable mode:

```bash
pip install -e .
```

GradNet targets Python 3.10+ and depends on PyTorch, PyTorch Lightning, torchdiffeq, NumPy, and tqdm (installed automatically by the command above). Install the optional NetworkX helpers with `pip install gradnet[networkx]` when you need conversions to `networkx` graphs or plotting utilities that rely on it.

## Documentation
Full API documentation, tutorials, and background material live at [gradnet.readthedocs.io](https://gradnet.readthedocs.io/).

## Quickstart

### Learn a constrained adjacency
```python
import torch
from gradnet import GradNet

num_nodes = 10
model = GradNet(
    num_nodes=num_nodes,
    budget=1.0,
    undirected=True,
)

adjacency = model()  # full (num_nodes, num_nodes) tensor
```
Pass a sparse COO mask via the `mask` argument to switch to the sparse backend and optimize only selected edges.

### Integrate a graph-driven ODE
```python
from gradnet import integrate_ode

# simple linear dynamics \dot{x} = Ax

def vector_field(t, x, A):
    return A @ x

x0 = torch.randn(num_nodes)
t_grid = torch.linspace(0.0, 1.0, 51)
sol_t, sol_x = integrate_ode(model, vector_field, x0, t_grid)
```

### Optimize with your own loss
```python
from gradnet import GradNet, fit

# encourage sparse, small-magnitude updates
def loss_fn(g: GradNet):
    delta = g.get_delta_adj()
    return delta.abs().mean()

fit(gradnet=model, loss_fn=loss_fn, num_updates=200, learning_rate=1e-2)
```
The trainer handles optimizer setup, logging, and checkpointing while you focus on defining the objective.

## Modules at a glance
- `gradnet.GradNet`: wraps dense and sparse parameterizations, supports directed/undirected networks, masking, custom edge-building costs etc.
- `gradnet.integrate_ode`: torchdiffeq-powered solver with adjoint and event support for adjacency-dependent dynamics.
- `gradnet.fit`: PyTorch Lightning loop that optimizes a `GradNet` using user-supplied loss functions.
- `gradnet.utils`: various helpers functions.

## License
GradNet is released under the BSD 3-Clause License. See `LICENSE` for details.
