# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GradNet is a Python library for differentiable parameterizations of graph adjacency matrices with budget and structure constraints, paired with ODE solvers and a PyTorch Lightning training loop. The package is on PyPI as `gradnet`.

## Commands

### Install (development)
```bash
pip install -e .
# With optional extras:
pip install -e ".[networkx,examples]"
```

### Run tests
```bash
pytest tests/
# Run a single test file:
pytest tests/test_utils_shortest_path.py
# Run a specific test:
pytest tests/test_utils_shortest_path.py::test_shortest_path_full_matrix
```

### Build docs
```bash
cd docs && make html
```

## Architecture

The package lives in [src/gradnet/](src/gradnet/) with lazy imports in [__init__.py](src/gradnet/__init__.py) that expose three public symbols: `GradNet`, `integrate_ode`, and `fit`.

### Core modules

**[gradnet.py](src/gradnet/gradnet.py)** — The main `GradNet` nn.Module. Contains:
- `DenseParameterization`: trainable dense delta matrix mapped through sign/positivity/symmetry/norm constraints
- `SparseParameterization`: same idea but for sparse COO edge subsets (given via a `mask`)
- `GradNet`: user-facing wrapper that holds a base adjacency and a parameterization; `forward()` returns `base_adj + delta_adj`; `renorm_params()` re-projects parameters onto the budget constraint after each optimizer step
- Module-level helpers: `normalize`, `positivize`, `symmetrize`

**[ode.py](src/gradnet/ode.py)** — `integrate_ode(gn, f, x0, tt, ...)`. Wraps `torchdiffeq` (odeint / odeint_adjoint / odeint_event). Vector field signature is `f(t, x, A, **f_kwargs)` where `A` is the adjacency from `gn()`. Handles device/dtype alignment automatically.

**[trainer.py](src/gradnet/trainer.py)** — `fit(gn, loss_fn, num_updates, ...)`. Wraps `GradNetLightning` (a `pl.LightningModule`) around a `GradNet`. One Lightning epoch = one optimizer step. Loss function protocol: `loss_fn(gn, **loss_kwargs) -> Tensor | (Tensor, dict)`. Calls `gn.renorm_params()` after each step by default (`post_step_renorm=True`).

**[utils.py](src/gradnet/utils.py)** — Helpers including `random_seed`, `prune_edges`, `shortest_path`, and internal `_to_like_struct` (moves tensors/numpy arrays to a target device/dtype).

### Key design patterns

- **Budget constraint enforcement**: `GradNet` stores raw unconstrained parameters; `renorm_params()` re-projects them to satisfy the `budget` norm after each optimizer step, keeping them on the constraint manifold.
- **Dense vs sparse backends**: passing a sparse COO `mask` to `GradNet` switches to `SparseParameterization` and optimizes only the masked edges. Without a mask, `DenseParameterization` is used.
- **Adjoint support**: `integrate_ode` registers the `GradNet` module with the `_VectorField` wrapper so `odeint_adjoint` can discover its parameters automatically.
- **Sparsity guarantee**: sparse mode always returns a sparse tensor for the delta; the `mask` defines which edges can be updated.

### Dependencies
- `torch>=2.0`, `pytorch-lightning>=1.9,<3`, `torchdiffeq>=0.2.3`, `numpy>=1.21`, `tqdm>=4.64`, `matplotlib`, `tensorboard`
- Optional: `networkx>=2.6` (graph conversion/plotting), `imageio-ffmpeg` (examples with animations)
