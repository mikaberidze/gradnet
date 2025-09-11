# gradnet

Trainable graph adjacency parameterizations with dense and sparse outputs, plus
ODE integration utilities and a lightweight PyTorch Lightning training loop.

Features
- GradNet: learn constrained adjacency deltas (dense or sparse) with simple projections
- ODE: integrate dynamics that depend on an adjacency via torchdiffeq
- Trainer: minimal Lightning wrapper to optimize GradNet with custom losses

Requirements
- Python >= 3.10
- Dependencies: torch, numpy, networkx, pytorch-lightning, tqdm, torchdiffeq

Install
- From source (editable):
  pip install -e .

Quickstart
- Learn a dense adjacency
  ```python
  from gradnet import GradNet
  import torch
  N = 10
  model = GradNet(num_nodes=N, budget=1.0, matrix_encoding="dense")
  A = model()  # (N, N) dense tensor
  ```

- Integrate an ODE using A
  ```python
  from gradnet import integrate_ode
  def vf(t, x, A):
      return A @ x
  x0 = torch.randn(N)
  tt = torch.linspace(0, 1, 11)
  t_out, x_out = integrate_ode(model, vf, x0, tt)
  ```

- Train with a custom loss
  ```python
  from gradnet import fit, GradNet
  def loss_fn(m: GradNet):
      A = m()
      # toy objective: promote small weights
      loss = (A.abs()).mean()
      return loss
  fit(gradnet=model, loss_fn=loss_fn, num_updates=100)
  ```

Modules
- gradnet.GradNet: main model exposing dense/sparse encodings and NetworkX export
- gradnet.integrate_ode: torchdiffeq wrapper with optional adjoint and events
- gradnet.fit: Lightning loop with progress and optional checkpoints

Examples
- See `examples/` for starter notebooks and scripts.
