:orphan:

Getting Started
===============

Install the package (editable mode recommended during development)::

   pip install -e .

Basic usage
-----------

Create a small GradNet, obtain an adjacency, and integrate a simple ODE::

   import torch
   from gradnet import GradNet, integrate_ode

   N = 3
   gn = GradNet(
       num_nodes=N,
       budget=1.0,
       mask=torch.ones((N, N)) - torch.eye(N),
       adj0=torch.zeros((N, N)),
       undirected=True,
   )

   def vf(t, x, A):
       return A @ x

   x0 = torch.ones(N)
   tt = torch.linspace(0, 1, 11)
   tt_out, x_out = integrate_ode(gn, vf, x0, tt)
