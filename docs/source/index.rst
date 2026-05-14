.. gradnet documentation master file, created by
   sphinx-quickstart on Wed Sep 10 00:18:24 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GradNet documentation
=====================

GradNet is a PyTorch-based framework for AI-enabled optimization of networks. 
Define static or dynamical objectives and constraints, 
then discover the optimal network structures.

It encodes the network structure as a differentiable object with optional
budget and structure constraints. It lets the users directly optimize
static objectives using a lightweight built-in training loop.
Alternatively, built-in ODE solvers can be used to define and
optimize dynamical objectives.


.. figure:: _static/gradient_descent.png
   :alt: Illustration of the gradient-based optimization pipeline for network structure.
   :align: center
   :width: 650px

   Illustration of the gradient-based optimization pipeline for network structures.

.. figure:: _static/rewiring_net.gif
   :alt: A random network rewires itself with GradNet to optimize synchronization in the Kuramoto model.
   :align: center
   :class: home-hero
   :width: 400px

   A random network rewires itself with GradNet to optimize synchronization in the Kuramoto model.

Installation
------------

.. code-block:: bash

   pip install gradnet

Quick links
-----------

- :doc:`GradNet </api/gradnet>` – differentiable adjacency matrix model.
- :doc:`fit </api/fit>` – lightweight default trainer for a GradNet + loss
  function (a self-contained training loop with logging, checkpointing, and a
  loss dtype/device safety net).
- :doc:`pl_fit </api/pl_fit>` – optional PyTorch Lightning trainer, available
  via ``pip install gradnet[pl]``, for mixed precision and the full PL feature
  set.
- :doc:`integrate_ode </api/integrate_ode>` – integrate GradNet-defined ODEs
  into control or simulation workflows.

Project links
-------------

- `GitHub repository <https://github.com/mikaberidze/gradnet>`_

.. toctree::
   :maxdepth: 1
   :caption: API Reference:
   :titlesonly:

   GradNet (class) <api/gradnet>
   fit (function) <api/fit>
   pl_fit (function) <api/pl_fit>
   integrate_ode (function) <api/integrate_ode>
   trainer (module) <api/trainer>
   pl_trainer (module) <api/pl_trainer>
   utils (module) <api/utils>

.. _tutorials-nav:

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Tutorials:
   :titlesonly:

   tutorials/*
