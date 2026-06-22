fit (function)
==============

Train a :class:`gradnet.GradNet` with a user-defined loss function using the
built-in trainer. Provides device placement, optional TensorBoard/CSV
logging, best/periodic/last checkpointing, a tqdm progress bar, and a loss
dtype/device safety net that auto-casts mismatched losses and warns once per
``fit()`` call.

Loss functions are normally called as ``loss_fn(gn, **loss_kwargs)``. If the
loss function signature defines a parameter ``step``,
``fit`` passes the current zero-based optimization step automatically, for
example ``loss_fn(gn, step=step, **loss_kwargs)``. The parameter name
``"step"`` is reserved for this trainer-provided value and must not be supplied
in ``loss_kwargs``.

.. currentmodule:: gradnet

.. autofunction:: fit
   :noindex:
