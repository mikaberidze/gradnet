fit (function)
==============

Train a :class:`gradnet.GradNet` with a user-defined loss function using the
built-in trainer. Provides device placement, optional TensorBoard/CSV
logging, best/periodic/last checkpointing, a tqdm progress bar, and a loss
dtype/device safety net that auto-casts mismatched losses and warns once per
``fit()`` call.

.. currentmodule:: gradnet

.. autofunction:: fit
   :noindex:
