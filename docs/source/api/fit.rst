fit (function)
==============

Train a :class:`gradnet.GradNet` with a user-defined loss function using the
lightweight default trainer. Provides device placement, optional
TensorBoard/CSV logging, best/periodic/last checkpointing, a tqdm progress
bar, and a loss dtype/device safety net that auto-casts mismatched losses and
warns once per ``fit()`` call.

For PyTorch Lightning features (mixed precision, multi-GPU, custom PL
callbacks/loggers), install the optional extra ``pip install gradnet[pl]``
and call :func:`gradnet.pl_fit` instead.

.. currentmodule:: gradnet

.. autofunction:: fit
   :noindex:
