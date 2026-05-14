pl_fit (function)
=================

Train a :class:`gradnet.GradNet` using the full-featured PyTorch Lightning
trainer. Accepts a superset of :func:`gradnet.fit`'s arguments — adds
``accelerator``, ``precision``, multi-GPU strategies, native ``pl.Callback``
and ``LightningLoggerBase`` instances, and the rest of PL's surface.

Requires the optional ``pl`` extra::

    pip install gradnet[pl]

Without that extra installed, ``import gradnet`` still works, but accessing
``gradnet.pl_fit`` raises :class:`ImportError` with installation instructions.

.. currentmodule:: gradnet.pl_trainer

.. autofunction:: fit
