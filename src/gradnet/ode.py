"""ODE integration utilities with optional adjoint and event support.

This module provides a thin wrapper around :mod:`torchdiffeq` to integrate
ordinary differential equations whose dynamics may depend on a static
adjacency matrix (e.g., produced by a :class:`gradnet.GradNet`). It offers:

- A single entry point :func:`integrate_ode` for forward solves, with optional
  adjoint sensitivity via :func:`torchdiffeq.odeint_adjoint`.
- Event-based termination via :func:`torchdiffeq.odeint_event` to stop an
  integration when a user-defined scalar function crosses zero.
- Careful device/dtype alignment for initial conditions, time grids, and
  keyword arguments.

The public API mirrors the style used in :mod:`gradnet.trainer` so the
docstrings render well when building documentation with Sphinx.
"""
from __future__ import annotations
from typing import Callable, Any, Optional, Union, NamedTuple, Mapping, Sequence
import torch
import torch.nn as nn
from .utils import _to_like_struct


class _VectorField(nn.Module):
    """Internal wrapper so the adjoint can discover parameters.

    The adjoint method inspects ``.parameters()`` of the module passed to
    :mod:`torchdiffeq`. This wrapper registers the provided GradNet (if any)
    and any ``nn.Module`` instances found inside ``f``'s keyword arguments, so
    their parameters are included in the default adjoint parameter set.

    Args:
      f (Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]):
        Vector field callable with signature
        ``f(t, x, A, **kwargs) -> dxdt``.
      A (torch.Tensor): Static adjacency or parameter tensor provided to ``f``.
      kwargs (Mapping[str, Any] | None): Keyword arguments forwarded to ``f``.
      gn_module (torch.nn.Module | None): Optional module (e.g., a
        :class:`gradnet.GradNet`) to be registered so its parameters are
        visible to the adjoint.
      params_modules (dict[str, torch.nn.Module] | None): Mapping of names to
        ``nn.Module`` instances found in ``kwargs`` that should also be
        registered.
    """
    def __init__(
        self,
        f: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        A: torch.Tensor,
        kwargs: Mapping[str, Any] | None,
        gn_module: Optional[nn.Module] = None,
        params_modules: Optional[dict[str, nn.Module]] = None,
    ):
        super().__init__()
        self._f = f
        self.A = A
        if isinstance(gn_module, nn.Module):
            self.gn = gn_module  # register so adjoint sees gn.parameters()
        if params_modules:
            for k, m in params_modules.items():
                self.add_module(f"param_mod_{k}", m)
        self._kwargs = {} if kwargs is None else kwargs
        self.register_buffer("_zero", torch.tensor(0.0))  # device anchor

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the user vector field as ``f(t, x, A, **kwargs)``."""
        return self._f(t, x, self.A, **self._kwargs)


def integrate_ode(
    gn: Union[Callable[[], torch.Tensor], torch.Tensor],
    f: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    x0: Union[torch.Tensor, float, int],
    tt: torch.Tensor,
    *,
    f_kwargs: Mapping[str, Any] | None = None,   # kwargs for f / event_fn

    method: str = "dopri5",
    rtol: float = 1e-4,
    atol: float = 1e-4,
    solver_options: Optional[dict] = None,

    adjoint: bool = False,
    adjoint_options: Optional[dict] = None,             # e.g., {'norm': 'seminorm'}
    adjoint_params: Optional[Sequence[torch.Tensor]] = None,  # optional override

    event_fn: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,

    track_gradients: bool = True
):
    """Integrate an ODE ``dx/dt = f(t, x, A, **f_kwargs)`` using torchdiffeq.

    This is a convenience wrapper around ``torchdiffeq.odeint`` with optional
    adjoint sensitivities and event-based termination. The vector field is
    called as ``f(t, x, A, **f_kwargs)`` where ``A`` is the network adjacency matrix
    represented as a (potentially sparse) torch.Tensor.

    Args:
      gn (Callable[[], torch.Tensor] | torch.Tensor): Tensor ``A`` or a
        zero-arg callable returning ``A``. If an ``nn.Module`` is provided, its
        parameters are included in the default adjoint parameter set.
      f (Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]):
        Vector field returning ``dx/dt`` with the same shape as ``x``.
      x0 (torch.Tensor | float | int): Initial state (scalars are promoted).
      tt (torch.Tensor): 1D time grid (monotone; may decrease for reverse-time
        event searches).
      f_kwargs (Mapping[str, Any] | None, optional): Keyword arguments passed to
        ``f`` (and ``event_fn`` if provided). Tensors/NumPy arrays are moved/cast
        to match ``A``.
      method (str, optional): Integrator, e.g., adaptive stepsize ``"dopri5"`` (default),
        or fixed-step ``"rk4"``, see more options in `torchdiffeq documentation
        <https://github.com/rtqichen/torchdiffeq>`_).
      rtol (float, optional): Relative tolerance.
      atol (float, optional): Absolute tolerance.
      solver_options (dict | None, optional): Additional solver options.
      adjoint (bool, optional): If ``True``, use the adjoint method.
      adjoint_options (dict | None, optional): Options for adjoint solve
        (e.g., ``{"norm": "seminorm"}``).
      adjoint_params (Sequence[torch.Tensor] | None, optional): Explicit list of
        parameters for adjoint gradients. Defaults to parameters discovered in
        the wrapped vector field (including modules in ``f_kwargs``).
      event_fn (Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None, optional):
        Optional scalar function ``g(t, x, A, **f_kwargs)``; the integration
        stops on zero-crossing.
      track_gradients (bool, optional): Enable autograd during the solve.

    Returns:
      tuple: ``(tt_out, y_out)`` where ``y_out`` has shape ``(len(tt_out), *x0.shape)``.
        If an event is used, outputs are truncated at the detected event time.

    Raises:
      TypeError: If ``f_kwargs`` is not a mapping (and not ``None``).

    Examples:
      Basic integration without events::

        import torch
        from gradnet.ode import integrate_ode

        A = torch.tensor([[0., 1.], [-1., 0.]])

        def vf(t, x, A):
            return A @ x

        x0 = torch.tensor([1., 0.])
        tt = torch.linspace(0, 1, steps=11)
        t_out, x_out = integrate_ode(A, vf, x0, tt)

      Event-driven integration until ``x[0]`` crosses zero::

        def event(t, x, A):
            return x[0]  # stop when it crosses 0

        tt_partial, x_partial = integrate_ode(
            A, vf, x0, tt, event_fn=event
        )

      (*) See also the `torchdiffeq documentation <https://github.com/rtqichen/torchdiffeq>`_ for supported methods and options.
    """
    # 0) Build adjacency once, keep a handle to gn if it's a module (for adjoint param discovery)
    if callable(gn):
        gn_module = gn if isinstance(gn, nn.Module) else None
        A = gn()
    else:
        gn_module = None
        A = gn
    if not isinstance(A, torch.Tensor):
        A = torch.as_tensor(A)
    # Ensure dense adjacency for downstream vector fields that use dense-style indexing
    if hasattr(A, "layout") and A.layout != torch.strided:
        A = A.to_dense()

    # 1) Align inputs to A’s device/dtype
    x0 = _to_like_struct(x0, A)
    tt = _to_like_struct(tt, A)

    # params must be kwargs if provided
    if f_kwargs is None:
        f_kwargs = {}
    elif isinstance(f_kwargs, Mapping):
        f_kwargs = _to_like_struct(f_kwargs, A)
    else:
        raise TypeError("`f_kwargs` must be a Mapping of keyword arguments (or None).")

    # Collect any nn.Modules inside params for adjoint to see them automatically
    params_modules = {k: v for k, v in f_kwargs.items() if isinstance(v, nn.Module)}

    # 2) Vector field module
    vf = _VectorField(
        f=f,
        A=A,
        kwargs=f_kwargs,
        gn_module=gn_module,
        params_modules=params_modules if params_modules else None,
    ).to(A.device, A.dtype)

    # 3) Choose solver interface and kwargs
    from torchdiffeq import odeint, odeint_adjoint, odeint_event
    ode_interface = odeint_adjoint if adjoint else odeint
    solver_options = {} if solver_options is None else solver_options
    # Do not inject unsupported keys into options (e.g., dtype for RK4).
    solver_options["dtype"] = gn.dtype  # inject dtype, it doesn't deduce this automatically from A
    base_kwargs = dict(rtol=rtol, atol=atol, method=method, options=solver_options)

    if adjoint:
        if adjoint_options is not None:
            base_kwargs["adjoint_options"] = adjoint_options
        if adjoint_params is not None:
            base_kwargs["adjoint_params"] = tuple(adjoint_params)
        # else: default is tuple(vf.parameters()), which now includes gn / any nn.Modules in params

    # 4) Gradient mode toggle
    prev_grad_mode = torch.is_grad_enabled()
    torch.set_grad_enabled(track_gradients)
    try:
        # Event-aware path
        if event_fn is not None:
            def _efn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                return event_fn(t, x, A, **f_kwargs)

            t0 = tt[0]

            _ret = odeint_event(
                vf, x0, t0,
                event_fn=_efn,
                odeint_interface=ode_interface,
                **base_kwargs
            )
            # torchdiffeq versions differ: some return (t, x), others (t, x, index)
            if isinstance(_ret, (tuple, list)) and len(_ret) == 3:
                t_event, x_event, _ = _ret
            else:
                t_event, x_event = _ret  # type: ignore[misc]

            # Build output grid up to the event, preserving time direction.
            # If times decrease (reverse-time), include points >= t_event and append t_event.
            decreasing = (tt[-1] - tt[0]) < 0
            if decreasing:
                mask = tt >= t_event
            else:
                mask = tt <= t_event
            tt_partial = tt[mask]
            if tt_partial.numel() == 0 or tt_partial[-1] != t_event:
                tt_partial = torch.cat([tt_partial, t_event.unsqueeze(0)], dim=0)

            x_partial = ode_interface(vf, x0, tt_partial, **base_kwargs)
            return tt_partial, x_partial

        # Standard solve
        y = ode_interface(vf, x0, tt, **base_kwargs)
        return tt, y

    finally:
        torch.set_grad_enabled(prev_grad_mode)
