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
from typing import Callable, Any, Optional, Union, Mapping, Sequence
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint, odeint_event


class _VectorField(nn.Module):
    """Internal wrapper that exposes vector-field parameters to adjoint solves."""

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

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the user vector field as ``f(t, x, A, **kwargs)``."""
        return self._f(t, x, self.A, **self._kwargs)


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return the corresponding real dtype for ``dtype``."""
    return torch.zeros((), dtype=dtype).real.dtype


def _promoted_state_dtype(
    A: torch.Tensor, x0: Union[torch.Tensor, float, int]
) -> torch.dtype:
    """Choose a solve dtype that preserves complex-valued states."""
    x0_dtype = x0.dtype if isinstance(x0, torch.Tensor) else torch.as_tensor(x0).dtype
    return torch.promote_types(A.dtype, x0_dtype)


def _to_device_struct(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors/NumPy arrays in ``obj`` to ``device``."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device)
    if isinstance(obj, np.ndarray):  # also catches np.matrix
        return torch.as_tensor(obj).to(device=device)
    if isinstance(obj, np.generic):  # NumPy scalar (e.g., np.float32(3.0))
        return torch.as_tensor(obj, device=device)
    if isinstance(obj, Mapping):
        return obj.__class__({k: _to_device_struct(v, device) for k, v in obj.items()})
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):  # namedtuple
        return obj.__class__(*[_to_device_struct(v, device) for v in obj])
    if isinstance(obj, (list, tuple)):
        typ = obj.__class__
        return typ(_to_device_struct(v, device) for v in obj)
    return obj  # nn.Module or anything else stays as-is


def integrate_ode(
    gn: Union[Callable[[], torch.Tensor], torch.Tensor],
    f: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    x0: Union[torch.Tensor, float, int],
    tt: torch.Tensor,
    *,
    f_kwargs: Mapping[str, Any] | None = None,  # kwargs for f / event_fn
    method: str = "dopri5",
    rtol: float = 1e-4,
    atol: float = 1e-4,
    solver_options: Optional[dict] = None,
    adjoint: bool = False,
    adjoint_options: Optional[dict] = None,  # e.g., {'norm': 'seminorm'}
    adjoint_params: Optional[Sequence[torch.Tensor]] = None,  # optional override
    event_fn: Optional[
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    track_gradients: bool = True,
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
        ``f`` (and ``event_fn`` if provided). Tensors/NumPy arrays are moved to
        the adjacency device without forcing them to ``A.dtype``.
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
      tuple: ``(tt_out, x_out)`` where ``x_out`` has shape ``(len(tt_out), *x0.shape)``.
        If an event is used, ``(tt_out, x_out, t_event, x_event)`` ``tt_out`` and ``x_out`` are
        truncated at the detected event time. ``t_event`` and ``x_event`` are the differentiable
        event time and state.

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

        tt_partial, x_partial, t_event, x_event = integrate_ode(
            A, vf, x0, tt, event_fn=event
        )

      (*) See also the `torchdiffeq documentation <https://github.com/rtqichen/torchdiffeq>`_ for supported methods and options.
    """
    # Build adjacency once and keep a module handle for adjoint parameter discovery.
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

    # Align the solve state to a promoted dtype so complex inputs stay complex.
    state_dtype = _promoted_state_dtype(A, x0)
    time_dtype = _real_dtype(state_dtype)
    x0 = torch.as_tensor(x0, device=A.device, dtype=state_dtype)
    tt = torch.as_tensor(tt, device=A.device, dtype=time_dtype)

    # params must be kwargs if provided
    if f_kwargs is None:
        f_kwargs = {}
    elif isinstance(f_kwargs, Mapping):
        f_kwargs = _to_device_struct(f_kwargs, A.device)
    else:
        raise TypeError("`f_kwargs` must be a Mapping of keyword arguments (or None).")

    # Collect any nn.Modules inside params for adjoint to see them automatically
    params_modules = {k: v for k, v in f_kwargs.items() if isinstance(v, nn.Module)}

    # Wrap the vector field so adjoint sees relevant module parameters.
    vf = _VectorField(
        f=f,
        A=A,
        kwargs=f_kwargs,
        gn_module=gn_module,
        params_modules=params_modules if params_modules else None,
    ).to(A.device)

    # Select solver interface and shared ODE kwargs.
    ode_interface = odeint_adjoint if adjoint else odeint
    solver_options = {} if solver_options is None else solver_options
    base_kwargs = dict(rtol=rtol, atol=atol, method=method, options=solver_options)

    if adjoint:
        if adjoint_options is not None:
            base_kwargs["adjoint_options"] = adjoint_options
        if adjoint_params is not None:
            base_kwargs["adjoint_params"] = tuple(adjoint_params)
        # else: default is tuple(vf.parameters()), which now includes gn / any nn.Modules in params

    # Temporarily switch gradient tracking mode for this solve.
    with torch.set_grad_enabled(track_gradients):
        # Event-aware path
        if event_fn is not None:

            def _efn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                return event_fn(t, x, A, **f_kwargs)

            t0 = tt[0]
            t1 = tt[-1]
            decreasing = (t1 - t0) < 0

            # Cap the event so integration always terminates by the requested end.
            def _efn_capped(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                g = _efn(t, x)
                t_cap = (t - t1) if decreasing else (t1 - t)
                t_cap = t_cap.to(dtype=g.dtype, device=g.device)
                return torch.minimum(g, t_cap)

            _ret = odeint_event(
                vf,
                x0,
                t0,
                event_fn=_efn_capped,
                odeint_interface=ode_interface,
                **base_kwargs,
            )
            # torchdiffeq versions may return (t, x) or (t, x, index).
            t_event, x_event = _ret[0], _ret[1]

            # Ensure the stop time does not extend beyond the integration grid.
            t_stop = (
                torch.maximum(t_event, t1) if decreasing else torch.minimum(t_event, t1)
            )
            # Snap tiny endpoint roundoff to the exact requested endpoint.
            if t_stop.is_floating_point():
                eps = torch.finfo(t_stop.dtype).eps
                scale = torch.maximum(torch.abs(t1), torch.ones_like(t1))
                near_end = torch.abs(t_stop - t1) <= (128.0 * eps) * scale
                t_stop = torch.where(near_end, t1, t_stop)

            # Build output grid up to the event, preserving time direction.
            # If times decrease (reverse-time), include points >= t_event and append t_event.
            if decreasing:
                mask = tt >= t_stop
            else:
                mask = tt <= t_stop
            tt_partial = tt[mask]
            if tt_partial.numel() == 0 or not torch.equal(tt_partial[-1], t_stop):
                tt_partial = torch.cat([tt_partial, t_stop.unsqueeze(0)], dim=0)

            x_partial = ode_interface(vf, x0, tt_partial, **base_kwargs)
            # Make sure returned (t_event, x_event) correspond to the capped output.
            return tt_partial, x_partial, tt_partial[-1], x_partial[-1]

        # Standard solve
        y = ode_interface(vf, x0, tt, **base_kwargs)
        return tt, y
