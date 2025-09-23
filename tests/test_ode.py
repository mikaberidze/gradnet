import os
import sys
import math

import pytest

# Skip the entire module if required deps are missing
torch = pytest.importorskip("torch")
pytest.importorskip("torchdiffeq")

import torch.nn as nn

# Ensure `src/` is on sys.path for local imports when using a src layout
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir, 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gradnet.ode import integrate_ode
from gradnet.gradnet import GradNet


def _close(a, b, atol=1e-5, rtol=1e-5):
    return torch.allclose(torch.as_tensor(a), torch.as_tensor(b), atol=atol, rtol=rtol)


def test_integrate_basic_linear_no_event():
    # dx/dt = a * x, a from A
    a = 0.3
    A = torch.tensor([[a]], dtype=torch.float32)

    def vf(t, x, A):
        return A[0, 0] * x

    x0 = torch.tensor([1.0])
    tt = torch.linspace(0.0, 1.0, steps=11)

    t_out, y = integrate_ode(A, vf, x0, tt)

    assert torch.is_tensor(t_out) and torch.is_tensor(y)
    assert t_out.shape == tt.shape
    assert y.shape == (tt.numel(), 1)
    assert _close(t_out, tt)  # same time grid returned


def test_dtype_and_alignment_and_solver_options():
    # Ensure x0/tt/f_kwargs are coerced to A's dtype/device and options work
    A = torch.tensor([[1.0]], dtype=torch.float64)
    coeff_np = 2.0  # will be coerced to float64 tensor internally via f_kwargs

    def vf(t, x, A, coeff):
        return (A[0, 0] * coeff) * 0.0 * x + 0.0 * x  # intentionally zero dynamics

    x0 = torch.tensor([3.0], dtype=torch.float32)
    tt = torch.linspace(0.0, 1.0, steps=11, dtype=torch.float32)

    t_out, y = integrate_ode(
        A,
        vf,
        x0,
        tt,
        f_kwargs={"coeff": coeff_np},
        method="rk4",
        solver_options={"step_size": 0.1},
    )

    assert t_out.dtype == torch.float64
    assert y.dtype == torch.float64
    # zero dynamics -> constant solution
    assert _close(y[0], y[-1]) and _close(y[0], torch.tensor([3.0], dtype=torch.float64))


def test_f_kwargs_type_error():
    A = torch.tensor([[1.0]])

    def vf(t, x, A):
        return A[0, 0] * 0.0 * x

    with pytest.raises(TypeError):
        integrate_ode(A, vf, torch.tensor([1.0]), torch.linspace(0.0, 1.0, 3), f_kwargs=[1, 2, 3])


class _AModule(nn.Module):
    dtype = torch.float64
    def __init__(self, a=0.3):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a)))

    def forward(self):
        return self.a.view(1, 1)


def _analytic_grad_linear(x0, a, T):
    # For dx/dt = a*x, L = 0.5 * x(T)^2, grad dL/da = x0^2 * T * exp(2*a*T)
    return (x0 ** 2) * T * math.exp(2.0 * a * T)


@pytest.mark.parametrize("use_adjoint", [False, True])
def test_adjoint_and_standard_grads_through_gn_module(use_adjoint):
    mod = _AModule(a=0.2)

    def vf(t, x, A):
        return A[0, 0] * x

    x0 = torch.tensor([1.2])
    T = 1.0
    tt = torch.linspace(0.0, T, steps=21)

    t_out, y = integrate_ode(mod, vf, x0, tt, adjoint=use_adjoint, rtol=1e-7, atol=1e-7)

    loss = 0.5 * (y[-1, 0] ** 2)
    loss.backward()

    assert mod.a.grad is not None
    # Compare to analytic gradient with a loose tolerance (numerical ODE solve)
    expected = _analytic_grad_linear(float(x0[0]), float(mod.a.detach()), T)
    assert abs(mod.a.grad.item() - expected) / max(1.0, abs(expected)) < 2e-3


def test_adjoint_params_override_works():
    mod = _AModule(a=0.1)

    def vf(t, x, A):
        return A[0, 0] * x

    x0 = torch.tensor([1.0])
    tt = torch.linspace(0.0, 0.5, steps=11)

    _, y = integrate_ode(
        mod,
        vf,
        x0,
        tt,
        adjoint=True,
        adjoint_params=(mod.a,),  # explicit override
        rtol=1e-6,
        atol=1e-6,
    )

    (0.5 * y[-1, 0] ** 2).backward()
    assert mod.a.grad is not None and torch.isfinite(mod.a.grad)


def test_params_modules_in_kwargs_participate_in_adjoint():
    # b parameter appears inside f_kwargs and should be registered via _VectorField
    class Scale(nn.Module):
        def __init__(self, b=0.4):
            super().__init__()
            self.b = nn.Parameter(torch.tensor(float(b)))

    scale = Scale(b=0.4)
    A = torch.tensor([[0.0]])  # unused in vf

    def vf(t, x, A, scale: Scale):
        return scale.b * x

    x0 = torch.tensor([1.1])
    T = 0.8
    tt = torch.linspace(0.0, T, steps=17)

    _, y = integrate_ode(A, vf, x0, tt, f_kwargs={"scale": scale}, adjoint=True, rtol=1e-7, atol=1e-7)
    loss = 0.5 * (y[-1, 0] ** 2)
    loss.backward()
    assert scale.b.grad is not None and torch.isfinite(scale.b.grad)


def test_event_forward_stops_at_zero_crossing():
    # x' = -1, x0=1 -> event at t=1 when x(t)=0
    def vf(t, x, A):
        return -torch.ones_like(x)

    def event_fn(t, x, A):
        return x[0]  # stop when x crosses 0

    A = torch.tensor([[0.0]])
    x0 = torch.tensor([1.0])
    tt = torch.linspace(0.0, 2.0, steps=21)

    tt_p, x_p = integrate_ode(A, vf, x0, tt, event_fn=event_fn)

    # last time equals event time and is about 1.0
    assert abs(tt_p[-1].item() - 1.0) < 1e-3
    # final state near zero
    assert abs(x_p[-1, 0].item()) < 1e-3
    # monotonic increasing times
    assert torch.all(tt_p[1:] >= tt_p[:-1])


def test_track_gradients_flag_controls_requires_grad():
    mod = _AModule(a=0.123)

    def vf(t, x, A):
        return A[0, 0] * x

    x0 = torch.tensor([1.0])
    tt = torch.linspace(0.0, 0.2, steps=5)

    # With gradients
    _, y = integrate_ode(mod, vf, x0, tt, track_gradients=True)
    assert y.requires_grad is True

    # Without gradients
    _, y2 = integrate_ode(mod, vf, x0, tt, track_gradients=False)
    assert y2.requires_grad is False


@pytest.mark.parametrize("method", ["rk4", "dopri5"])  # require torchdiffeq to support both
def test_solver_methods_run_and_shapes(method):
    A = torch.tensor([[0.5]])

    def vf(t, x, A):
        return A[0, 0] * x

    x0 = torch.tensor([1.0])
    tt = torch.linspace(0.0, 1.0, steps=7)
    t_out, y = integrate_ode(A, vf, x0, tt, method=method)
    assert t_out.shape == tt.shape
    assert y.shape == (tt.numel(), 1)


def test_dtype_alignment_with_gradnet_and_nested_kwargs():
    # Build a tiny GradNet producing adjacency with specific dtype
    N = 2
    gn = GradNet(
        num_nodes=N,
        budget=1.0,
        mask=torch.ones((N, N)) - torch.eye(N),
        adj0=torch.zeros((N, N)),
        delta_sign="free",
        final_sign="free",
        undirected=True,
        rand_init_weights=False,
        use_budget_up=True,
        cost_matrix=torch.ones((N, N)),
        cost_aggr_norm=1,
        device="cpu",
        dtype=torch.float64,
    )

    def vf(t, x, A, nested):
        # `nested` contains tensors/lists that should be cast to float64
        c = nested[0]["tensor"][0]
        return (A[0, 1] + c) * x

    x0 = torch.tensor([1.0], dtype=torch.float32)
    tt = torch.linspace(0.0, 0.1, steps=3, dtype=torch.float32)
    f_kwargs = {"nested": [{"tensor": torch.tensor([0.25], dtype=torch.float32)}]}

    t_out, y = integrate_ode(gn, vf, x0, tt, f_kwargs=f_kwargs)
    assert t_out.dtype == torch.float64
    assert y.dtype == torch.float64


@pytest.mark.xfail(reason="torchdiffeq may underflow dt for adjoint+event+reverse-time on some versions", strict=False)
def test_event_with_adjoint_and_reverse_time():
    # x' = +1, reverse-time search from t0=1 to t=0 when x crosses 0
    def vf(t, x, A):
        return torch.ones_like(x)

    def event_fn(t, x, A):
        return x[0]

    A = torch.tensor([[0.0]])
    x0 = torch.tensor([-0.1])
    tt = torch.tensor([1.0, 0.0])

    tt_p, x_p = integrate_ode(
        A,
        vf,
        x0,
        tt,
        event_fn=event_fn,
        event_options={"reverse_time": True},
        adjoint=True,
        method="rk4",  # fixed-step avoids adaptive underflow in some torchdiffeq versions
        solver_options={"step_size": 0.05},
    )
    # event at x=0; since x' = +1 and starting at -0.1 at t=1 going backward,
    # the event should be slightly before t=1
    assert tt_p.numel() >= 1


def test_sparse_gradnet_works_with_ode_by_providing_dense_A():
    """GradNet can produce a sparse adjacency; ODE should handle it.

    Desired behavior: integrate_ode should pass a dense adjacency to the
    vector field even if the underlying GradNet is sparse-backed, so common
    dense indexing like ``A[0, 1]`` is valid inside ``vf``.
    """
    N = 3
    # Sparse mask allowing a small set of undirected edges
    mask_idx = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    mask_val = torch.ones(mask_idx.shape[1], dtype=torch.float32)
    mask_sparse = torch.sparse_coo_tensor(mask_idx, mask_val, (N, N)).coalesce()

    # Sparse zero adj0 to keep the resulting adjacency sparse
    adj0_idx = torch.empty((2, 0), dtype=torch.long)
    adj0_val = torch.empty((0,), dtype=torch.float32)
    adj0_sparse = torch.sparse_coo_tensor(adj0_idx, adj0_val, (N, N)).coalesce()

    gn = GradNet(
        num_nodes=N,
        budget=1.0,
        mask=mask_sparse,
        adj0=adj0_sparse,
        delta_sign="nonnegative",
        final_sign="nonnegative",
        undirected=True,
        rand_init_weights=False,
        use_budget_up=True,
        cost_matrix=torch.ones((N, N)),
        cost_aggr_norm=1,
        device="cpu",
        dtype=torch.float32,
    )

    # Vector field uses dense-style indexing into A; this will error if A is sparse
    def vf(t, x, A):
        return (A[0, 1]) * x

    x0 = torch.tensor([1.0], dtype=torch.float32)
    tt = torch.linspace(0.0, 0.1, steps=5, dtype=torch.float32)

    # Expect successful integration once integrate_ode ensures dense A
    t_out, y = integrate_ode(gn, vf, x0, tt)
    assert t_out.shape == tt.shape
    assert y.shape == (tt.numel(), 1)
