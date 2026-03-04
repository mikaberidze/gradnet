import os
import sys
import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchdiffeq")

# Ensure `src/` is on sys.path for local imports when using a src layout
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir, "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gradnet.gradnet import DenseParameterization, GradNet, SparseParameterization
from gradnet.ode import integrate_ode


def _make_grid_mask(rows: int, cols: int, *, dtype: torch.dtype) -> torch.Tensor:
    n = rows * cols
    mask = torch.zeros((n, n), dtype=dtype)

    def node_id(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            u = node_id(r, c)
            if r + 1 < rows:
                v = node_id(r + 1, c)
                mask[u, v] = 1.0
                mask[v, u] = 1.0
            if c + 1 < cols:
                v = node_id(r, c + 1)
                mask[u, v] = 1.0
                mask[v, u] = 1.0
    return mask


def _dense_to_sparse_mask(mask_dense: torch.Tensor) -> torch.Tensor:
    idx = torch.nonzero(mask_dense, as_tuple=False).T
    vals = torch.ones(idx.shape[1], dtype=mask_dense.dtype, device=mask_dense.device)
    return torch.sparse_coo_tensor(idx, vals, mask_dense.shape).coalesce()


def _as_dense(A: torch.Tensor) -> torch.Tensor:
    if hasattr(A, "layout") and A.layout != torch.strided:
        return A.to_dense()
    return A


def _heat_diffusion_vf(t, x, A, **_kwargs):
    degree = A.sum(dim=1)
    return A @ x - degree * x


def _target_heat_event(t, x, A, threshold, target_idx, **_kwargs):
    del t, A
    # Some torchdiffeq event implementations may pass a stacked state (e.g. [2, n]).
    x_target = x[..., target_idx]
    if x_target.ndim > 0:
        x_target = x_target.reshape(-1)[-1]
    return threshold - x_target


def _event_time(
    model: GradNet,
    *,
    x0: torch.Tensor,
    tt: torch.Tensor,
    threshold: float,
    target_idx: int,
    track_gradients: bool = True,
) -> torch.Tensor:
    _, _, t_event, _ = integrate_ode(
        model,
        _heat_diffusion_vf,
        x0,
        tt,
        f_kwargs={"threshold": threshold, "target_idx": target_idx},
        event_fn=_target_heat_event,
        method="rk4",
        solver_options={"step_size": 0.1},
        rtol=1e-5,
        atol=1e-6,
        track_gradients=track_gradients,
    )
    return t_event


def _optimize_event_time(
    model: GradNet,
    *,
    x0: torch.Tensor,
    tt: torch.Tensor,
    threshold: float,
    target_idx: int,
    lr: float,
    num_updates: int,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    t_start = time.perf_counter()
    for _ in range(num_updates):
        opt.zero_grad(set_to_none=True)
        t_evt = _event_time(
            model,
            x0=x0,
            tt=tt,
            threshold=threshold,
            target_idx=target_idx,
            track_gradients=True,
        )
        t_evt.backward()
        opt.step()
        model.renorm_params()
        history.append(float(t_evt.detach().cpu()))

    elapsed = time.perf_counter() - t_start
    final_t = float(
        _event_time(
            model,
            x0=x0,
            tt=tt,
            threshold=threshold,
            target_idx=target_idx,
            track_gradients=False,
        )
        .detach()
        .cpu()
    )
    with torch.no_grad():
        final_adj = _as_dense(model().detach().clone())
    return history, final_t, final_adj, elapsed


def test_sparse_and_dense_encodings_match_on_grid_diffusion_event_time():
    torch.manual_seed(0)
    dtype = torch.float64
    rows, cols = 3, 3
    n = rows * cols
    source_idx = 0
    target_idx = n - 1
    threshold = 0.03

    mask_dense = _make_grid_mask(rows, cols, dtype=dtype)
    mask_sparse = _dense_to_sparse_mask(mask_dense)

    common = dict(
        num_nodes=n,
        budget=12.0,
        adj0=torch.zeros((n, n), dtype=dtype),
        delta_sign="nonnegative",
        final_sign="nonnegative",
        undirected=True,
        rand_init_weights=False,
        use_budget_up=True,
        cost_matrix=torch.ones((n, n), dtype=dtype),
        cost_aggr_norm=1,
        device="cpu",
        dtype=dtype,
    )

    gn_dense = GradNet(mask=mask_dense, **common)
    gn_sparse = GradNet(mask=mask_sparse, **common)
    assert isinstance(gn_dense.param, DenseParameterization)
    assert isinstance(gn_sparse.param, SparseParameterization)

    dense_raw0 = torch.zeros((n, n), dtype=dtype)
    dense_raw0[mask_dense > 0] = 1.0
    gn_dense.set_initial_state(dense_raw0)
    gn_sparse.set_initial_state(torch.ones_like(gn_sparse.param.delta_adj_raw))

    A0_dense = _as_dense(gn_dense().detach())
    A0_sparse = _as_dense(gn_sparse().detach())
    assert torch.allclose(A0_dense, A0_sparse, atol=1e-10, rtol=1e-10)

    x0 = torch.zeros(n, dtype=dtype)
    x0[source_idx] = 1.0
    tt = torch.linspace(0.0, 12.0, steps=121, dtype=dtype)

    t0_dense = float(_event_time(gn_dense, x0=x0, tt=tt, threshold=threshold, target_idx=target_idx).detach().cpu())
    t0_sparse = float(_event_time(gn_sparse, x0=x0, tt=tt, threshold=threshold, target_idx=target_idx).detach().cpu())
    assert t0_dense == pytest.approx(t0_sparse, rel=1e-6, abs=1e-8)
    assert t0_dense < float(tt[-1])

    _, dense_final_t, dense_final_adj, dense_elapsed = _optimize_event_time(
        gn_dense,
        x0=x0,
        tt=tt,
        threshold=threshold,
        target_idx=target_idx,
        lr=0.03,
        num_updates=100,
    )
    _, sparse_final_t, sparse_final_adj, sparse_elapsed = _optimize_event_time(
        gn_sparse,
        x0=x0,
        tt=tt,
        threshold=threshold,
        target_idx=target_idx,
        lr=0.03,
        num_updates=100,
    )

    assert dense_elapsed > 0.0
    assert sparse_elapsed > 0.0
    assert dense_final_t <= t0_dense + 1e-6
    assert sparse_final_t <= t0_sparse + 1e-6
    assert dense_final_t == pytest.approx(sparse_final_t, rel=1e-2, abs=2e-2)
    assert torch.allclose(dense_final_adj, sparse_final_adj, atol=2e-2, rtol=2e-2)
