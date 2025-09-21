import os
import sys
import math

import pytest

# Skip the entire module if required deps are missing
torch = pytest.importorskip("torch")

# Ensure `src/` is on sys.path for local imports when using a src layout
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir, 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gradnet.gradnet import (
    normalize,
    positivize,
    symmetrize,
    DenseParameterization,
    SparseParameterization,
    GradNet,
)
from gradnet.utils import to_networkx
from gradnet.trainer import fit


def _p_cost_norm(x, cost, p: int):
    # cost may be dense or sparse; ensure dense elementwise product
    if hasattr(cost, "layout") and cost.layout != torch.strided:
        cost = cost.to_dense()
    return (torch.abs(cost * x) ** p).sum() ** (1.0 / p)


def test_helpers_normalize_and_transforms():
    A = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    C = torch.ones_like(A)
    out = normalize(A, norm_val=10.0, cost_aggr_norm=2, cost_matrix=C, strict=True)
    s = _p_cost_norm(out, C, 2)
    assert torch.allclose(s, torch.tensor(10.0), atol=1e-6)

    # strict=False should not upscale
    B = torch.tensor([[0.1, 0.0], [0.0, 0.0]])
    out2 = normalize(B, norm_val=10.0, cost_aggr_norm=1, cost_matrix=C, strict=False)
    assert torch.allclose(out2, B)  # unchanged because upscaling is disabled

    # transforms
    X = torch.tensor([[0.0, 2.0], [-3.0, 0.0]])
    assert torch.equal(positivize(X), X ** 2)
    S = symmetrize(torch.tensor([[0.0, 1.0], [2.0, 0.0]]))
    assert torch.allclose(S, torch.tensor([[0.0, 1.5], [1.5, 0.0]]))


def test_denseparam_forward_pipeline_budget_and_mask():
    N = 3
    budget = 5.0
    mask = torch.ones((N, N)) - torch.eye(N)
    cost = torch.ones((N, N))

    dp = DenseParameterization(
        num_nodes=N,
        budget=budget,
        mask=mask,
        cost_matrix=cost,
        delta_sign="nonnegative",
        undirected=True,
        use_budget_up=True,
        cost_aggr_norm=1,
        rand_init_weights=False,
    )
    # set deterministic raw to ones
    dp.set_initial_state(torch.ones((N, N)))

    delta = dp()
    # mask zeroes diagonal
    assert torch.all(delta.diag() == 0)
    # p=1 budget with unit cost
    s = _p_cost_norm(delta, cost, 1)
    assert torch.allclose(s, torch.tensor(budget), atol=1e-5)
    # symmetry and positivity
    assert torch.allclose(delta, delta.T)
    assert torch.all(delta >= 0)


def test_denseparam_no_upscale_when_disabled():
    N = 3
    mask = torch.ones((N, N)) - torch.eye(N)
    cost = torch.ones((N, N))
    dp = DenseParameterization(
        num_nodes=N,
        budget=1000.0,  # huge budget
        mask=mask,
        cost_matrix=cost,
        delta_sign="free",
        undirected=False,
        use_budget_up=False,  # do not upscale
        cost_aggr_norm=2,
        rand_init_weights=False,
    )
    # a tiny raw with a single nonzero entry
    raw = torch.zeros((N, N))
    raw[0, 1] = 0.01
    dp.set_initial_state(raw)
    delta = dp()
    # since strict=False, result should not be further upscaled by normalize;
    # it should match the masked raw after internal renormalization
    expected = (dp.delta_adj_raw.detach() * mask)
    assert torch.allclose(delta.detach(), expected)


def test_sparseparam_budget_and_mirroring():
    # Undirected graph with edges (0,1) and (1,2)
    N = 3
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)  # i<j pairs
    E = edge_index.shape[1]
    cost_p_sum = torch.ones((E,)) * 2.0  # undirected sum per edge for p=1
    sp = SparseParameterization(
        num_nodes=N,
        budget=3.0,
        edge_index=edge_index,
        cost_p_sum=cost_p_sum,
        delta_sign="nonnegative",
        undirected=True,
        use_budget_up=True,
        cost_aggr_norm=1,
        rand_init_weights=False,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    # set raw weights to ones -> renorm will scale to sqrt(E)
    sp.set_initial_state(torch.ones((E,)))
    delta_sp = sp()
    assert delta_sp.layout == torch.sparse_coo
    delta_sp = delta_sp.coalesce()
    ii, jj = delta_sp.indices()
    # Mirroring present: entries (0,1),(1,0) and (1,2),(2,1)
    got_pairs = set(map(tuple, torch.stack([ii, jj], dim=1).tolist()))
    assert got_pairs == {(0, 1), (1, 0), (1, 2), (2, 1)}
    assert delta_sp.values().numel() == 2 * E
    # Check budget p=1 using the vector formulation
    vals = sp.delta_adj_raw
    p = 1
    s = (torch.abs(vals) ** p * cost_p_sum).sum() ** (1.0 / p)
    scale = sp.budget / max(1e-8, s)
    assert torch.allclose(
        (torch.abs(vals * scale) ** p * cost_p_sum).sum() ** (1.0 / p),
        torch.tensor(sp.budget),
        atol=1e-5,
    )


def test_gradnet_dense_and_sparse_backends_and_forward_addition():
    # Dense mask path
    N = 3
    mask_dense = torch.ones((N, N)) - torch.eye(N)
    gn_dense = GradNet(
        num_nodes=N,
        budget=2.0,
        mask=mask_dense,
        adj0=torch.eye(N),
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
    assert isinstance(gn_dense.param, DenseParameterization)
    A = gn_dense()
    assert A.shape == (N, N)
    # diagonal retains adj0
    assert torch.allclose(torch.diag(A), torch.ones(N))

    # Sparse mask path
    mask_idx = torch.tensor([[0, 1, 2], [1, 2, 0]])  # include a cycle (0,1),(1,2),(2,0)
    mask_val = torch.ones(mask_idx.shape[1])
    mask_sparse = torch.sparse_coo_tensor(mask_idx, mask_val, (N, N)).coalesce()
    gn_sparse = GradNet(
        num_nodes=N,
        budget=1.5,
        mask=mask_sparse,
        adj0=torch.zeros((N, N)),
        delta_sign="free",
        final_sign="free",
        undirected=True,
        rand_init_weights=False,
        use_budget_up=True,
        cost_matrix=torch.ones((N, N)),
        cost_aggr_norm=2,
        device="cpu",
        dtype=torch.float64,
    )
    assert isinstance(gn_sparse.param, SparseParameterization)
    delta = gn_sparse.get_delta_adj()
    assert delta.layout == torch.sparse_coo
    A2 = gn_sparse()
    # adj0 is dense zeros; adding sparse delta yields dense tensor per implementation
    assert A2.layout == torch.strided

    # final_sign projection enforces the desired sign cone on the output
    gn_nonpos = GradNet(
        num_nodes=N,
        budget=1.0,
        mask=mask_dense,
        adj0=torch.zeros((N, N)),
        delta_sign="nonnegative",
        final_sign="nonpositive",
        undirected=True,
        rand_init_weights=False,
        use_budget_up=True,
        cost_matrix=torch.ones((N, N)),
        cost_aggr_norm=1,
        device="cpu",
        dtype=torch.float32,
    )
    gn_nonpos.set_initial_state(torch.ones((N, N)))
    A3 = gn_nonpos()
    assert torch.all(A3 <= 0)


def test_prepare_edge_list_and_cost_aggregation_and_gather_helpers():
    N = 4
    # Mask with diagonal entries to be dropped and duplicate undirected entries
    idx = torch.tensor([[0, 0, 1, 2, 3], [0, 1, 2, 1, 3]])
    val = torch.ones(idx.shape[1])
    mask_sparse = torch.sparse_coo_tensor(idx, val, (N, N)).coalesce()
    cost_dense = torch.zeros((N, N))
    # set costs for (0,1) and (1,0), (1,2) only one side to test missing handling
    cost_dense[0, 1] = 2.0
    cost_dense[1, 0] = 2.0
    cost_dense[1, 2] = 3.0

    # access private helpers through the class instance
    gn = GradNet(
        num_nodes=N,
        budget=1.0,
        mask=mask_sparse,
        adj0=None,
        undirected=True,
        device="cpu",
        dtype=torch.float32,
    )

    edge_index, cost_p_sum = gn._prepare_edge_list(
        mask=mask_sparse, cost_matrix=cost_dense, undirected=True, p=2, dtype=torch.float32, device=torch.device("cpu")
    )
    # unique undirected edges among [(0,1),(1,2)] from the mask (diagonal dropped)
    expected = set([(0, 1), (1, 2)])
    got = set(map(tuple, edge_index.t().tolist()))
    assert got == expected
    # p=2: for (0,1) -> 2^2+2^2=8, for (1,2) -> 3^2+0^2=9
    assert torch.allclose(cost_p_sum, torch.tensor([8.0, 9.0]))

    # _gather_sparse_values tested implicitly via cost aggregation above, but also check sparse path
    cost_sparse = cost_dense.to_sparse().coalesce()
    edge_index2, cost_p_sum2 = gn._prepare_edge_list(
        mask=mask_sparse, cost_matrix=cost_sparse, undirected=True, p=2, dtype=torch.float32, device=torch.device("cpu")
    )
    assert torch.equal(edge_index2, edge_index)
    assert torch.allclose(cost_p_sum2, cost_p_sum)


def test_to_networkx_prunes_and_types():
    N = 3
    mask = torch.ones((N, N)) - torch.eye(N)
    gn = GradNet(
        num_nodes=N,
        budget=1.0,
        mask=mask,
        adj0=torch.zeros((N, N)),
        undirected=True,
        rand_init_weights=False,
        use_budget_up=True,
        device="cpu",
        dtype=torch.float32,
    )
    # Make a tiny budget so some edges fall below threshold
    net = to_networkx(gn, pruning_threshold=1e-6)
    assert net.number_of_nodes() == N
    # undirected graph expected
    import networkx as nx

    assert isinstance(net, nx.Graph)
    # edges non-empty if budget > 0
    assert net.number_of_edges() >= 1


def test_to_networkx_directed_graph_edges():
    N = 3
    mask = torch.ones((N, N)) - torch.eye(N)
    gn = GradNet(
        num_nodes=N,
        budget=1.0,
        mask=mask,
        adj0=torch.zeros((N, N)),
        undirected=False,
        rand_init_weights=False,
        use_budget_up=True,
        device="cpu",
        dtype=torch.float32,
    )
    net = to_networkx(gn, pruning_threshold=0.0)
    import networkx as nx
    assert isinstance(net, nx.DiGraph)
    # With directed graph, (i,j) and (j,i) are distinct; since initialization is
    # symmetric here, expect both directions to exist for at least one pair.
    assert any(net.has_edge(i, j) and net.has_edge(j, i) for i in range(N) for j in range(N) if i != j)
    # Total directed edges equals N*(N-1) for a fully masked dense case.
    assert net.number_of_edges() == N * (N - 1)


def test_to_networkx_directed_with_asymmetric_mask():
    # Use an asymmetric mask (strictly upper triangular) to induce asymmetric edges
    N = 4
    full = torch.ones((N, N))
    mask = torch.triu(full, diagonal=1)  # i<j allowed, j<i disallowed
    gn = GradNet(
        num_nodes=N,
        budget=1.0,
        mask=mask,
        adj0=torch.zeros((N, N)),
        undirected=False,
        rand_init_weights=False,
        use_budget_up=True,
        device="cpu",
        dtype=torch.float32,
    )
    net = to_networkx(gn, pruning_threshold=0.0)
    import networkx as nx
    assert isinstance(net, nx.DiGraph)
    # Expect edges only in upper-triangular direction
    for i in range(N):
        for j in range(N):
            if i < j:
                assert net.has_edge(i, j)
                assert not net.has_edge(j, i)
            elif i > j:
                assert not net.has_edge(i, j)


def test_denseparam_dof_renorm_scale():
    N = 5
    mask = torch.ones((N, N)) - torch.eye(N)
    dp = DenseParameterization(
        num_nodes=N,
        budget=1.0,
        mask=mask,
        cost_matrix=torch.ones((N, N)),
        delta_sign="free",
        undirected=True,
        use_budget_up=True,
        cost_aggr_norm=1,
        rand_init_weights=False,
    )
    # ones init -> renorm to sqrt(DOF) where DOF = N*(N-1)/2 (off-diagonal undirected)
    dof = N * (N - 1) // 2
    norm = torch.linalg.norm(dp.delta_adj_raw.detach())
    assert torch.allclose(norm, torch.tensor(dof, dtype=norm.dtype).sqrt(), atol=1e-5)


def test_prepare_edge_list_none_cost_returns_unit_weights():
    N = 4
    # Undirected unique edges: (0,1),(1,2)
    mask_idx = torch.tensor([[0, 1], [1, 2]])
    mask_val = torch.ones(2)
    mask_sp = torch.sparse_coo_tensor(mask_idx, mask_val, (N, N)).coalesce()
    gn = GradNet(num_nodes=N, budget=1.0, mask=mask_sp, undirected=True, device="cpu", dtype=torch.float32)
    edge_index, cost_p_sum = gn._prepare_edge_list(
        mask=mask_sp, cost_matrix=None, undirected=True, p=1, dtype=torch.float32, device=torch.device("cpu")
    )
    assert torch.allclose(cost_p_sum, torch.full((edge_index.shape[1],), 2.0))


def test_gradnet_export_and_from_config_roundtrip():
    mask = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    adj0 = torch.tensor([[0.0, 0.5], [0.5, 0.0]])
    cost = torch.ones((2, 2))

    gn = GradNet(
        num_nodes=2,
        budget=1.0,
        mask=mask,
        adj0=adj0,
        delta_sign="nonnegative",
        final_sign="nonnegative",
        undirected=True,
        rand_init_weights=False,
        use_budget_up=True,
        cost_matrix=cost,
        cost_aggr_norm=1,
        device="cpu",
        dtype=torch.float32,
    )

    config = gn.export_config()
    gn_rebuilt = GradNet.from_config(config)

    assert gn_rebuilt.num_nodes == gn.num_nodes
    assert math.isclose(gn_rebuilt.budget, gn.budget)
    assert gn_rebuilt.delta_sign == gn.delta_sign
    assert gn_rebuilt.final_sign == gn.final_sign
    assert gn_rebuilt.undirected == gn.undirected
    assert gn_rebuilt.use_budget_up == gn.use_budget_up
    assert gn_rebuilt.cost_aggr_norm == gn.cost_aggr_norm
    assert torch.allclose(gn_rebuilt.mask, gn.mask)
    assert torch.allclose(gn_rebuilt.adj0, gn.adj0)
    assert torch.allclose(gn_rebuilt.cost_matrix, gn.cost_matrix)


def test_gradnet_from_checkpoint_roundtrip(tmp_path):
    pytest.importorskip("pytorch_lightning")

    mask = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    cost = torch.ones((2, 2), dtype=torch.float32)

    gn = GradNet(
        num_nodes=2,
        budget=1.0,
        mask=mask,
        adj0=torch.zeros((2, 2), dtype=torch.float32),
        delta_sign="nonnegative",
        final_sign="nonnegative",
        undirected=True,
        rand_init_weights=False,
        use_budget_up=True,
        cost_matrix=cost,
        cost_aggr_norm=1,
        device="cpu",
        dtype=torch.float32,
    )

    def loss_fn(model: GradNet):
        A = model()
        loss = (A ** 2).mean()
        return loss, {"adj_sum": A.sum()}

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer, best_ckpt = fit(
        gn=gn,
        loss_fn=loss_fn,
        num_updates=2,
        optim_cls=torch.optim.SGD,
        optim_kwargs={"lr": 0.1},
        enable_checkpointing=True,
        checkpoint_dir=str(ckpt_dir),
        logger=False,
        accelerator="cpu",
    )

    assert isinstance(best_ckpt, str) and os.path.exists(best_ckpt)

    orig_state = {k: v.detach().clone() for k, v in gn.state_dict().items()}

    reloaded = GradNet.from_checkpoint(best_ckpt)

    for key, value in orig_state.items():
        assert torch.allclose(reloaded.state_dict()[key], value)

    config = trainer.lightning_module.hparams.get("gradnet_config")
    assert config is not None
    assert config["num_nodes"] == gn.num_nodes
    assert torch.allclose(config["mask"], gn.mask.cpu())
