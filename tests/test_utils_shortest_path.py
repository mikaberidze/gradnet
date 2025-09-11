import pytest

try:
    import scipy  # noqa: F401
    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    SCIPY_AVAILABLE = False

# Skip whole module if SciPy or Torch missing
pytestmark = pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy is required for shortest_path tests")
torch = pytest.importorskip("torch")


from gradnet.utils import shortest_path  # noqa: E402


def make_path_graph(N: int = 5, weights=None, sparse: bool = False, dtype=torch.float64):
    """Build a path graph adjacency of size N with given edge weights.
    Nodes are 0..N-1 and edges are (i, i+1) for i in 0..N-2.
    """
    if weights is None:
        weights = [float(i + 1) for i in range(N - 1)]  # 1,2,3,...
    assert len(weights) == N - 1
    if not sparse:
        A = torch.zeros((N, N), dtype=dtype)
        for i, w in enumerate(weights):
            A[i, i + 1] = w
            A[i + 1, i] = w
        return A
    # sparse COO
    rows = []
    cols = []
    vals = []
    for i, w in enumerate(weights):
        rows += [i, i + 1]
        cols += [i + 1, i]
        vals += [w, w]
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=dtype)
    return torch.sparse_coo_tensor(indices, values, size=(N, N)).coalesce()


def make_star_graph(N: int = 5, center: int = 0, weights=None, sparse: bool = False, dtype=torch.float64):
    """Build a star graph adjacency of size N with given edge weights from center to leaves.
    weights is a list of length N-1 mapping to leaves in ascending node order excluding center.
    """
    assert 0 <= center < N
    leaves = [i for i in range(N) if i != center]
    if weights is None:
        # Make them distinct to exercise invert path costs
        weights = [float(i + 1) for i in range(len(leaves))]  # 1,2,3,...
    assert len(weights) == N - 1
    if not sparse:
        A = torch.zeros((N, N), dtype=dtype)
        for leaf, w in zip(leaves, weights):
            A[center, leaf] = w
            A[leaf, center] = w
        return A
    # sparse COO
    rows = []
    cols = []
    vals = []
    for leaf, w in zip(leaves, weights):
        rows += [center, leaf]
        cols += [leaf, center]
        vals += [w, w]
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=dtype)
    return torch.sparse_coo_tensor(indices, values, size=(N, N)).coalesce()


def expected_path_dist_matrix(weights):
    N = len(weights) + 1
    def cost(w):
        return float(w)
    D = [[0.0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                D[i][j] = 0.0
            else:
                a, b = sorted((i, j))
                s = 0.0
                for k in range(a, b):
                    s += cost(weights[k])
                D[i][j] = s
    return torch.tensor(D, dtype=torch.float64)


def expected_star_dist_matrix(N: int, center: int, weights):
    leaves = [i for i in range(N) if i != center]
    w_by_node = {leaf: float(w) for leaf, w in zip(leaves, weights)}
    def cost(w):
        return float(w)
    D = [[0.0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                D[i][j] = 0.0
            elif i == center:
                # center to leaf j
                D[i][j] = cost(w_by_node[j])
            elif j == center:
                # leaf i to center
                D[i][j] = cost(w_by_node[i])
            else:
                # leaf i to leaf j via center
                D[i][j] = cost(w_by_node[i]) + cost(w_by_node[j])
    return torch.tensor(D, dtype=torch.float64)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("graph_type", ["path", "star"])
def test_shortest_path_full_matrix(graph_type, sparse):
    dtype = torch.float64
    if graph_type == "path":
        # N=5, weights 1,2,3,4
        weights = [1.0, 2.0, 3.0, 4.0]
        A = make_path_graph(N=5, weights=weights, sparse=sparse, dtype=dtype)
        D_expected = expected_path_dist_matrix(weights)
    else:  # star
        # N=5, center=0, weights to leaves 1..4
        weights = [1.0, 2.0, 3.0, 4.0]
        A = make_star_graph(N=5, center=0, weights=weights, sparse=sparse, dtype=dtype)
        D_expected = expected_star_dist_matrix(N=5, center=0, weights=weights)

    D = shortest_path(A, pair="full")

    assert isinstance(D, torch.Tensor)
    assert D.shape == D_expected.shape
    # Compare with tolerance due to float ops
    assert torch.allclose(D.cpu(), D_expected.to(D.dtype), rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("graph_type", ["path", "star"])
def test_shortest_path_single_pair_matches_full(graph_type, sparse):
    dtype = torch.float64
    if graph_type == "path":
        weights = [1.0, 2.0, 3.0, 4.0]  # N=5
        A = make_path_graph(N=5, weights=weights, sparse=sparse, dtype=dtype)
        # Compare (0, 4) pair
        D_full = shortest_path(A, pair="full")
        d_pair = shortest_path(A, pair=(0, 4))
        assert torch.is_tensor(d_pair) and d_pair.ndim == 0
        assert torch.allclose(d_pair, D_full[0, 4], rtol=1e-7, atol=1e-9)
    else:
        weights = [1.0, 2.0, 3.0, 4.0]  # N=5, center 0
        A = make_star_graph(N=5, center=0, weights=weights, sparse=sparse, dtype=dtype)
        # Compare leaves (1, 2)
        D_full = shortest_path(A, pair="full")
        d_pair = shortest_path(A, pair=(1, 2))
        assert torch.is_tensor(d_pair) and d_pair.ndim == 0
        assert torch.allclose(d_pair, D_full[1, 2], rtol=1e-7, atol=1e-9)


def test_gradients_single_pair_path_dense():
    # Dense path graph N=5, weights 1..4
    dtype = torch.float64
    weights = [1.0, 2.0, 3.0, 4.0]
    A = make_path_graph(N=5, weights=weights, sparse=False, dtype=dtype).detach().clone().requires_grad_(True)

    # distance from 0 -> 4 uses edges (0,1),(1,2),(2,3),(3,4) in that orientation
    d = shortest_path(A, pair=(0, 4))
    d.backward()

    assert A.grad is not None
    grad = A.grad.detach()

    # Expected gradients
    expected = [1.0, 1.0, 1.0, 1.0]

    path_edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    for (u, v), gexp in zip(path_edges, expected):
        assert torch.isclose(grad[u, v], torch.tensor(gexp, dtype=dtype), rtol=1e-7, atol=1e-9)

    # Non-path (and opposite-orientation) entries should have ~0 gradient
    N = 5
    for u in range(N):
        for v in range(N):
            if (u, v) in path_edges:
                continue
            # diagonals and all other off-diagonals should be zero
            assert torch.isclose(grad[u, v], torch.tensor(0.0, dtype=dtype), atol=1e-10)



def test_gradients_single_pair_star_dense():
    # Dense star graph N=5, center=0, leaf weights 1..4 for nodes 1..4
    dtype = torch.float64
    weights = [1.0, 2.0, 3.0, 4.0]
    A = make_star_graph(N=5, center=0, weights=weights, sparse=False, dtype=dtype).detach().clone().requires_grad_(True)

    # Pair leaf1->leaf2 traverses edges (0,2) then (1,0) in predecessor backtrack
    d = shortest_path(A, pair=(1, 2))
    d.backward()

    assert A.grad is not None
    grad = A.grad.detach()

    # Edges used: (0,2) with weight 2, and (1,0) with weight 1
    assert torch.isclose(grad[0, 2], torch.tensor(1.0, dtype=dtype), rtol=1e-7, atol=1e-9)
    assert torch.isclose(grad[1, 0], torch.tensor(1.0, dtype=dtype), rtol=1e-7, atol=1e-9)

    # Others ~0
    N = 5
    used = {(0, 2), (1, 0)}
    for u in range(N):
        for v in range(N):
            if (u, v) in used:
                continue
            assert torch.isclose(grad[u, v], torch.tensor(0.0, dtype=dtype), atol=1e-10)


    # Removed pruning tests: simplified API has no pruning/invert options
