import torch
import random
import numpy as np
import time
from functools import wraps
import torch.linalg as LA
from typing import Mapping, Optional



def _random_seed(seed):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper


def prune_edges(del_adj: torch.Tensor, threshold: float) -> torch.Tensor:
    """Prunes edges in a given adjacency tensor based on a threshold value."""
    norm = torch.abs(del_adj).sum()
    pruned = torch.where(torch.abs(del_adj) < threshold, torch.zeros_like(del_adj), del_adj)
    pruned_norm = torch.abs(pruned).sum()
    if pruned_norm < 1e-12:  # all edges pruned
        return torch.zeros_like(del_adj)
    return pruned * (norm / pruned_norm)


def positions_to_distance_matrix(positions: torch.Tensor, norm: float = 2.0):
    """Compute the pairwise distance matrix from node positions using a given norm."""
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    return LA.vector_norm(diff, ord=norm, dim=-1)


def reg_loss(del_adj: torch.Tensor) -> torch.Tensor:
    """
    Regularization loss for sparsifying the delta adjacency.
    Computes sum(sigmoid(abs(del_adj))).
    """
    # f = lambda x: torch.sigmoid(x)
    f = lambda x: torch.log(x + 1)
    return torch.sum(f(torch.abs(del_adj)))/del_adj.shape[-1]


def _to_like_struct(obj, like: torch.Tensor):
    """Recursively move/cast tensors (and NumPy) inside obj to like.device/dtype; leave others as-is."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device=like.device, dtype=like.dtype)
    if isinstance(obj, np.ndarray):  # also catches np.matrix
        # as_tensor shares memory on CPU; then we move/cast to match `like`
        t = torch.as_tensor(obj)  # stays on CPU first
        return t.to(device=like.device, dtype=like.dtype)
    if isinstance(obj, np.generic):  # NumPy scalar (e.g., np.float32(3.0))
        return torch.tensor(obj, device=like.device, dtype=like.dtype)
    if isinstance(obj, Mapping):
        return obj.__class__({k: _to_like_struct(v, like) for k, v in obj.items()})
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):  # namedtuple
        return obj.__class__(*[_to_like_struct(v, like) for v in obj])
    if isinstance(obj, (list, tuple)):
        typ = obj.__class__
        return typ(_to_like_struct(v, like) for v in obj)
    return obj  # nn.Module or anything else stays as-is 


def to_networkx(gn, pruning_threshold: float = 1e-8):
    """Export the current adjacency to a NetworkX graph.

    Edges with absolute weight below ``pruning_threshold`` are dropped.
    Supports both dense and sparse internal representations.

    :param pruning_threshold: Minimum absolute weight to keep an edge.
    :type pruning_threshold: float
    :return: A ``networkx.Graph`` if ``undirected`` else a ``DiGraph``.
    :rtype: networkx.Graph | networkx.DiGraph
    """
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - exercised when the extra is absent
        raise ImportError(
            "to_networkx requires the optional 'networkx' extra; install it with"
            " `pip install gradnet[networkx]`."
        ) from exc
    net = nx.Graph() if gn.undirected else nx.DiGraph()
    net.add_nodes_from(range(gn.num_nodes))

    A = gn()
    if isinstance(A, torch.Tensor) and A.layout != torch.strided:
        A = A.coalesce()
        idx = A.indices().t().tolist()
        vals = A.values().detach().cpu().tolist()
        if gn.undirected:
            seen = set()
            for (i, j), w in zip(idx, vals):
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                if abs(w) > pruning_threshold:
                    net.add_edge(a, b, weight=float(w))
        else:
            for (i, j), w in zip(idx, vals):
                if abs(w) > pruning_threshold:
                    net.add_edge(int(i), int(j), weight=float(w))
    else:
        adj = A.detach().cpu()
        m = gn.mask.to_dense() if isinstance(gn.mask, torch.Tensor) and gn.mask.layout != torch.strided else gn.mask
        for i in range(gn.num_nodes):
            j_range = range(i + 1, gn.num_nodes) if gn.undirected else range(gn.num_nodes)
            for j in j_range:
                w = float(adj[i, j])
                if abs(w) > pruning_threshold and (m[i, j] != 0):
                    net.add_edge(i, j, weight=w)
    return net


def plot_adjacency_heatmap(
    gn,
    *,
    ax=None,
    title: str = None,
    xlabel: str = "$j$",
    ylabel: str = "$i$",
    cbar_label: str = "$A_{ij}$",
    imshow_kwargs: Optional[dict] = None,
):
    """Plot an adjacency matrix as a heatmap.

    - If ``ax`` is ``None``, creates a new figure and axes.
    - The colorbar attaches to ``ax.figure``.
    - Accepts a GradNet-like object (callable with no args), a Torch tensor,
      or any array-like representing an adjacency.
    """
    import matplotlib.pyplot as plt
    
    # Resolve input to a NumPy array adjacency
    if isinstance(gn, torch.Tensor):
        data = gn.detach().cpu().numpy()
    elif callable(gn):  # GradNet or similar returning adjacency via __call__
        A = gn()
        data = A.detach().cpu().numpy() if isinstance(A, torch.Tensor) else np.asarray(A)
    else:
        data = np.asarray(gn)
    imshow_kwargs = {} if imshow_kwargs is None else dict(imshow_kwargs)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    im = ax.imshow(data, **imshow_kwargs)
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return im


def plot_graph(
    gn,
    *,
    ax=None,
    pruning_threshold: float = 1e-8,
    layout: str = "spring",
    node_size: float = 15.0,
    edgecolors: str = "black",
    draw_kwargs: Optional[dict] = None,
    add_colorbar: bool = True,
    colorbar_label: str = None,
):
    """Draw the NetworkX representation of ``gn``.

    - If ``ax`` is ``None``, creates a new figure and axes.
    - Uses ``to_networkx`` and derives edge widths from weights.
    - ``layout`` can be a ``networkx.draw_*`` name or a callable.
    - If `add_colorbar=True`, adds a colorbar when `node_color` is array-like.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    import numpy as np
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "plot_graph requires the optional 'networkx' extra; install it with"
            " `pip install gradnet[networkx]`."
        ) from exc

    if ax is None:
        _, ax = plt.subplots()

    net = to_networkx(gn, pruning_threshold=pruning_threshold)
    edge_weights = list(nx.get_edge_attributes(net, "weight").values())
    if not edge_weights:
        edge_weights = None

    draw_kwargs = {} if draw_kwargs is None else dict(draw_kwargs)
    draw_kwargs.setdefault("nodelist", sorted(net.nodes()))
    draw_kwargs.setdefault("node_size", node_size)
    draw_kwargs.setdefault("width", edge_weights)
    draw_kwargs.setdefault("edgecolors", edgecolors)

    draw_fn = getattr(nx, f"draw_{layout}") if isinstance(layout, str) else layout
    if not callable(draw_fn):
        raise ValueError(f"layout '{layout}' is not callable")

    # Draw the network
    draw_fn(net, ax=ax, **draw_kwargs)

    # Optionally add a colorbar
    if add_colorbar and "node_color" in draw_kwargs:
        node_color = draw_kwargs["node_color"]
        if hasattr(node_color, "__len__") and not isinstance(node_color, str):
            cmap = draw_kwargs.get("cmap", plt.cm.viridis)
            sm = ScalarMappable(cmap=cmap)
            sm.set_array(np.asarray(node_color))
            ax.figure.colorbar(sm, ax=ax, label=colorbar_label)

    return net



def _shortest_path(A: torch.Tensor, pair="full"):
    """Compute shortest path distances with SciPy and preserve Torch grads.

    - Accepts an adjacency tensor ``A`` (dense or sparse PyTorch).
    - Edge costs equal the provided weights. Zeros off-diagonal denote absence
      of edges.
    - ``pair`` may be ``"full"`` for all-pairs distances or a tuple ``(i, j)``
      for a single-source, single-target distance.
    - Uses SciPy's Dijkstra to get predecessors and reconstructs distances by
      summing Torch weights along chosen paths so gradients flow.
    - For sparse Torch tensors, converts to SciPy CSR; otherwise uses a dense
      NumPy array. Dense/sparse behavior is preserved.

    Returns:
      - If ``pair == 'full'``: ``torch.Tensor`` of shape ``(N, N)`` with grads.
      - If ``pair`` is ``(i, j)``: a scalar ``torch.Tensor`` distance.
    """
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path as sp_shortest_path
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError("scipy is required for shortest_path computation") from e

    if not isinstance(A, torch.Tensor):
        raise TypeError("A must be a torch.Tensor (dense or sparse)")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency must be a square 2D matrix")

    N = A.shape[0]

    # Infer undirectedness by symmetry (tolerant). Symmetric => undirected.
    Adense_for_sym = A.to_dense() if A.layout != torch.strided else A
    undirected = torch.allclose(Adense_for_sym, Adense_for_sym.T)

    # Build SciPy graph (costs) from Torch
    if A.layout != torch.strided:
        Ac = A.coalesce()
        ii = Ac.indices()[0].detach().cpu().numpy()
        jj = Ac.indices()[1].detach().cpu().numpy()
        vv_np = Ac.values().detach().cpu().numpy()
        C = csr_matrix((vv_np, (ii, jj)), shape=(N, N))
    else:
        # zeros off-diagonal represent no edge for csgraph; keep diagonal zeros
        C = A.detach().cpu().numpy()

    directed = not undirected

    # Helper to reconstruct Torch-summed cost along predecessor path
    def _reconstruct_cost_from_predecessors(src: int, dst: int, pred_row: np.ndarray) -> torch.Tensor:
        if src == dst:
            # return a dense scalar to avoid sparse/dense copy issues downstream
            return torch.zeros((), dtype=A.dtype, device=A.device)
        k = int(dst)
        if k < 0 or k >= N:
            return torch.tensor(float("inf"), dtype=A.dtype, device=A.device)
        total = torch.zeros((), dtype=A.dtype, device=A.device)
        Adense = A.to_dense() if (A.layout != torch.strided) else A
        while k != src:
            pk = int(pred_row[k])
            if pk == -9999 or pk < 0:
                return torch.tensor(float("inf"), dtype=A.dtype, device=A.device)  # unreachable
            total = total + Adense[pk, k]
            k = pk
        return total

    if pair == "full":
        # Compute all-pairs predecessors once
        dist_np, pred_np = sp_shortest_path(C, directed=directed, return_predecessors=True, unweighted=False)
        # Reconstruct distances via Torch sums along chosen paths
        out = torch.empty((N, N), device=A.device, dtype=A.dtype)
        Adense = A.to_dense() if (A.layout != torch.strided) else A

        def cost_entry(u, v):
            return Adense[u, v]

        for i in range(N):
            pred_row = pred_np[i]
            for j in range(N):
                if i == j:
                    out[i, j] = torch.zeros((), device=A.device, dtype=A.dtype)
                    continue
                # unreachable?
                if not np.isfinite(dist_np[i, j]):
                    out[i, j] = torch.tensor(float("inf"), device=A.device, dtype=A.dtype)
                    continue
                # backtrack using predecessors and sum torch costs along the path
                k = j
                total = torch.zeros((), device=A.device, dtype=A.dtype)
                while k != i:
                    pk = int(pred_row[k])
                    if pk == -9999 or pk < 0:
                        total = torch.tensor(float("inf"), device=A.device, dtype=A.dtype)
                        break
                    total = total + cost_entry(pk, k)
                    k = pk
                out[i, j] = total
        return out

    # pair = (i, j): single-source, single-target
    if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
        raise ValueError("pair must be 'full' or a tuple (i, j)")
    i, j = int(pair[0]), int(pair[1])
    dist_np, pred_np = sp_shortest_path(C, directed=directed, indices=i, return_predecessors=True, unweighted=False)
    # If unreachable
    if not np.isfinite(dist_np[j]):
        return torch.tensor(float("inf"), dtype=A.dtype, device=A.device)
    return _reconstruct_cost_from_predecessors(i, j, pred_np)
