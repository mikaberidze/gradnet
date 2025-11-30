import torch
import random
import numpy as np
import time
import warnings
import tempfile
from functools import wraps
from pathlib import Path
import torch.linalg as LA
from typing import Mapping, Optional, Union
import csv



def random_seed(seed):
    '''Set random seed for reproducibility. Works with torch, numpy, and random.'''
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def prune_edges(
    del_adj: torch.Tensor,
    *,
    threshold: Optional[float] = None,
    target_edge_number: Optional[int] = None,
    renorm: bool = True,
) -> torch.Tensor:
    """Prune edges in an adjacency-like tensor.

    Use either a numeric ``threshold`` (keeps entries with ``abs(x) >= threshold``)
    or specify ``target_edge_number`` to automatically determine a threshold that
    yields exactly that many unpruned entries. Exactly one of these must be
    provided.

    If ``renorm`` is True, rescales the pruned tensor to match the original L1
    norm. If all entries are pruned, returns an all-zero tensor.
    """
    # Validate mutually exclusive arguments
    if (threshold is None) == (target_edge_number is None):
        raise ValueError("Provide exactly one of 'threshold' or 'target_edge_number'.")

    # Determine threshold via target count if requested
    if threshold is None:
        k = int(target_edge_number)  # type: ignore[arg-type]
        abs_vals = torch.abs(del_adj).reshape(-1)
        # Consider only strictly positive magnitudes so zeros are always pruned
        pos_vals = abs_vals[abs_vals > 0]
        M = int(pos_vals.numel())
        if k < 0 or k > M:
            raise ValueError(
                f"target_edge_number must be between 0 and {M} for the given tensor; got {k}."
            )
        if M == 0:
            # Nothing to keep regardless of k; everything is zero already
            pruned = torch.zeros_like(del_adj)
            return pruned

        # Handle boundary cases explicitly
        if k == 0:
            # Threshold above the maximum magnitude prunes everything
            vmax = pos_vals.max().item()
            # Use machine epsilon for floating tensors; fall back to small constant otherwise
            try:
                eps = float(torch.finfo(pos_vals.dtype).eps)
            except TypeError:
                eps = 1e-12
            threshold = vmax + eps
        elif k == M:
            # Keep everything with positive magnitude
            threshold = pos_vals.min().item()
        else:
            # Find a threshold in (v_{k+1}, v_k] to keep exactly k entries.
            # Sort magnitudes descending and examine neighbors around k.
            sorted_vals = torch.sort(pos_vals, descending=True).values
            v_k = sorted_vals[k - 1].item()
            v_next = sorted_vals[k].item()
            if v_k == v_next:
                # Exact k is impossible due to ties at the boundary
                raise ValueError(
                    "Cannot achieve the requested target_edge_number exactly due to "
                    "duplicate magnitudes at the pruning boundary."
                )
            threshold = (v_k + v_next) / 2.0

    # Apply pruning using the resolved threshold
    norm = torch.abs(del_adj).sum()
    pruned = torch.where(torch.abs(del_adj) < float(threshold), torch.zeros_like(del_adj), del_adj)
    if not renorm:
        return pruned
    pruned_norm = torch.abs(pruned).sum()
    if pruned_norm < 1e-12:  # all edges pruned
        return torch.zeros_like(del_adj)
    return pruned * (norm / pruned_norm)


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
    add_colorbar: bool = False,
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


def load_scalars(log_dir: Union[str, Path]):
    """Return shared steps and a dict of scalar series from Lightning logs.

    The ``log_dir`` can be either a specific version directory
    (e.g., ``lightning_logs/gradnet/version_3``) or the parent folder that
    contains multiple ``version_*`` subdirectories (e.g.,
    ``lightning_logs/gradnet``). This function prefers CSV logs when present
    and falls back to TensorBoard event files if available.

    Returns ``(steps, series)`` where ``steps`` is a single list of integers
    (epoch/step) shared by all metrics, and ``series`` is a mapping
    ``{name: values}`` with values aligned to ``steps``. Missing values are
    filled with ``nan``.

    Usage:
        >>> steps, series = load_scalar_series('lightning_logs/gradnet')
        >>> loss = series['loss']

    :param log_dir: Path to a logger directory or its parent.
    :return: tuple[list[int], dict[str, list[float]]]
    """
    root = Path(log_dir)

    def _is_version_dir(p: Path) -> bool:
        if not p.is_dir():
            return False
        if (p / 'metrics.csv').exists():
            return True
        for f in p.iterdir():
            if f.is_file() and f.name.startswith('events.out.tfevents'):
                return True
        return False

    def _find_version_dir(base: Path) -> Path:
        if _is_version_dir(base):
            return base
        candidates = [d for d in base.iterdir() if d.is_dir() and d.name.startswith('version')]
        if not candidates:
            return base  # best effort; may still contain event files directly
        def _ver_num(p: Path) -> int:
            name = p.name
            try:
                return int(name.split('_')[-1])
            except Exception:
                return -1
        candidates.sort(key=lambda p: (_ver_num(p), p.stat().st_mtime))
        return candidates[-1]

    version_dir = _find_version_dir(root)

    # 1) Try CSV (CSVLogger)
    csv_path = version_dir / 'metrics.csv'
    if csv_path.exists():
        # aggregate rows per x (epoch preferred, else step/global_step, else row index)
        per_x: dict[int, dict[str, float]] = {}
        metric_names: set[str] = set()
        next_row_index = 0
        with csv_path.open(newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            long_format = ('metric' in fieldnames and 'value' in fieldnames)
            for row in reader:
                # compute x for this row
                epoch = row.get('epoch')
                step = row.get('step') or row.get('global_step')
                x: Optional[int] = None
                for cand in (epoch, step):
                    if cand is not None and str(cand) != '':
                        try:
                            x = int(float(cand))
                            break
                        except Exception:
                            pass
                if x is None:
                    x = next_row_index
                next_row_index += 1

                if long_format:
                    name = row.get('metric')
                    val = row.get('value')
                    if name is None or val is None or val == '' or str(val).lower() == 'nan':
                        continue
                    try:
                        v = float(val)
                    except Exception:
                        continue
                    metric_names.add(name)
                    per_x.setdefault(x, {})[name] = v
                else:
                    # wide format: one row per x, multiple metric columns
                    for name, val in row.items():
                        if name in {"epoch", "step", "global_step", "time", "created_at"}:
                            continue
                        if val is None or val == '' or str(val).lower() == 'nan':
                            continue
                        try:
                            v = float(val)
                        except Exception:
                            continue
                        metric_names.add(name)
                        per_x.setdefault(x, {})[name] = v

        if not per_x:
            return [], {}
        steps = sorted(per_x.keys())
        # Build aligned series with NaNs for missing values
        series: dict[str, list[float]] = {name: [float('nan')] * len(steps) for name in sorted(metric_names)}
        index_of = {s: i for i, s in enumerate(steps)}
        for s, vals in per_x.items():
            i = index_of[s]
            for name, v in vals.items():
                series[name][i] = float(v)
        return steps, series

    # 2) Try TensorBoard events (TensorBoardLogger)
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore
    except Exception:
        # tensorboard not available and no CSV found
        raise RuntimeError(
            f"No metrics.csv found in '{version_dir}' and 'tensorboard' is not installed. "
            "Install it with `pip install tensorboard`, or enable CSVLogger."
        )

    # If the provided directory is not a version dir, EventAccumulator can still
    # discover event files within it.
    ea = EventAccumulator(str(version_dir), size_guidance={'scalars': 0})
    ea.Reload()
    scalar_tags = list(ea.Tags().get('scalars', []))
    if not scalar_tags:
        return [], {}
    # unify steps across all tags
    step_set: set[int] = set()
    per_tag_events: dict[str, list] = {}
    for tag in scalar_tags:
        ev = ea.Scalars(tag)
        per_tag_events[tag] = ev
        for e in ev:
            step_set.add(int(e.step))
    steps = sorted(step_set)
    idx = {s: i for i, s in enumerate(steps)}
    series: dict[str, list[float]] = {tag: [float('nan')] * len(steps) for tag in scalar_tags}
    for tag, ev in per_tag_events.items():
        for e in ev:
            ii = idx[int(e.step)]
            series[tag][ii] = float(e.value)
    return steps, series


def animate_adjacency(
    checkpoints: Union[str, Path],
    *,
    output_path: Optional[Union[str, Path]] = None,
    fps: int = 30,
    dpi: int = 100,
    figsize: Optional[tuple[float, float]] = None,
    title_template: Optional[str] = "Checkpoint {index}: {name}",
    imshow_kwargs: Optional[Mapping] = None,
):
    """Animate adjacency heatmaps for GradNet checkpoints named ``gn-periodic-*.ckpt``."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation as mpl_animation
    except Exception as exc:
        warnings.warn(f"Matplotlib is required for animations: {exc}")
        return None

    from .gradnet import GradNet

    fps = max(1, int(fps))
    root = Path(checkpoints)
    if root.is_dir():
        ckpts = sorted(p for p in root.glob('gn-periodic-*.ckpt') if p.is_file())
    else:
        matches = root.is_file() and root.name.startswith('gn-periodic-') and root.suffix == '.ckpt'
        ckpts = [root] if matches else []

    if not ckpts:
        warnings.warn("No checkpoints matching 'gn-periodic-*.ckpt' were found.")
        return None

    adjacencies = []
    for path in ckpts:
        model = GradNet.from_checkpoint(str(path), map_location='cpu')
        adjacencies.append(model.to_numpy())

    show_kwargs = {} if imshow_kwargs is None else dict(imshow_kwargs)
    show_kwargs.setdefault('vmin', min(float(adj.min()) for adj in adjacencies))
    show_kwargs.setdefault('vmax', max(float(adj.max()) for adj in adjacencies))

    fig, ax = plt.subplots(figsize=figsize)
    im = plot_adjacency_heatmap(
        adjacencies[0],
        ax=ax,
        title=title_template.format(index=0, name=ckpts[0].name) if title_template else None,
        imshow_kwargs=show_kwargs,
    )

    def _update(index: int):
        im.set_data(adjacencies[index])
        if title_template:
            ax.set_title(title_template.format(index=index, name=ckpts[index].name))
        return [im]

    ani = mpl_animation.FuncAnimation(fig, _update, frames=len(adjacencies), interval=1000.0 / fps)

    saved_path: Optional[Path] = None
    temporary_path: Optional[Path] = None
    ffmpeg_error: Optional[Exception] = None
    try:
        from matplotlib.animation import FFMpegWriter
    except Exception as exc:
        ffmpeg_error = exc
    else:
        ffmpeg_error = None
        target_path: Optional[Path]
        if output_path:
            target_path = Path(output_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                tmp.close()
                temporary_path = Path(tmp.name)
                target_path = temporary_path
            except Exception as exc:
                warnings.warn(f"Unable to allocate temporary file for MP4 animation: {exc}")
                target_path = None

        if target_path is not None:
            try:
                ani.save(str(target_path), writer=FFMpegWriter(fps=fps), dpi=dpi)
                saved_path = target_path
            except Exception as exc:
                warnings.warn(f"Failed to save animation to {target_path}: {exc}")
                if temporary_path and temporary_path.exists():
                    temporary_path.unlink(missing_ok=True)
                temporary_path = None
                saved_path = None
    if saved_path is None and output_path and ffmpeg_error is not None:
        warnings.warn(f"FFMpeg writer unavailable; failed to create MP4 animation: {ffmpeg_error}")

    displayed = False
    try:
        from IPython.display import HTML, Video, display

        if saved_path and saved_path.exists():
            # Embed MP4 first to minimize notebook size when possible.
            display(Video(str(saved_path), embed=True))
            displayed = True
        else:
            try:
                display(HTML(ani.to_html5_video()))
                displayed = True
            except Exception:
                display(HTML(ani.to_jshtml()))
                displayed = True
    except Exception:
        pass

    if not displayed:
        try:
            plt.show()
            displayed = True
        except Exception:
            pass

    if not displayed:
        warnings.warn('Unable to display the animation; consider running inside a notebook environment.')

    plt.close(fig)

    return saved_path or ani


def positions_to_distance_matrix(positions: torch.Tensor, norm: float = 2.0):
    """Compute the pairwise distance matrix from node positions using a given norm."""
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    return LA.vector_norm(diff, ord=norm, dim=-1)


def regularization_loss(del_adj: torch.Tensor) -> torch.Tensor:
    """
    Regularization loss for sparsifying the delta adjacency.
    Computes sum(sigmoid(abs(del_adj))).
    """
    # f = lambda x: torch.sigmoid(x)
    f = lambda x: torch.log(x + 1)
    return torch.sum(f(torch.abs(del_adj)))/del_adj.shape[-1]


################################################################################# private utils

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

def _timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper
