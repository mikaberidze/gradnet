"""Core GradNet module and parameterizations.

This module provides:

- Utility helpers for transforming/normalizing adjacency-like tensors
  (:func:`normalize`, :func:`positivize`, :func:`symmetrize`).
- Parameterization backends for mapping trainable parameters to a constrained
  perturbation of an adjacency matrix: :class:`DenseParameterization` for
  dense masks and :class:`SparseParameterization` for sparse edge lists.
- The user-facing :class:`GradNet` wrapper that owns mask/cost/base-adjacency
  and exposes a simple ``forward`` returning the full adjacency.

Docstrings mirror the style used in :mod:`gradnet.ode` and
:mod:`gradnet.trainer` for high-quality Sphinx rendering.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
import warnings

# ----------------------------------------------------------------------------
# Global Helper Functions (dtype/device-safe)
# ----------------------------------------------------------------------------
def normalize(matrix: torch.Tensor,
              norm_val: float,
              cost_aggr_norm: int = 1,
              cost_matrix: Optional[torch.Tensor] = None,
              strict: bool = True) -> torch.Tensor:
    """Scale a matrix to satisfy a cost-weighted p-norm budget.

    Scales ``matrix`` so that ``|| cost_matrix * matrix ||_p == norm_val``
    (or ``<=`` when ``strict=False``), using the same dtype/device as the
    input.

    Args:
      matrix (torch.Tensor): Input tensor to scale.
      norm_val (float): Target norm value (budget).
      cost_aggr_norm (int, optional): Aggregation norm ``p`` used for the
        cost-weighted p-norm. Must be a positive integer.
      cost_matrix (torch.Tensor | None, optional): Per-entry cost tensor; if
        ``None``, uses ones like ``matrix``. May be dense or sparse; dense
        arithmetic is used.
      strict (bool, optional): If ``False``, scales by ``min(scale, 1)`` to
        avoid upscaling beyond the current norm.

    Returns:
      torch.Tensor: Scaled matrix with cost-weighted p-norm equal to
        ``norm_val`` (or not exceeding it when ``strict=False``).
    """
    if cost_matrix is None:
        cost_matrix = torch.ones_like(matrix)

    if not isinstance(cost_aggr_norm, int) or cost_aggr_norm <= 0:
        raise ValueError("cost_aggr_norm must be a positive integer")
    p = cost_aggr_norm

    # If matrix is dense but cost_matrix is sparse, densify cost for elementwise ops
    if hasattr(cost_matrix, "layout") and matrix.layout == torch.strided and cost_matrix.layout != torch.strided:
        cost_matrix = cost_matrix.to_dense()

    eps = matrix.new_tensor(1e-8)
    s = (torch.abs(cost_matrix * matrix)**p).sum()**(1.0 / p)

    norm_val_t = matrix.new_tensor(norm_val)
    scale = norm_val_t / torch.clamp(s, min=eps)

    if not strict:
        scale = torch.minimum(scale, s.new_tensor(1.0))

    return matrix * scale


def positivize(matrix: torch.Tensor, q: int=1, eps: float=1e-10) -> torch.Tensor:
    """Map unconstrained entries to nonnegative values via squaring.

    Args:
      matrix (torch.Tensor): Input tensor.

    Returns:
      torch.Tensor: Elementwise square of ``matrix``.
    """
    return (matrix**2+eps**q)**(1/q) - eps


def symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    """Return the averaged symmetric part of a square matrix.

    Computes ``0.5 * (M + M^T)`` along the last two axes.

    Args:
      matrix (torch.Tensor): Square matrix or a batch thereof.

    Returns:
      torch.Tensor: Symmetrized matrix.
    """
    return 0.5 * (matrix + matrix.transpose(-1, -2))


# ----------------------------------------------------------------------------
# Parameterization Submodule (Option 1)
# ----------------------------------------------------------------------------
class DenseParameterization(nn.Module):
    """Dense parameterization of a delta adjacency matrix.

    Maintains a dense, trainable ``delta_adj_raw`` and projects it to a
    constrained perturbation ``delta`` through the following pipeline::

        raw -> (symmetrize?) -> mask -> (positivize?) -> normalize (budget)

    Args:
      num_nodes (int): Number of nodes (matrix dimension).
      budget (float): Target cost-weighted p-norm for the normalized ``delta``.
      mask (torch.Tensor): Dense mask selecting active entries (1 for active,
        0 for inactive). Nonzero diagonal entries are allowed but typically
        masked out by users.
      cost_matrix (torch.Tensor): Per-entry cost tensor for the normalization.
      delta_sign (str, optional): Sign constraint for the perturbation. One of
        ``{"free", "nonnegative", "nonpositive"}``.
      undirected (bool, optional): If ``True``, symmetrize before
        masking/normalizing.
      use_budget_up (bool, optional): If ``True``, always scale up to the
        budget; otherwise do not upscale.
      cost_aggr_norm (int, optional): Aggregation norm ``p`` for the
        cost-weighted p-norm.
      rand_init_weights (bool | float, optional): Initialization mix coefficient
        ``a``. Cast to float and clamped to ``[0,1]``. Initial raw parameters are
        set to ``a * 1 + (1 - a) * U(0,1)``.
    """
    def __init__(self,
                 num_nodes: int,
                 budget: float,
                 mask: torch.Tensor,
                 cost_matrix: torch.Tensor,
                 *,
                 delta_sign: str = "nonnegative",
                 undirected: bool = True,
                 use_budget_up: bool = False,
                 cost_aggr_norm: int = 1,
                 rand_init_weights: Union[bool, float] = True):
        super().__init__()

        self.num_nodes = int(num_nodes)
        self.budget = float(budget)
        allowed_signs = {"free", "nonnegative", "nonpositive"}
        ds = str(delta_sign).lower()
        if ds not in allowed_signs:
            raise ValueError(f"delta_sign must be one of {sorted(allowed_signs)}; got {delta_sign!r}")
        self.delta_sign = ds
        self.undirected = bool(undirected)
        self.use_budget_up = bool(use_budget_up)
        self.cost_aggr_norm = int(cost_aggr_norm)

        # non-trainable buffers
        # Enforce zero diagonal on mask
        m = torch.as_tensor(mask)
        if hasattr(m, "layout") and m.layout != torch.strided:
            mc = m.coalesce()
            ii, jj = mc.indices()
            keep = ii != jj
            m = torch.sparse_coo_tensor(
                torch.stack([ii[keep], jj[keep]], dim=0),
                mc.values()[keep],
                mc.shape,
                device=mc.device,
                dtype=mc.dtype,
            ).coalesce()
        else:
            m = m.clone()
            if m.ndim >= 2 and m.shape[-1] == m.shape[-2]:
                m.fill_diagonal_(0)
        self.register_buffer("mask", m)
        self.register_buffer("cost_matrix", torch.as_tensor(cost_matrix))

        # trainable parameter
        shape = (self.num_nodes, self.num_nodes)
        # Mixed initialization: a*ones + (1-a)*rand, where a in [0,1]
        try:
            a = float(rand_init_weights)
        except Exception:
            a = 1.0 if bool(rand_init_weights) else 0.0
        a = max(0.0, min(1.0, a))
        if use_budget_up: 
            unif = torch.ones(shape, device=self.mask.device, dtype=self.mask.dtype)
        else:
            unif = torch.zeros(shape, device=self.mask.device, dtype=self.mask.dtype)
        rnd = torch.rand(shape, device=self.mask.device, dtype=self.mask.dtype)
        # Reverse semantics so a controls randomness: a=1 -> random, a=0 -> ones
        delta0 = (1.0 - a) * unif + a * rnd
        self.delta_adj_raw = nn.Parameter(delta0, requires_grad=True)

        # Normalize initial scale for stability
        self.renorm_params()

    # --------- Convenience properties ------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.delta_adj_raw.device

    @property
    def dtype(self) -> torch.dtype:
        return self.delta_adj_raw.dtype

    def extra_repr(self) -> str:
        return (f"num_nodes={self.num_nodes}, budget={self.budget}, "
                f"delta_sign={self.delta_sign!r}, undirected={self.undirected}, "
                f"use_budget_up={self.use_budget_up}, p={self.cost_aggr_norm}, "
                f"dtype={self.dtype}, device={self.device}")

    # --------- State management -------------------------------------------------
    @torch.no_grad()
    def set_initial_state(self, delta_adj_raw_0: torch.Tensor):
        """Set the internal raw parameter and re-normalize.

        Args:
          delta_adj_raw_0 (torch.Tensor): Tensor with the same shape as
            ``delta_adj_raw``.

        Raises:
          ValueError: If the provided tensor shape mismatches.
        """
        delta_adj_raw_0 = torch.as_tensor(delta_adj_raw_0, device=self.device, dtype=self.dtype)
        if delta_adj_raw_0.shape != self.delta_adj_raw.shape:
            raise ValueError(f"Shape mismatch: got {tuple(delta_adj_raw_0.shape)}, "
                             f"expected {tuple(self.delta_adj_raw.shape)}.")
        self.delta_adj_raw.copy_(delta_adj_raw_0)
        self.renorm_params()

    @torch.no_grad()
    def renorm_params(self):
        """Renormalize the raw parameters to a DOF-aware scale.

        Computes a target scale proportional to ``sqrt(D)`` where ``D`` is the
        number of active degrees of freedom implied by ``mask`` and
        ``undirected``. This makes the initial magnitude less sensitive to the
        mask sparsity or graph size, improving optimization stability.
        """
        # In dense encoding, simply count active mask entries; if undirected,
        # approximate DOF by halving the count (i,j) and (j,i) pairs.
        m = self.mask
        if hasattr(m, "layout") and m.layout != torch.strided:
            m = m.to_dense()
        nz = int((m != 0).sum().item())
        dof = int(nz / 2) if self.undirected else nz
        if dof <= 0:
            dof = self.num_nodes  # safe fallback
        eps = self.delta_adj_raw.new_tensor(1e-12)
        delta_adj_norm = torch.linalg.norm(self.delta_adj_raw)
        if delta_adj_norm <= eps:
            return  # avoid divide-by-zero
        target = self.delta_adj_raw.new_tensor(float(dof)) ** 0.5  #! keep this note
        scale = target / torch.clamp(delta_adj_norm, min=eps)
        self.delta_adj_raw.mul_(scale)  # in-place scaling

    # --------- Build current delta ---------------------------------------------
    def forward(self) -> torch.Tensor:
        """Project raw parameters to a constrained ``delta`` matrix.

        Applies optional symmetrization and positivity, then masks inactive
        entries and finally scales to match the cost-weighted p-norm budget.

        Returns:
          torch.Tensor: Normalized perturbation matrix ``delta``.
        """
        delta = self.delta_adj_raw

        if self.undirected:
            delta = symmetrize(delta)

        if self.delta_sign == "nonnegative":
            delta = positivize(delta)
        elif self.delta_sign == "nonpositive":
            delta = -positivize(delta)

        delta = delta * self.mask
        
        delta = normalize(delta,
                          self.budget,
                          cost_aggr_norm=self.cost_aggr_norm,
                          cost_matrix=self.cost_matrix,
                          strict=self.use_budget_up)
        return delta


class SparseParameterization(nn.Module):
    """Sparse, edge-list parameterization for masked adjacencies.

    This backend stores a 1D trainable vector of length ``E`` (active edges)
    and constructs a sparse COO tensor for the ``delta`` matrix. In undirected
    mode, only ``(i < j)`` edges are parameterized and mirrored on output.
    """
    def __init__(
        self,
        *,
        num_nodes: int,
        budget: float,
        edge_index: torch.Tensor,  # [2, E]
        cost_p_sum: torch.Tensor,  # [E]
        delta_sign: str = "nonnegative",
        undirected: bool = True,
        use_budget_up: bool = False,
        cost_aggr_norm: int = 1,
        rand_init_weights: Union[bool, float] = True,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """Construct a sparse edge-list parameterization.

        Args:
          num_nodes (int): Number of nodes ``N`` (matrix dimension).
          budget (float): Target cost-weighted p-norm of the perturbation.
          edge_index (torch.Tensor): Integer tensor of shape ``(2, E)`` giving
            the edge list. In undirected mode, edges must satisfy ``i < j``.
          cost_p_sum (torch.Tensor): Positive tensor of shape ``(E,)``
            containing, for each edge, the sum of costs to the power ``p``
            used in the normalization. For undirected graphs this is typically
            ``|c_ij|^p + |c_ji|^p``; for directed, ``|c_ij|^p``.
          delta_sign (str, optional): Sign constraint for the perturbation.
            One of ``{"free", "nonnegative", "nonpositive"}``.
          undirected (bool, optional): If ``True``, mirror ``(i, j)`` entries
            to ``(j, i)`` when building the sparse matrix.
          use_budget_up (bool, optional): If ``True``, always scale up to the
            budget; otherwise avoid upscaling.
          cost_aggr_norm (int, optional): Aggregation norm ``p`` for the
            cost-weighted p-norm.
          rand_init_weights (bool | float, optional): Initialization mix
            coefficient ``a``. Cast to float and clamped to ``[0,1]``.
            Raw edge weights are set to ``a * 1 + (1 - a) * U(0,1)``.
          dtype (torch.dtype): Parameter/buffer dtype for this module.
          device (torch.device): Device for parameters/buffers.
        """
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.budget = float(budget)
        allowed_signs = {"free", "nonnegative", "nonpositive"}
        ds = str(delta_sign).lower()
        if ds not in allowed_signs:
            raise ValueError(f"delta_sign must be one of {sorted(allowed_signs)}; got {delta_sign!r}")
        self.delta_sign = ds
        self.undirected = bool(undirected)
        self.use_budget_up = bool(use_budget_up)
        self.cost_aggr_norm = int(cost_aggr_norm)

        self.register_buffer("edge_index", edge_index.to(device=device))
        self.register_buffer("cost_p_sum", cost_p_sum.to(device=device, dtype=dtype))

        E = self.edge_index.shape[1]
        try:
            a = float(rand_init_weights)
        except Exception:
            a = 1.0 if bool(rand_init_weights) else 0.0
        a = max(0.0, min(1.0, a))
        if use_budget_up: 
            unif = torch.ones((E,), device=device, dtype=dtype)
        else:
            unif = torch.zerps((E,), device=device, dtype=dtype)
        rnd = torch.rand((E,), device=device, dtype=dtype)
        # Reverse semantics so a controls randomness: a=1 -> random, a=0 -> ones
        w0 = (1.0 - a) * unif + a * rnd
        self.delta_adj_raw = nn.Parameter(w0, requires_grad=True)

    @property
    def device(self) -> torch.device:
        return self.delta_adj_raw.device

    @property
    def dtype(self) -> torch.dtype:
        return self.delta_adj_raw.dtype

    def extra_repr(self) -> str:
        E = int(self.edge_index.shape[1])
        return (f"num_nodes={self.num_nodes}, edges={E}, budget={self.budget}, "
                f"delta_sign={self.delta_sign!r}, undirected={self.undirected}, "
                f"use_budget_up={self.use_budget_up}, p={self.cost_aggr_norm}, "
                f"dtype={self.dtype}, device={self.device}")

    @torch.no_grad()
    def set_initial_state(self, delta_adj_raw_0: torch.Tensor):
        """Set the internal raw edge weights and re-normalize.

        Args:
          delta_adj_raw_0 (torch.Tensor): 1D tensor with length ``E``.

        Raises:
          ValueError: If shape mismatches the internal parameter.
        """
        delta_adj_raw_0 = torch.as_tensor(delta_adj_raw_0, device=self.device, dtype=self.dtype)
        if delta_adj_raw_0.shape != self.delta_adj_raw.shape:
            raise ValueError(f"Shape mismatch: got {tuple(delta_adj_raw_0.shape)}, expected {tuple(self.delta_adj_raw.shape)}.")
        self.delta_adj_raw.copy_(delta_adj_raw_0)
        self.renorm_params()

    @torch.no_grad()
    def renorm_params(self):
        """Scale raw edge parameters to a constant norm (``~sqrt(E)``)."""
        eps = self.delta_adj_raw.new_tensor(1e-12)
        wnorm = torch.linalg.norm(self.delta_adj_raw)
        if wnorm <= eps:
            return
        E = self.edge_index.shape[1]
        target = self.delta_adj_raw.new_tensor(float(E)) ** 0.5  #! keep this note
        scale = target / torch.clamp(wnorm, min=eps)
        self.delta_adj_raw.mul_(scale)

    def forward(self) -> torch.Tensor:
        """Project raw edge weights to a sparse, normalized ``delta``.

        Applies optional positivity in vector space, scales to match the
        cost-weighted p-norm budget, and constructs a COO matrix. In
        undirected mode, edges are mirrored.

        Returns:
          torch.Tensor: Coalesced sparse COO tensor of shape ``(N, N)``.
        """
        w = self.delta_adj_raw

        if self.delta_sign == "nonnegative":
            w = positivize(w)
        elif self.delta_sign == "nonpositive":
            w = -positivize(w)

        p = max(1, int(self.cost_aggr_norm))
        eps = w.new_tensor(1e-8)
        s = (torch.abs(w) ** p * self.cost_p_sum).sum() ** (1.0 / p)
        norm_val_t = w.new_tensor(self.budget)
        scale = norm_val_t / torch.clamp(s, min=eps)
        if not self.use_budget_up:
            scale = torch.minimum(scale, s.new_tensor(1.0))
        vals = w * scale

        if self.undirected:
            i, j = self.edge_index
            ii = torch.cat([i, j], dim=0)
            jj = torch.cat([j, i], dim=0)
            vv = torch.cat([vals, vals], dim=0)
            return torch.sparse_coo_tensor(torch.stack([ii, jj], dim=0), vv, (self.num_nodes, self.num_nodes), device=self.device, dtype=self.dtype).coalesce()
        else:
            return torch.sparse_coo_tensor(self.edge_index, vals, (self.num_nodes, self.num_nodes), device=self.device, dtype=self.dtype).coalesce()


# ----------------------------------------------------------------------------
# GradNet (Thin Wrapper Using Parameterization Submodule)
# ----------------------------------------------------------------------------
class GradNet(nn.Module):
    """User-facing GradNet: learn a constrained ``delta`` over a base adjacency.

    This thin wrapper owns the mask, cost matrix, and base adjacency ``adj0``,
    and delegates the trainable parameters to either a dense or sparse
    parameterization depending on mask layout and size.
    """
    def __init__(self,
                 num_nodes: int,
                 budget: float,
                 mask = None,
                 adj0 = None,
                 delta_sign: str = "nonnegative",
                 final_sign: str = "nonnegative",
                 undirected: bool = True,
                 rand_init_weights: Union[bool, float] = True,
                 use_budget_up: bool = True,
                 cost_matrix = None,
                 cost_aggr_norm: int = 1,
                 *,
                device: Optional[str] = None,
                dtype: Optional[str] = None):
        """Construct a GradNet instance.

        Args:
          num_nodes (int): Number of nodes (matrix dimension).
          budget (float): Target cost-weighted p-norm of the perturbation.
          mask (torch.Tensor | None, optional): Active-entry mask. Dense masks
            result in a dense parameterization; sparse COO masks may trigger a
            sparse backend if sufficiently small. If ``None``, defaults to
            all-ones off-diagonal.
          adj0 (torch.Tensor | None, optional): Base adjacency. If ``None``,
            uses zeros.
          delta_sign (str, optional): Sign constraint for ``delta``. One of
            ``{"free", "nonnegative", "nonpositive"}``.
          final_sign (str, optional): Sign constraint applied to the returned
            adjacency. One of ``{"free", "nonnegative", "nonpositive"}``.
          undirected (bool, optional): If ``True``, symmetrize ``delta`` and
            expect a symmetric cost matrix.
          rand_init_weights (bool | float, optional): Initialization mix
            coefficient ``a``. Cast to float and clamped to ``[0,1]``.
            ``a = 1.0`` or ``True`` yields fully random ``U(0,1)``; ``a = 0.0`` or
            ``False`` yields uniform ones. Intermediate values yield interpolation.
          use_budget_up (bool, optional): If ``True``, always scale up to the
            budget.
          cost_matrix (torch.Tensor | None, optional): Per-entry costs for
            normalization; defaults to ones.
          cost_aggr_norm (int, optional): Aggregation norm ``p`` for the
            cost-weighted p-norm.
          device (torch.device | str | None, optional): Target device for
            buffers/parameters. If ``None``, inferred from input tensors or
            defaults to CPU.
          dtype (torch.dtype | str | None, optional): Target dtype for
            buffers/parameters. If ``None``, inferred from input tensors or
            from PyTorch defaults.
        """
        super().__init__()

        # ---- Standard device/dtype negotiation --------------------------------
        # 1) If any input tensor is provided, use its device/dtype as defaults.
        # 2) Otherwise, fall back to explicit kwargs, else CPU + torch.get_default_dtype().
        infer_from = next((t for t in (adj0, mask, cost_matrix) if isinstance(t, torch.Tensor)), None)
        dev = torch.device(device) if device is not None else (
            infer_from.device if infer_from is not None else torch.device("cpu")
        )
        if dtype is None:
            dt = infer_from.dtype if infer_from is not None else torch.get_default_dtype()
        else:
            if isinstance(dtype, torch.dtype):
                dt = dtype
            elif isinstance(dtype, str):
                key = dtype.split(".")[-1].lower()
                candidate = getattr(torch, key, None)
                if not isinstance(candidate, torch.dtype):
                    raise ValueError(f"Unsupported dtype string '{dtype}'")
                dt = candidate
            else:
                raise TypeError("dtype must be a torch.dtype, str, or None")

        # ---- Public config -----------------------------------------------------
        self.num_nodes = int(num_nodes)
        self.budget = float(budget)
        allowed_signs = {"free", "nonnegative", "nonpositive"}
        ds = str(delta_sign).lower()
        fs = str(final_sign).lower()
        if ds not in allowed_signs:
            raise ValueError(f"delta_sign must be one of {sorted(allowed_signs)}; got {delta_sign!r}")
        if fs not in allowed_signs:
            raise ValueError(f"final_sign must be one of {sorted(allowed_signs)}; got {final_sign!r}")
        self.delta_sign = ds
        self.final_sign = fs
        self.undirected = bool(undirected)
        self.use_budget_up = bool(use_budget_up)
        self.cost_aggr_norm = int(cost_aggr_norm)

        # ---- Helpers -----------------------------------------------------------
        def _coerce(x, make_fallback):
            """
            Convert x to a detached tensor on (device,dtype). 
            If None, create via fallback_fn().
            """
            if x is None:
                t = make_fallback()
            else:
                if isinstance(x, torch.Tensor):
                    # preserve sparse layout when provided
                    t = x.to(device=dev, dtype=dt)
                else:
                    t = torch.as_tensor(x, device=dev, dtype=dt)
            return t.detach()

        N = self.num_nodes

        # Default mask: all except diagonal
        mask_default = torch.ones((N, N), device=dev, dtype=dt) - torch.eye(N, device=dev, dtype=dt)
        mask_buf = _coerce(mask, lambda: mask_default)
        # Enforce zero diagonal on mask (dense or sparse)
        if isinstance(mask_buf, torch.Tensor) and mask_buf.layout != torch.strided:
            mbc = mask_buf.coalesce()
            ii, jj = mbc.indices()
            keep = ii != jj
            mask_buf = torch.sparse_coo_tensor(
                torch.stack([ii[keep], jj[keep]], dim=0),
                mbc.values()[keep],
                mbc.shape,
                device=mbc.device,
                dtype=mbc.dtype,
            ).coalesce()
        else:
            mask_buf = mask_buf.clone()
            if mask_buf.ndim >= 2 and mask_buf.shape[-1] == mask_buf.shape[-2]:
                mask_buf.fill_diagonal_(0)
        self.register_buffer("mask", mask_buf)

        # Default cost_matrix: ones
        cost_buf = _coerce(cost_matrix, lambda: torch.ones((N, N), device=dev, dtype=dt))
        self.register_buffer("cost_matrix", cost_buf)

        # Default adj0: zeros
        adj0_buf = _coerce(adj0, lambda: torch.zeros((N, N), device=dev, dtype=dt))
        self.register_buffer("adj0", adj0_buf)

        # ---- Parameterization submodule ---------------------------------------
        if isinstance(self.mask, torch.Tensor) and self.mask.layout != torch.strided:
            edge_index, cost_p_sum = self._prepare_edge_list(
                mask=self.mask,
                cost_matrix=self.cost_matrix if cost_matrix is not None else None,
                undirected=self.undirected,
                p=self.cost_aggr_norm,
                dtype=dt,
                device=dev,
            )
            self.param = SparseParameterization(
                num_nodes=N,
                budget=self.budget,
                edge_index=edge_index,
                cost_p_sum=cost_p_sum,
                delta_sign=self.delta_sign,
                undirected=self.undirected,
                use_budget_up=self.use_budget_up,
                cost_aggr_norm=self.cost_aggr_norm,
                # Backends expect bool; pass truthiness and override below if needed
                rand_init_weights=rand_init_weights,
                dtype=dt,
                device=dev,
            )
        else:
            # Ensure dense cost for elementwise ops
            if isinstance(self.cost_matrix, torch.Tensor) and self.cost_matrix.layout != torch.strided:
                self.cost_matrix = self.cost_matrix.to_dense()
            self.param = DenseParameterization(
                num_nodes=N,
                budget=self.budget,
                mask=self.mask,
                cost_matrix=self.cost_matrix,
                delta_sign=self.delta_sign,
                undirected=self.undirected,
                use_budget_up=self.use_budget_up,
                cost_aggr_norm=self.cost_aggr_norm,
                # Backends expect bool; pass truthiness and override below if needed
                rand_init_weights=rand_init_weights,
            )

    # --------- Convenience properties ------------------------------------------
    @property
    def device(self) -> torch.device:
        return self.param.device

    @property
    def dtype(self) -> torch.dtype:
        return self.param.dtype

    def extra_repr(self) -> str:
        return (f"num_nodes={self.num_nodes}, budget={self.budget}, "
                f"delta_sign={self.delta_sign!r}, final_sign={self.final_sign!r}, undirected={self.undirected}, "
                f"use_budget_up={self.use_budget_up}, p={self.cost_aggr_norm}, "
                f"dtype={self.dtype}, device={self.device}")

    # --------- Minimal serialization helpers ----------------------------------
    def export_config(self) -> dict:
        """Return a CPU-side configuration snapshot for later reconstruction."""
        def _clone_cpu(x):
            if isinstance(x, torch.Tensor):
                return x.detach().clone().cpu()
            return x

        return {
            "num_nodes": self.num_nodes,
            "budget": self.budget,
            "mask": _clone_cpu(self.mask),
            "adj0": _clone_cpu(self.adj0),
            "delta_sign": self.delta_sign,
            "final_sign": self.final_sign,
            "undirected": self.undirected,
            "use_budget_up": self.use_budget_up,
            "cost_matrix": _clone_cpu(self.cost_matrix),
            "cost_aggr_norm": self.cost_aggr_norm,
        }

    @classmethod
    def from_config(cls, config: dict) -> "GradNet":
        """Rebuild a ``GradNet`` from :meth:`export_config` output."""
        cfg = dict(config)
        mask = cfg.pop("mask", None)
        adj0 = cfg.pop("adj0", None)
        cost_matrix = cfg.pop("cost_matrix", None)
        return cls(
            mask=mask,
            adj0=adj0,
            cost_matrix=cost_matrix,
            rand_init_weights=False,
            **cfg,
        )

    # --------- State management passthroughs -----------------------------------
    @torch.no_grad()
    def set_initial_state(self, delta_adj_raw_0: torch.Tensor):
        """Forward to the parameterization's ``set_initial_state`` and renormalize."""
        self.param.set_initial_state(delta_adj_raw_0)

    @torch.no_grad()
    def renorm_params(self):
        """Renormalize internal parameters using the backend's strategy."""
        self.param.renorm_params()

    # --------- Build current delta / adjacency ---------------------------------
    def get_delta_adj(self) -> torch.Tensor:
        """Return the normalized perturbation matrix ``delta`` from the backend."""
        return self.param()

    def forward(self) -> torch.Tensor:
        """Return the full adjacency ``A = adj0 + delta``.

        Handles dense/sparse combinations between ``adj0`` and ``delta`` and
        returns either a dense or a sparse tensor accordingly.
        """
        delta = self.get_delta_adj()
        A0 = self.adj0
        # Handle dense/sparse combinations
        if isinstance(A0, torch.Tensor) and A0.layout != torch.strided:
            if isinstance(delta, torch.Tensor) and delta.layout != torch.strided:
                adj = (A0.coalesce() + delta.coalesce()).coalesce()
            else:
                adj = A0.to_dense() + delta
        else:
            if isinstance(delta, torch.Tensor) and delta.layout != torch.strided:
                adj = A0 + delta.to_dense()
            else:
                adj = A0 + delta

        if self.final_sign != "free":
            if isinstance(adj, torch.Tensor) and adj.layout != torch.strided:
                adj = adj.coalesce()
                values = positivize(adj.values(), q=2)
                if self.final_sign == "nonpositive":
                    values = -values
                adj = torch.sparse_coo_tensor(
                    adj.indices(),
                    values,
                    adj.shape,
                    device=adj.device,
                    dtype=values.dtype,
                ).coalesce()
            else:
                adj = positivize(adj, q=2)
                if self.final_sign == "nonpositive":
                    adj = -adj

        return adj

    def to_numpy(self):
        """Return the full adjacency as a NumPy array on CPU."""
        A = self()
        if isinstance(A, torch.Tensor) and A.layout != torch.strided:
            return A.detach().to_dense().cpu().numpy()
        else:
            return A.detach().cpu().numpy()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        map_location: Optional[Union[str, torch.device]] = "cpu",
    ) -> "GradNet":
        """Load a ``GradNet`` from a PyTorch Lightning checkpoint. Checkpoints are stored by fit."""
        ckpt = torch.load(checkpoint_path, map_location=map_location)
        config = ckpt.get("hyper_parameters", {}).get("gradnet_config")
        if config is None:
            raise ValueError("Checkpoint missing 'gradnet_config'; ensure training used updated GradNetLightning.")

        model = cls.from_config(config)

        from .trainer import GradNetLightning  # lazy import to avoid cycles

        def _noop_loss_fn(_gn: "GradNet", **_):
            return torch.zeros((), device=model.device, dtype=model.dtype)

        module = GradNetLightning.load_from_checkpoint(
            checkpoint_path,
            map_location=map_location,
            gn=model,
            loss_fn=_noop_loss_fn,
            loss_kwargs={},
            optim_cls=torch.optim.SGD,
            optim_kwargs={"lr": 0.0},
        )
        return module.gn

    # --------------------- Internal helpers -----------------------------------
    def _prepare_edge_list(
        self,
        *,
        mask: torch.Tensor,
        cost_matrix: Optional[torch.Tensor],
        undirected: bool,
        p: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edge-index representation and per-edge cost for sparse masks.

        Expects a sparse COO mask and (optionally) a sparse/dense cost matrix.
        Returns unique edges and the associated cost p-sum.

        :return: Tuple ``(edge_index, cost_p_sum)`` where ``edge_index`` is a
            ``2 x E`` tensor of indices and ``cost_p_sum`` is length ``E``.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        if mask.layout == torch.strided:
            raise ValueError("Expected a sparse mask tensor for edge-list mode")
        N = self.num_nodes
        m = mask.coalesce()
        ii, jj = m.indices()
        # Zero diagonal: drop any present and warn
        keep = ii != jj
        dropped = int((~keep).sum().item())
        if dropped > 0:
            warnings.warn(f"Mask has {dropped} diagonal entries; they will be ignored (set to 0).", RuntimeWarning)
        ii = ii[keep]
        jj = jj[keep]

        if undirected:
            a = torch.minimum(ii, jj)
            b = torch.maximum(ii, jj)
            keys = a * N + b
            uk = torch.unique(keys, sorted=True)
            ei = (uk // N).to(torch.long)
            ej = (uk % N).to(torch.long)
            edge_index = torch.stack([ei, ej], dim=0)
        else:
            keys = ii * N + jj
            uk = torch.unique(keys, sorted=True)
            ei = (uk // N).to(torch.long)
            ej = (uk % N).to(torch.long)
            edge_index = torch.stack([ei, ej], dim=0)

        E = edge_index.shape[1]

        # Handle cost matrix
        if cost_matrix is None:
            cost_p_sum = torch.full((E,), 2.0 if undirected else 1.0, device=device, dtype=dtype)
            return edge_index.to(device=device), cost_p_sum

        # Warn on asymmetry in undirected mode
        if undirected:
            if cost_matrix.layout == torch.strided:
                cm = cost_matrix
                if cm.shape != (N, N):
                    raise ValueError("cost_matrix shape mismatch")
                if not torch.allclose(cm, cm.transpose(-1, -2)):
                    warnings.warn("Undirected requested but cost_matrix is not symmetric.", RuntimeWarning)
            else:
                cm = cost_matrix.coalesce()
                ri, rj = edge_index
                c_ij = _gather_sparse_values(cm, ri, rj, default=0.0)
                c_ji = _gather_sparse_values(cm, rj, ri, default=0.0)
                if torch.any(c_ij != c_ji):
                    warnings.warn("Undirected requested but cost_matrix has asymmetric values on masked edges.", RuntimeWarning)

        # Build cost_p_sum and warn on missing costs
        if cost_matrix.layout == torch.strided:
            ri, rj = edge_index
            c_ij = torch.abs(cost_matrix[ri, rj]) ** p
            if undirected:
                c_ji = torch.abs(cost_matrix[rj, ri]) ** p
                cost_p_sum = (c_ij + c_ji).to(dtype=dtype, device=device)
            else:
                cost_p_sum = c_ij.to(dtype=dtype, device=device)
        else:
            cm = cost_matrix.coalesce()
            ri, rj = edge_index
            c_ij = torch.abs(_gather_sparse_values(cm, ri, rj, default=0.0)) ** p
            missing_ij = (c_ij == 0)
            if undirected:
                c_ji = torch.abs(_gather_sparse_values(cm, rj, ri, default=0.0)) ** p
                missing_ji = (c_ji == 0)
                missing = (missing_ij | missing_ji)
                cost_p_sum = (c_ij + c_ji).to(dtype=dtype, device=device)
            else:
                missing = missing_ij
                cost_p_sum = c_ij.to(dtype=dtype, device=device)
            miss_count = int(missing.sum().item())
            if miss_count > 0:
                warnings.warn(f"Cost matrix missing {miss_count} entries for masked edges; assuming 0 cost.", RuntimeWarning)

        return edge_index.to(device=device), cost_p_sum


def _gather_sparse_values(cm: torch.Tensor, ri: torch.Tensor, rj: torch.Tensor, default: float = 0.0) -> torch.Tensor:
    """Gather values from a coalesced COO sparse matrix at (ri, rj) pairs.

    Missing entries are filled with ``default``. Accepts a dense matrix as a
    convenience and gathers via advanced indexing in that case.

    :param cm: Sparse COO (or dense) matrix to query. Must be square.
    :type cm: torch.Tensor
    :param ri: Row indices.
    :type ri: torch.Tensor
    :param rj: Column indices.
    :type rj: torch.Tensor
    :param default: Value used for missing entries.
    :type default: float
    :return: Values gathered at ``(ri, rj)`` with missing entries filled.
    :rtype: torch.Tensor
    """
    if cm.layout == torch.strided:
        return cm[ri, rj]
    N = cm.shape[0]
    idx = cm.indices()
    vals = cm.values()
    keys = idx[0] * N + idx[1]
    qkeys = ri * N + rj
    sk, order = torch.sort(keys)
    svals = vals[order]
    pos = torch.searchsorted(sk, qkeys)
    pos = torch.clamp(pos, max=max(0, sk.numel() - 1))
    match = (sk[pos] == qkeys) if sk.numel() > 0 else torch.zeros_like(qkeys, dtype=torch.bool)
    out = torch.full(qkeys.shape, fill_value=float(default), device=vals.device, dtype=vals.dtype)
    if sk.numel() > 0:
        out[match] = svals[pos[match]]
    return out
