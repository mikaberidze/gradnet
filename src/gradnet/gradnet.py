"""Core GradNet module and parameterizations.

This module provides:

- Utility helpers for transforming/normalizing adjacency-like tensors
  (:func:`normalize`, :func:`square`, :func:`symmetrize`).
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
from typing import Callable, Optional, Tuple, Union
import warnings
from contextlib import contextmanager


# ----------------------------------------------------------------------------
# GradNet (Thin Wrapper Using Parameterization Submodule)
# ----------------------------------------------------------------------------
class GradNet(nn.Module):
    """User-facing GradNet: learn a constrained ``delta`` over a base adjacency.

    This thin wrapper owns the mask, cost matrix, and base adjacency ``adj0``,
    and delegates the trainable parameters to either a dense or sparse
    parameterization depending on mask layout.
    """

    def __init__(
        self,
        num_nodes: int,
        budget: Optional[float],
        mask=None,
        adj0=None,
        delta_sign: str = "nonnegative",
        final_sign: str = "free",
        directed: bool = False,
        rand_init_weights: Union[bool, float] = True,
        strict_budget: bool = True,
        cost_matrix=None,
        cost_aggr_norm: int = 1,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ):
        """Construct a GradNet instance.

        Args:
          num_nodes (int): Number of nodes (matrix dimension).
          budget (float | None): Target cost-weighted p-norm of the
            perturbation. If ``None``, no budget normalization is enforced.
          mask (torch.Tensor | None, optional): Active-entry mask. Dense masks
            result in a dense parameterization; sparse COO masks use the sparse
            backend. If ``None``, defaults to all-ones off-diagonal.
          adj0 (torch.Tensor | None, optional): Base adjacency. If ``None``,
            uses a zero matrix matching the selected backend layout.
          delta_sign (str, optional): Sign constraint for ``delta``. One of
            ``{"free", "nonnegative", "nonpositive"}``.
          final_sign (str, optional): Sign constraint applied to the returned
            adjacency. One of ``{"free", "nonnegative", "nonpositive"}``.
          directed (bool, optional): If ``False``, symmetrize ``delta`` and
            expect a symmetric cost matrix.
          rand_init_weights (bool | float, optional): Initialization mix
            coefficient ``a``. Cast to float and clamped to ``[0,1]``.
            ``a = 1.0`` or ``True`` yields fully random ``U(0,1)``; ``a = 0.0`` or
            ``False`` yields uniform ones. Intermediate values yield interpolation.
          strict_budget (bool, optional): If ``True``, always scale up/down
            to the exact budget. If ``False``, scale down only.
          cost_matrix (torch.Tensor | None, optional): Per-entry costs for
            normalization; defaults to ones. In sparse backend mode, omitted
            costs remain implicit (unit costs) and no dense default matrix is
            materialized.
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

        # ---- Public config -----------------------------------------------------
        self.num_nodes = int(num_nodes)
        self.budget = None if budget is None else float(budget)
        allowed_signs = {"free", "nonnegative", "nonpositive"}
        ds = str(delta_sign).lower()
        fs_requested = str(final_sign).lower()
        if ds not in allowed_signs:
            raise ValueError(
                f"delta_sign must be one of {sorted(allowed_signs)}; got {delta_sign!r}"
            )
        if fs_requested not in allowed_signs:
            raise ValueError(
                f"final_sign must be one of {sorted(allowed_signs)}; got {final_sign!r}"
            )
        if {ds, fs_requested} == {"nonnegative", "nonpositive"}:
            warnings.warn(
                "delta_sign and final_sign request opposite cones; final projection may violate delta_sign and strict budget behavior.",
                RuntimeWarning,
            )
        self.delta_sign = ds
        self.final_sign = fs_requested
        self.directed = bool(directed)
        self.strict_budget = bool(strict_budget)
        self.cost_aggr_norm = int(cost_aggr_norm)

        dev, dt = self._resolve_device_dtype(adj0, mask, cost_matrix, device, dtype)
        N = self.num_nodes
        self.register_buffer("mask", self._prep_mask(mask, N, dev, dt))
        use_sparse_backend = _is_sparse_tensor(self.mask)
        self.register_buffer(
            "cost_matrix", self._prep_cost(cost_matrix, use_sparse_backend, N, dev, dt)
        )
        self.register_buffer(
            "adj0", self._prep_adj0(adj0, use_sparse_backend, N, dev, dt)
        )
        self._validate_undirected_inputs(
            mask_provided=mask is not None,
            adj0_provided=adj0 is not None,
            cost_provided=cost_matrix is not None,
        )
        self._warn_if_adj0_violates_final_sign(fs_requested)
        if fs_requested != "free" and ds == fs_requested:
            self.final_sign = "free"
        self.param = self._build_param(
            use_sparse_backend=use_sparse_backend,
            rand_init_weights=rand_init_weights,
            device=dev,
            dtype=dt,
        )

    @staticmethod
    def _resolve_device_dtype(
        adj0,
        mask,
        cost_matrix,
        device: Optional[Union[str, torch.device]],
        dtype: Optional[Union[str, torch.dtype]],
    ) -> Tuple[torch.device, torch.dtype]:
        """Resolve target device/dtype from inputs and explicit overrides."""
        infer_from = next(
            (t for t in (adj0, mask, cost_matrix) if isinstance(t, torch.Tensor)), None
        )
        dev = (
            torch.device(device)
            if device is not None
            else (infer_from.device if infer_from is not None else torch.device("cpu"))
        )
        if dtype is None:
            dt = (
                infer_from.dtype
                if infer_from is not None
                else torch.get_default_dtype()
            )
        elif isinstance(dtype, torch.dtype):
            dt = dtype
        elif isinstance(dtype, str):
            key = dtype.split(".")[-1].lower()
            candidate = getattr(torch, key, None)
            if not isinstance(candidate, torch.dtype):
                raise ValueError(f"Unsupported dtype string '{dtype}'")
            dt = candidate
        else:
            raise TypeError("dtype must be a torch.dtype, str, or None")
        return dev, dt

    def _prep_mask(
        self,
        mask,
        N: int,
        dev: torch.device,
        dt: torch.dtype,
    ) -> torch.Tensor:
        """Prepare and sanitize the mask buffer once."""
        mask_buf = _coerce(
            mask, lambda: torch.ones((N, N), device=dev, dtype=dt), device=dev, dtype=dt
        )
        return _zero_mask_diagonal(mask_buf)

    def _prep_adj0(
        self,
        adj0,
        use_sparse: bool,
        N: int,
        dev: torch.device,
        dt: torch.dtype,
    ) -> torch.Tensor:
        """Prepare the base adjacency buffer with backend-aligned layout."""
        if adj0 is None and use_sparse:
            return _make_sparse_zero_matrix((N, N), device=dev, dtype=dt)
        return _coerce(
            adj0,
            lambda: torch.zeros((N, N), device=dev, dtype=dt),
            device=dev,
            dtype=dt,
        )

    def _prep_cost(
        self,
        cost_matrix,
        use_sparse: bool,
        N: int,
        dev: torch.device,
        dt: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Prepare the cost buffer (or ``None`` for sparse implicit unit-cost mode)."""
        if cost_matrix is None and use_sparse:
            return None
        cost_buf = _coerce(
            cost_matrix,
            lambda: torch.ones((N, N), device=dev, dtype=dt),
            device=dev,
            dtype=dt,
        )
        if not use_sparse and _is_sparse_tensor(cost_buf):
            return cost_buf.to_dense()
        return cost_buf

    def _validate_undirected_inputs(
        self,
        *,
        mask_provided: bool,
        adj0_provided: bool,
        cost_provided: bool,
    ) -> None:
        """Require symmetric inputs in undirected mode for user-provided tensors."""
        if self.directed:
            return
        checks = (
            ("mask", self.mask, mask_provided),
            ("adj0", self.adj0, adj0_provided),
            ("cost_matrix", self.cost_matrix, cost_provided),
        )
        for name, tensor, provided in checks:
            if not provided or tensor is None:
                continue
            if not _is_symmetric_matrix(tensor):
                raise ValueError(
                    f"directed=False requires {name} to be symmetric; got shape {tuple(tensor.shape)}."
                )

    def _warn_if_adj0_violates_final_sign(self, requested_final_sign: str) -> None:
        """Warn if ``adj0`` violates the requested output sign cone."""
        if requested_final_sign == "free":
            return
        if _violates_sign_constraint(self.adj0, requested_final_sign):
            warnings.warn(
                f"adj0 violates requested final_sign={requested_final_sign!r}; output sign may be incompatible with adj0.",
                RuntimeWarning,
            )

    def _build_param(
        self,
        *,
        use_sparse_backend: bool,
        rand_init_weights: Union[bool, float],
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module:
        """Instantiate the dense/sparse parameterization backend."""
        if use_sparse_backend:
            edge_index, cost_p_sum = self._prepare_edge_list(
                mask=self.mask,
                cost_matrix=self.cost_matrix,
                directed=self.directed,
                p=self.cost_aggr_norm,
                dtype=dtype,
                device=device,
            )
            return SparseParameterization(
                num_nodes=self.num_nodes,
                budget=self.budget,
                edge_index=edge_index,
                cost_p_sum=cost_p_sum,
                delta_sign=self.delta_sign,
                directed=self.directed,
                strict_budget=self.strict_budget,
                cost_aggr_norm=self.cost_aggr_norm,
                rand_init_weights=rand_init_weights,
            )
        return DenseParameterization(
            num_nodes=self.num_nodes,
            budget=self.budget,
            mask=self.mask,
            cost_matrix=self.cost_matrix,
            delta_sign=self.delta_sign,
            directed=self.directed,
            strict_budget=self.strict_budget,
            cost_aggr_norm=self.cost_aggr_norm,
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
        return (
            f"num_nodes={self.num_nodes}, budget={self.budget}, "
            f"delta_sign={self.delta_sign!r}, final_sign={self.final_sign!r}, directed={self.directed}, "
            f"strict_budget={self.strict_budget}, p={self.cost_aggr_norm}, "
            f"dtype={self.dtype}, device={self.device}"
        )

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
            "directed": self.directed,
            "strict_budget": self.strict_budget,
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

    def should_renorm_after_step(self) -> bool:
        """Return whether post-update parameter renormalization is advised.

        This is ``True`` only when the model enforces a budget and strict
        budget scaling is enabled.
        """
        return (self.budget is not None) and self.strict_budget

    # --------- Build current delta / adjacency ---------------------------------
    def get_delta_adj(self, noise_amplitude: float = 0.0) -> torch.Tensor:
        """Return the normalized perturbation matrix ``delta`` from the backend.

        Args:
          noise_amplitude (float, optional): Standardized magnitude for the
            stochastic perturbation applied to the raw parameters before
            constraints. Defaults to 0 (deterministic).
        """
        return self.param(noise_amplitude=noise_amplitude)

    def forward(self, noise_amplitude: float = 0.0) -> torch.Tensor:
        """Return the full adjacency ``A = adj0 + delta``.

        Handles dense/sparse combinations between ``adj0`` and ``delta`` and
        returns either a dense or a sparse tensor accordingly. When
        ``noise_amplitude > 0`` the same stochastic perturbation as
        :meth:`get_delta_adj` is injected before constraints.

        Args:
          noise_amplitude (float, optional): Magnitude of the Gaussian noise
            applied to ``delta_adj_raw`` prior to constraint handling.
        """
        delta = self.get_delta_adj(noise_amplitude=noise_amplitude)
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
                values = smooth_abs(adj.values())
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
                adj = smooth_abs(adj)
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
        with _suppress_torch_weights_warning():
            ckpt = torch.load(checkpoint_path, map_location=map_location)
        config = ckpt.get("hyper_parameters", {}).get("gradnet_config")
        if config is None:
            raise ValueError(
                "Checkpoint missing 'gradnet_config'; ensure training used updated GradNetLightning."
            )

        model = cls.from_config(config)

        from .trainer import GradNetLightning  # lazy import to avoid cycles

        def _noop_loss_fn(_gn: "GradNet", **_):
            return torch.zeros((), device=model.device, dtype=model.dtype)

        with _suppress_torch_weights_warning():
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
    @staticmethod
    def _prepare_edge_list(
        *,
        mask: torch.Tensor,
        cost_matrix: Optional[torch.Tensor],
        directed: bool,
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
        N = int(mask.shape[0])
        m = mask.coalesce()
        ii, jj = m.indices()
        # Zero diagonal: drop any present and warn
        keep = ii != jj
        dropped = int((~keep).sum().item())
        if dropped > 0:
            warnings.warn(
                f"Mask has {dropped} diagonal entries; they will be ignored (set to 0).",
                RuntimeWarning,
            )
        ii = ii[keep]
        jj = jj[keep]

        if not directed:
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
            cost_p_sum = torch.full(
                (E,), 1.0 if directed else 2.0, device=device, dtype=dtype
            )
            return edge_index.to(device=device), cost_p_sum

        # Warn on asymmetry in directed=False mode.
        if not directed:
            if cost_matrix.layout == torch.strided:
                cm = cost_matrix
                if cm.shape != (N, N):
                    raise ValueError("cost_matrix shape mismatch")
                if not torch.allclose(cm, cm.transpose(-1, -2)):
                    warnings.warn(
                        "directed=False requested but cost_matrix is not symmetric.",
                        RuntimeWarning,
                    )
            else:
                cm = cost_matrix.coalesce()
                ri, rj = edge_index
                c_ij = _gather_sparse_values(cm, ri, rj, default=0.0)
                c_ji = _gather_sparse_values(cm, rj, ri, default=0.0)
                if torch.any(c_ij != c_ji):
                    warnings.warn(
                        "directed=False requested but cost_matrix has asymmetric values on masked edges.",
                        RuntimeWarning,
                    )

        # Build cost_p_sum and warn on missing costs
        if cost_matrix.layout == torch.strided:
            ri, rj = edge_index
            c_ij = torch.abs(cost_matrix[ri, rj]) ** p
            if not directed:
                c_ji = torch.abs(cost_matrix[rj, ri]) ** p
                cost_p_sum = (c_ij + c_ji).to(dtype=dtype, device=device)
            else:
                cost_p_sum = c_ij.to(dtype=dtype, device=device)
        else:
            cm = cost_matrix.coalesce()
            ri, rj = edge_index
            c_ij = torch.abs(_gather_sparse_values(cm, ri, rj, default=0.0)) ** p
            missing_ij = c_ij == 0
            if not directed:
                c_ji = torch.abs(_gather_sparse_values(cm, rj, ri, default=0.0)) ** p
                missing_ji = c_ji == 0
                missing = missing_ij | missing_ji
                cost_p_sum = (c_ij + c_ji).to(dtype=dtype, device=device)
            else:
                missing = missing_ij
                cost_p_sum = c_ij.to(dtype=dtype, device=device)
            miss_count = int(missing.sum().item())
            if miss_count > 0:
                warnings.warn(
                    f"Cost matrix missing {miss_count} entries for masked edges; assuming 0 cost.",
                    RuntimeWarning,
                )

        return edge_index.to(device=device), cost_p_sum


# ----------------------------------------------------------------------------
# Parameterization Submodule (Option 1)
# ----------------------------------------------------------------------------
class DenseParameterization(nn.Module):
    """Dense parameterization of a delta adjacency matrix.

    Maintains a dense, trainable ``delta_adj_raw`` and projects it to a
    constrained perturbation ``delta`` through the following pipeline::

        raw -> (symmetrize?) -> mask -> (square?) -> normalize (when budget is set)

    Args:
      num_nodes (int): Number of nodes (matrix dimension).
      budget (float | None): Target cost-weighted p-norm for ``delta``.
        If ``None``, skip normalization.
      mask (torch.Tensor): Dense mask selecting active entries (1 for active,
        0 for inactive). Nonzero diagonal entries are allowed but typically
        masked out by users.
      cost_matrix (torch.Tensor): Per-entry cost tensor for the normalization.
      delta_sign (str, optional): Sign constraint for the perturbation. One of
        ``{"free", "nonnegative", "nonpositive"}``.
      directed (bool, optional): If ``False``, symmetrize before
        masking/normalizing.
      strict_budget (bool, optional): If ``True``, always scale up/down to the
        exact budget; if ``False``, scale down only.
      cost_aggr_norm (int, optional): Aggregation norm ``p`` for the
        cost-weighted p-norm.
      rand_init_weights (bool | float, optional): Initialization mix coefficient
        ``a``. Cast to float and clamped to ``[0,1]``. Initial raw parameters are
        set to ``(1 - a) * base + a * U(0,1)``, where ``base`` is ones for strict
        budget mode and zeros otherwise.
    """

    def __init__(
        self,
        num_nodes: int,
        budget: Optional[float],
        mask: torch.Tensor,
        cost_matrix: torch.Tensor,
        *,
        delta_sign: str = "nonnegative",
        directed: bool = False,
        strict_budget: bool = False,
        cost_aggr_norm: int = 1,
        rand_init_weights: Union[bool, float] = True,
    ):
        super().__init__()

        self.num_nodes = int(num_nodes)
        self.budget = None if budget is None else float(budget)
        allowed_signs = {"free", "nonnegative", "nonpositive"}
        ds = str(delta_sign).lower()
        if ds not in allowed_signs:
            raise ValueError(
                f"delta_sign must be one of {sorted(allowed_signs)}; got {delta_sign!r}"
            )
        self.delta_sign = ds
        self.directed = bool(directed)
        self.strict_budget = bool(strict_budget)
        self.cost_aggr_norm = int(cost_aggr_norm)

        # non-trainable buffers
        self.register_buffer("mask", torch.as_tensor(mask))
        self.register_buffer("cost_matrix", torch.as_tensor(cost_matrix))

        # trainable parameter
        shape = (self.num_nodes, self.num_nodes)
        delta0 = _init_raw_weights(
            shape,
            rand_init_weights,
            self.strict_budget,
            device=self.mask.device,
            dtype=self.mask.dtype,
        )
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

    def degrees_of_freedom(self) -> int:
        """Return an estimate of the active degrees of freedom."""
        m = self.mask
        if hasattr(m, "layout") and m.layout != torch.strided:
            m = m.to_dense()
        nz = int((m != 0).sum().item())
        dof = nz if self.directed else int(nz / 2)
        if dof <= 0:
            dof = self.num_nodes  # safe fallback to stay compatible with init logic
        return dof

    def extra_repr(self) -> str:
        return (
            f"num_nodes={self.num_nodes}, budget={self.budget}, "
            f"delta_sign={self.delta_sign!r}, directed={self.directed}, "
            f"strict_budget={self.strict_budget}, p={self.cost_aggr_norm}, "
            f"dtype={self.dtype}, device={self.device}"
        )

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
        delta_adj_raw_0 = torch.as_tensor(
            delta_adj_raw_0, device=self.device, dtype=self.dtype
        )
        if delta_adj_raw_0.shape != self.delta_adj_raw.shape:
            raise ValueError(
                f"Shape mismatch: got {tuple(delta_adj_raw_0.shape)}, "
                f"expected {tuple(self.delta_adj_raw.shape)}."
            )
        self.delta_adj_raw.copy_(delta_adj_raw_0)
        self.renorm_params()

    @torch.no_grad()
    def renorm_params(self):
        """Renormalize the raw parameters to a DOF-aware scale.

        Computes a target scale proportional to ``sqrt(D)`` where ``D`` is the
        number of active degrees of freedom implied by ``mask`` and
        ``directed``. This makes the initial magnitude less sensitive to the
        mask sparsity or graph size, improving optimization stability.
        """
        dof = self.degrees_of_freedom()
        eps = self.delta_adj_raw.new_tensor(1e-12)
        delta_adj_norm = torch.linalg.norm(self.delta_adj_raw)
        if delta_adj_norm <= eps:
            return  # avoid divide-by-zero
        target = self.delta_adj_raw.new_tensor(float(dof)) ** 0.5
        scale = target / torch.clamp(delta_adj_norm, min=eps)
        self.delta_adj_raw.mul_(scale)  # in-place scaling

    # --------- Build current delta ---------------------------------------------
    def forward(self, noise_amplitude: float = 0.0) -> torch.Tensor:
        """Project raw parameters to a constrained ``delta`` matrix.

        Applies optional symmetrization and positivity, then masks inactive
        entries and finally scales to match the cost-weighted p-norm budget.
        When ``noise_amplitude > 0``, injects Gaussian noise with norm
        ``sqrt(dof) * noise_amplitude`` before constraint handling.

        Args:
          noise_amplitude (float, optional): Multiplicative factor for the
            Gaussian perturbation applied to ``delta_adj_raw``. Defaults to 0.

        Returns:
          torch.Tensor: Normalized perturbation matrix ``delta``.
        """
        delta = self.delta_adj_raw
        amp = float(noise_amplitude)
        if amp != 0.0:
            noise = torch.randn_like(delta)
            eps = delta.new_tensor(1e-12)
            noise_norm = torch.linalg.norm(noise)
            dof = max(1, self.degrees_of_freedom())
            target = (delta.new_tensor(float(dof)) ** 0.5) * abs(amp)
            noise = noise * (target / torch.clamp(noise_norm, min=eps))
            delta = delta + noise

        if not self.directed:
            delta = symmetrize(delta)

        if self.delta_sign == "nonnegative":
            delta = square(delta)
        elif self.delta_sign == "nonpositive":
            delta = -square(delta)

        delta = delta * self.mask

        if self.budget is not None:
            delta = normalize(
                delta,
                self.budget,
                cost_aggr_norm=self.cost_aggr_norm,
                cost_matrix=self.cost_matrix,
                scale_up=self.strict_budget,
            )
        return delta


# ----------------------------------------------------------------------------
# Parameterization Submodule (Option 2)
# ----------------------------------------------------------------------------
class SparseParameterization(nn.Module):
    """Sparse, edge-list parameterization for masked adjacencies.

    This backend stores a 1D trainable vector of length ``E`` (active edges)
    and constructs a sparse COO tensor for the ``delta`` matrix. In
    ``directed=False`` mode, only ``(i < j)`` edges are parameterized and
    mirrored on output.
    """

    def __init__(
        self,
        *,
        num_nodes: int,
        budget: Optional[float],
        edge_index: torch.Tensor,  # [2, E]
        cost_p_sum: torch.Tensor,  # [E]
        delta_sign: str = "nonnegative",
        directed: bool = False,
        strict_budget: bool = False,
        cost_aggr_norm: int = 1,
        rand_init_weights: Union[bool, float] = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """Construct a sparse edge-list parameterization.

        Args:
          num_nodes (int): Number of nodes ``N`` (matrix dimension).
          budget (float | None): Target cost-weighted p-norm of the
            perturbation. If ``None``, skip normalization.
          edge_index (torch.Tensor): Integer tensor of shape ``(2, E)`` giving
            the edge list. When ``directed=False``, edges must satisfy ``i < j``.
          cost_p_sum (torch.Tensor): Positive tensor of shape ``(E,)``
            containing, for each edge, the sum of costs to the power ``p``
            used in the normalization. For ``directed=False`` this is typically
            ``|c_ij|^p + |c_ji|^p``; for directed, ``|c_ij|^p``.
          delta_sign (str, optional): Sign constraint for the perturbation.
            One of ``{"free", "nonnegative", "nonpositive"}``.
          directed (bool, optional): If ``False``, mirror ``(i, j)`` entries
            to ``(j, i)`` when building the sparse matrix.
          strict_budget (bool, optional): If ``True``, always scale up/down to
            the exact budget; if ``False``, scale down only.
          cost_aggr_norm (int, optional): Aggregation norm ``p`` for the
            cost-weighted p-norm.
          rand_init_weights (bool | float, optional): Initialization mix
            coefficient ``a``. Cast to float and clamped to ``[0,1]``.
            Raw edge weights are set to ``(1 - a) * base + a * U(0,1)``,
            where ``base`` is ones for strict budget mode and zeros otherwise.
          dtype (torch.dtype | None, optional): Parameter/buffer dtype. If
            omitted, inferred from ``cost_p_sum``.
          device (torch.device | None, optional): Parameter/buffer device. If
            omitted, inferred from ``edge_index``.
        """
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.budget = None if budget is None else float(budget)
        allowed_signs = {"free", "nonnegative", "nonpositive"}
        ds = str(delta_sign).lower()
        if ds not in allowed_signs:
            raise ValueError(
                f"delta_sign must be one of {sorted(allowed_signs)}; got {delta_sign!r}"
            )
        self.delta_sign = ds
        self.directed = bool(directed)
        self.strict_budget = bool(strict_budget)
        self.cost_aggr_norm = int(cost_aggr_norm)

        if device is None:
            device = edge_index.device
        if dtype is None:
            dtype = (
                cost_p_sum.dtype
                if torch.is_floating_point(cost_p_sum)
                else torch.get_default_dtype()
            )
        self.register_buffer("edge_index", edge_index.to(device=device))
        self.register_buffer("cost_p_sum", cost_p_sum.to(device=device, dtype=dtype))

        E = self.edge_index.shape[1]
        w0 = _init_raw_weights(
            (E,),
            rand_init_weights,
            self.strict_budget,
            device=device,
            dtype=dtype,
        )
        self.delta_adj_raw = nn.Parameter(w0, requires_grad=True)

    @property
    def device(self) -> torch.device:
        return self.delta_adj_raw.device

    @property
    def dtype(self) -> torch.dtype:
        return self.delta_adj_raw.dtype

    def degrees_of_freedom(self) -> int:
        """Return the number of learnable edge weights (E)."""
        return int(self.edge_index.shape[1])

    def extra_repr(self) -> str:
        E = int(self.edge_index.shape[1])
        return (
            f"num_nodes={self.num_nodes}, edges={E}, budget={self.budget}, "
            f"delta_sign={self.delta_sign!r}, directed={self.directed}, "
            f"strict_budget={self.strict_budget}, p={self.cost_aggr_norm}, "
            f"dtype={self.dtype}, device={self.device}"
        )

    @torch.no_grad()
    def set_initial_state(self, delta_adj_raw_0: torch.Tensor):
        """Set the internal raw edge weights and re-normalize.

        Args:
          delta_adj_raw_0 (torch.Tensor): 1D tensor with length ``E``.

        Raises:
          ValueError: If shape mismatches the internal parameter.
        """
        delta_adj_raw_0 = torch.as_tensor(
            delta_adj_raw_0, device=self.device, dtype=self.dtype
        )
        if delta_adj_raw_0.shape != self.delta_adj_raw.shape:
            raise ValueError(
                f"Shape mismatch: got {tuple(delta_adj_raw_0.shape)}, expected {tuple(self.delta_adj_raw.shape)}."
            )
        self.delta_adj_raw.copy_(delta_adj_raw_0)
        self.renorm_params()

    @torch.no_grad()
    def renorm_params(self):
        """Scale raw edge parameters to a backend-aligned constant norm.

        Directed mode uses ``sqrt(E)`` for ``E`` learnable edges.
        ``directed=False`` mode uses ``sqrt(E/2)`` so the per-directed-entry
        raw scale matches the dense backend after post-step renormalization.
        """
        eps = self.delta_adj_raw.new_tensor(1e-12)
        wnorm = torch.linalg.norm(self.delta_adj_raw)
        if wnorm <= eps:
            return
        dof = self.degrees_of_freedom()
        eff_dof = float(dof) * (1.0 if self.directed else 0.5)
        target = self.delta_adj_raw.new_tensor(max(eff_dof, 1e-12)) ** 0.5
        scale = target / torch.clamp(wnorm, min=eps)
        self.delta_adj_raw.mul_(scale)

    def forward(self, noise_amplitude: float = 0.0) -> torch.Tensor:
        """Project raw edge weights to a sparse, normalized ``delta``.

        Applies optional positivity in vector space, scales to match the
        cost-weighted p-norm budget, and constructs a COO matrix. In
        ``directed=False`` mode, edges are mirrored.
        When ``noise_amplitude > 0``,
        adds Gaussian noise with norm ``sqrt(dof) * noise_amplitude`` to the
        raw edge weights before enforcing constraints.

        Args:
          noise_amplitude (float, optional): Multiplicative noise factor for
            the raw edge weights. Defaults to 0.

        Returns:
          torch.Tensor: Coalesced sparse COO tensor of shape ``(N, N)``.
        """
        w = self.delta_adj_raw
        amp = float(noise_amplitude)
        if amp != 0.0:
            noise = torch.randn_like(w)
            eps = w.new_tensor(1e-12)
            noise_norm = torch.linalg.norm(noise)
            dof = max(1, self.degrees_of_freedom())
            target = (w.new_tensor(float(dof)) ** 0.5) * abs(amp)
            noise = noise * (target / torch.clamp(noise_norm, min=eps))
            w = w + noise

        if self.delta_sign == "nonnegative":
            w = square(w)
        elif self.delta_sign == "nonpositive":
            w = -square(w)

        p = max(1, int(self.cost_aggr_norm))
        if self.budget is None:
            vals = w
        else:
            eps = w.new_tensor(1e-8)
            s = (torch.abs(w) ** p * self.cost_p_sum).sum() ** (1.0 / p)
            norm_val_t = w.new_tensor(self.budget)
            scale = norm_val_t / torch.clamp(s, min=eps)
            if not self.strict_budget:
                scale = torch.minimum(scale, s.new_tensor(1.0))
            vals = w * scale

        if not self.directed:
            i, j = self.edge_index
            ii = torch.cat([i, j], dim=0)
            jj = torch.cat([j, i], dim=0)
            vv = torch.cat([vals, vals], dim=0)
            return torch.sparse_coo_tensor(
                torch.stack([ii, jj], dim=0),
                vv,
                (self.num_nodes, self.num_nodes),
                device=self.device,
                dtype=self.dtype,
            ).coalesce()
        else:
            return torch.sparse_coo_tensor(
                self.edge_index,
                vals,
                (self.num_nodes, self.num_nodes),
                device=self.device,
                dtype=self.dtype,
            ).coalesce()


# ----------------------------------------------------------------------------
# Global Helper Functions (dtype/device-safe)
# ----------------------------------------------------------------------------
def normalize(
    matrix: torch.Tensor,
    norm_val: float,
    cost_aggr_norm: int = 1,
    cost_matrix: Optional[torch.Tensor] = None,
    scale_up: bool = True,
) -> torch.Tensor:
    """Scale a matrix to satisfy a cost-weighted p-norm budget.

    Scales ``matrix`` so that ``|| cost_matrix * matrix ||_p == norm_val``
    (or ``<=`` when ``scale_up=False``), using the same dtype/device as the
    input.

    Args:
      matrix (torch.Tensor): Input tensor to scale.
      norm_val (float): Target norm value (budget).
      cost_aggr_norm (int, optional): Aggregation norm ``p`` used for the
        cost-weighted p-norm. Must be a positive integer.
      cost_matrix (torch.Tensor | None, optional): Per-entry cost tensor; if
        ``None``, uses ones like ``matrix``. May be dense or sparse; dense
        arithmetic is used.
      scale_up (bool, optional): If ``False``, scales by ``min(scale, 1)`` to
        avoid upscaling beyond the current norm.

    Returns:
      torch.Tensor: Scaled matrix with cost-weighted p-norm equal to
        ``norm_val`` (or not exceeding it when ``scale_up=False``).
    """
    if cost_matrix is None:
        cost_matrix = torch.ones_like(matrix)

    if not isinstance(cost_aggr_norm, int) or cost_aggr_norm <= 0:
        raise ValueError("cost_aggr_norm must be a positive integer")
    p = cost_aggr_norm

    # If matrix is dense but cost_matrix is sparse, densify cost for elementwise ops
    if (
        hasattr(cost_matrix, "layout")
        and matrix.layout == torch.strided
        and cost_matrix.layout != torch.strided
    ):
        cost_matrix = cost_matrix.to_dense()

    eps = matrix.new_tensor(1e-8)
    s = (torch.abs(cost_matrix * matrix) ** p).sum() ** (1.0 / p)

    norm_val_t = matrix.new_tensor(norm_val)
    scale = norm_val_t / torch.clamp(s, min=eps)

    if not scale_up:
        scale = torch.minimum(scale, s.new_tensor(1.0))

    return matrix * scale


def smooth_abs(matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Smooth absolute value."""
    if eps == 0:
        return torch.abs(matrix)
    return matrix * torch.tanh(matrix / eps)


def square(matrix: torch.Tensor) -> torch.Tensor:
    """element-wise square"""
    return matrix**2


def symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    """Return the averaged symmetric part of a square matrix.

    Computes ``0.5 * (M + M^T)`` along the last two axes.

    Args:
      matrix (torch.Tensor): Square matrix or a batch thereof.

    Returns:
      torch.Tensor: Symmetrized matrix.
    """
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _is_sparse_tensor(x: object) -> bool:
    """Return ``True`` when ``x`` is a non-strided PyTorch tensor."""
    return isinstance(x, torch.Tensor) and x.layout != torch.strided


def _is_symmetric_matrix(
    matrix: torch.Tensor, *, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    """Return ``True`` if ``matrix`` is square and symmetric."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    if _is_sparse_tensor(matrix):
        mc = matrix.coalesce()
        mt = mc.transpose(0, 1).coalesce()
        if not torch.equal(mc.indices(), mt.indices()):
            return False
        return bool(torch.allclose(mc.values(), mt.values(), rtol=rtol, atol=atol))
    return bool(torch.allclose(matrix, matrix.transpose(-1, -2), rtol=rtol, atol=atol))


def _violates_sign_constraint(matrix: torch.Tensor, sign: str) -> bool:
    """Return whether ``matrix`` has entries outside the requested sign cone."""
    s = str(sign).lower()
    if s == "free":
        return False
    vals = matrix.coalesce().values() if _is_sparse_tensor(matrix) else matrix
    if s == "nonnegative":
        return bool(torch.any(vals < 0))
    if s == "nonpositive":
        return bool(torch.any(vals > 0))
    raise ValueError(f"Unknown sign constraint: {sign!r}")


def _make_sparse_zero_matrix(
    shape: Tuple[int, int], *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Create a coalesced sparse COO all-zero matrix."""
    idx = torch.empty((2, 0), dtype=torch.long, device=device)
    vals = torch.empty((0,), dtype=dtype, device=device)
    return torch.sparse_coo_tensor(
        idx, vals, shape, device=device, dtype=dtype
    ).coalesce()


def _coerce(
    x: object,
    make_fallback: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert input to a detached tensor on ``(device, dtype)``."""
    if x is None:
        t = make_fallback()
    elif isinstance(x, torch.Tensor):
        # preserve sparse layout when provided
        t = x.to(device=device, dtype=dtype)
    else:
        t = torch.as_tensor(x, device=device, dtype=dtype)
    return t.detach()


def _zero_mask_diagonal(mask: torch.Tensor) -> torch.Tensor:
    """Return ``mask`` with diagonal entries removed/set to zero."""
    if _is_sparse_tensor(mask):
        mc = mask.coalesce()
        ii, jj = mc.indices()
        keep = ii != jj
        return torch.sparse_coo_tensor(
            torch.stack([ii[keep], jj[keep]], dim=0),
            mc.values()[keep],
            mc.shape,
            device=mc.device,
            dtype=mc.dtype,
        ).coalesce()
    m = mask.clone()
    if m.ndim >= 2 and m.shape[-1] == m.shape[-2]:
        m.fill_diagonal_(0)
    return m


def _init_raw_weights(
    shape: Tuple[int, ...],
    a: Union[bool, float],
    strict_budget: bool,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Initialize raw parameters with a budget-aware base + random mix."""
    try:
        mix = float(a)
    except Exception:
        mix = 1.0 if bool(a) else 0.0
    mix = max(0.0, min(1.0, mix))
    base = torch.ones(shape, device=device, dtype=dtype)
    if not strict_budget:
        base = torch.zeros(shape, device=device, dtype=dtype)
    rnd = torch.rand(shape, device=device, dtype=dtype)
    # Keep existing behavior: mix=1 -> random, mix=0 -> base.
    return (1.0 - mix) * base + mix * rnd


def _gather_sparse_values(
    cm: torch.Tensor, ri: torch.Tensor, rj: torch.Tensor, default: float = 0.0
) -> torch.Tensor:
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
    match = (
        (sk[pos] == qkeys)
        if sk.numel() > 0
        else torch.zeros_like(qkeys, dtype=torch.bool)
    )
    out = torch.full(
        qkeys.shape, fill_value=float(default), device=vals.device, dtype=vals.dtype
    )
    if sk.numel() > 0:
        out[match] = svals[pos[match]]
    return out


@contextmanager
def _suppress_torch_weights_warning():
    """Silence torch's weights_only FutureWarning for trusted checkpoints."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"You are using `torch.load` with `weights_only=False`.*",
            category=FutureWarning,
        )
        yield
