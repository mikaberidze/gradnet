"""
Grid-mask training to minimize resistance distance (Kirchhoff index).

This example:
  - Builds a 2D grid mask over N = rows * cols nodes
  - Trains edge weights (under a fixed budget) to reduce resistance distance
  - Plots the resulting weighted graph with a grid layout and edge thicknesses

Run from repo root:
  python examples/5_grid_resistance.py
"""
from __future__ import annotations
import math
import argparse
from typing import Dict, Tuple

import torch
import torch.linalg as LA
import matplotlib.pyplot as plt
import networkx as nx

from gradnet import GradNet, fit
from gradnet.utils import to_networkx


def make_grid_mask(rows: int, cols: int, *, device=None, dtype=None) -> torch.Tensor:
    """Return a dense 0/1 mask for 4-neighbor grid edges (undirected).

    Nodes are indexed row-major: i = r * cols + c.
    Diagonal is zero; (i,j) and (j,i) both set to 1 for neighbors.
    """
    N = int(rows) * int(cols)
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    M = torch.zeros((N, N), device=device, dtype=dtype)

    def idx(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            i = idx(r, c)
            if r + 1 < rows:
                j = idx(r + 1, c)
                M[i, j] = 1
                M[j, i] = 1
            if c + 1 < cols:
                j = idx(r, c + 1)
                M[i, j] = 1
                M[j, i] = 1
    return M


def grid_positions(rows: int, cols: int) -> Dict[int, Tuple[float, float]]:
    """Return a dict of node -> (x,y) on a regular grid for NetworkX."""
    pos: Dict[int, Tuple[float, float]] = {}
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            pos[i] = (float(c), -float(r))
    return pos


def resistance_distance_loss(gn: GradNet, eps: float = 1e-6) -> torch.Tensor:
    """Kirchhoff index = sum of reciprocals of nonzero Laplacian eigenvalues.

    Uses symmetric eigen decomposition for stability. Adds a small epsilon to
    avoid numerical issues when eigenvalues are near zero.
    """
    A = gn()  # full adjacency
    Ad = A.to_dense() if (isinstance(A, torch.Tensor) and A.layout != torch.strided) else A
    deg = Ad.sum(dim=1)
    L = torch.diag(deg) - Ad
    # eigenvalues in ascending order; L is symmetric PSD
    eigs = LA.eigvalsh(L)
    # skip the zero eigenvalue (connected components -> multiplicity)
    eigs_sorted, _ = torch.sort(eigs)
    nonzero = eigs_sorted[1:]  # drop the smallest eigenvalue (should be ~0)
    loss = torch.sum(1.0 / (nonzero + eps))
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6, help="Grid rows")
    ap.add_argument("--cols", type=int, default=6, help="Grid cols")
    ap.add_argument("--budget", type=float, default=-1.0, help="Total edge-weight budget (<=0 uses a size-based default)")
    ap.add_argument("--updates", type=int, default=400, help="Optimization steps")
    ap.add_argument("--lr", type=float, default=5e-2, help="Learning rate")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--device", type=str, default="cpu", help="cpu | cuda | mps")
    args = ap.parse_args()

    # Helper: choose a mild default budget that scales with grid size
    # If user passed default sentinel, compute here
    rows, cols = args.rows, args.cols
    device = torch.device(args.device)

    # Construct mask and model
    mask = make_grid_mask(rows, cols, device=device, dtype=torch.float32)
    N = rows * cols
    cost = torch.ones((N, N), device=device, dtype=torch.float32)

    # If budget == -1, set a grid-size-aware default
    budget = args.budget
    if budget <= 0:
        # proportional to number of edges in the grid (approx 2*rows*cols)
        approx_edges = 2 * rows * cols - rows - cols
        budget = float(approx_edges)

    gn = GradNet(
        num_nodes=N,
        budget=budget,
        mask=mask,
        cost_matrix=cost,
        undirected=True,
        positive=True,
        rand_init_weights=0.5,  # mix of ones and uniform for stable start
        use_budget_up=False,
        cost_aggr_norm=1,
        adj0=torch.zeros((N, N), device=device, dtype=torch.float32),
    )

    # Optimize edge weights
    fit(
        gn=gn,
        loss_fn=resistance_distance_loss,
        loss_kwargs={},
        num_updates=args.updates,
        optim_kwargs={"lr": args.lr},
        logger=False,
        enable_checkpointing=False,
        verbose=True,
    )

    # Plot with grid layout, edge thickness by weight
    net = to_networkx(gn, pruning_threshold=1e-8)
    pos = grid_positions(rows, cols)
    weights = nx.get_edge_attributes(net, "weight")
    edge_widths = list(weights.values())

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    nx.draw(
        net,
        pos=pos,
        ax=ax,
        with_labels=False,
        node_size=40,
        width=edge_widths,
        edgecolors="black",
        )
    ax.set_aspect("equal")
    ax.set_title(f"Grid {rows}x{cols} (budget={budget:g}) – edge widths=weights")
    ax.set_axis_off()
    plt.show()


if __name__ == "__main__":
    main()
