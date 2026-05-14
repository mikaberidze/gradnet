"""Lightweight default trainer for :class:`gradnet.GradNet`.

A small, self-contained training loop with optional TensorBoard/CSV
logging, best-loss + periodic checkpointing, a tqdm progress bar, and a
loss dtype/device safety net that auto-casts mismatched losses and
warns once per ``fit()`` call.

Differences from :func:`gradnet.pl_fit` (the PyTorch Lightning trainer
available via ``pip install gradnet[pl]``):

* ``device`` replaces ``accelerator``
* No ``precision`` parameter — control dtype with ``gn.to(dtype=...)``
  before calling ``fit``; mixed precision lives in ``pl_fit``
* No multi-GPU / DDP / TPU / arbitrary PL callbacks or loggers
"""

from __future__ import annotations

import csv
import os
import random
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

try:
    from tqdm.auto import tqdm
except Exception:
    from tqdm import tqdm

from .gradnet import GradNet
from .utils import _to_like_struct

# --------------------------------------------------------------------------- Protocols


class LossFn(Protocol):
    """Loss callable: ``loss_fn(gn, **loss_kwargs) -> Tensor | (Tensor, metrics)``."""

    def __call__(
        self,
        model: GradNet,
        **loss_kwargs: Any,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, Dict[str, Union[float, int, torch.Tensor]]],
    ]: ...


class Callback(Protocol):
    """Three-hook training callback. Stateless callbacks can omit hooks."""

    def on_fit_start(self, trainer: "GradNetTrainer") -> None: ...
    def on_step_end(
        self,
        trainer: "GradNetTrainer",
        step: int,
        loss: float,
        metrics: Dict[str, float],
    ) -> None: ...
    def on_fit_end(self, trainer: "GradNetTrainer") -> None: ...


class Logger(Protocol):
    """Metric sink. Implement to plug in custom logging backends."""

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None: ...
    def finalize(self) -> None: ...


# --------------------------------------------------------------------------- Trainer state


class GradNetTrainer:
    """Container exposed to callbacks while :func:`fit` is running."""

    def __init__(self, max_epochs: int):
        self.max_epochs = max_epochs
        self.current_epoch: int = 0
        self.callback_metrics: Dict[str, float] = {}


# --------------------------------------------------------------------------- Built-in callback


class _EpochTQDM:
    """tqdm progress bar driven by the Callback protocol."""

    def on_fit_start(self, trainer: GradNetTrainer) -> None:
        self._bar = tqdm(total=trainer.max_epochs, desc="Updates", dynamic_ncols=True)

    def on_step_end(self, trainer, step, loss, metrics):
        self._bar.set_postfix({k: f"{v:.4g}" for k, v in metrics.items()})
        self._bar.update(1)

    def on_fit_end(self, trainer):
        self._bar.close()


# --------------------------------------------------------------------------- Loggers


class TensorBoardLogger:
    """Writes scalar metrics via :class:`torch.utils.tensorboard.SummaryWriter`."""

    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter

        os.makedirs(log_dir, exist_ok=True)
        self._writer = SummaryWriter(log_dir)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        for k, v in metrics.items():
            self._writer.add_scalar(k, v, global_step=step)

    def finalize(self) -> None:
        self._writer.close()


class CSVLogger:
    """Appends each step's metrics to ``<log_dir>/metrics.csv``.

    Format matches :func:`gradnet.utils.load_scalars` (columns: ``epoch``,
    then one column per metric).
    """

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self._path = Path(log_dir) / "metrics.csv"
        self._file = self._path.open("w", newline="")
        self._writer: Optional[csv.DictWriter] = None

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        row = {"epoch": step, **metrics}
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def finalize(self) -> None:
        self._file.close()


# --------------------------------------------------------------------------- Checkpointing


class CheckpointManager:
    """Saves the best-loss checkpoint, optional periodic snapshots, and an optional ``last.ckpt``.

    Filename layout matches the PL trainer so existing utilities
    (``utils.animate_adjacency``, ``GradNet.from_checkpoint``) keep working.
    """

    def __init__(
        self,
        dirpath: Optional[str],
        save_last: bool = False,
        every_n: Optional[int] = None,
    ):
        self._dir = Path(dirpath or "checkpoints")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._save_last = save_last
        self._every_n = every_n
        self._best_loss = float("inf")
        self._best_path: Optional[str] = None

    def _dump(self, path: Path, gn: nn.Module, cfg) -> None:
        torch.save({"state_dict": gn.state_dict(), "gradnet_config": cfg}, path)

    def on_step_end(self, step: int, loss: float, gn: nn.Module, cfg) -> None:
        if loss < self._best_loss:
            self._best_loss = loss
            new_path = self._dir / f"gn-{step:05d}.ckpt"
            self._dump(new_path, gn, cfg)
            if self._best_path and Path(self._best_path) != new_path:
                Path(self._best_path).unlink(missing_ok=True)
            self._best_path = str(new_path)
        if self._every_n and (step + 1) % self._every_n == 0:
            self._dump(self._dir / f"gn-periodic-{step:05d}.ckpt", gn, cfg)

    def on_fit_end(self, step: int, gn: nn.Module, cfg) -> None:
        if self._save_last:
            self._dump(self._dir / "last.ckpt", gn, cfg)

    @property
    def best_model_path(self) -> Optional[str]:
        return self._best_path


# --------------------------------------------------------------------------- Helpers


def _resolve_device(device: Union[str, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_max_time(s: str) -> float:
    """Parse a ``[DD:]HH:MM:SS`` wall-clock budget into seconds."""
    parts = [int(p) for p in s.split(":")]
    if not 1 <= len(parts) <= 4:
        raise ValueError(f"max_time must be 'DD:HH:MM:SS' (up to 4 parts), got {s!r}")
    parts = [0] * (4 - len(parts)) + parts
    dd, hh, mm, ss = parts
    return dd * 86400 + hh * 3600 + mm * 60 + ss


def _print_model_summary(gn: nn.Module) -> None:
    total = sum(p.numel() for p in gn.parameters())
    trainable = sum(p.numel() for p in gn.parameters() if p.requires_grad)
    print(f"GradNet | params: {total:,} total, {trainable:,} trainable")


_DTYPE_DEVICE_ERROR_RE = re.compile(
    r"dtype|same device|expected all tensors to be on the same device",
    re.IGNORECASE,
)


def _looks_like_dtype_or_device_error(exc: Exception) -> bool:
    return bool(_DTYPE_DEVICE_ERROR_RE.search(str(exc)))


def _warn_loss_mismatch(
    got_dtype: Optional[torch.dtype],
    got_device: Optional[torch.device],
    want_dtype: torch.dtype,
    want_device: torch.device,
    *,
    stage: str,
    original_error: Optional[str] = None,
) -> None:
    """Loud, one-shot warning describing a loss dtype/device mismatch."""
    msg = (
        "loss tensor dtype/device mismatch\n"
        f"  detected at:   {stage}\n"
        f"  loss returned: dtype={got_dtype} device={got_device}\n"
        f"  GradNet wants: dtype={want_dtype} device={want_device}\n"
        "\n"
        f"The trainer auto-cast the loss to dtype={want_dtype}, device={want_device}\n"
        "to keep training running, but this likely indicates a mismatch\n"
        "inside your loss function — any of:\n"
        "  • a tensor of a different dtype/device mixed into the graph\n"
        "  • a numpy / Python scalar promoted unexpectedly\n"
        "  • a manually-created helper without explicit dtype= / device=\n"
        "\n"
        "Recommended fixes:\n"
        "  • Pass helpers via `loss_kwargs=` — they are auto-cast to gn's\n"
        "    dtype/device.\n"
        "  • Or construct helpers inside loss_fn with explicit dtype/device:\n"
        "        A = gn()\n"
        "        helper = torch.tensor([...], dtype=A.dtype, device=A.device)\n"
        "        helper2 = some_array.to(dtype=A.dtype, device=A.device)\n"
        "\n"
        "This warning is shown only once per fit() call."
    )
    if original_error:
        msg += f"\n\nOriginal error during backward:\n  {original_error}"

    banner = "=" * 64
    body = f"{banner}\nWARNING: {msg}\n{banner}"
    if sys.stderr.isatty():
        body = f"\033[31m{body}\033[0m"
    print(body, file=sys.stderr)

    warnings.warn(msg, category=UserWarning, stacklevel=3)


def _validate_callbacks(callbacks) -> List[Callback]:
    if callbacks is None:
        return []
    cbs = list(callbacks)
    try:
        import pytorch_lightning as pl  # type: ignore
    except ImportError:
        return cbs
    for cb in cbs:
        if isinstance(cb, pl.Callback):
            raise TypeError(
                "pl.Callback instances require gradnet[pl]; "
                "use gradnet.pl_fit(...) for the PyTorch Lightning trainer."
            )
    return cbs


def _resolve_logger(
    logger: Union[Logger, bool, None],
    log_dir: Optional[str],
    verbose: bool,
) -> Optional[Logger]:
    if not logger:  # None or False
        return None
    if logger is True:
        path = log_dir or "trainer_logs/gradnet"
        try:
            return TensorBoardLogger(path)
        except Exception as exc:
            if verbose:
                warnings.warn(
                    f"TensorBoard unavailable ({exc}); falling back to CSVLogger.",
                    RuntimeWarning,
                )
            return CSVLogger(path)
    # Reject PL logger instances before duck-typing (they also expose log_metrics/finalize)
    try:
        from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase  # type: ignore

        if isinstance(logger, LightningLoggerBase):
            raise TypeError(
                "PyTorch Lightning loggers require gradnet[pl]; "
                "use gradnet.pl_fit(...) for the PyTorch Lightning trainer."
            )
    except ImportError:
        pass
    if hasattr(logger, "log_metrics") and hasattr(logger, "finalize"):
        return logger
    raise TypeError(f"Unsupported `logger` value: {logger!r}")


# --------------------------------------------------------------------------- fit


def fit(
    *,
    gn: GradNet,
    loss_fn: LossFn,
    loss_kwargs: Optional[Mapping[str, Any]] = None,
    num_updates: int,
    optim_cls: type = torch.optim.Adam,
    optim_kwargs: Optional[Dict[str, Any]] = None,
    sched_cls: Optional[type] = None,
    sched_kwargs: Optional[Dict[str, Any]] = None,
    device: Union[str, torch.device] = "auto",
    logger: Union[Logger, bool, None] = False,
    log_dir: Optional[str] = None,
    enable_checkpointing: bool = False,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every_n: Optional[int] = None,
    save_last: bool = False,
    callbacks: Optional[List[Callback]] = None,
    max_time: Optional[str] = None,
    grad_clip_val: float = 0.0,
    post_step_renorm: bool = True,
    compile_model: bool = False,
    seed: Optional[int] = None,
    deterministic: Optional[Union[bool, str]] = None,
    verbose: bool = True,
) -> Tuple[GradNetTrainer, Optional[str]]:
    """Optimise ``gn`` for ``num_updates`` steps.

    Each step evaluates ``loss_fn(gn, **loss_kwargs)``, runs
    ``loss.backward()``, optionally clips gradients, steps the optimiser
    (and scheduler if any), optionally renormalises ``gn``, and dispatches
    metrics to logger/checkpoint/callbacks.

    Returns ``(trainer, best_ckpt_path)``. ``trainer.callback_metrics`` holds
    the last step's metrics; ``best_ckpt_path`` is ``None`` when
    ``enable_checkpointing=False``.

    See :func:`gradnet.pl_fit` for mixed precision, multi-GPU, and the full
    PyTorch Lightning feature set.
    """
    if seed is not None:
        _seed_everything(seed)
    if deterministic is not None:
        torch.use_deterministic_algorithms(bool(deterministic))

    if loss_kwargs is not None and not isinstance(loss_kwargs, Mapping):
        raise TypeError(
            "`loss_kwargs` must be a Mapping of keyword arguments (or None)."
        )

    callbacks = _validate_callbacks(callbacks)
    tr_logger = _resolve_logger(logger, log_dir, verbose)

    target_device = _resolve_device(device)
    gn.to(target_device)
    cfg = (
        gn.export_config() if isinstance(gn, GradNet) else None
    )  # capture before compile
    if compile_model:
        try:
            gn = torch.compile(gn)
        except Exception as e:
            warnings.warn(f"torch.compile failed; continuing uncompiled. Error: {e}")

    ref_dtype, ref_device = gn.dtype, gn.device
    loss_kwargs = _to_like_struct(loss_kwargs, gn) if loss_kwargs else {}

    opt = optim_cls(gn.parameters(), **(optim_kwargs or {"lr": 1e-2}))
    sched = sched_cls(opt, **(sched_kwargs or {})) if sched_cls else None

    ckpt = (
        CheckpointManager(
            checkpoint_dir, save_last=save_last, every_n=checkpoint_every_n
        )
        if enable_checkpointing
        else None
    )

    trainer = GradNetTrainer(max_epochs=int(num_updates))
    cbs: List[Callback] = list(callbacks)
    if verbose:
        _print_model_summary(gn)
        cbs.append(_EpochTQDM())

    for cb in cbs:
        cb.on_fit_start(trainer)

    deadline = time.monotonic() + _parse_max_time(max_time) if max_time else None
    mismatch_warned = False

    for step in range(trainer.max_epochs):
        trainer.current_epoch = step

        out = loss_fn(gn, **loss_kwargs)
        loss, metrics = (out, {}) if isinstance(out, torch.Tensor) else out

        if (loss.dtype != ref_dtype) or (loss.device != ref_device):
            if not mismatch_warned:
                _warn_loss_mismatch(
                    loss.dtype,
                    loss.device,
                    ref_dtype,
                    ref_device,
                    stage="loss return",
                )
                mismatch_warned = True
            loss = loss.to(dtype=ref_dtype, device=ref_device)

        try:
            loss.backward()
        except RuntimeError as e:
            if not mismatch_warned and _looks_like_dtype_or_device_error(e):
                _warn_loss_mismatch(
                    None,
                    None,
                    ref_dtype,
                    ref_device,
                    stage="backward",
                    original_error=str(e),
                )
                mismatch_warned = True
            raise

        if grad_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(gn.parameters(), grad_clip_val)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if sched:
            sched.step()

        if post_step_renorm and hasattr(gn, "renorm_params"):
            policy = getattr(gn, "should_renorm_after_step", None)
            if (not callable(policy)) or bool(policy()):
                gn.renorm_params()

        metrics_f = {
            "loss": float(loss.detach().item()),
            **{
                k: float(v.item() if isinstance(v, torch.Tensor) else v)
                for k, v in metrics.items()
            },
        }
        trainer.callback_metrics = metrics_f
        if tr_logger:
            tr_logger.log_metrics(metrics_f, step)
        if ckpt:
            ckpt.on_step_end(step, metrics_f["loss"], gn, cfg)
        for cb in cbs:
            cb.on_step_end(trainer, step, metrics_f["loss"], metrics_f)

        if deadline and time.monotonic() >= deadline:
            break

    if ckpt:
        ckpt.on_fit_end(trainer.current_epoch, gn, cfg)
    for cb in cbs:
        cb.on_fit_end(trainer)
    if tr_logger:
        tr_logger.finalize()

    return trainer, (ckpt.best_model_path if ckpt else None)
