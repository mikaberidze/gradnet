"""Default trainer for :class:`gradnet.GradNet`.

A small, self-contained training loop with optional TensorBoard/CSV
logging, best-loss + periodic checkpointing, a tqdm progress bar, and a
loss dtype/device safety net that auto-casts mismatched losses and
warns once per ``fit()`` call.
"""

from __future__ import annotations

import csv
import inspect
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
    """Protocol for loss callables accepted by :func:`fit`.

    A loss function is called once per optimization step as
    ``loss_fn(gn, **loss_kwargs)``. If its signature defines a keyword-compatible
    ``step`` parameter, :func:`fit` calls it as
    ``loss_fn(gn, step=step, **loss_kwargs)`` with the zero-based optimization
    step. It may be a plain function, lambda, bound method, callable object, or
    any object whose ``__call__`` method has this shape. The first argument is
    the :class:`gradnet.GradNet` being optimized; keyword arguments come from
    ``loss_kwargs``. The ``step`` keyword is reserved for the trainer and cannot
    be supplied through ``loss_kwargs``.

    Implementations must return either a scalar differentiable
    :class:`torch.Tensor`, or ``(loss, metrics)`` where ``loss`` is that tensor
    and ``metrics`` is a dictionary of scalar values. Metric values may be
    floats, ints, or scalar tensors and are logged after conversion to floats.
    """

    def __call__(
        self,
        model: GradNet,
        **loss_kwargs: Any,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, Dict[str, Union[float, int, torch.Tensor]]],
    ]: ...


class Callback(Protocol):
    """Protocol for callback objects accepted by :func:`fit`.

    A callback is any object implementing the three lifecycle hooks below.
    ``fit`` calls ``on_fit_start`` before the first update, ``on_step_end``
    after every optimizer step, and ``on_fit_end`` after the training loop and
    checkpoint finalization. Hooks receive the live :class:`GradNetTrainer`
    state; step-end hooks also receive the zero-based step index, scalar loss,
    and metrics dictionary.
    Callbacks may call ``trainer.request_stop(reason)`` to end training after
    the current step-end callback pass.

    If a callback has no work for a lifecycle point, implement that hook as a
    no-op method. The trainer invokes all three hooks unconditionally.
    """

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
        self.should_stop: bool = False
        self.stop_reason: Optional[str] = None

    def request_stop(self, reason: Optional[str] = None) -> None:
        """Ask the active training loop to stop after the current callback pass."""
        self.should_stop = True
        self.stop_reason = reason


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
    """Saves the best-loss checkpoint, optional periodic snapshots, and an optional ``last.ckpt``."""

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

_RESERVED_STEP_KWARG = "step"


def _looks_like_dtype_or_device_error(exc: Exception) -> bool:
    return bool(_DTYPE_DEVICE_ERROR_RE.search(str(exc)))


def _loss_fn_accepts_keyword_step(loss_fn: LossFn) -> bool:
    """Return whether ``loss_fn(gn, step=...)`` is an explicit valid call."""
    try:
        signature = inspect.signature(loss_fn)
    except (TypeError, ValueError):
        return False
    param = signature.parameters.get(_RESERVED_STEP_KWARG)
    if param is None:
        return False
    return param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ) and _signature_accepts_model_and_step(signature)


def _signature_accepts_model_and_step(signature: inspect.Signature) -> bool:
    try:
        signature.bind_partial(None, **{_RESERVED_STEP_KWARG: 0})
    except TypeError:
        return False
    return True


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
    """Optimize a :class:`gradnet.GradNet` with a user-defined loss function.

    This is a small, self-contained training loop for GradNet objects. Each
    update evaluates ``loss_fn(gn, **loss_kwargs)``. If ``loss_fn`` defines a
    keyword-compatible ``step`` parameter, the trainer instead evaluates
    ``loss_fn(gn, step=step, **loss_kwargs)`` with the current zero-based update
    index. The loop backpropagates through the returned loss, optionally clips
    gradients, steps the optimizer and scheduler, optionally renormalizes the
    GradNet parameters, and dispatches scalar metrics to loggers, checkpointing,
    and callbacks.

    Args:
      gn (GradNet): GradNet instance to optimize. It is moved to ``device`` and
        trained in place.
      loss_fn (LossFn): Callable evaluated once per update as
        ``loss_fn(gn, **loss_kwargs)``. If its signature defines a
        keyword-compatible ``step`` parameter, ``fit`` passes the current
        zero-based update index as ``step``. It may be a function, lambda, bound
        method, or callable object. It must return either a scalar differentiable tensor ``loss``,
        or ``(loss, metrics)`` where metrics is a mapping with string keys and scalar values.
      loss_kwargs (Mapping[str, Any] | None, optional): Keyword arguments passed
        to ``loss_fn``. Tensor-like values are moved/cast to match ``gn`` before
        training. If the returned loss tensor has a different dtype or device,
        the trainer auto-casts it to match ``gn`` and emits a one-time warning.
        The parameter name ``"step"`` is reserved for the current optimization
        step and cannot be supplied through ``loss_kwargs``.
      num_updates (int): Maximum number of optimizer updates to run.
      optim_cls (type, optional): Optimizer class. Defaults to
        :class:`torch.optim.Adam`.
      optim_kwargs (dict | None, optional): Keyword arguments for ``optim_cls``.
        Defaults to ``{"lr": 1e-2}`` when omitted.
      sched_cls (type | None, optional): Optional learning-rate scheduler class.
      sched_kwargs (dict | None, optional): Keyword arguments for ``sched_cls``.
      device (str | torch.device, optional): Training device. ``"auto"`` selects
        CUDA if available, then MPS if available, otherwise CPU.
      logger (Logger | bool | None, optional): ``False``/``None`` disables
        logging. ``True`` creates a TensorBoard logger when available and falls
        back to CSV. A custom logger must implement ``log_metrics`` and
        ``finalize``.
      log_dir (str | None, optional): Directory for the built-in logger.
      enable_checkpointing (bool, optional): Save best-loss checkpoints and any
        requested periodic/last checkpoints.
      checkpoint_dir (str | None, optional): Checkpoint directory. Defaults to
        ``"checkpoints"`` when checkpointing is enabled.
      checkpoint_every_n (int | None, optional): Save a periodic checkpoint every
        ``n`` updates when set.
      save_last (bool, optional): Save ``last.ckpt`` at the end of training when
        checkpointing is enabled.
      callbacks (list[Callback] | None, optional): Callback objects receiving
        fit-start, step-end, and fit-end hooks. Each object must implement
        ``on_fit_start(trainer)``, ``on_step_end(trainer, step, loss, metrics)``,
        and ``on_fit_end(trainer)``; use no-op methods for lifecycle points that
        do not need custom work. A callback can stop training after the current
        step by calling ``trainer.request_stop(reason)``.
      max_time (str | None, optional): Optional wall-clock budget in
        ``[DD:]HH:MM:SS`` form. Training stops after the current update once the
        budget is reached.
      grad_clip_val (float, optional): If positive, gradient norm clipping value.
      post_step_renorm (bool, optional): If ``True`` and ``gn`` exposes
        ``renorm_params()``, renormalize after optimizer steps according to
        ``gn.should_renorm_after_step()`` when that policy exists.
      compile_model (bool, optional): Try to wrap ``gn`` with
        :func:`torch.compile`; failures warn and continue uncompiled.
      seed (int | None, optional): Seed Python, NumPy, and torch RNGs before
        training.
      deterministic (bool | str | None, optional): If provided, passed as a bool
        to :func:`torch.use_deterministic_algorithms`.
      verbose (bool, optional): Print a model summary and show a tqdm progress
        bar.

    Returns:
      tuple: ``(trainer, best_ckpt_path)``. ``trainer.callback_metrics`` holds
        the last recorded metrics, including ``"loss"``. ``best_ckpt_path`` is
        the best-loss checkpoint path, or ``None`` when checkpointing is
        disabled.

    Raises:
      TypeError: If ``loss_kwargs`` is not a mapping and not ``None``.
      ValueError: If ``loss_kwargs`` contains the reserved ``"step"`` key.

    Examples:
      Basic training with a tensor loss::

        import torch
        from gradnet import GradNet, fit

        gn = GradNet(num_nodes=5, budget=5.0)

        def loss_fn(gn):
            return -gn().sum()

        trainer, best_ckpt = fit(
            gn=gn,
            loss_fn=loss_fn,
            num_updates=100,
            optim_kwargs={"lr": 1e-2},
            verbose=False,
        )

      Returning extra metrics and using a callback::

        class Tracker:
            def on_fit_start(self, trainer):
                self.losses = []

            def on_step_end(self, trainer, step, loss, metrics):
                self.losses.append(loss)

            def on_fit_end(self, trainer):
                pass

        def loss_with_metrics(gn, target_sum):
            A = gn()
            loss = (A.sum() - target_sum).pow(2)
            return loss, {"edge_sum": A.sum()}

        tracker = Tracker()
        trainer, _ = fit(
            gn=gn,
            loss_fn=loss_with_metrics,
            loss_kwargs={"target_sum": torch.tensor(4.0)},
            num_updates=50,
            callbacks=[tracker],
            logger=True,
        )

      Using the current optimization step inside a loss::

        def scheduled_loss(gn, step):
            warmup = min(1.0, (step + 1) / 10)
            return -warmup * gn().sum()

        fit(gn=gn, loss_fn=scheduled_loss, num_updates=100)
    """
    if seed is not None:
        _seed_everything(seed)
    if deterministic is not None:
        torch.use_deterministic_algorithms(bool(deterministic))

    if loss_kwargs is not None and not isinstance(loss_kwargs, Mapping):
        raise TypeError(
            "`loss_kwargs` must be a Mapping of keyword arguments (or None)."
        )
    if loss_kwargs is not None and _RESERVED_STEP_KWARG in loss_kwargs:
        raise ValueError(
            'The loss_kwargs parameter name "step" is reserved for the current '
            'optimization step passed by fit(). Remove "step" from '
            "`loss_kwargs`; define a `step` parameter on `loss_fn` to receive "
            "it automatically."
        )

    callbacks = list(callbacks) if callbacks else []
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
    pass_step_to_loss = _loss_fn_accepts_keyword_step(loss_fn)

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

        if pass_step_to_loss:
            out = loss_fn(gn, step=step, **loss_kwargs)
        else:
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

        if trainer.should_stop:
            break
        if deadline and time.monotonic() >= deadline:
            break

    if ckpt:
        ckpt.on_fit_end(trainer.current_epoch, gn, cfg)
    for cb in cbs:
        cb.on_fit_end(trainer)
    if tr_logger:
        tr_logger.finalize()

    return trainer, (ckpt.best_model_path if ckpt else None)
