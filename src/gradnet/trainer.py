"""Utilities to train a :class:`gradnet.GradNet` with PyTorch Lightning.

This module provides a thin Lightning wrapper and a convenience function
(:func:`fit`) to optimize a ``GradNet`` for a fixed number of
updates.
"""
from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple, Union, Mapping, Any, Protocol
import logging
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase
from pytorch_lightning.callbacks import Callback
try:  # prefer notebook progress bar when the stack supports it
    from tqdm import TqdmWarning  # type: ignore[attr-defined]
except (ImportError, AttributeError):
    TqdmWarning = Warning  # fallback when tqdm lacks TqdmWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=TqdmWarning)
    try:
        from tqdm.auto import tqdm
    except Exception:
        from tqdm import tqdm  # noqa: F401  # CLI fallback without warnings
from .utils import _to_like_struct
from .gradnet import GradNet
try:# PL >= 1.6-ish
    from pytorch_lightning.utilities.warnings import PossibleUserWarning
except Exception: # Fallback for older PL where it's just a UserWarning
    PossibleUserWarning = UserWarning
warnings.filterwarnings(  # silence few data-loader workers worning. We don't need data-loader workers
    "ignore",
    message=r"The 'train_dataloader' does not have many workers.*",
    category=PossibleUserWarning,
)
warnings.filterwarnings(  # silence GPU not used warning
    "ignore",
    message=r"GPU available but not used.*",
    category=PossibleUserWarning,
)

class LossFn(Protocol):
    """Protocol for loss functions used with :func:`fit`.

    Implementations must accept a :class:`gradnet.GradNet` and may accept
    arbitrary keyword arguments. They should return either a scalar loss
    tensor, or a tuple ``(loss, metrics_dict)`` where ``metrics_dict`` maps
    metric names to floats/ints/tensors.
    """
    def __call__(
        self,
        model: GradNet,
        **loss_kwargs: Any,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, Dict[str, Union[float, int, torch.Tensor]]],
    ]: ...


class _OneItem(Dataset):
    """A trivial dataset that always yields a single empty batch.

    Used to drive the Lightning training loop with one update per epoch
    without relying on external data.
    """
    def __len__(self): return 1
    def __getitem__(self, idx): return {}


class GradNetLightning(pl.LightningModule):
    """LightningModule wrapper around a ``GradNet`` and a user loss.

    This module performs manual optimisation: it evaluates ``loss_fn`` to obtain
    a scalar loss (and optional metrics), applies gradient clipping (if
    configured), steps the optimizer, optionally renormalises the model
    parameters, and logs metrics under ``monitor_key``.

    Parameters
    ----------
    gn : torch.nn.Module
        Model to optimise. Typically a :class:`gradnet.GradNet`; any ``nn.Module``
        is accepted. If the module exposes ``renorm_params()``, that method is
        invoked after each optimizer step when ``post_step_renorm`` is ``True``.
    loss_fn : LossFn
        Callable evaluated on every optimisation step as
        ``loss_fn(gn, **loss_kwargs)``. Must return either a scalar loss tensor or
        a ``(loss, metrics_dict)`` tuple.
    loss_kwargs : Mapping[str, Any] | None, optional
        Extra keyword arguments forwarded to ``loss_fn`` and converted via
        :func:`gradnet.utils._to_like_struct` so tensors follow ``gn``'s device
        and dtype.
    optim_cls : type[torch.optim.Optimizer]
        Optimiser class instantiated over ``gn.parameters()``.
    optim_kwargs : dict, optional
        Arguments passed to ``optim_cls`` (e.g., ``{"lr": 1e-2}``).
    sched_cls : type | None, optional
        Optional LR scheduler class applied on top of the optimiser.
    sched_kwargs : dict | None, optional
        Keyword arguments for ``sched_cls``.
    grad_clip_val : float, optional
        Gradient-norm clipping threshold. ``0.0`` disables clipping.
    post_step_renorm : bool, optional
        Call ``gn.renorm_params()`` after each optimiser step when available.
    monitor_key : str, optional
        Metric name under which the primary loss is logged.
    compile_model : bool, optional
        Attempt to wrap the model with :func:`torch.compile` during ``setup``;
        fall back silently when compilation fails.
    """
    def __init__(
        self,
        *,
        gn: nn.Module,
        loss_fn: LossFn,
        loss_kwargs: Mapping[str, Any] | None = None,   # kwargs for the loss function
        optim_cls: type[torch.optim.Optimizer],
        optim_kwargs: dict,
        sched_cls: Optional[type] = None,
        sched_kwargs: Optional[dict] = None,
        grad_clip_val: float = 0.0,
        post_step_renorm: bool = True,
        monitor_key: str = "loss",
        compile_model: bool = False,
    ):
        super().__init__()
        gradnet_config = None
        if isinstance(gn, GradNet) and hasattr(gn, "export_config"):
            gradnet_config = gn.export_config()
        self.save_hyperparameters({"gradnet_config": gradnet_config}, logger=False)
        self.gn = gn
        self.loss_fn = loss_fn
        self.loss_kwargs = loss_kwargs
        self.optim_cls = optim_cls
        self.optim_kwargs = optim_kwargs
        self.sched_cls = sched_cls
        self.sched_kwargs = sched_kwargs or {}
        self.grad_clip_val = float(grad_clip_val)
        self.post_step_renorm = bool(post_step_renorm)
        self.monitor_key = monitor_key
        self.compile_model = bool(compile_model)

        self.automatic_optimization = False  # manual optimization

    def setup(self, stage: Optional[str] = None):
        """Optional model compilation with ``torch.compile``.

        :param stage: Lightning training stage (unused).
        :type stage: str | None
        """
        if self.compile_model:
            try:
                self.gn = torch.compile(self.gn)  # type: ignore[attr-defined]
            except Exception as e:
                pl.utilities.rank_zero.rank_zero_warn(f"torch.compile failed; continuing uncompiled. Error: {e}")

    def forward(self):
        """Return the model output (full adjacency in ``GradNet``).

        :return: Forward pass of ``gn``.
        :rtype: torch.Tensor
        """
        return self.gn()

    def training_step(self, batch, batch_idx):
        """One optimization step driven by the user loss.

        Computes ``loss_fn(gn, **loss_kwargs)``, backpropagates, clips
        gradients if configured, takes an optimizer step, optionally calls
        ``gn.renorm_params()``, and logs loss/metrics.

        :param batch: Dummy batch (unused).
        :param batch_idx: Training step index.
        :return: Detached loss tensor.
        :rtype: torch.Tensor
        """
        # compute loss (+ optional metrics)
        out = self.loss_fn(self.gn, **self.loss_kwargs)
        loss, metrics = (out, {}) if isinstance(out, torch.Tensor) else out

        opt = self.optimizers()
        self.manual_backward(loss)

        if self.grad_clip_val > 0:
            self.clip_gradients(opt, gradient_clip_val=self.grad_clip_val, gradient_clip_algorithm="norm")

        opt.step()
        opt.zero_grad(set_to_none=True)

        # required: renormalize after each update
        if self.post_step_renorm and hasattr(self.gn, "renorm_params"):
            self.gn.renorm_params()

        self.log(self.monitor_key, loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=1)
        for k, v in metrics.items():
            v = v if isinstance(v, torch.Tensor) else torch.tensor(float(v), device=loss.device)
            self.log(k, v, prog_bar=False, on_epoch=True, on_step=False, sync_dist=True, batch_size=1)

        return loss.detach()

    def configure_optimizers(self):
        """Construct optimizer (and optional LR scheduler) for Lightning.

        :return: Optimizer or an optimizer+lr_scheduler dict per Lightning API.
        :rtype: torch.optim.Optimizer | dict
        """
        opt = self.optim_cls(self.gn.parameters(), **self.optim_kwargs)
        if self.sched_cls is None:
            return opt
        sched = self.sched_cls(opt, **self.sched_kwargs)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch", "frequency": 1, "name": "lr"},
        }


class _EpochTQDM(Callback):
    """Minimal epoch-wise TQDM progress bar callback.

    Shows total updates, and displays numeric metrics
    collected in ``trainer.callback_metrics``.
    """
    def on_fit_start(self, trainer, *_):
        self.bar = tqdm(total=trainer.max_epochs, desc="Updates", dynamic_ncols=True)
    def on_train_epoch_end(self, trainer, *_):
        self.bar.set_postfix({k: v.item() if hasattr(v, "item") else v
                              for k, v in trainer.callback_metrics.items()
                              if isinstance(v, (int, float)) or hasattr(v, "item")})
        self.bar.update(1)
    def on_fit_end(self, *_):
        self.bar.close()


def _resolve_logger(
    logger: LightningLoggerBase | bool | None,
    *,
    verbose: bool,
) -> LightningLoggerBase | bool:
    """Return a Lightning logger instance or ``False``.

    When ``logger`` is ``True`` this attempts to build a ``TensorBoardLogger``
    and falls back to ``CSVLogger`` if TensorBoard is unavailable. Logging is
    disabled when ``verbose`` is ``False`` to mirror the previous behaviour.
    """
    if not verbose:
        return False

    if isinstance(logger, LightningLoggerBase):
        return logger

    if not logger:  # covers False and None
        return False

    try:
        from pytorch_lightning.loggers import TensorBoardLogger

        return TensorBoardLogger(save_dir="lightning_logs", name="gradnet")
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        warnings.warn(
            "TensorBoard logger unavailable; using CSVLogger instead. "
            "Install the `tensorboard` package to re-enable TensorBoard logging. "
            f"Original error: {exc}",
            RuntimeWarning,
        )
        from pytorch_lightning.loggers import CSVLogger

        return CSVLogger(save_dir="lightning_logs", name="gradnet")


def fit(
    *,
    gn: GradNet,
    loss_fn: LossFn,
    loss_kwargs: Mapping[str, Any] | None = None,   # kwargs for the loss function
    num_updates: int,
    optim_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
    optim_kwargs: Optional[dict] = None,
    sched_cls: Optional[type] = None,
    sched_kwargs: Optional[dict] = None,
    # runtime
    precision: Union[str, int] = "32-true",
    accelerator: str = "auto",
    # logging/ckpt
    logger: LightningLoggerBase | bool | None = False,
    enable_checkpointing: bool = False,
    checkpoint_dir: Optional[str] = None,
    monitor: str = "loss",
    mode: str = "min",
    save_top_k: int = 1,
    save_last: bool = True,
    callbacks: Optional[list[pl.Callback]] = None,
    max_time: Optional[str] = None,
    # extras
    grad_clip_val: float = 0.0,
    post_step_renorm: bool = True,
    compile_model: bool = False,
    seed: Optional[int] = None,
    deterministic: Optional[Union[bool, str]] = None,
    verbose: bool = True,
):
    """Optimise a :class:`gradnet.GradNet` for a fixed number of updates.

    One trainer epoch corresponds to a single optimiser step, so
    ``num_updates`` equals the number of optimisation steps executed. Each step
    evaluates ``loss_fn(gn, **loss_kwargs)`` and drives manual optimisation via
    :class:`GradNetLightning`.

    Parameters
    ----------
    gn : GradNet
        Network to optimise.
    loss_fn : LossFn
        Callable invoked as ``loss_fn(gn, **loss_kwargs)`` and returning either a
        scalar loss tensor or ``(loss, metrics_dict)``.
    loss_kwargs : Mapping[str, Any] | None, optional
        Extra keyword arguments forwarded to ``loss_fn``. When provided, tensors
        and arrays are coerced to ``gn``'s device/dtype via
        :func:`gradnet.utils._to_like_struct`.
    num_updates : int
        Number of optimisation steps to run.
    optim_cls : type[torch.optim.Optimizer], optional
        Optimiser class constructed as ``optim_cls(gn.parameters(), **optim_kwargs)``.
    optim_kwargs : dict | None, optional
        Keyword arguments for the optimiser. Defaults to ``{"lr": 1e-2}`` when
        ``None``.
    sched_cls : type | None, optional
        Optional learning-rate scheduler applied to the optimiser.
    sched_kwargs : dict | None, optional
        Keyword arguments for ``sched_cls``.
    precision : str | int, optional
        Forwarded to ``pl.Trainer(precision=...)`` (e.g., ``"32-true"``, ``16``).
    accelerator : str, optional
        Passed to ``pl.Trainer(accelerator=...)`` (``"auto"``, ``"cpu"``, ``"gpu"``, etc.).
    logger : LightningLoggerBase | bool | None, optional
        Logger configuration forwarded to ``pl.Trainer``. Use ``True`` for the
        default logger (falls back to ``CSVLogger`` when TensorBoard is
        unavailable), ``False`` to disable logging, or supply a Lightning logger
        instance.
    enable_checkpointing : bool, optional
        Enable the default ``ModelCheckpoint`` callback. When ``True`` the
        callback is appended automatically using the ``monitor``/``mode`` settings.
    checkpoint_dir : str | None, optional
        Directory used by ``ModelCheckpoint`` when checkpointing is enabled.
    monitor : str, optional
        Metric key to monitor for checkpoint selection and loss logging.
    mode : str, optional
        Whether to minimise (``"min"``) or maximise (``"max"``) ``monitor``.
    save_top_k : int, optional
        Number of best checkpoints to keep.
    save_last : bool, optional
        Whether to always save the final checkpoint.
    callbacks : list[pl.Callback] | None, optional
        Additional Lightning callbacks to register.
    max_time : str | None, optional
        Training time limit forwarded to ``pl.Trainer(max_time=...)``.
    grad_clip_val : float, optional
        Gradient-norm clipping threshold applied before optimiser steps.
    post_step_renorm : bool, optional
        Call ``gn.renorm_params()`` after each optimiser step when available.
    compile_model : bool, optional
        Attempt to wrap ``gn`` with :func:`torch.compile` during setup.
    seed : int | None, optional
        When provided, seeds PyTorch Lightning via ``pl.seed_everything``.
    deterministic : bool | str | None, optional
        If not ``None``, passed to ``torch.use_deterministic_algorithms``.
    verbose : bool, optional
        Show progress via :class:`tqdm.auto.tqdm`.

    Returns
    -------
    tuple[pl.Trainer, str | None]
        The configured trainer and the best checkpoint path (``None`` when
        checkpointing is disabled).

    Raises
    ------
    TypeError
        If ``loss_kwargs`` is neither ``None`` nor a mapping.

    Examples
    --------
    >>> from pytorch_lightning.loggers import TensorBoardLogger
    >>> logger = TensorBoardLogger(save_dir="logs", name="demo")
    >>> trainer, best_ckpt = fit(
    ...     gn=model,
    ...     loss_fn=loss,
    ...     num_updates=100,
    ...     logger=logger,
    ... )

    .. seealso::
       PyTorch Optimizers (``torch.optim``), PyTorch LR Schedulers
       (``torch.optim.lr_scheduler``), and PyTorch Lightning's Trainer and
       Callbacks documentation for accepted values of ``precision`` and
       ``accelerator`` and for available callback types.
    """
    if seed is not None:
        pl.seed_everything(seed, workers=True)
    if deterministic is not None:
        torch.use_deterministic_algorithms(bool(deterministic))

    # params must be kwargs if provided
    if loss_kwargs is None:
        loss_kwargs = {}
    elif isinstance(loss_kwargs, Mapping):
        loss_kwargs = _to_like_struct(loss_kwargs, gn)
    else:
        raise TypeError("`f_kwargs` must be a Mapping of keyword arguments (or None).")


    module = GradNetLightning(
        gn=gn,
        loss_fn=loss_fn,
        loss_kwargs=loss_kwargs,
        optim_cls=optim_cls,
        optim_kwargs=optim_kwargs or {"lr": 1e-2},
        sched_cls=sched_cls,
        sched_kwargs=sched_kwargs,
        grad_clip_val=grad_clip_val,
        post_step_renorm=post_step_renorm,
        monitor_key=monitor,
        compile_model=compile_model,
    )

    cb = list(callbacks or [])
    ckpt = None
    if enable_checkpointing:
        ckpt = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="gn-{epoch:05d}-{loss:.6f}",
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=save_last,
            auto_insert_metric_name=False,
        )
        cb.append(ckpt)

    # progress bar only when verbose
    if verbose:
        cb.append(_EpochTQDM())

    # Silence PL info logs if verbose is False
    prev_levels: dict[str, int] = {}
    if not verbose:
        for name in ("pytorch_lightning", "lightning"):
            lg = logging.getLogger(name)
            prev_levels[name] = lg.level
            lg.setLevel(logging.ERROR)

    trainer_logger = _resolve_logger(logger, verbose=verbose)

    trainer = pl.Trainer(
        max_epochs=int(num_updates),
        accelerator=accelerator,
        precision=precision,
        logger=trainer_logger,
        enable_checkpointing=enable_checkpointing,
        callbacks=cb,
        log_every_n_steps=1,
        max_time=max_time,
        enable_progress_bar=False,
        enable_model_summary=bool(verbose),
    )

    loader = DataLoader(_OneItem(), batch_size=1, shuffle=False, num_workers=0)
    trainer.fit(module, train_dataloaders=loader)

    # Restore previous PL logger levels if we changed them
    if not verbose:
        for name, lvl in prev_levels.items():
            logging.getLogger(name).setLevel(lvl)

    return trainer, (ckpt.best_model_path if (enable_checkpointing and ckpt is not None) else None)
