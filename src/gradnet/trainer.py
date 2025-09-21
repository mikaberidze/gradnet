"""Utilities to train a :class:`gradnet.GradNet` with PyTorch Lightning.

This module provides a thin Lightning wrapper and a convenience function
(:func:`fit`) to optimize a ``GradNet`` for a fixed number of
updates.
"""
from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple, Union, Mapping, Any, Protocol
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase
from pytorch_lightning.callbacks import Callback
from tqdm.auto import tqdm
from .utils import _to_like_struct
from .gradnet import GradNet
import warnings
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

    This module performs manual optimization: it calls ``loss_fn`` to obtain
    a scalar loss (and optional metrics), applies gradient clipping (if
    configured), steps the optimizer, optionally renormalizes the model
    parameters, and logs metrics under ``monitor_key``.

    Parameters
    ----------
    :param gn: The model to optimize. Typically a ``GradNet`` instance;
        any ``nn.Module`` is accepted. If it defines ``renorm_params()``,
        that method is called after each optimizer step when
        ``post_step_renorm=True``.
    :type gn: torch.nn.Module
    :param loss_fn: Callable evaluated each step as
        ``loss_fn(gn, **loss_kwargs)``. Must return a scalar loss
        tensor, or ``(loss, metrics_dict)``.
    :type loss_fn: LossFn
    :param loss_kwargs: Keyword arguments forwarded to ``loss_fn`` on every
        step. Mapped through ``utils._to_like_struct`` upstream to match the
        model's device/dtype.
    :type loss_kwargs: Mapping[str, Any] | None
    :param optim_cls: Optimizer class to construct over ``gn.parameters()``.
    :type optim_cls: type[torch.optim.Optimizer]
    :param optim_kwargs: Keyword arguments for ``optim_cls`` (e.g., ``{"lr": 1e-2}``).
    :type optim_kwargs: dict
    :param sched_cls: Optional LR scheduler class to wrap the optimizer.
    :type sched_cls: type | None
    :param sched_kwargs: Keyword arguments for ``sched_cls``.
    :type sched_kwargs: dict | None
    :param grad_clip_val: Gradient-norm clipping value; ``0.0`` disables clipping.
    :type grad_clip_val: float
    :param post_step_renorm: Call ``gn.renorm_params()`` after each step
        if available.
    :type post_step_renorm: bool
    :param monitor_key: Metric name under which the primary loss is logged.
    :type monitor_key: str
    :param compile_model: Attempt to wrap the model with ``torch.compile`` in
        ``setup``; continue uncompiled on failure.
    :type compile_model: bool
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
    logger: Optional[LightningLoggerBase] = True,
    enable_checkpointing: bool = True,
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
    """Optimize a GradNet object for a fixed number of updates using
    adapted PyTorch Lightning.

    Each update calls:
    ``loss_fn(gn, **loss_kwargs)`` to compute a scalar loss (and
    optional metrics). One epoch corresponds to one optimizer update, so
    ``num_updates`` equals the number of optimization steps.

    Parameters
    ----------
    :param gn: The network to optimize. Must be a ``GradNet`` instance.
    :type gn: gradnet.GradNet
    :param loss_fn: Callable invoked as ``loss_fn(gn, **loss_kwargs)``.
        Must return either a scalar loss tensor or a tuple
        ``(loss, metrics_dict)``, where ``metrics_dict`` maps metric names to
        numbers or tensors. The main loss is logged under ``monitor``.
    :type loss_fn: LossFn
    :param loss_kwargs: Keyword arguments forwarded to ``loss_fn``. If a mapping
        is provided, tensors/NumPy arrays inside are recursively moved/cast to
        match ``gn``'s device/dtype via ``utils._to_like_struct``.
    :type loss_kwargs: Mapping[str, Any] | None, optional
    :param num_updates: Number of optimizer steps to run (epochs == updates).
    :type num_updates: int
    :param optim_cls: Optimizer class constructed as
        ``optim_cls(gn.parameters(), **optim_kwargs)``. Typical choices are
        from ``torch.optim`` (e.g., ``Adam``, ``SGD``).
    :type optim_cls: type[torch.optim.Optimizer], optional
    :param optim_kwargs: Keyword arguments for ``optim_cls`` (e.g., ``{"lr": 1e-2}``).
        If ``None``, defaults to ``{"lr": 1e-2}``.
    :type optim_kwargs: dict | None, optional
    :param sched_cls: Optional LR scheduler class instantiated as
        ``sched_cls(optimizer, **sched_kwargs)``. Common choices are from
        ``torch.optim.lr_scheduler`` (e.g., ``StepLR``, ``ExponentialLR``).
        Note: Schedulers that require a monitored metric (e.g.,
        ``ReduceLROnPlateau``) are not configured with ``monitor`` here and may
        need customization.
    :type sched_cls: type | None, optional
    :param sched_kwargs: Keyword arguments for ``sched_cls``.
    :type sched_kwargs: dict | None, optional
    :param precision: Forwarded to ``pl.Trainer(precision=...)``. Accepts an
        ``int`` (e.g., ``32``) or a string such as ``"32-true"``,
        ``"16-mixed"``, or ``"bf16-mixed"`` when supported by your setup.
    :type precision: str | int, optional
    :param accelerator: Forwarded to ``pl.Trainer(accelerator=...)`` (e.g.,
        ``"auto"``, ``"cpu"``, ``"gpu"``, ``"mps"``), depending on your
        Lightning version and hardware.
    :type accelerator: str, optional
    :param logger: Forwarded to ``pl.Trainer(logger=...)``. Can be ``True``/``False``
        or a Lightning logger instance. For TensorBoard, pass
        ``pytorch_lightning.loggers.TensorBoardLogger(...)``.
    :type logger: LightningLoggerBase | bool | None, optional
    :param enable_checkpointing: Whether to enable checkpointing. When ``True``,
        a ``ModelCheckpoint`` callback is added using the options below.
    :type enable_checkpointing: bool, optional
    :param checkpoint_dir: Directory for checkpoints (``ModelCheckpoint.dirpath``).
    :type checkpoint_dir: str | None, optional
    :param monitor: Name of the logged metric to monitor for checkpointing and
        main loss logging. The LightningModule logs the loss under this key.
    :type monitor: str, optional
    :param mode: Whether to minimize or maximize ``monitor`` when selecting the
        best checkpoints (``"min"`` for losses, ``"max"`` for scores).
    :type mode: str, optional
    :param save_top_k: How many best checkpoints to keep.
    :type save_top_k: int, optional
    :param save_last: Whether to always save the last checkpoint.
    :type save_last: bool, optional
    :param callbacks: Additional ``pytorch_lightning.Callback`` instances to use.
        A progress callback is always appended, and a ``ModelCheckpoint`` is
        appended when ``enable_checkpointing=True``.
    :type callbacks: list[pl.Callback] | None, optional
    :param max_time: Training time limit forwarded to ``pl.Trainer(max_time=...)``.
        Commonly a string like ``"DD:HH:MM:SS"`` (e.g., ``"00:01:00:00"`` for 1 hour),
        but confirm accepted formats with your Lightning version.
    :type max_time: str | None, optional
    :param grad_clip_val: If ``> 0``, apply gradient-norm clipping before the
        optimizer step.
    :type grad_clip_val: float, optional
    :param post_step_renorm: After each optimizer step, call
        ``gn.renorm_params()`` if available.
    :type post_step_renorm: bool, optional
    :param compile_model: If ``True``, attempts ``torch.compile(gn)`` during
        setup; on failure, logs a warning and continues uncompiled.
    :type compile_model: bool, optional
    :param seed: If provided, calls ``pl.seed_everything(seed, workers=True)``.
    :type seed: int | None, optional
    :param deterministic: If provided, calls
        ``torch.use_deterministic_algorithms(bool(deterministic))``.
    :type deterministic: bool | str | None, optional
    :raises TypeError: If ``loss_kwargs`` is not a mapping (and not ``None``).
    :return: Tuple ``(trainer, best_ckpt_path)`` where ``trainer`` is the
        instantiated ``pl.Trainer`` and ``best_ckpt_path`` is the path to the
        best checkpoint (or ``None`` when checkpointing is disabled).
    :rtype: tuple[pl.Trainer, str | None]

    .. tip::
       To use TensorBoard logging, pass a TensorBoard logger instance, e.g.::

           from pytorch_lightning.loggers import TensorBoardLogger
           logger = TensorBoardLogger(save_dir="logs", name="exp")
           fit(gn=model, loss_fn=loss, num_updates=100, logger=logger)

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

    trainer = pl.Trainer(
        max_epochs=int(num_updates),
        accelerator=accelerator,
        precision=precision,
        logger=(logger if verbose else False),
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


#TODO figure out logging (plot loss and metrics)
