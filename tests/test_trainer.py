import os
import types
import pathlib

import pytest
import sys
import os

# Ensure `src/` is on sys.path for local imports when using a src layout
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir, 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from gradnet.trainer import (
    _OneItem,
    GradNetLightning,
    _EpochTQDM,
    fit,
)


class DummyNet(nn.Module):
    """Simple module to exercise training and renorm behavior."""
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))
        self.renorm_calls = 0
        self.compiled_set = False

    def forward(self):
        # returns a tiny tensor to keep things extremely lightweight
        return self.w * torch.ones(1)

    def renorm_params(self):
        self.renorm_calls += 1


def simple_loss(model: nn.Module, scale: float = 1.0):
    # scalar differentiable loss using the model output
    out = model()
    loss = (out.pow(2).sum()) * float(scale)
    # also return a static metric for logging coverage
    return loss, {"metric": 1.0}


def test_oneitem_dataset_contract():
    ds = _OneItem()
    assert len(ds) == 1
    assert isinstance(ds[0], dict)
    assert ds[0] == {}


def test_gradnetlightning_forward_calls_model():
    net = DummyNet()
    module = GradNetLightning(
        gn=net,
        loss_fn=simple_loss,
        loss_kwargs={"scale": 1.0},
        optim_cls=torch.optim.SGD,
        optim_kwargs={"lr": 0.1},
    )
    y = module.forward()
    assert torch.is_tensor(y)
    assert y.shape == torch.Size([1])


def test_configure_optimizers_without_scheduler():
    net = DummyNet()
    module = GradNetLightning(
        gn=net,
        loss_fn=simple_loss,
        loss_kwargs={"scale": 1.0},
        optim_cls=torch.optim.Adam,
        optim_kwargs={"lr": 1e-2},
    )
    opt = module.configure_optimizers()
    assert isinstance(opt, torch.optim.Optimizer)


def test_configure_optimizers_with_scheduler():
    net = DummyNet()
    module = GradNetLightning(
        gn=net,
        loss_fn=simple_loss,
        loss_kwargs={"scale": 1.0},
        optim_cls=torch.optim.SGD,
        optim_kwargs={"lr": 0.1},
        sched_cls=torch.optim.lr_scheduler.StepLR,
        sched_kwargs={"step_size": 1, "gamma": 0.9},
    )
    cfg = module.configure_optimizers()
    assert isinstance(cfg, dict)
    assert "optimizer" in cfg and "lr_scheduler" in cfg


def test_fit_runs_and_calls_renorm(tmp_path: pathlib.Path):
    net = DummyNet()
    num_updates = 3
    trainer, best_ckpt = fit(
        gn=net,
        loss_fn=simple_loss,
        loss_kwargs={"scale": 1.0},
        num_updates=num_updates,
        optim_cls=torch.optim.SGD,
        optim_kwargs={"lr": 0.05},
        enable_checkpointing=False,  # keep filesystem clean for this one
        logger=False,
        accelerator="cpu",
        grad_clip_val=0.0,
        post_step_renorm=True,
    )
    assert isinstance(trainer, pl.Trainer)
    assert best_ckpt is None
    # renorm called once per update
    assert net.renorm_calls == num_updates


def test_checkpointing_every_n(tmp_path: pathlib.Path):
    net = DummyNet()
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer, best = fit(
        gn=net,
        loss_fn=simple_loss,
        loss_kwargs={"scale": 1.0},
        num_updates=2,
        optim_cls=torch.optim.Adam,
        optim_kwargs={"lr": 1e-2},
        enable_checkpointing=True,
        checkpoint_dir=str(ckpt_dir),
        checkpoint_every_n=2,
        save_last=True,
        logger=False,
        accelerator="cpu",
    )
    module = trainer.lightning_module
    assert module.monitor_key == "loss"
    # best checkpoint path should be a file path when checkpointing enabled
    assert isinstance(best, str) and len(best) > 0
    assert os.path.exists(best)
    ckpt_cb = next(c for c in trainer.callbacks if isinstance(c, ModelCheckpoint))
    assert ckpt_cb.every_n_epochs == 2
    assert ckpt_cb.monitor == "loss"
    assert ckpt_cb.mode == "min"
    assert ckpt_cb.save_top_k == 1
    assert ckpt_cb.save_last is True


def test_checkpointing_defaults_best_only(tmp_path: pathlib.Path):
    net = DummyNet()
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer, best = fit(
        gn=net,
        loss_fn=simple_loss,
        loss_kwargs={"scale": 1.0},
        num_updates=2,
        optim_cls=torch.optim.Adam,
        optim_kwargs={"lr": 1e-2},
        enable_checkpointing=True,
        checkpoint_dir=str(ckpt_dir),
        logger=False,
        accelerator="cpu",
    )
    ckpt_cb = next(c for c in trainer.callbacks if isinstance(c, ModelCheckpoint))
    assert ckpt_cb.every_n_epochs is None
    assert ckpt_cb.save_last is False
    assert isinstance(best, str) and os.path.exists(best)


def test_checkpointing_every_n_validation(tmp_path: pathlib.Path):
    net = DummyNet()
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError):
        fit(
            gn=net,
            loss_fn=simple_loss,
            loss_kwargs={"scale": 1.0},
            num_updates=1,
            optim_cls=torch.optim.SGD,
            optim_kwargs={"lr": 0.05},
            enable_checkpointing=True,
            checkpoint_dir=str(ckpt_dir),
            checkpoint_every_n=0,
            logger=False,
            accelerator="cpu",
        )


def test_callbacks_are_appended(tmp_path: pathlib.Path):
    net = DummyNet()
    class MyCB(pl.Callback):
        pass
    cb = MyCB()
    trainer, _ = fit(
        gn=net,
        loss_fn=simple_loss,
        loss_kwargs={"scale": 1.0},
        num_updates=1,
        optim_cls=torch.optim.SGD,
        optim_kwargs={"lr": 0.05},
        enable_checkpointing=False,
        callbacks=[cb],
        logger=False,
        accelerator="cpu",
    )
    # our callback present
    assert any(isinstance(c, MyCB) for c in trainer.callbacks)
    # progress bar callback appended
    assert any(isinstance(c, _EpochTQDM) for c in trainer.callbacks)


def test_seed_and_deterministic_called(monkeypatch, tmp_path: pathlib.Path):
    net = DummyNet()
    called = {"seed": None, "det": None}

    def fake_seed_everything(seed, workers=False):
        called["seed"] = (seed, workers)
        return {"seed": seed}

    def fake_use_deterministic(flag):
        called["det"] = flag

    monkeypatch.setattr(pl, "seed_everything", fake_seed_everything)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", fake_use_deterministic)

    fit(
        gn=net,
        loss_fn=simple_loss,
        loss_kwargs={"scale": 1.0},
        num_updates=1,
        optim_cls=torch.optim.SGD,
        optim_kwargs={"lr": 0.05},
        seed=123,
        deterministic=True,
        enable_checkpointing=False,
        logger=False,
        accelerator="cpu",
    )

    assert called["seed"] == (123, True)
    assert called["det"] is True


def test_compile_model_flag(monkeypatch, tmp_path: pathlib.Path):
    net = DummyNet()

    def fake_compile(model):
        # mark on original instance so we can assert later
        if hasattr(model, "compiled_set"):
            model.compiled_set = True
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)

    trainer, _ = fit(
        gn=net,
        loss_fn=simple_loss,
        loss_kwargs={"scale": 1.0},
        num_updates=1,
        optim_cls=torch.optim.SGD,
        optim_kwargs={"lr": 0.05},
        compile_model=True,
        enable_checkpointing=False,
        logger=False,
        accelerator="cpu",
    )

    assert net.compiled_set is True
    assert trainer.lightning_module.compile_model is True


def test_loss_kwargs_type_error():
    net = DummyNet()
    with pytest.raises(TypeError):
        fit(
            gn=net,
            loss_fn=simple_loss,
            loss_kwargs=[1, 2, 3],  # not a Mapping
            num_updates=1,
            optim_cls=torch.optim.SGD,
            optim_kwargs={"lr": 0.05},
            enable_checkpointing=False,
            logger=False,
            accelerator="cpu",
        )


def test_grad_clip_val_runs():
    net = DummyNet()
    # ensure training completes with gradient clipping enabled
    fit(
        gn=net,
        loss_fn=simple_loss,
        loss_kwargs={"scale": 1.0},
        num_updates=1,
        optim_cls=torch.optim.SGD,
        optim_kwargs={"lr": 0.05},
        grad_clip_val=0.1,
        enable_checkpointing=False,
        logger=False,
        accelerator="cpu",
    )


def test_no_renorm_when_disabled():
    net = DummyNet()
    fit(
        gn=net,
        loss_fn=simple_loss,
        loss_kwargs={"scale": 1.0},
        num_updates=2,
        optim_cls=torch.optim.SGD,
        optim_kwargs={"lr": 0.05},
        post_step_renorm=False,
        enable_checkpointing=False,
        logger=False,
        accelerator="cpu",
    )
    assert net.renorm_calls == 0


def test_precision_string_runs_cpu():
    net = DummyNet()
    fit(
        gn=net,
        loss_fn=simple_loss,
        loss_kwargs={"scale": 1.0},
        num_updates=1,
        optim_cls=torch.optim.SGD,
        optim_kwargs={"lr": 0.05},
        precision="32",
        enable_checkpointing=False,
        logger=False,
        accelerator="cpu",
    )
