import os
import sys
import time
import warnings
from pathlib import Path

import pytest
import torch

THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir, "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from gradnet import GradNet
from gradnet.trainer import (
    fit,
    CSVLogger,
    GradNetTrainer,
    _looks_like_dtype_or_device_error,
    _parse_max_time,
    _resolve_device,
)


# --------------------------------------------------------------------- fixtures

@pytest.fixture
def gn():
    return GradNet(num_nodes=5, budget=5.0)


def neg_sum_loss(g):
    return -g().sum()


def _count_mismatch_warnings(caught):
    return sum("loss tensor dtype/device mismatch" in str(w.message) for w in caught)


# --------------------------------------------------------------------- 1. one-step sanity

def test_loss_decreases_over_5_updates(gn):
    initial = float(neg_sum_loss(gn).item())
    trainer, _ = fit(gn=gn, loss_fn=neg_sum_loss, num_updates=5, verbose=False)
    assert isinstance(trainer, GradNetTrainer)
    assert trainer.callback_metrics["loss"] <= initial


# --------------------------------------------------------------------- 2. loss return shapes

def test_loss_returns_metrics_tuple(gn):
    def loss_with_metrics(g):
        return -g().sum(), {"extra": 42.0, "abs_sum": g().abs().sum()}

    trainer, _ = fit(gn=gn, loss_fn=loss_with_metrics, num_updates=2, verbose=False)
    metrics = trainer.callback_metrics
    assert "loss" in metrics
    assert metrics["extra"] == 42.0
    assert "abs_sum" in metrics


# --------------------------------------------------------------------- 3. gradient clipping

def test_grad_clipping_calls_torch_clip(monkeypatch, gn):
    calls = []
    real = torch.nn.utils.clip_grad_norm_

    def spy(*args, **kwargs):
        calls.append(kwargs.get("max_norm", args[1] if len(args) > 1 else None))
        return real(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)

    fit(gn=gn, loss_fn=neg_sum_loss, num_updates=3, grad_clip_val=1.0, verbose=False)
    assert len(calls) == 3

    calls.clear()
    fit(gn=gn, loss_fn=neg_sum_loss, num_updates=3, grad_clip_val=0.0, verbose=False)
    assert calls == []


# --------------------------------------------------------------------- 4. optimizer / scheduler

def test_custom_optimizer_and_scheduler_step_once_per_update(gn):
    sched_steps = [0]

    class CountingSched:
        def __init__(self, opt, **_): pass
        def step(self): sched_steps[0] += 1

    fit(
        gn=gn, loss_fn=neg_sum_loss, num_updates=4,
        optim_cls=torch.optim.SGD, optim_kwargs={"lr": 0.1},
        sched_cls=CountingSched,
        verbose=False,
    )
    assert sched_steps[0] == 4


# --------------------------------------------------------------------- 5. renorm policy

def test_renorm_skipped_when_policy_returns_false(monkeypatch, gn):
    calls = []
    real_renorm = gn.renorm_params
    monkeypatch.setattr(gn, "renorm_params", lambda: (calls.append(1), real_renorm())[-1])
    monkeypatch.setattr(gn, "should_renorm_after_step", lambda: False)
    fit(gn=gn, loss_fn=neg_sum_loss, num_updates=3, verbose=False)
    assert calls == []


# --------------------------------------------------------------------- 6. CSV logger

def test_csv_logger_is_readable_by_load_scalars(gn, tmp_path):
    from gradnet.utils import load_scalars

    logger = CSVLogger(str(tmp_path))
    fit(gn=gn, loss_fn=neg_sum_loss, num_updates=4, logger=logger, verbose=False)

    steps, series = load_scalars(tmp_path)
    assert steps == [0, 1, 2, 3]
    assert "loss" in series
    assert len(series["loss"]) == 4


# --------------------------------------------------------------------- 7. checkpointing

def test_best_checkpoint_loadable_via_from_checkpoint(gn, tmp_path):
    _, best = fit(
        gn=gn, loss_fn=neg_sum_loss, num_updates=5,
        enable_checkpointing=True, checkpoint_dir=str(tmp_path),
        verbose=False,
    )
    assert best and Path(best).exists()
    reloaded = GradNet.from_checkpoint(best, map_location="cpu")
    assert reloaded.num_nodes == gn.num_nodes


def test_periodic_checkpoints_match_animate_glob(gn, tmp_path):
    fit(
        gn=gn, loss_fn=neg_sum_loss, num_updates=10,
        enable_checkpointing=True, checkpoint_dir=str(tmp_path),
        checkpoint_every_n=2,
        verbose=False,
    )
    periodic = sorted(tmp_path.glob("gn-periodic-*.ckpt"))
    assert len(periodic) == 5  # steps 1, 3, 5, 7, 9 (0-indexed; step+1 % 2 == 0)


def test_save_last_creates_last_ckpt(gn, tmp_path):
    fit(
        gn=gn, loss_fn=neg_sum_loss, num_updates=3,
        enable_checkpointing=True, checkpoint_dir=str(tmp_path),
        save_last=True,
        verbose=False,
    )
    assert (tmp_path / "last.ckpt").exists()


# --------------------------------------------------------------------- 8. max_time

def test_parse_max_time_grammar():
    assert _parse_max_time("00:00:01") == 1
    assert _parse_max_time("00:01:30") == 90
    assert _parse_max_time("01:00:00") == 3600
    assert _parse_max_time("01:00:00:00") == 86400


def test_max_time_stops_loop_early(monkeypatch, gn):
    # Make time.monotonic look like a long time has passed after the first step.
    times = iter([0.0, 0.0, 999.0, 999.0, 999.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(times))

    trainer, _ = fit(
        gn=gn, loss_fn=neg_sum_loss, num_updates=100, max_time="00:00:01",
        verbose=False,
    )
    assert trainer.current_epoch < 99


# --------------------------------------------------------------------- 9. seed reproducibility

def test_seed_makes_runs_identical():
    # fit()'s seed only affects post-construction RNG, so seed before building the model too.
    def run():
        torch.manual_seed(42)
        g = GradNet(num_nodes=5, budget=5.0)
        trainer, _ = fit(gn=g, loss_fn=neg_sum_loss, num_updates=5, seed=42, verbose=False)
        return trainer.callback_metrics["loss"]
    assert run() == run()


# --------------------------------------------------------------------- 10. device resolution

def test_resolve_device():
    assert isinstance(_resolve_device("auto"), torch.device)
    assert _resolve_device("cpu") == torch.device("cpu")
    assert _resolve_device(torch.device("cpu")) == torch.device("cpu")


def test_resolve_device_rejects_invalid_alias():
    # "gpu" is not a torch device name → torch.device("gpu") raises.
    with pytest.raises((RuntimeError, ValueError)):
        _resolve_device("gpu")


# --------------------------------------------------------------------- 11. callback lifecycle

def test_callback_lifecycle_order_and_count(gn):
    events = []

    class Tracker:
        def on_fit_start(self, t): events.append("start")
        def on_step_end(self, t, step, loss, metrics): events.append(("step", step))
        def on_fit_end(self, t): events.append("end")

    fit(gn=gn, loss_fn=neg_sum_loss, num_updates=3, callbacks=[Tracker()], verbose=False)
    assert events == ["start", ("step", 0), ("step", 1), ("step", 2), "end"]


def test_callback_can_request_stop(gn):
    events = []

    class StopAfterTwo:
        def on_fit_start(self, t):
            events.append("start")

        def on_step_end(self, t, step, loss, metrics):
            events.append(("step", step))
            if step == 1:
                t.request_stop("target reached")

        def on_fit_end(self, t):
            events.append(("end", t.current_epoch, t.stop_reason))

    trainer, _ = fit(
        gn=gn,
        loss_fn=neg_sum_loss,
        num_updates=10,
        callbacks=[StopAfterTwo()],
        verbose=False,
    )

    assert events == ["start", ("step", 0), ("step", 1), ("end", 1, "target reached")]
    assert trainer.should_stop is True
    assert trainer.stop_reason == "target reached"


# --------------------------------------------------------------------- 13. dtype safety net (loss return)

@pytest.mark.parametrize("model_dtype,loss_dtype", [
    (torch.float32, torch.float64),
    (torch.float64, torch.float32),
])
def test_dtype_safety_net_casts_and_warns_once(model_dtype, loss_dtype):
    # MPS rejects float64, so pin to CPU for these dtype-mixing scenarios.
    g = GradNet(num_nodes=5, budget=5.0).to(model_dtype)

    def loss_fn(gn):
        return (-gn().sum()).to(loss_dtype)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer, _ = fit(gn=g, loss_fn=loss_fn, num_updates=5, device="cpu", verbose=False)
    assert _count_mismatch_warnings(caught) == 1
    assert trainer.callback_metrics["loss"] < 0  # made progress


# --------------------------------------------------------------------- 14. device safety net

def test_device_safety_net():
    if not torch.cuda.is_available():
        pytest.skip("needs a second device (CUDA) to exercise device mismatch")
    g = GradNet(num_nodes=5, budget=5.0)  # on cpu

    def loss_fn(gn):
        return (-gn().sum()).to("cuda")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer, _ = fit(gn=g, loss_fn=loss_fn, num_updates=3, verbose=False)
    assert _count_mismatch_warnings(caught) == 1
    assert trainer.callback_metrics["loss"] < 0


# --------------------------------------------------------------------- 15. deep-graph mismatch at backward

def test_backward_dtype_error_emits_warning_then_reraises(gn):
    class _RaiseDtypeOnBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x

        @staticmethod
        def backward(ctx, grad_output):
            raise RuntimeError("expected dtype Float but got Double")

    def loss_fn(g):
        return _RaiseDtypeOnBackward.apply(-g().sum())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="expected dtype"):
            fit(gn=gn, loss_fn=loss_fn, num_updates=1, verbose=False)
    assert _count_mismatch_warnings(caught) == 1


def test_looks_like_dtype_or_device_error_matches_common_messages():
    assert _looks_like_dtype_or_device_error(RuntimeError("expected dtype Float but got Double"))
    assert _looks_like_dtype_or_device_error(RuntimeError("Expected all tensors to be on the same device"))
    assert _looks_like_dtype_or_device_error(RuntimeError("Tensors must have the same dtype"))
    assert not _looks_like_dtype_or_device_error(RuntimeError("Index out of bounds"))


# --------------------------------------------------------------------- 16. mismatch warning dedup

def test_dtype_mismatch_warning_emitted_only_once_across_many_steps():
    g = GradNet(num_nodes=5, budget=5.0)  # float32

    def loss_fn(gn):
        return (-gn().sum()).to(torch.float64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit(gn=g, loss_fn=loss_fn, num_updates=10, device="cpu", verbose=False)
    assert _count_mismatch_warnings(caught) == 1


# --------------------------------------------------------------------- 17. native dtype routing

def test_native_dtype_routing_no_warning():
    g = GradNet(num_nodes=5, budget=5.0).double()  # MPS lacks float64, pin to CPU

    def loss_fn(gn):
        return -gn().sum()  # float64 throughout, matches gn

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer, _ = fit(gn=g, loss_fn=loss_fn, num_updates=3, device="cpu", verbose=False)
    assert _count_mismatch_warnings(caught) == 0
    assert trainer.callback_metrics["loss"] < 0
