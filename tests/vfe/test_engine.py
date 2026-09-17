"""Training machinery: log averaging, LR schedule, clipping, random streams,
checkpoints and gradient accumulation."""

import io
import os

import pytest
import torch
from torch import nn

from vfe.apis.train import RngStreams, train_step
from vfe.engine import StepLrScheduler, clip_grads
from vfe.engine.checkpoint import resume_checkpoint, save_checkpoint
from vfe.engine.train_log import LogBuffer


def test_log_buffer_weighted_average_of_last_n():
    buf = LogBuffer()
    for value, count in ((1.0, 1), (3.0, 1), (6.0, 2)):
        buf.update({"loss": value}, count)
    buf.average(2)
    assert buf.output["loss"] == pytest.approx((3.0 + 2 * 6.0) / 3)
    buf.average()
    assert buf.output["loss"] == pytest.approx((1.0 + 3.0 + 12.0) / 4)


def test_step_lr_schedule_with_linear_warmup():
    param = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1e-3)
    scheduler = StepLrScheduler(optimizer, step=[4], warmup="linear", warmup_iters=500,
                                warmup_ratio=1.0 / 3)

    def lr_at(epoch, global_iter):
        scheduler.before_epoch(epoch)
        scheduler.before_iter(global_iter)
        return optimizer.param_groups[0]["lr"]

    assert lr_at(0, 0) == pytest.approx(1e-3 / 3)
    assert lr_at(0, 250) == pytest.approx(1e-3 * (1 - 0.5 * (2 / 3)))
    assert lr_at(0, 500) == pytest.approx(1e-3)
    assert lr_at(3, 50_000) == pytest.approx(1e-3)
    assert lr_at(4, 60_000) == pytest.approx(1e-4)


def test_clip_grads():
    a, b = nn.Parameter(torch.ones(3)), nn.Parameter(torch.ones(4))
    assert clip_grads([a, b], max_norm=1.0) is None  # no gradients yet
    a.grad = torch.full((3,), 3.0)
    b.grad = torch.full((4,), 4.0)
    total = clip_grads([a, b], max_norm=1.0)
    assert total.item() == pytest.approx((9 * 3 + 16 * 4) ** 0.5)
    assert torch.cat([a.grad, b.grad]).norm().item() == pytest.approx(1.0, rel=1e-5)


def test_rng_streams_are_independent_and_survive_a_weights_only_load():
    torch.manual_seed(0)
    streams = RngStreams(2, torch.device("cpu"))
    with streams.use(0):
        torch.rand(5)  # consume stream 0 only
    with streams.use(1):
        first_of_stream_1 = torch.rand(3)
    torch.manual_seed(0)
    assert torch.equal(first_of_stream_1, torch.rand(3))  # stream 1 started where both began

    buf = io.BytesIO()
    torch.save(streams.state_dict(), buf)
    buf.seek(0)
    restored = RngStreams(2, torch.device("cpu"))
    restored.load_state_dict(torch.load(buf, weights_only=True))
    with streams.use(0):
        expected = torch.rand(4)
    with restored.use(0):
        assert torch.equal(torch.rand(4), expected)

    with pytest.raises(ValueError):
        RngStreams(3, torch.device("cpu")).load_state_dict(streams.state_dict())


class TinyDetector(nn.Module):
    """Returns a loss dict the way detectors' forward_train does."""

    CLASSES = ("a", "b")

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 1)

    def forward(self, x, y):
        return {"loss_mse": ((self.lin(x) - y) ** 2).mean(), "acc": torch.tensor(50.0)}


def batch(seed):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(4, 3, generator=g), "y": torch.randn(4, 1, generator=g)}


def test_train_step_accumulates_the_mean_gradient_exactly():
    torch.manual_seed(0)
    model = TinyDetector()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = [batch(1), batch(2)]
    log_vars = train_step(model, batches, optimizer, torch.device("cpu"))
    accumulated = [p.grad.clone() for p in model.parameters()]

    optimizer.zero_grad()
    losses = []
    for b in batches:
        loss = model(**b)["loss_mse"]
        (loss / 2).backward()
        losses.append(loss.item())
    for got, expected in zip(accumulated, (p.grad for p in model.parameters()), strict=True):
        assert torch.equal(got, expected)
    assert log_vars["loss"] == 0.0 + losses[0] / 2 + losses[1] / 2
    assert log_vars["acc"] == 50.0


def test_checkpoint_round_trip_resumes_model_and_optimizer(tmp_path):
    torch.manual_seed(0)
    model = TinyDetector()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.05)
    for seed in (1, 2):
        train_step(model, [batch(seed)], optimizer, torch.device("cpu"))
        optimizer.step()
    path = str(tmp_path / "epoch_2.pth")
    save_checkpoint(model, path, optimizer, meta={"epoch": 2, "iter": 20},
                    extra={"rng_states": [RngStreams(1, torch.device("cpu")).state_dict()]})
    assert os.readlink(tmp_path / "latest.pth") == "epoch_2.pth"
    assert not (tmp_path / "epoch_2.pth.tmp").exists()

    torch.manual_seed(123)
    fresh = TinyDetector()
    fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-2, weight_decay=0.05)
    checkpoint = resume_checkpoint(fresh, str(tmp_path / "latest.pth"), fresh_optimizer)
    assert checkpoint["meta"]["epoch"] == 2 and checkpoint["meta"]["iter"] == 20
    assert checkpoint["meta"]["CLASSES"] == ("a", "b")
    for p, q in zip(model.parameters(), fresh.parameters(), strict=True):
        assert torch.equal(p, q)

    # Both continue identically.
    for m, opt in ((model, optimizer), (fresh, fresh_optimizer)):
        train_step(m, [batch(3)], opt, torch.device("cpu"))
        opt.step()
    for p, q in zip(model.parameters(), fresh.parameters(), strict=True):
        assert torch.equal(p, q)


def test_resume_rejects_weights_only_checkpoints(tmp_path):
    model = TinyDetector()
    path = str(tmp_path / "weights.pth")
    torch.save({"state_dict": model.state_dict()}, path)
    with pytest.raises(KeyError):
        resume_checkpoint(model, path, torch.optim.SGD(model.parameters(), lr=0.1))
