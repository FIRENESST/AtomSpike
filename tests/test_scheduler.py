from __future__ import annotations

import pytest
import torch

from atomspike.train.sched import WSDConfig, WSDScheduler


def test_wsd_first_optimizer_step_is_not_zero_lr() -> None:
    param = torch.nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.SGD([param], lr=1.0)
    sched = WSDScheduler(
        opt,
        WSDConfig(total_steps=10, warmup_frac=0.2, stable_frac=0.7, decay_frac=0.1),
    )

    assert opt.param_groups[0]["lr"] == pytest.approx(0.5)
    param.grad = torch.ones_like(param)
    opt.step()
    sched.step()

    assert param.item() == pytest.approx(0.5)
    assert sched.step_count == 1
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0)


def test_wsd_resume_prepares_exact_next_learning_rate() -> None:
    p0 = torch.nn.Parameter(torch.tensor(0.0))
    opt0 = torch.optim.SGD([p0], lr=2.0)
    cfg = WSDConfig(total_steps=20, warmup_frac=0.1, stable_frac=0.7, decay_frac=0.2)
    sched0 = WSDScheduler(opt0, cfg)
    for _ in range(7):
        sched0.step()

    p1 = torch.nn.Parameter(torch.tensor(0.0))
    opt1 = torch.optim.SGD([p1], lr=2.0)
    sched1 = WSDScheduler(opt1, cfg)
    sched1.load_state_dict(sched0.state_dict())

    assert sched1.step_count == 7
    assert opt1.param_groups[0]["lr"] == pytest.approx(opt0.param_groups[0]["lr"])
    sched0.step()
    sched1.step()
    assert opt1.param_groups[0]["lr"] == pytest.approx(opt0.param_groups[0]["lr"])
