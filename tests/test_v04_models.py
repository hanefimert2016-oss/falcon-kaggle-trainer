import torch

from flm.models import (
    ByteCausalLM,
    TextConfig,
    ComputerUseModel,
    ComputerUseConfig,
)
from flm.runtime import select_runtime


def test_text_model_forward_backward():
    cfg = TextConfig(
        seq_len=16,
        n_layer=1,
        n_head=2,
        n_embd=32,
    )
    model = ByteCausalLM(cfg)
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    _, loss = model(x, y)
    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()


def test_computer_use_forward_backward():
    cfg = ComputerUseConfig(
        task_len=16,
        action_len=12,
        image_size=32,
        patch=8,
        embd=32,
        text_layers=1,
        vision_layers=1,
        n_head=2,
    )
    model = ComputerUseModel(cfg)
    images = torch.rand(2, 3, 32, 32)
    task = torch.randint(0, 256, (2, 16))
    action_in = torch.randint(0, 256, (2, 12))
    action_target = torch.randint(0, 256, (2, 12))
    op_target = torch.tensor([0, 1])
    _, _, loss = model(
        images,
        task,
        action_in,
        op_target,
        action_target,
    )
    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()


def test_cpu_runtime():
    runtime = select_runtime("cpu")
    assert runtime.kind == "cpu"
    assert str(runtime.device) == "cpu"
