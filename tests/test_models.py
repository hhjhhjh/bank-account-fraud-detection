import pytest

torch = pytest.importorskip("torch")

from baf.models import BaselineMLP, ResidualBlock, ResidualMLP


def test_residual_block_is_identity_when_its_branch_is_zero():
    block = ResidualBlock(width=4, dropout=0.1)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.zero_()
    features = torch.tensor([[-1.0, 0.5, -0.25, 2.0]], requires_grad=True)

    output = block(features)
    output.sum().backward()

    torch.testing.assert_close(output, features)
    torch.testing.assert_close(features.grad, torch.ones_like(features))


def test_residual_mlp_outputs_one_logit_and_has_similar_capacity_to_e1():
    baseline = BaselineMLP(input_dim=57, hidden_dims=[128, 64], dropout=0.1)
    residual = ResidualMLP(input_dim=57, width=75, num_blocks=1, dropout=0.1)
    baseline_count = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    residual_count = sum(p.numel() for p in residual.parameters() if p.requires_grad)

    assert residual(torch.zeros(1, 57)).shape == (1,)
    assert residual(torch.zeros(4, 57)).shape == (4,)
    assert abs(residual_count - baseline_count) / baseline_count < 0.01
