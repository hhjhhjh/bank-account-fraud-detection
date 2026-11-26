import pytest

torch = pytest.importorskip("torch")

from baf.models import BaselineMLP, ResidualBlock, ResidualMLP
from baf.ft_transformer import FTTransformer, NumericalFeatureTokenizer, ReGLU


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


def test_feature_tokenizer_uses_separate_affine_vectors_for_each_column():
    tokenizer = NumericalFeatureTokenizer(input_dim=2, d_token=4)
    with torch.no_grad():
        tokenizer.weight.copy_(torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]]))
        tokenizer.bias.copy_(torch.tensor([[0., 1., 0., 1.], [1., 0., 1., 0.]]))
    features = torch.tensor([[2., 3.]], requires_grad=True)
    tokens = tokenizer(features)
    torch.testing.assert_close(tokens, torch.tensor([[[2., 5., 6., 9.], [13., 9., 7., 3.]]]))
    tokens.sum().backward()
    torch.testing.assert_close(features.grad, torch.tensor([[10., 10.]]))


def test_reglu_multiplies_values_by_positive_gates():
    features = torch.tensor([[2., -3., -1., 4.]])
    torch.testing.assert_close(ReGLU()(features), torch.tensor([[0., -12.]]))


def test_ft_transformer_has_finite_single_logits_and_feature_gradients():
    model = FTTransformer(4, d_token=8, n_heads=2, num_blocks=2, ffn_hidden=12, dropout=0.1)
    assert isinstance(model.blocks[0].attention_norm, torch.nn.Identity)
    assert isinstance(model.blocks[1].attention_norm, torch.nn.LayerNorm)
    features = torch.randn(3, 4, requires_grad=True)
    logits = model(features)
    assert logits.shape == (3,)
    logits.sum().backward()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0
    assert model(torch.zeros(1, 4)).shape == (1,)
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(2, 5))


@pytest.mark.parametrize("override", [
    {"d_token": 7}, {"n_heads": 0}, {"num_blocks": 0}, {"ffn_hidden": 0},
    {"input_dim": 0}, {"dropout": 1.0}, {"residual_dropout": -0.1},
])
def test_ft_transformer_rejects_invalid_model_configuration(override):
    kwargs = dict(input_dim=4, d_token=8, n_heads=2, num_blocks=1, ffn_hidden=12, dropout=0.1)
    with pytest.raises(ValueError):
        FTTransformer(**{**kwargs, **override})
