import pytest

torch = pytest.importorskip("torch")

from baf.losses import BinaryFocalLoss


def test_focal_loss_matches_formula_for_both_classes():
    logits = torch.tensor([-3.0, -0.5, 0.5, 3.0], dtype=torch.float64)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
    fraud_probability = logits.sigmoid()
    correct_probability = torch.where(labels.bool(), fraud_probability, 1 - fraud_probability)
    expected = (-(1 - correct_probability).square() * correct_probability.log()).mean()

    torch.testing.assert_close(BinaryFocalLoss(gamma=2)(logits, labels), expected)
    torch.testing.assert_close(
        BinaryFocalLoss(gamma=0)(logits, labels),
        torch.nn.functional.binary_cross_entropy_with_logits(logits, labels),
    )


def test_focal_loss_and_gradients_are_finite_for_extreme_logits():
    logits = torch.tensor([-1000.0, -1000.0, 1000.0, 1000.0], requires_grad=True)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loss = BinaryFocalLoss(gamma=0.5)(logits, labels)
    loss.backward()

    assert loss.item() == pytest.approx(500)
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("gamma", [-1, float("nan"), float("inf"), True])
def test_focal_loss_rejects_invalid_gamma(gamma):
    with pytest.raises(ValueError, match="focal_gamma"):
        BinaryFocalLoss(gamma)
