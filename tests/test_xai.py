import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("captum")

from baf.xai import explain_inputs, group_attributions, select_local_cases


def test_ig_completeness_for_linear_logit_model():
    model = torch.nn.Linear(3, 1, bias=True)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -2.0, 3.0]]))
    inputs = torch.tensor([[2.0, 4.0, -1.0], [1.0, 1.0, 1.0]])
    baseline = torch.zeros(1, 3)
    config = {"initial_steps": 16, "max_steps": 64, "internal_batch_size": 32,
              "completeness_atol": 1e-5, "completeness_rtol": 1e-5}
    attrs, delta, steps = explain_inputs(lambda x: model(x).squeeze(-1), inputs, baseline, config)
    np.testing.assert_allclose(attrs, (inputs * model.weight).detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(delta, 0, atol=1e-5)
    assert steps.tolist() == [16, 16]


def test_grouping_sums_signed_onehot_and_indicator_before_absolute_value():
    mapping = [{"original_feature": name} for name in ("a", "a", "b", "a")]
    grouped, names = group_attributions(np.array([[2.0, -1.0, -3.0, 0.5]]), mapping)
    assert names == ["a", "b"]
    np.testing.assert_array_equal(grouped, [[1.5, -3.0]])


def test_case_selection_is_deterministic_and_reports_missing_outcomes():
    labels = np.array([1, 1, 0, 0, 1, 0])
    scores = np.array([1.0, 0.2, 0.8, -1.0, 0.4, 0.9])
    selected = select_local_cases(labels, scores, 0.5)
    assert selected == select_local_cases(labels, scores, 0.5)
    assert labels[selected["TP"]] == labels[selected["FN"]] == 1
    assert labels[selected["FP"]] == 0
    missing = select_local_cases(np.array([1, 0]), np.array([1.0, -1.0]), 0.5)
    assert missing["FN"] is None and missing["FP"] is None
