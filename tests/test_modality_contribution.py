import pytest

from residual_actmodel.modality_contribution import summarize_modality_losses


def test_summarize_modality_losses_computes_relative_drop_from_full():
    result = summarize_modality_losses(
        {
            "full": 10.0,
            "no_visual": 11.0,
            "no_current_force": 12.0,
        }
    )

    assert result["visual"]["loss"] == pytest.approx(11.0)
    assert result["visual"]["absolute_increase"] == pytest.approx(1.0)
    assert result["visual"]["relative_increase_pct"] == pytest.approx(10.0)
    assert result["current_force"]["relative_increase_pct"] == pytest.approx(20.0)


def test_summarize_modality_losses_rejects_missing_full_baseline():
    with pytest.raises(ValueError, match="full"):
        summarize_modality_losses({"no_visual": 11.0})
