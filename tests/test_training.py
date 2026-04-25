from __future__ import annotations

from womens_health_ai.settings import paths


def test_training_generates_required_artifacts(trained_metadata):
    assert paths.bundle_path.exists()
    assert paths.model_info_path.exists()
    assert "logistic_regression" in trained_metadata["models"]
    assert "decision_tree" in trained_metadata["models"]
    assert "recall" in trained_metadata["champion_model"]["metrics"]
    assert "shap_reports" in trained_metadata
    assert "eda_reports" in trained_metadata

