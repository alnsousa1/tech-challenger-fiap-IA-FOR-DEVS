from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _discover_base_dir() -> Path:
    env_override = os.getenv("WOMENS_HEALTH_AI_HOME")
    if env_override:
        return Path(env_override).resolve()

    current_file = Path(__file__).resolve()
    for candidate in [current_file.parent, *current_file.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate

    current_workdir = Path.cwd().resolve()
    for candidate in [current_workdir, *current_workdir.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate

    return current_workdir


BASE_DIR = _discover_base_dir()


@dataclass(frozen=True)
class ProjectPaths:
    base_dir: Path = BASE_DIR
    artifacts_dir: Path = BASE_DIR / "artifacts"
    reports_dir: Path = BASE_DIR / "reports"
    reports_eda_dir: Path = BASE_DIR / "reports" / "eda"
    reports_explainability_dir: Path = BASE_DIR / "reports" / "explainability"
    static_dir: Path = BASE_DIR / "src" / "womens_health_ai" / "api" / "static"
    bundle_path: Path = BASE_DIR / "artifacts" / "model_bundle.joblib"
    model_info_path: Path = BASE_DIR / "artifacts" / "model_info.json"


@dataclass(frozen=True)
class ProjectSettings:
    project_name: str = "Women's Health AI Diagnostic Support"
    project_version: str = "1.0.0"
    dataset_name: str = "Breast Cancer Wisconsin (Diagnostic)"
    target_column: str = "diagnosis_malignant"
    target_label_column: str = "diagnosis_label"
    positive_class_label: str = "malignant"
    negative_class_label: str = "benign"
    test_size: float = 0.20
    random_state: int = 42
    cv_folds: int = 5
    shap_background_size: int = 120
    shap_sample_size: int = 80
    model_threshold: float = float(os.getenv("MODEL_THRESHOLD", "0.50"))
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "info")
    malignant_recall_rationale: str = (
        "Recall is the primary healthcare metric because false negatives can delay "
        "diagnosis for malignant cases. F1-score complements recall by balancing "
        "the cost of missed cancers with the operational burden of false positives."
    )


paths = ProjectPaths()
settings = ProjectSettings()
