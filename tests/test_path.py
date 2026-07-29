from pathlib import Path

from ovi_core.path import get_model_path, get_models_root, get_repo_root


def test_repo_root_matches_workspace_root():
    assert Path(get_repo_root()).resolve() == Path(__file__).resolve().parents[1]


def test_models_root_points_to_repo_models_directory():
    assert Path(get_models_root()).resolve() == Path(get_repo_root()).resolve() / "models"


def test_model_path_joins_named_model_dir():
    assert Path(get_model_path("demo-model")).resolve() == Path(get_models_root()).resolve() / "demo-model"
