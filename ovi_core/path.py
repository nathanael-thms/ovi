import os
from pathlib import Path


# This file centralizes the logic for determining directories
def get_repo_root() -> str:
    """Return the repository root based on this module's location."""
    return str(Path(__file__).resolve().parents[1])


def get_models_root() -> str:
    """Return the models directory."""
    return os.path.join(get_repo_root(), "models")


def get_model_path(model_name: str) -> str:
    """Return the absolute path for a named model directory."""
    return os.path.join(get_models_root(), model_name)
