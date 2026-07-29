# Copyright 2026 Nathanael Thomas
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ovi_core/path.py
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
