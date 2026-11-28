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

# ovi_core/exceptions.py

class OviEngineError(Exception):
    """Base exception for all OviEngine errors."""
    pass

class ModelfileConfigError(OviEngineError):
    """Raised when a Modelfile contains invalid parameters or conflicting settings."""
    def __init__(self, model_name: str, reason: str):
        self.model_name = model_name
        self.reason = reason
        super().__init__(f"Invalid Modelfile config for '{model_name}': {reason}")

class ModelNotFoundError(OviEngineError):
    """Raised when the OpenVINO IR model files are missing."""
    def __init__(self, model_name: str, model_dir: str):
        self.model_name = model_name
        self.model_dir = model_dir
        super().__init__(f"OpenVINO IR model '{model_name}' not found at: {model_dir}")


class ModelCompilationError(OviEngineError):
    """Raised when OpenVINO fails to compile or load the model onto the target device."""
    def __init__(self, model_name: str, device: str, original_error: Exception):
        self.model_name = model_name
        self.device = device
        self.original_error = original_error
        super().__init__(
            f"Native OpenVINO compilation failed for '{model_name}' on {device}: {original_error}"
        )
