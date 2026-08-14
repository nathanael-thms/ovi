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
 
# ovi_core/parse_modelfile.py

from ovi_core.path import get_model_file_path

def get_device_from_modelfile(model_name: str) -> str:
    """
    Reads the Modelfile for the specified model and extracts the device information.
    Returns the device as a string (e.g., "CPU", "GPU").
    If the Modelfile does not exist or does not contain a device entry, returns "CPU" by default.
    """
    modelfile_path = get_model_file_path(model_name)

    # 1. Set a default value first!
    device = "CPU"

    try:
        with open(modelfile_path, 'r') as f:
            for line in f:
                if line.startswith("DEVICE="):
                    device = line.split("=", 1)[1].strip()
    except FileNotFoundError:
        print(f"Warning: Modelfile not found for model '{model_name}'. Defaulting to CPU.")

    # 2. This is now safe because 'device' is guaranteed to exist!
    return device