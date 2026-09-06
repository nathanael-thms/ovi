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

import os

from ovi_core.path import get_model_file_path

def get_device_from_modelfile(model_name: str) -> str:
    """
    Reads the Modelfile for the specified model and extracts the device information.
    Returns the device as a string (e.g., "CPU", "GPU").
    If the Modelfile does not exist or does not contain a device entry, returns "CPU" by default.
    """

    modelfile_path = get_model_file_path(model_name)

    # Default device
    device = "CPU"

    try:
        with open(modelfile_path, 'r') as f:
            for line in f:
                if line.startswith("DEVICE "):
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        candidate = parts[1].strip().strip('"').strip("'")
                        # Validate the candidate device
                        if candidate in ("CPU", "GPU", "NPU", "AUTO"):
                            device = candidate
    except FileNotFoundError:
        print(f"Warning: Modelfile not found for model '{model_name}'. Defaulting to CPU.")

    return device

def get_parameters_from_modelfile(model_name: str) -> dict:
    """
    Reads the Modelfile for the specified model and extracts valid OpenVINO GenAI
    generation parameters. Returns a dictionary with correctly typed values.
    """

    available_parameters = {
        "max_new_tokens": int,
        "temperature": float,
        "top_k": int,
        "top_p": float,
        "repetition_penalty": float,
        "presence_penalty": float,
        "frequency_penalty": float,
        "num_beams": int,
        "no_repeat_ngram_size": int,
        "max_length": int,
        "min_new_tokens": int,
        "max_ngram_size": int,
        "min_p": float,
        "diversity_penalty": float,
        "length_penalty": float,
        "ignore_eos": bool,
        "echo": bool,
        "logprobs": int,
        "stop_strings": str,
        "stop_token_ids": str
    }

    aliases = {
        "stop": "stop_strings",
        "stop_sequence": "stop_strings",
        "stop_sequences": "stop_strings",
        "stop_token": "stop_token_ids",
        "stop_tokens": "stop_token_ids",
        "num_beam_groups": "num_beams",
        "beam_width": "num_beams",
        "no_repeat_ngram": "no_repeat_ngram_size",
        "min_tokens": "min_new_tokens",
        "ngram_size": "max_ngram_size",
        "min_probability": "min_p",
        "diversity": "diversity_penalty",
        "length": "length_penalty",
        "ignore_end_of_sequence": "ignore_eos",
        "echo_prompt": "echo",
        "log_probabilities": "logprobs",
        "temp": "temperature",
        "presence": "presence_penalty",
        "frequency": "frequency_penalty",
        "beam_size": "num_beams",
        "stop_string": "stop_strings"
    }

    modelfile_path = get_model_file_path(model_name)
    parameters = {}

    if os.path.isfile(modelfile_path):
        with open(modelfile_path, "r") as f:
            for line in f:
                if not line.startswith("PARAMETER "):
                    continue

                _, key, raw_value = line.split(" ", 2)
                key = key.strip()
                raw_value = raw_value.strip()

                # Apply alias if present
                if key in aliases:
                    key = aliases[key]

                # Skip unknown parameters
                if key not in available_parameters:
                    continue

                cast_type = available_parameters[key]

                try:
                    if cast_type is bool:
                        value = raw_value.lower() in ("1", "true", "yes", "on")
                    elif key == "stop_token_ids":
                        value = [int(x.strip()) for x in raw_value.split(",")]
                    else:
                        value = cast_type(raw_value)
                except ValueError:
                    continue

                parameters[key] = value

    return parameters