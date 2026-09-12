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
import sys

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
        "stop_token_ids": set
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

                parts = line.split(None, 2)
                if len(parts) < 3:
                    continue

                _, key, raw_value = parts
                key = key.strip()

                # Strip comments ONLY when they appear as " #"
                if " #" in raw_value:
                    raw_value = raw_value.split(" #", 1)[0]

                # Strip surrounding quotes
                raw_value = raw_value.strip().strip('"').strip("'")

                # Apply alias
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
                        value = [
                            int(x) for x in raw_value.split(",")
                            if x.strip().isdigit()
                        ]
                    else:
                        value = cast_type(raw_value)
                except ValueError:
                    continue

                # Store or merge values
                if key == "stop_token_ids" and key in parameters:
                    parameters[key].extend(value)
                else:
                    parameters[key] = value

        # Convert list-type token fields into sets
        if "stop_token_ids" in parameters and isinstance(parameters["stop_token_ids"], list):
            parameters["stop_token_ids"] = set(parameters["stop_token_ids"])

        # Generation parameter legality validator
        def fail(msg: str):
            print(f"Error: {msg}")
            sys.exit(1)

        # num_beams sanity
        nb = parameters.get("num_beams", 1)
        if not isinstance(nb, int) or nb <= 0:
            fail("Invalid num_beams in Modelfile: expected a positive integer.")
        if nb > 16:
            fail("Invalid num_beams in Modelfile: maximum value is 16.")
        parameters["num_beams"] = nb

        if nb > 1:
            if "top_k" in parameters and parameters["top_k"] != 0:
                fail("Invalid Modelfile config: beam search requires top_k=0.")
            if "top_p" in parameters and parameters["top_p"] != 0.0:
                fail("Invalid Modelfile config: beam search requires top_p=0.0.")
            if "min_p" in parameters and parameters["min_p"] != 0.0:
                fail("Invalid Modelfile config: beam search requires min_p=0.0.")
            if "presence_penalty" in parameters and parameters["presence_penalty"] != 0.0:
                fail("Invalid Modelfile config: beam search forbids presence_penalty.")
            if "frequency_penalty" in parameters and parameters["frequency_penalty"] != 0.0:
                fail("Invalid Modelfile config: beam search forbids frequency_penalty.")
            if "diversity_penalty" in parameters and parameters["diversity_penalty"] != 0.0:
                fail("Invalid Modelfile config: beam search forbids diversity_penalty.")
            if "repetition_penalty" in parameters and parameters["repetition_penalty"] != 1.0:
                fail("Invalid Modelfile config: beam search requires repetition_penalty=1.0.")
            if "temperature" in parameters and parameters["temperature"] != 1.0:
                fail("Invalid Modelfile config: beam search requires temperature=1.0.")
            if "max_ngram_size" in parameters and parameters["max_ngram_size"] != 0:
                fail("Invalid Modelfile config: beam search requires max_ngram_size=0.")

        # length_penalty legality
        lp = parameters.get("length_penalty", 0.0)
        if not isinstance(lp, (int, float)) or lp < 0:
            fail("Invalid length_penalty in Modelfile: expected a non-negative number.")

        # logprobs legality
        lg = parameters.get("logprobs", 0)
        if not isinstance(lg, int) or lg < 0 or lg > 10:
            fail("Invalid logprobs in Modelfile: expected an integer between 0 and 10.")

        # max_length / min_new_tokens legality
        max_len = parameters.get("max_length", None)
        min_new = parameters.get("min_new_tokens", 0)

        if isinstance(max_len, int) and isinstance(min_new, int):
            if max_len < min_new:
                fail("Invalid Modelfile config: max_length cannot be less than min_new_tokens.")

        # max_new_tokens ≤ max_length
        mnt = parameters.get("max_new_tokens", None)
        if isinstance(max_len, int) and isinstance(mnt, int):
            if mnt > max_len:
                fail("Invalid Modelfile config: max_new_tokens cannot exceed max_length.")

        # stop_token_ids legality
        if "stop_token_ids" in parameters:
            ids = parameters["stop_token_ids"]
            if not isinstance(ids, (list, set)):
                fail("Invalid stop_token_ids in Modelfile: expected a list or set of non-negative integers.")
            normalized = {x for x in ids if isinstance(x, int) and x >= 0}
            if len(normalized) != len(ids):
                fail("Invalid stop_token_ids in Modelfile: all values must be non-negative integers.")
            parameters["stop_token_ids"] = normalized

        # stop_strings legality
        if "stop_strings" in parameters:
            if not isinstance(parameters["stop_strings"], str):
                fail("Invalid stop_strings in Modelfile: expected a string.")

        # Streaming compatibility
        if parameters.get("stream", False) and parameters.get("num_beams", 1) > 1:
            fail("Invalid Modelfile config: streaming cannot be combined with num_beams > 1.")

        return parameters

    return parameters