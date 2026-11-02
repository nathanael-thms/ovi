# Copyright 2026 Nathanael Thomas
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://apache.org
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ovi_core/modelfile_editor.py

import os
import subprocess
import textwrap

import sys

from ovi_core.path import get_model_file_path


def open_modelfile_editor(model_name: str) -> None:
    DEFAULT_MODELFILE_TEMPLATE = """
    # ovi Modelfile
    # For syntax, see https://ovi.nattho.com/models/#syntax
    # Currently all teh lines are commented out, you may uncomment, remove, and edit these lines as you wish

    # Select the target device
    # DEVICE CPU
    
    # System prompt
    # SYSTEM "You are a helpful coding assistant."
    
    # Generation settings
    # PARAMETER max_new_tokens 256
    # PARAMETER temperature 0.7
    # PARAMETER top_p 0.9
    # PARAMETER stop_strings "END"
    # PARAMETER stop_token_ids 12,13,99
    """

    modelfile_path = get_model_file_path(model_name)

    abs_path = os.path.abspath(modelfile_path)
    model_dir = os.path.dirname(abs_path)

    # Check if the parent model directory exists before proceeding
    if not os.path.exists(model_dir):
        print("Model Not Found")
        return

    # Write the template only if the file does not already exist
    if not os.path.exists(abs_path):
        with open(abs_path, 'w', encoding='utf-8') as file:
            # textwrap.dedent removes any accidental leading indentation from multi-line strings
            file.write(textwrap.dedent(DEFAULT_MODELFILE_TEMPLATE).strip() + "\n")

    editor = os.environ.get('VISUAL') or os.environ.get('EDITOR') or 'nano'

    try:
        subprocess.run([editor, abs_path], check=True)
    except FileNotFoundError:
        try:
            subprocess.run(['vi', abs_path], check=True)
        except Exception as e:
            sys.exit(f"Failed to open editor: {e}")
    except subprocess.CalledProcessError as e:
        sys.exit(f"Editor exited with error: {e}")
