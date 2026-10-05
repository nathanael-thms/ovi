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

# ovi_core/add_model.py

import curses
import os
import shutil
import subprocess

from huggingface_hub import HfApi
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.shortcuts import prompt

from ovi_core.menu import draw_menu
from ovi_core.path import get_models_root
from ovi_core.path import get_repo_root

MODEL_FILE_NAME = "openvino_model.xml"


def method_menu(stdscr: curses.window) -> str:
    """Display the model-selection menu and return the chosen action."""
    stdscr.keypad(True)

    options = [
        "Add a model from a local directory",
        "Add a model from Hugging Face Hub",
        "Exit"
    ]
    current_row = 0

    while True:
        draw_menu(stdscr, current_row, options)
        key = stdscr.getch()

        if key == curses.KEY_UP:
            current_row = (current_row - 1) % len(options)
        elif key == curses.KEY_DOWN:
            current_row = (current_row + 1) % len(options)
        elif key in (curses.KEY_ENTER, 10, 13):
            return options[current_row]
        elif key in (curses.KEY_LEFT, ord("b"), ord("B")):
            return "Back"
        elif key in (27, ord("q"), ord("Q")):
            raise SystemExit(0)


def add_model(method: str | None = None, path: str | None = None, model: str | None = None, name: str | None = None) -> None:
    """Dispatch a direct add action if a complete arg-pair is supplied; otherwise use the interactive menu."""

    if method == "hf" and model is not None and name is not None:
        add_model_from_hf_hub(model=model, name=name)
        return

    if method == "local" and path is not None and name is not None:
        add_model_from_local_directory(path=path, name=name)
        return

    try:
        option = curses.wrapper(method_menu)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt. Exiting ovi")
        raise SystemExit(0) from None

    if option == "Exit":
        print("Exiting ovi")
        raise SystemExit(0)
    if option == "Add a model from a local directory":
        add_model_from_local_directory()
    elif option == "Add a model from Hugging Face Hub":
        add_model_from_hf_hub()


def add_model_from_local_directory(path:str | None = None, name:str | None = None,) -> None:
    """Copy a local OpenVINO model directory into the managed models directory."""
    if path is None:
        print("Please enter the path to the local directory containing the model:")
        path = input().strip()

    if not path or not os.path.isdir(path):
        print(f"Error: The provided path '{path}' is not a valid directory.")
        raise SystemExit(1)

    required_xml = os.path.join(path, MODEL_FILE_NAME)
    if not os.path.exists(required_xml):
        print(
            "WARNING: The provided directory does not contain the required OpenVINO IR model "
            f"files ({MODEL_FILE_NAME})."
        )
        print("Would you like to continue adding this model? (y/n):")
        if input().strip().lower() != "y":
            print("Aborting model addition.")
            raise SystemExit(1)

    if name is None:
        print("Please enter the name you wish to identify the model by:")
        name = input().strip()
    if not name:
        print("No model name provided. Aborting.")
        raise SystemExit(1)

    new_model_path = os.path.join(get_models_root(), name)
    if os.path.exists(new_model_path):
        print(f"Error: Model directory '{new_model_path}' already exists.")
        raise SystemExit(1)

    try:
        shutil.copytree(path, new_model_path)
        print(f"Successfully added model '{name}'")
    except OSError as exc:
        print(f"Error: Failed to copy model files: {exc}")
        raise SystemExit(1) from exc


class SafeFuzzyCompleter(Completer):
    """Allow simple substring matching without regex tokenization surprises."""

    def __init__(self, choices):
        self.choices = choices

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lower()
        for choice in self.choices:
            if text in choice.lower():
                yield Completion(choice, start_position=-len(text))


def add_model_from_hf_hub(model: str | None = None, name: str | None = None) -> None:
    """Pull a model from the OpenVINO organization on Hugging Face."""
    print("Connecting to Hugging Face Hub to grab models...")
    api = HfApi()

    try:
        models = api.list_models(author="OpenVINO")
        model_list = [model_obj.id for model_obj in models]

        if model is None:
            if not model_list:
                print("No models found under the OpenVINO organization.")
                return

            print("\n=== OpenVINO HuggingFace Search Menu ===")
            print(" -> Type characters to search & filter models dynamically.")
            print(" -> Use UP and DOWN arrow keys to navigate suggestions dropdown.")
            print(" -> Press ENTER to select.\n")

            selected_model = prompt(
                "Search OpenVINO Models: ",
                completer=SafeFuzzyCompleter(model_list),
                complete_while_typing=True,
            ).strip()

            if not selected_model:
                print("No model selected. Aborting.")
                return

            if selected_model not in model_list:
                print(f"\nWARNING: '{selected_model}' matches no official repository under OpenVINO.")
                return
        else:
            selected_model = model

        if name is None:
            default_model_name = selected_model.rsplit("/", 1)[-1]
            print("\nPlease enter the name you wish to identify the model by:")
            name = input(f"[{default_model_name}] ").strip() or default_model_name

        if not name:
            print("No model name provided. Aborting.")
            return

        new_model_path = os.path.join(get_models_root(), name)
        if os.path.exists(new_model_path) and os.listdir(new_model_path):
            print(f"Error: Model directory '{new_model_path}' already exists and is not empty.")
            raise SystemExit(1)
        os.makedirs(new_model_path, exist_ok=True)

        print(f"\nDownloading weights for '{selected_model}' to {new_model_path}...")
        try:
            subprocess.run(
                [f"{get_repo_root()}/ovi-env/bin/hf", "download", selected_model, "--local-dir", new_model_path],
                check=True
            )
            print(f"\nSuccessfully downloaded and configured model '{name}'!")
        except subprocess.CalledProcessError as exc:
            print(f"\nError: The 'hf' downloader failed with exit code {exc.returncode}.")
            raise SystemExit(1) from exc

    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc
