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

import argparse
import curses
import os
import subprocess
import readline
import sys

# Third-party Hugging Face Hub library integration
from huggingface_hub import HfApi

# Third-party prompt_toolkit library for interactive fuzzy search
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.shortcuts import prompt

from ovi_core.menu import draw_menu
from ovi_core.path import get_models_root

def method_menu(stdscr: curses.window) -> str:
    # Set up the terminal for menu navigation, and draw the initial menu
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
        elif key in [curses.KEY_ENTER, 10, 13]:
            return options[current_row]
        elif key in [curses.KEY_LEFT]:
            return "Back"
        elif key in [27, ord('q'), ord('Q')]:
            sys.exit(0)

def add_model() -> None:
    """
    Asks the user which method of adding a model they would like to use, and then calls the appropriate function.

    Available methods:
    1. Add a model from a local directory.
    2. Add a model from Hugging Face Hub.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Add a model from a local directory")
    parser.add_argument("--hf", action="store_true", help="Add a model from Hugging Face Hub")

    args = parser.parse_args()

    if args.local:
        add_model_from_local_directory()
    elif args.hf:
        add_model_from_hf_hub()
    else:
        try:
            option = curses.wrapper(method_menu)

            if option == "Exit":
                print("Exiting ovi")
                sys.exit(0)
            elif option == "Add a model from a local directory":
                add_model_from_local_directory()
            elif option == "Add a model from Hugging Face Hub":
                add_model_from_hf_hub()

        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt. Exiting ovi")
            sys.exit(0)

def add_model_from_local_directory() -> None:
    """
    Function for adding a model from a local directory.
    """
    print("Please enter the path to the local directory containing the model:")
    model_path = input().strip()

    if not model_path or not os.path.isdir(model_path):
        print(f"Error: The provided path '{model_path}' is not a valid directory.")
        sys.exit(1)
    else:
        required_xml = os.path.join(model_path, "openvino_model.xml")
        if not os.path.exists(required_xml):
            print(
                "WARNING: The provided directory does not contain the required OpenVINO IR model files (openvino_model.xml).")
            print("Would you like to continue adding this model? (y/n):")
            choice = input().strip().lower()
            if choice != 'y':
                print("Aborting model addition.")
                sys.exit(1)

    print("Please enter the name you wish to identify the model by:")
    model_name = input().strip()

    new_model_path = os.path.join(get_models_root(), model_name)
    command = f'mkdir -p "{new_model_path}" && cp -RL "{model_path}"/* "{new_model_path}/"'
    try:
        # Run the shell string securely and block until it completes
        subprocess.run(command, shell=True, check=True)
        print(f"Successfully added model '{model_name}'")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to copy model files. Terminal exited with code {e.returncode}.")
        sys.exit(1)


def add_model_from_hf_hub() -> None:
    """
    Pulls a model from the HF hub under OpenVINO toolkit organisation.
    Provides an interactive, fuzzy-searchable selection interface.
    """

    print("Connecting to Hugging Face Hub to grab models...")
    api = HfApi()

    try:
        models = api.list_models(author="OpenVINO")
        model_list = [model.id for model in models]

        if not model_list:
            print("No models found under the OpenVINO organization.")
            return

        print("\n=== OpenVINO HuggingFace Search Menu ===")
        print(" -> Type characters to search & filter models dynamically.")
        print(" -> Use UP and DOWN arrow keys to navigate suggestions dropdown.")
        print(" -> Press ENTER to select.\n")

        class SafeFuzzyCompleter(Completer):
            def __init__(self, choices):
                self.choices = choices

            def get_completions(self, document, complete_event):
                # Grab whatever the user has typed in the buffer so far (e.g., 'qwen2.')
                text = document.text_before_cursor.lower()

                # Filter down choices manually using simple sub-string matching
                # This completely sidesteps broken regex tokenization splits on '.', '-' and '/'
                for choice in self.choices:
                    if text in choice.lower():
                        # Display the full repository name, positioning the text drop down accurately
                        yield Completion(choice, start_position=-len(text))

        # Instantiate our safe completer
        fuzzy_model_completer = SafeFuzzyCompleter(model_list)

        # Prompt with active text entry and real-time dropdown filtering
        selected_model = prompt(
            "Search OpenVINO Models: ",
            completer=fuzzy_model_completer,
            complete_while_typing=True
        ).strip()

        if not selected_model:
            print("No model selected. Aborting.")
            return

        if selected_model not in model_list:
            print(f"\nWARNING: '{selected_model}' matches no official repository under OpenVINO.")
            return

        # Isolate the model name (e.g., transforms 'OpenVINO/qwen2.5-7b' into 'qwen2.5-7b')
        default_model_name = selected_model.split('/')[-1]

        # Inject the string into the terminal input buffer for the next input() call
        readline.set_startup_hook(lambda: readline.insert_text(default_model_name))

        try:
            print("\nPlease enter the name you wish to identify the model by:")
            model_name = input().strip()
        finally:
            # Clear the hook immediately so it doesn't leak into future inputs
            readline.set_startup_hook()

        if not model_name:
            print("No model name provided. Aborting.")
            return

        new_model_path = os.path.join(get_models_root(), model_name)

        # Create directory if it doesn't exist, but don't overwrite existing content
        os.makedirs(new_model_path, exist_ok=True)

        print(f"\nDownloading weights for '{selected_model}' to {new_model_path}...")
        try:
            # check=True will raise an error automatically if the download fails
            subprocess.run(
                ["hf", "download", selected_model, "--local-dir", new_model_path],
                check=True
            )
            print(f"\nSuccessfully downloaded and configured model '{model_name}'!")

        except subprocess.CalledProcessError as e:
            print(f"\nError: The 'hf' downloader failed with exit code {e.returncode}.")
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
