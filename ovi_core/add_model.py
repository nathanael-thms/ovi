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

# ovi_core/add_model.py

import argparse
import curses
import sys
import os
import subprocess

from ovi_core.menu import draw_menu
from ovi_core.path import get_models_root

def method_menu(stdscr: curses.window) -> str:
    # Set up the terminal for menu navigation, and draw the initial menu
    stdscr.keypad(True)

    options = ["Add a model from a local directory", "Exit"]
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

    Later on it will include a method to add from HF hub
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Add a model from a local directory")

    args = parser.parse_args()

    if args.local:
        add_model_from_local_directory()
    else:
        try:
            option = curses.wrapper(method_menu)

            if option == "Exit":
                print("Exiting ovi")
                sys.exit(0)
            elif option == "Add a model from a local directory":
                add_model_from_local_directory()

        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt. Exiting ovi")
            sys.exit(0)

def add_model_from_local_directory() -> None:
    """
    Function for adding a model from a local directory.
    This function will be implemented in the future to handle the logic of adding a model from a local directory.
    """
    print("Please enter the path to the local directory containing the model:")
    model_path = input().strip()

    # Verify the provided path exists and is a directory + ensure it contains the required OpenVINO IR model files
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
