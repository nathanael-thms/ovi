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

import curses
import os

from ovi_core.path import get_models_root


def init_colors() -> None:
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)


def draw_menu(stdscr: curses.window, selected_index: int, options: list[str]) -> None:
    # Clear the screen and set up colors
    stdscr.erase()
    curses.curs_set(0)
    init_colors()

    # Get terminal dimensions and check if the terminal is too small
    height, width = stdscr.getmaxyx()
    if width < 20 or height < len(options) + 6:
        try:
            stdscr.addstr(0, 0, "Terminal too small!")
        except curses.error:
            pass
        stdscr.refresh()
        return

    # Draw the menu options with the selected option highlighted
    try:
        line_width = min(60, max(0, width - 1))

        stdscr.addstr(0, 0, "↑/↓ navigate • ← back • enter launch • esc/q quit.\n", curses.A_DIM)
        stdscr.addstr("=" * line_width + "\n\n")

        for index, option in enumerate(options):
            if index == selected_index:
                stdscr.addstr(" ▸ ", curses.A_BOLD)
                stdscr.addstr(f"{option}\n", curses.color_pair(1) | curses.A_BOLD)
            else:
                stdscr.addstr(f"   {option}\n")
    except curses.error:
        stdscr.erase()
        stdscr.addstr(0, 0, "Terminal too small!")
    finally:
        stdscr.refresh()


def run_menu(stdscr: curses.window, options: list[str], empty_message: str | None = None, draw_fn=draw_menu) -> str:
    stdscr.keypad(True)
    current_row = 0

    if not options:
        if empty_message:
            stdscr.erase()
            stdscr.addstr(0, 0, empty_message)
            stdscr.refresh()
            stdscr.getch()
        return "Exit"

    while True:
        draw_fn(stdscr, current_row, options)
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


def main_menu(stdscr: curses.window) -> str:
    options = ["Launch a model", "Add a model", "Exit"]
    return run_menu(stdscr, options, draw_fn=draw_menu)


def model_menu(stdscr: curses.window, models_dir: str | None = None) -> str:
    models_root = get_models_root() if models_dir is None else models_dir
    models = [d for d in os.listdir(models_root) if os.path.isdir(os.path.join(models_root, d))]
    return run_menu(stdscr, models, empty_message="No models found. Press any key to exit", draw_fn=draw_menu)


def method_menu(stdscr: curses.window) -> str:
    options = [
        "Add a model from a local directory",
        "Add a model from Hugging Face Hub",
        "Exit",
    ]
    return run_menu(stdscr, options, draw_fn=draw_menu)
