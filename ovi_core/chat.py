# ovi_core/chat.py

import atexit
import os
import openvino_genai as ov_genai

from ovi_core.load import OviEngine
from ovi_core.parse_modelfile import get_device_from_modelfile

try:
    import readline
except ImportError:  # pragma: no cover - readline is standard on Unix
    readline = None

_HISTORY_FILE = os.path.expanduser("~/.ovi_history")
_HISTORY_LENGTH = 1000


def _stream_callback(token_subword: str) -> bool:
    print(token_subword, end="", flush=True)
    return False  # False continues execution; True aborts generation


def _configure_readline() -> None:
    # Configure readline for command history and navigation
    if readline is None:
        return

    try:
        readline.read_history_file(_HISTORY_FILE)
    except FileNotFoundError:
        pass
    except OSError:
        pass

    readline.set_history_length(_HISTORY_LENGTH)
    atexit.register(_write_history_file)


def _write_history_file() -> None:
    # Write the command history to the history file on exit
    if readline is None:
        return

    try:
        readline.write_history_file(_HISTORY_FILE)
    except OSError:
        pass


def _record_history_entry(entry: str) -> None:
    # Return if the entry is empty or readline is not available
    if not entry.strip() or readline is None:
        return

    # Use the appropriate method to add the entry to the history, depending on the readline version
    add_history = getattr(readline, "add_history", None)
    if add_history is None:
        add_history = getattr(readline, "add_history_entry", None)

    # If neither method is available, return without adding the entry
    if add_history is None:
        return

    # Add the entry to the history, handling any potential exceptions
    try:
        add_history(entry)
    except (AttributeError, OSError):
        pass


def start_chat_loop(model_name: str):
    engine_data = OviEngine.get_pipeline(model_name)

    # Safely unpack the native components out of the container
    pipe = engine_data["pipeline"]
    system_prompt = engine_data.get("system_prompt", "")

    # Initialize the native OpenVINO ChatHistory container
    history = ov_genai.ChatHistory()
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    _configure_readline()
    # Display a message and instructions for the user
    print(f"\nConnected to raw model '{model_name}'. Type '/exit' to quit.")
    print("Use ↑/↓ to browse recent prompts.")
    # Display a separator line
    print("-" * 60)

    try:
        while True:
            user_input = input("\n>>> ")
            if not user_input.strip():
                continue

            # Check if the last entry in the history is the same as the current input to avoid duplicates
            if readline is not None and readline.get_current_history_length() > 0:
                last_entry = readline.get_history_item(readline.get_current_history_length())
                if last_entry != user_input:
                    _record_history_entry(user_input)

            if user_input.lower() in ("/exit", "/quit"):
                print("bye")
                return

            # Append the user prompt to our session tracking container
            history.append({"role": "user", "content": user_input})

            # Core native text streaming block using the history payload
            pipe.generate(history, streamer=_stream_callback)
            print()  # Terminal formatting newline

    except KeyboardInterrupt:
        print("\nChat session stopped.")