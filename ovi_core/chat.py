# ovi_core/chat.py

import atexit
import os

from ovi_core.load import OviEngine

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

    # # Add the entry to the history, handling any potential exceptions
    # try:
    #     add_history(entry)
    # except (AttributeError, OSError):
    #     pass


def start_chat_loop(model_name: str, device: str = "CPU"):
    # Start a chat loop with the specified model and device
    pipe = OviEngine.get_pipeline(model_name, device)

    pipe.start_chat()
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

            _record_history_entry(user_input)

            if user_input.lower() in ("/exit", "/quit"):
                print("bye")
                return

            # Core native text streaming block
            pipe.generate(user_input, streamer=_stream_callback)
            print()  # Terminal formatting newline

    except KeyboardInterrupt:
        print("\nChat session stopped.")
    finally:
        pipe.finish_chat()
