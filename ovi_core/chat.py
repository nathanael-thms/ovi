# ovi_core/chat.py

import os
import sys

script_path = os.path.realpath(__file__)
script_dir = os.path.dirname(script_path)
root_dir = os.path.dirname(script_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
models_dir = os.path.join(script_dir, "models")

from ovi_core.load import OviEngine

def _stream_callback(token_subword: str) -> bool:
    print(token_subword, end="", flush=True)
    return False  # False continues execution; True aborts generation

def start_chat_loop(model_name: str, device: str = "CPU"):
    # Retrieves the pre-loaded engine or compiles it instantly on the fly
    pipe = OviEngine.get_pipeline(model_name, device)
    
    pipe.start_chat()
    print(f"\nConnected to raw model '{model_name}'. Type '/exit' to quit.")
    print("-" * 60)

    try:
        while True:
            user_input = input("\n>>> ")
            if not user_input.strip():
                continue

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
        OviEngine.mark_idle()
