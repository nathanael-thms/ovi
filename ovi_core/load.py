# ovi_core/load.py
import os
import sys
import threading
import time
import openvino_genai as ov_genai


class OviEngine:
    # Persistent global memory variables
    _current_model_name = None
    _current_device = None
    _pipeline_instance = None
    _cooldown_seconds = 300
    _cooldown_timer = None
    _cooldown_deadline = None

    @classmethod
    def _cancel_cooldown_timer(cls):
        if cls._cooldown_timer is not None:
            cls._cooldown_timer.cancel()
            cls._cooldown_timer = None
        cls._cooldown_deadline = None

    @classmethod
    def _schedule_unload_after_cooldown(cls):
        if cls._pipeline_instance is None:
            return

        cls._cancel_cooldown_timer()
        cls._cooldown_deadline = time.monotonic() + cls._cooldown_seconds
        cls._cooldown_timer = threading.Timer(
            cls._cooldown_seconds,
            cls._unload_if_idle,
        )
        cls._cooldown_timer.daemon = True
        cls._cooldown_timer.start()

    @classmethod
    def _unload_if_idle(cls):
        if cls._pipeline_instance is None:
            return
        if cls._cooldown_deadline is None:
            cls.unload()
            return
        if time.monotonic() >= cls._cooldown_deadline:
            cls.unload()

    @classmethod
    def get_pipeline(cls, model_name: str, device: str = "CPU"):
        """
        Retrieves the raw OpenVINO model pipeline from memory.
        Compiles the IR graph only if it isn't already running.
        """
        print("Checking if model is already loaded")
        if (
            cls._pipeline_instance is not None
            and cls._current_model_name == model_name
            and cls._current_device == device
        ):
            cls._cancel_cooldown_timer()
            return cls._pipeline_instance

        if cls._pipeline_instance is not None:
            print(f"Unloading '{cls._current_model_name}'...")
            cls.unload()

        script_path = os.path.realpath(__file__)
        script_dir = os.path.dirname(script_path)
        grandparent_dir = os.path.dirname(script_dir)
        models_root = os.path.join(grandparent_dir, "models")
        model_dir = os.path.join(models_root, model_name)

        required_xml = os.path.join(model_dir, "openvino_model.xml")
        if not os.path.exists(required_xml):
            print(f"Error: Compiled OpenVINO IR model not found at: {model_dir}")
            sys.exit(1)

        print(f"Loading model '{model_name}' onto target device: {device}...")

        try:
            cls._pipeline_instance = ov_genai.LLMPipeline(model_dir, device)
            cls._current_model_name = model_name
            cls._current_device = device
            cls._cancel_cooldown_timer()
            return cls._pipeline_instance

        except Exception as e:
            print(f"Native OpenVINO Compilation Failed: {e}")
            sys.exit(1)

    @classmethod
    def mark_idle(cls):
        """Start the grace period after a chat session ends."""
        if cls._pipeline_instance is None:
            return False

        print(f"Model '{cls._current_model_name}' will stay warm for {cls._cooldown_seconds} seconds.")
        cls._schedule_unload_after_cooldown()
        return True

    @classmethod
    def wait_for_idle_timeout(cls, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cls._pipeline_instance is None and cls._current_model_name is None:
                return True
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        return cls._pipeline_instance is None and cls._current_model_name is None

    @classmethod
    def unload(cls):
        """Manually flushes the active model out of RAM/VRAM."""
        cls._cancel_cooldown_timer()
        if cls._pipeline_instance is not None:
            print(f"Unloading '{cls._current_model_name}'...")
            cls._pipeline_instance = None
            cls._current_model_name = None
            cls._current_device = None
            return True
        cls._current_model_name = None
        cls._current_device = None
        return False
