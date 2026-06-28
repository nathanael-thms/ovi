# ovi_core/load.py
import os
import sys
import openvino_genai as ov_genai

class OviEngine:
    # Persistent global memory variables
    _current_model_name = None
    _pipeline_instance = None

    @classmethod
    def get_pipeline(cls, model_name: str, device: str = "CPU"):
        """
        Retrieves the raw OpenVINO model pipeline from memory.
        Compiles the IR graph only if it isn't already running.
        """
        # 1. Return immediately if the exact model is already warm in memory
        print("Checking if model is already loaded")
        if cls._pipeline_instance is not None and cls._current_model_name == model_name:
            return cls._pipeline_instance

        # 2. Swap models safely if a different model was previously running
        if cls._pipeline_instance is not None:
            print(f"Unloading '{cls._current_model_name}'...")
            cls._pipeline_instance = None
            cls._current_model_name = None

        # 3. Resolve the path to the native IR folder
        script_path = os.path.realpath(__file__)
        script_dir = os.path.dirname(script_path)
        grandparent_dir = os.path.dirname(script_dir)
        models_root = os.path.join(grandparent_dir, "models")
        model_dir = os.path.join(models_root, model_name)

        # 4. Strict structural verification for raw OpenVINO IR format
        required_xml = os.path.join(model_dir, "openvino_model.xml")
        if not os.path.exists(required_xml):
            print(f"Error: Compiled OpenVINO IR model not found at: {model_dir}")
            sys.exit(1)

        print(f"Loading model '{model_name}' onto target device: {device}...")
        
        try:
            # 5. Compile directly into the targeted hardware runtime backend
            cls._pipeline_instance = ov_genai.LLMPipeline(model_dir, device)
            cls._current_model_name = model_name
            return cls._pipeline_instance
            
        except Exception as e:
            print(f"Native OpenVINO Compilation Failed: {e}")
            sys.exit(1)

    @classmethod
    def unload(cls):
        """Manually flushes the active model out of RAM/VRAM."""
        if cls._pipeline_instance is not None:
            print(f"Unloading '{cls._current_model_name}'...")
            cls._pipeline_instance = None
            cls._current_model_name = None
            return True
        return False
