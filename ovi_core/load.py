# ovi_core/load.py
import os
import sys
import openvino_genai as ov_genai


class OviEngine:
    # Persistent global memory variables
    _current_model_name = None
    _current_device = None
    _pipeline_instance = None
    
    @classmethod
    def get_pipeline(cls, model_name: str, device: str = "CPU"):
        """
        Retrieves the raw OpenVINO model pipeline from memory and compiles the IR graph.
        """
        
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
            return cls._pipeline_instance

        except Exception as e:
            print(f"Native OpenVINO Compilation Failed: {e}")
            sys.exit(1)
