# ovi_core/load.py
import os
import sys
import openvino_genai as ov_genai

from ovi_core.path import get_model_path


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

        # Get the model directory path based on the model name
        model_dir = get_model_path(model_name)


        # Ensure it is a true OpenVINO IR model directory by checking for the required XML file
        required_xml = os.path.join(model_dir, "openvino_model.xml")
        if not os.path.exists(required_xml):
            print(f"Error: OpenVINO IR model not found at: {model_dir}")
            sys.exit(1)

        print(f"Loading model '{model_name}' onto target device: {device}...")

        # Load the model into memory and compile the IR graph for the specified device
        try:
            cls._pipeline_instance = ov_genai.LLMPipeline(model_dir, device)
            return cls._pipeline_instance

        except Exception as e:
            print(f"Native OpenVINO Compilation Failed: {e}")
            sys.exit(1)
