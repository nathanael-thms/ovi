import importlib
import os
import sys
import tempfile
import types
import unittest


class FakePipeline:
    def __init__(self, model_dir, device):
        self.model_dir = model_dir
        self.device = device


class FakeOpenVinoGenAI(types.ModuleType):
    def __init__(self):
        super().__init__("openvino_genai")
        self.pipeline_count = 0

    class LLMPipeline(FakePipeline):
        def __init__(self, model_dir, device):
            super().__init__(model_dir, device)
            self.__class__._counter += 1

    LLMPipeline._counter = 0


class OviEngineWarmCacheTests(unittest.TestCase):
    def setUp(self):
        self.fake_module = FakeOpenVinoGenAI()
        sys.modules["openvino_genai"] = self.fake_module
        sys.modules.pop("ovi_core.load", None)
        self.load_module = importlib.import_module("ovi_core.load")
        self.load_module.OviEngine._cooldown_seconds = 0.1

        self.temp_dir = tempfile.TemporaryDirectory(dir="/tmp", prefix="ovi-test-")
        self.addCleanup(self.temp_dir.cleanup)
        self.repo_root = os.path.dirname(os.path.dirname(__file__))
        self.model_root = os.path.join(self.repo_root, "models", "demo-model")
        os.makedirs(self.model_root, exist_ok=True)
        with open(os.path.join(self.model_root, "openvino_model.xml"), "w", encoding="utf-8") as handle:
            handle.write("<model />")

        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)
        self.addCleanup(os.chdir, self.original_cwd)

    def tearDown(self):
        sys.modules.pop("ovi_core.load", None)
        sys.modules.pop("openvino_genai", None)

    def test_reuses_pipeline_during_cooldown(self):
        first = self.load_module.OviEngine.get_pipeline("demo-model", "CPU")
        self.load_module.OviEngine.mark_idle()

        second = self.load_module.OviEngine.get_pipeline("demo-model", "CPU")

        self.assertIs(second, first)
        self.assertEqual(self.fake_module.LLMPipeline._counter, 1)

    def test_unloads_after_cooldown(self):
        self.load_module.OviEngine.get_pipeline("demo-model", "CPU")
        self.load_module.OviEngine.mark_idle()

        self.assertTrue(self.load_module.OviEngine.wait_for_idle_timeout(timeout=1.0))
        self.assertIsNone(self.load_module.OviEngine._pipeline_instance)
        self.assertIsNone(self.load_module.OviEngine._current_model_name)


if __name__ == "__main__":
    unittest.main()
