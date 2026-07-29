import sys
import types

import pytest


@pytest.fixture(autouse=True)
def fake_openvino(monkeypatch):
    class FakePipeline:
        instances = []

        def __init__(self, model_dir, device):
            self.model_dir = model_dir
            self.device = device
            self.events = []
            type(self).instances.append(self)

        def start_chat(self):
            self.events.append("start")

        def generate(self, user_input, streamer):
            self.events.append(("generate", user_input))
            return streamer("token")

        def finish_chat(self):
            self.events.append("finish")

    monkeypatch.setitem(
        sys.modules,
        "openvino_genai",
        types.SimpleNamespace(LLMPipeline=FakePipeline),
    )
    return FakePipeline
