import importlib
import sys
from types import SimpleNamespace
import pytest


# Inheriting from list allows MockChatHistory to natively support .append()
class MockChatHistory(list):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def get_history(self):
        return self


def _load_chat_module(monkeypatch):
    # Force unload modules to ensure a clean test state
    sys.modules.pop("ovi_core.chat", None)
    sys.modules.pop("ovi_core.load", None)
    sys.modules.pop("ovi_core.parse_modelfile", None)

    # Intercept system modules before importlib runs
    for mod_name in ["openvino_genai", "openvino"]:
        if mod_name in sys.modules:
            setattr(sys.modules[mod_name], "ChatHistory", MockChatHistory)
        else:
            mock_mod = SimpleNamespace(ChatHistory=MockChatHistory)
            monkeypatch.setitem(sys.modules, mod_name, mock_mod)

    return importlib.import_module("ovi_core.chat")


def test_stream_callback_prints_tokens_and_returns_false(capsys, monkeypatch):
    chat_module = _load_chat_module(monkeypatch)

    assert chat_module._stream_callback("token") is False

    captured = capsys.readouterr()
    assert captured.out == "token"


def test_record_history_entry_records_non_empty_entries(monkeypatch):
    chat_module = _load_chat_module(monkeypatch)
    recorded = []
    fake_readline = SimpleNamespace(add_history=lambda entry: recorded.append(entry))

    monkeypatch.setattr(chat_module, "readline", fake_readline)

    chat_module._record_history_entry("hello")
    chat_module._record_history_entry("   ")

    assert recorded == ["hello"]


def test_start_chat_loop_starts_pipeline_and_finishes_cleanly(monkeypatch, fake_openvino):
    chat_module = _load_chat_module(monkeypatch)

    class FakePipeline:
        def __init__(self):
            self.events = []

        def generate(self, history_input, streamer, **kwargs):
            # Capture the structured history object passed to the model
            self.events.append(("generate", history_input))
            streamer("token")

    fake_pipeline = FakePipeline()

    # Explicitly guarantee ChatHistory exists inside the active chat_module namespace
    monkeypatch.setattr(chat_module, "ChatHistory", MockChatHistory, raising=False)

    import ovi_core.parse_modelfile
    monkeypatch.setattr(ovi_core.parse_modelfile, "get_device_from_modelfile", lambda model_name: "CPU")
    monkeypatch.setattr(ovi_core.parse_modelfile, "get_system_prompt_from_modelfile",
                        lambda model_name: "System Prompt")
    monkeypatch.setattr(ovi_core.parse_modelfile, "get_parameters_from_modelfile",
                        lambda model_name: {"temperature": 0.7})

    # Mock the engine pipeline retrieval to yield the formatted wrapper dictionary
    monkeypatch.setattr(chat_module.OviEngine, "get_pipeline", lambda model_name, device="CPU": {
        "pipeline": fake_pipeline
    })

    monkeypatch.setattr(chat_module, "_configure_readline", lambda: None)
    monkeypatch.setattr(chat_module, "_record_history_entry", lambda entry: None)

    inputs = iter(["hello", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    chat_module.start_chat_loop("demo-model")

    # Match the updated execution metrics exactly
    assert fake_pipeline.events == [("generate", [{"role": "user", "content": "hello"}])]
