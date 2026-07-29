import importlib
import sys
from types import SimpleNamespace


def _load_chat_module():
    sys.modules.pop("ovi_core.chat", None)
    sys.modules.pop("ovi_core.load", None)
    return importlib.import_module("ovi_core.chat")


def test_stream_callback_prints_tokens_and_returns_false(capsys):
    chat_module = _load_chat_module()

    assert chat_module._stream_callback("token") is False

    captured = capsys.readouterr()
    assert captured.out == "token"


def test_record_history_entry_records_non_empty_entries(monkeypatch):
    chat_module = _load_chat_module()
    recorded = []
    fake_readline = SimpleNamespace(add_history=lambda entry: recorded.append(entry))

    monkeypatch.setattr(chat_module, "readline", fake_readline)

    chat_module._record_history_entry("hello")
    chat_module._record_history_entry("   ")

    assert recorded == ["hello"]


def test_start_chat_loop_starts_pipeline_and_finishes_cleanly(monkeypatch, fake_openvino):
    chat_module = _load_chat_module()

    class FakePipeline:
        def __init__(self):
            self.events = []

        def start_chat(self):
            self.events.append("start")

        def generate(self, user_input, streamer):
            self.events.append(("generate", user_input))
            streamer("token")

        def finish_chat(self):
            self.events.append("finish")

    fake_pipeline = FakePipeline()

    monkeypatch.setattr(chat_module.OviEngine, "get_pipeline", lambda model_name, device="CPU": fake_pipeline)
    monkeypatch.setattr(chat_module, "_configure_readline", lambda: None)
    monkeypatch.setattr(chat_module, "_record_history_entry", lambda entry: None)
    inputs = iter(["hello", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    chat_module.start_chat_loop("demo-model")

    assert fake_pipeline.events == ["start", ("generate", "hello"), "finish"]
