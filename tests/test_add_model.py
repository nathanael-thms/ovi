import importlib
import sys


def _load_add_model_module():
    sys.modules.pop("ovi_core.add_model", None)
    return importlib.import_module("ovi_core.add_model")


def test_add_model_dispatches_local_flag(monkeypatch):
    module = _load_add_model_module()
    calls = {}

    monkeypatch.setattr(module, "add_model_from_local_directory", lambda: calls.setdefault("method", "local"))
    monkeypatch.setattr(module, "add_model_from_hf_hub", lambda: calls.setdefault("method", "hf"))
    monkeypatch.setattr(sys, "argv", ["ovi", "--local"])

    module.add_model()

    assert calls == {"method": "local"}


def test_add_model_from_local_directory_copies_valid_model(monkeypatch, tmp_path):
    module = _load_add_model_module()
    source_dir = tmp_path / "source-model"
    source_dir.mkdir()
    (source_dir / module.MODEL_FILE_NAME).write_text("<model />")

    target_root = tmp_path / "models"
    monkeypatch.setattr(module, "get_models_root", lambda: str(target_root))
    responses = iter([str(source_dir), "custom-model"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))

    module.add_model_from_local_directory()

    copied_dir = target_root / "custom-model"
    assert copied_dir.is_dir()
    assert (copied_dir / module.MODEL_FILE_NAME).read_text() == "<model />"


def test_add_model_from_hf_hub_uses_default_model_name(monkeypatch, tmp_path):
    module = _load_add_model_module()
    target_root = tmp_path / "models"
    target_root.mkdir()

    class FakeModel:
        id = "OpenVINO/qwen2.5-7b"

    fake_api = type("FakeAPI", (), {"list_models": lambda self, author=None: [FakeModel()]})()
    monkeypatch.setattr(module, "HfApi", lambda: fake_api)
    monkeypatch.setattr(module, "prompt", lambda *args, **kwargs: "OpenVINO/qwen2.5-7b")
    monkeypatch.setattr(module, "get_models_root", lambda: str(target_root))

    called = {}

    def fake_run(command, check=True):
        called["command"] = command
        return None

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr("builtins.input", lambda prompt="": "custom-name")

    module.add_model_from_hf_hub()

    assert called["command"] == ["hf", "download", "OpenVINO/qwen2.5-7b", "--local-dir", str(target_root / "custom-name")]
    assert (target_root / "custom-name").is_dir()
