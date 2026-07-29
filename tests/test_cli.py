import importlib.machinery
import importlib.util
import sys
from pathlib import Path


def _load_cli_module():
    for module_name in ("ovi_core.chat", "ovi_core.load", "ovi_core.path"):
        sys.modules.pop(module_name, None)

    module_path = Path(__file__).resolve().parents[1] / "ovi"
    loader = importlib.machinery.SourceFileLoader("ovi_cli", str(module_path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_subcommand_launches_chat_loop(monkeypatch, fake_openvino):
    ovi_module = _load_cli_module()
    calls = {}

    def fake_start_chat_loop(model_name, device="CPU"):
        calls["model_name"] = model_name
        calls["device"] = device

    monkeypatch.setattr(ovi_module.chat, "start_chat_loop", fake_start_chat_loop)
    monkeypatch.setattr(sys, "argv", ["ovi", "run", "demo-model"])

    ovi_module.main()

    assert calls["model_name"] == "demo-model"
    assert calls["device"] == "CPU"
