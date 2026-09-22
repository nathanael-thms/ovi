import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path


def _load_cli_module():
    for module_name in ("ovi", "ovi_core.chat", "ovi_core.load", "ovi_core.path"):
        sys.modules.pop(module_name, None)

    module_path = Path(__file__).resolve().parents[1] / "ovi"
    loader = importlib.machinery.SourceFileLoader("ovi", str(module_path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["ovi"] = module
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


def test_main_menu_and_option_run_paths(monkeypatch):
    ovi_module = _load_cli_module()
    monkeypatch.setattr(sys, "argv", ["ovi"])
    monkeypatch.setattr(ovi_module.curses, "wrapper", lambda fn: "Launch a model")
    monkeypatch.setattr(ovi_module, "option_run", lambda: None)

    assert ovi_module.main() is None

    monkeypatch.setattr(ovi_module.curses, "wrapper", lambda fn: "Exit")
    with __import__("pytest").raises(SystemExit):
        ovi_module.main()


def test_main_menu_and_model_menu_navigation(monkeypatch):
    ovi_module = _load_cli_module()

    class FakeWindow:
        def __init__(self, keys):
            self.keys = iter(keys)
            self.calls = []

        def keypad(self, enabled):
            self.enabled = enabled

        def erase(self):
            pass

        def addstr(self, *args, **kwargs):
            pass

        def refresh(self):
            pass

        def getch(self):
            return next(self.keys)

        def getmaxyx(self):
            return (25, 80)

    monkeypatch.setattr(ovi_module, "draw_menu", lambda stdscr, selected_index, options: None)
    fake = FakeWindow([ovi_module.curses.KEY_DOWN, ovi_module.curses.KEY_ENTER])
    assert ovi_module.main_menu(fake) == "Exit"

    fake = FakeWindow([ovi_module.curses.KEY_UP, ovi_module.curses.KEY_ENTER])
    assert ovi_module.main_menu(fake) == "Exit"

    fake = FakeWindow([ovi_module.curses.KEY_LEFT])
    assert ovi_module.main_menu(fake) == "Back"

    fake = FakeWindow([27])
    with __import__("pytest").raises(SystemExit):
        ovi_module.main_menu(fake)

    monkeypatch.setattr(ovi_module, "models_dir", "/tmp/ovi-models")
    monkeypatch.setattr(ovi_module.os, "listdir", lambda path: ["alpha", "beta"])
    monkeypatch.setattr(ovi_module.os.path, "isdir", lambda path: True)
    fake = FakeWindow([ovi_module.curses.KEY_UP, ovi_module.curses.KEY_ENTER])
    assert ovi_module.model_menu(fake) == "beta"

    fake = FakeWindow([ovi_module.curses.KEY_LEFT])
    assert ovi_module.model_menu(fake) == "Back"


def test_model_menu_returns_exit_for_empty_directory(monkeypatch):
    ovi_module = _load_cli_module()

    class FakeWindow:
        def keypad(self, enabled):
            self.enabled = enabled

        def erase(self):
            pass

        def addstr(self, *args, **kwargs):
            pass

        def refresh(self):
            pass

        def getch(self):
            return 0

    fake_window = FakeWindow()
    monkeypatch.setattr(ovi_module.os, "listdir", lambda path: [])

    assert ovi_module.model_menu(fake_window) == "Exit"


def test_main_handles_add_model_flag_and_help(monkeypatch):
    ovi_module = _load_cli_module()
    called = {}

    monkeypatch.setattr(ovi_module, "add_model", lambda: called.setdefault("add_model", True))
    monkeypatch.setattr(sys, "argv", ["ovi", "--add-model"])
    ovi_module.main()
    assert called == {"add_model": True}

    monkeypatch.setattr(sys, "argv", ["ovi", "help"])
    with __import__("pytest").raises(SystemExit):
        ovi_module.main()

    monkeypatch.setattr(sys, "argv", ["ovi", "--is-ovi-install"])
    assert ovi_module.main() is None


def test_option_run_launches_selected_model(monkeypatch):
    ovi_module = _load_cli_module()
    calls = {}

    monkeypatch.setattr(ovi_module.curses, "wrapper", lambda fn: "demo-model")
    monkeypatch.setattr(ovi_module.chat, "start_chat_loop", lambda model_name: calls.setdefault("model_name", model_name))
    ovi_module.option_run()
    assert calls == {"model_name": "demo-model"}

    monkeypatch.setattr(ovi_module.curses, "wrapper", lambda fn: "Back")
    ovi_module.option_run()

    monkeypatch.setattr(ovi_module.curses, "wrapper", lambda fn: "Exit")
    ovi_module.option_run()


def test_main_handles_keyboard_interrupt(monkeypatch):
    ovi_module = _load_cli_module()
    monkeypatch.setattr(sys, "argv", ["ovi"])
    monkeypatch.setattr(ovi_module.curses, "wrapper", lambda fn: (_ for _ in ()).throw(KeyboardInterrupt()))
    with __import__("pytest").raises(SystemExit):
        ovi_module.main()


def test_ovi_script_entrypoint_runs_main(monkeypatch):
    import runpy
    script_path = Path(__file__).resolve().parents[1] / "ovi"
    monkeypatch.setattr(sys, "argv", ["ovi", "--is-ovi-install"])
    runpy.run_path(str(script_path), run_name="__main__")


def test_ovi_script_relaunches_under_missing_venv(monkeypatch):
    import runpy
    script_path = Path(__file__).resolve().parents[1] / "ovi"
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setattr(sys, "base_prefix", sys.base_prefix)
    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr("os.execv", lambda *args: (_ for _ in ()).throw(SystemExit(0)))
    with __import__("pytest").raises(SystemExit):
        runpy.run_path(str(script_path), run_name="__main__")
