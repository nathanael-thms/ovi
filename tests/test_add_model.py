import curses
import importlib
import subprocess
import sys

import pytest

from ovi_core.menu import draw_menu


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


def test_add_model_dispatches_hf_flag(monkeypatch):
    module = _load_add_model_module()
    calls = {}

    monkeypatch.setattr(module, "add_model_from_local_directory", lambda: calls.setdefault("method", "local"))
    monkeypatch.setattr(module, "add_model_from_hf_hub", lambda: calls.setdefault("method", "hf"))
    monkeypatch.setattr(sys, "argv", ["ovi", "--hf"])

    module.add_model()

    assert calls == {"method": "hf"}


def test_add_model_uses_menu_selection(monkeypatch):
    module = _load_add_model_module()
    calls = {}

    monkeypatch.setattr(module.curses, "wrapper", lambda fn: "Add a model from Hugging Face Hub")
    monkeypatch.setattr(module, "add_model_from_local_directory", lambda: calls.setdefault("local", True))
    monkeypatch.setattr(module, "add_model_from_hf_hub", lambda: calls.setdefault("hf", True))
    monkeypatch.setattr(sys, "argv", ["ovi"])

    module.add_model()

    assert calls == {"hf": True}


def test_method_menu_handles_navigation_and_exit(monkeypatch):
    module = _load_add_model_module()

    class FakeWindow:
        def keypad(self, enabled):
            self.enabled = enabled

        def getch(self):
            return next(self.keys)

    fake = FakeWindow()
    fake.keys = iter([module.curses.KEY_DOWN, module.curses.KEY_ENTER, module.curses.KEY_LEFT, 27])
    monkeypatch.setattr(module, "draw_menu", lambda stdscr, current_row, options: None)

    assert module.method_menu(fake) == "Add a model from Hugging Face Hub"

    fake.keys = iter([module.curses.KEY_UP, module.curses.KEY_ENTER])
    assert module.method_menu(fake) == "Exit"

    fake.keys = iter([module.curses.KEY_LEFT])
    assert module.method_menu(fake) == "Back"

    fake.keys = iter([27])
    with pytest.raises(SystemExit):
        module.method_menu(fake)


def test_add_model_from_local_directory_rejects_invalid_and_existing_inputs(monkeypatch, tmp_path):
    module = _load_add_model_module()

    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    with pytest.raises(SystemExit):
        module.add_model_from_local_directory()

    valid_model_dir = tmp_path / "source-model"
    valid_model_dir.mkdir()
    (valid_model_dir / module.MODEL_FILE_NAME).write_text("<model />")
    target_root = tmp_path / "models"
    target_root.mkdir()
    existing_copy = target_root / "duplicate"
    existing_copy.mkdir()

    monkeypatch.setattr(module, "get_models_root", lambda: str(target_root))
    responses = iter([str(valid_model_dir), "duplicate"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
    with pytest.raises(SystemExit):
        module.add_model_from_local_directory()

    responses = iter([str(valid_model_dir), "new-model"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
    module.add_model_from_local_directory()
    assert (target_root / "new-model" / module.MODEL_FILE_NAME).read_text() == "<model />"


def test_add_model_from_local_directory_aborts_for_missing_model_file(monkeypatch, tmp_path):
    module = _load_add_model_module()
    source_dir = tmp_path / "source-model"
    source_dir.mkdir()
    target_root = tmp_path / "models"
    target_root.mkdir()

    monkeypatch.setattr(module, "get_models_root", lambda: str(target_root))
    responses = iter([str(source_dir), "n", "ignored"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))

    with pytest.raises(SystemExit):
        module.add_model_from_local_directory()

    responses = iter([str(source_dir), "y", "custom-name"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
    module.add_model_from_local_directory()
    assert (target_root / "custom-name").is_dir()

    responses = iter([str(source_dir), "custom-name"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
    monkeypatch.setattr(module.shutil, "copytree", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("copy failed")))
    with pytest.raises(SystemExit):
        module.add_model_from_local_directory()

    monkeypatch.setattr(sys, "argv", ["ovi"])
    monkeypatch.setattr(module.curses, "wrapper", lambda fn: "Exit")
    with pytest.raises(SystemExit):
        module.add_model()


def test_add_model_from_hf_hub_handles_empty_selection_and_download_success(monkeypatch, tmp_path):
    module = _load_add_model_module()
    target_root = tmp_path / "models"
    target_root.mkdir()

    class FakeModel:
        id = "OpenVINO/qwen2.5-7b"

    fake_api = type("FakeAPI", (), {"list_models": lambda self, author=None: []})()
    monkeypatch.setattr(module, "HfApi", lambda: fake_api)
    monkeypatch.setattr(module, "get_models_root", lambda: str(target_root))

    module.add_model_from_hf_hub()

    fake_api = type("FakeAPI", (), {"list_models": lambda self, author=None: [FakeModel()]})()
    monkeypatch.setattr(module, "HfApi", lambda: fake_api)
    monkeypatch.setattr(module, "prompt", lambda *args, **kwargs: "")
    monkeypatch.setattr("builtins.input", lambda prompt="": "custom-name")

    module.add_model_from_hf_hub()

    called = {}

    def fake_run(command, check=True):
        called["command"] = command
        return None

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "prompt", lambda *args, **kwargs: "OpenVINO/qwen2.5-7b")
    monkeypatch.setattr("builtins.input", lambda prompt="": "custom-name")

    module.add_model_from_hf_hub()
    assert called["command"] == ["hf", "download", "OpenVINO/qwen2.5-7b", "--local-dir", str(target_root / "custom-name")]
    assert (target_root / "custom-name").is_dir()


def test_add_model_from_hf_hub_fails_on_invalid_and_download_errors(monkeypatch, tmp_path):
    module = _load_add_model_module()
    target_root = tmp_path / "models"
    target_root.mkdir()

    class FakeModel:
        id = "OpenVINO/qwen2.5-7b"

    fake_api = type("FakeAPI", (), {"list_models": lambda self, author=None: [FakeModel()]})()
    monkeypatch.setattr(module, "HfApi", lambda: fake_api)
    monkeypatch.setattr(module, "prompt", lambda *args, **kwargs: "not-a-real-model")
    monkeypatch.setattr(module, "get_models_root", lambda: str(target_root))

    module.add_model_from_hf_hub()

    monkeypatch.setattr(module, "prompt", lambda *args, **kwargs: "OpenVINO/qwen2.5-7b")
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    with pytest.raises(SystemExit):
        module.add_model_from_hf_hub()

    monkeypatch.setattr(module, "prompt", lambda *args, **kwargs: "OpenVINO/qwen2.5-7b")
    monkeypatch.setattr("builtins.input", lambda prompt="": "existing-name")
    (target_root / "existing-name").mkdir()
    (target_root / "existing-name" / "filled").write_text("x")
    with pytest.raises(SystemExit):
        module.add_model_from_hf_hub()

    class ProblemAPI:
        def list_models(self, author=None):
            raise RuntimeError("hf failed")

    monkeypatch.setattr(module, "HfApi", lambda: ProblemAPI())
    with pytest.raises(SystemExit):
        module.add_model_from_hf_hub()

    def fake_run(command, check=True):
        raise subprocess.CalledProcessError(9, command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "prompt", lambda *args, **kwargs: "OpenVINO/qwen2.5-7b")
    monkeypatch.setattr("builtins.input", lambda prompt="": "good-name")
    with pytest.raises(SystemExit):
        module.add_model_from_hf_hub()


def test_draw_menu_handles_small_and_standard_terminal(monkeypatch):
    class FakeWindow:
        def __init__(self, maxyx, fail_on_addstr=False):
            self.maxyx = maxyx
            self.calls = []
            self.fail_on_addstr = fail_on_addstr

        def erase(self):
            self.calls.append("erase")

        def addstr(self, *args, **kwargs):
            if self.fail_on_addstr:
                self.fail_on_addstr = False
                raise curses.error("boom")
            self.calls.append((args, kwargs))

        def refresh(self):
            self.calls.append("refresh")

        def getmaxyx(self):
            return self.maxyx

    fake_small = FakeWindow((5, 10))
    monkeypatch.setattr("ovi_core.menu.curses.curs_set", lambda *args, **kwargs: None)
    monkeypatch.setattr("ovi_core.menu.curses.has_colors", lambda: True)
    monkeypatch.setattr("ovi_core.menu.curses.start_color", lambda: None)
    monkeypatch.setattr("ovi_core.menu.curses.use_default_colors", lambda: None)
    monkeypatch.setattr("ovi_core.menu.curses.init_pair", lambda *args, **kwargs: None)
    monkeypatch.setattr("ovi_core.menu.curses.color_pair", lambda value: 0)

    draw_menu(fake_small, 0, ["One", "Two"])
    assert fake_small.calls

    fake_large = FakeWindow((25, 80))
    draw_menu(fake_large, 1, ["One", "Two"])
    assert fake_large.calls

    fake_error = FakeWindow((25, 80), fail_on_addstr=True)
    draw_menu(fake_error, 0, ["One"])
    assert fake_error.calls


def test_safe_fuzzy_completer_matches_substrings():
    module = _load_add_model_module()
    completer = module.SafeFuzzyCompleter(["OpenVINO/qwen2.5-7b", "OpenVINO/phi-2"])
    completions = list(completer.get_completions(type("Doc", (), {"text_before_cursor": "qwen"})(), None))
    assert completions[0].text == "OpenVINO/qwen2.5-7b"
