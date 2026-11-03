import subprocess

import pytest

from ovi_core import modelfile_editor


def _configure_model_file(monkeypatch, path):
    monkeypatch.setattr(modelfile_editor, "get_model_file_path", lambda name: str(path))


def test_editor_creates_documented_template_and_uses_visual(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    modelfile_path = model_dir / "Modelfile"
    _configure_model_file(monkeypatch, modelfile_path)
    monkeypatch.setenv("VISUAL", "visual-editor")
    monkeypatch.setenv("EDITOR", "fallback-editor")
    commands = []
    monkeypatch.setattr(
        modelfile_editor.subprocess,
        "run",
        lambda command, check: commands.append((command, check)),
    )

    modelfile_editor.open_modelfile_editor("model")

    contents = modelfile_path.read_text(encoding="utf-8")
    assert contents.startswith("# ovi Modelfile\n")
    assert "# DEVICE CPU" in contents
    assert '# SYSTEM "You are a helpful coding assistant."' in contents
    assert "# PARAMETER stop_token_ids 12,13,99" in contents
    assert contents.endswith("\n")
    assert commands == [(["visual-editor", str(modelfile_path.resolve())], True)]


def test_editor_preserves_existing_modelfile_and_uses_editor(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    modelfile_path = model_dir / "Modelfile"
    modelfile_path.write_text("DEVICE GPU\n", encoding="utf-8")
    _configure_model_file(monkeypatch, modelfile_path)
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setenv("EDITOR", "custom-editor")
    commands = []
    monkeypatch.setattr(
        modelfile_editor.subprocess,
        "run",
        lambda command, check: commands.append(command),
    )

    modelfile_editor.open_modelfile_editor("model")

    assert modelfile_path.read_text(encoding="utf-8") == "DEVICE GPU\n"
    assert commands == [["custom-editor", str(modelfile_path.resolve())]]


def test_editor_defaults_to_nano(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _configure_model_file(monkeypatch, model_dir / "Modelfile")
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    commands = []
    monkeypatch.setattr(
        modelfile_editor.subprocess,
        "run",
        lambda command, check: commands.append(command),
    )

    modelfile_editor.open_modelfile_editor("model")

    assert commands[0][0] == "nano"


def test_editor_uses_vi_when_preferred_editor_is_missing(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    modelfile_path = model_dir / "Modelfile"
    _configure_model_file(monkeypatch, modelfile_path)
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setenv("EDITOR", "unavailable-editor")
    commands = []

    def run(command, check):
        commands.append(command)
        if command[0] == "unavailable-editor":
            raise FileNotFoundError("editor not found")

    monkeypatch.setattr(modelfile_editor.subprocess, "run", run)

    modelfile_editor.open_modelfile_editor("model")

    assert commands == [
        ["unavailable-editor", str(modelfile_path.resolve())],
        ["vi", str(modelfile_path.resolve())],
    ]


def test_editor_reports_missing_model_directory(monkeypatch, tmp_path, capsys):
    modelfile_path = tmp_path / "missing" / "Modelfile"
    _configure_model_file(monkeypatch, modelfile_path)
    commands = []
    monkeypatch.setattr(modelfile_editor.subprocess, "run", lambda *args, **kwargs: commands.append(args))

    modelfile_editor.open_modelfile_editor("missing")

    assert not modelfile_path.exists()
    assert commands == []
    assert "Model Not Found" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("failed_editor", "expected_message"),
    [
        ("initial-editor", "Editor exited with error"),
        ("vi", "Failed to open editor"),
    ],
)
def test_editor_reports_editor_failures(monkeypatch, tmp_path, failed_editor, expected_message):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    modelfile_path = model_dir / "Modelfile"
    _configure_model_file(monkeypatch, modelfile_path)
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setenv("EDITOR", "initial-editor")

    def run(command, check):
        if command[0] == "initial-editor" and failed_editor == "vi":
            raise FileNotFoundError("initial editor not found")
        if command[0] == failed_editor:
            raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(modelfile_editor.subprocess, "run", run)

    with pytest.raises(SystemExit, match=expected_message):
        modelfile_editor.open_modelfile_editor("model")
