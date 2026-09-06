import os
import textwrap

import pytest

from ovi_core import parse_modelfile


def write_modelfile(path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))


def test_get_device_from_modelfile_returns_specified_device(tmp_path, monkeypatch):
    mf = tmp_path / "Modelfile"
    write_modelfile(mf, '''
    DEVICE GPU
    ''')

    monkeypatch.setattr(parse_modelfile, "get_model_file_path", lambda model_name: str(mf))

    assert parse_modelfile.get_device_from_modelfile("unused") == "GPU"


def test_get_device_from_modelfile_defaults_to_cpu_on_missing_or_invalid(tmp_path, monkeypatch, capsys):
    # missing file
    missing = tmp_path / "nope" / "Modelfile"
    monkeypatch.setattr(parse_modelfile, "get_model_file_path", lambda model_name: str(missing))
    assert parse_modelfile.get_device_from_modelfile("m") == "CPU"

    # invalid device value -> default CPU
    mf = tmp_path / "Modelfile"
    write_modelfile(mf, '''
    DEVICE UNKNOWN
    ''')
    monkeypatch.setattr(parse_modelfile, "get_model_file_path", lambda model_name: str(mf))
    assert parse_modelfile.get_device_from_modelfile("m") == "CPU"


def test_get_parameters_from_modelfile_parses_values_and_aliases_and_lists(tmp_path, monkeypatch):
    mf = tmp_path / "Modelfile"
    write_modelfile(mf, '''
    # basic numeric types
    PARAMETER max_new_tokens 128
    PARAMETER temp "0.5"   # alias for temperature
    PARAMETER top_p 0.9

    # booleans (various true forms)
    PARAMETER ignore_end_of_sequence true
    PARAMETER echo yes

    # stop token ids from multiple lines and aliases
    PARAMETER stop_tokens "1, 2"
    PARAMETER stop_token 3

    # stop strings
    PARAMETER stop_sequence "###"

    # invalid value should be ignored
    PARAMETER max_new_tokens not_an_int

    # unknown parameter should be ignored
    PARAMETER not_a_param 123
    ''')

    monkeypatch.setattr(parse_modelfile, "get_model_file_path", lambda model_name: str(mf))

    params = parse_modelfile.get_parameters_from_modelfile("unused")

    # numeric and alias parsing
    assert params.get("max_new_tokens") == 128
    assert pytest.approx(params.get("temperature")) == 0.5
    assert pytest.approx(params.get("top_p")) == 0.9

    # booleans
    assert params.get("ignore_eos") is True
    assert params.get("echo") is True

    # stop token ids should combine and be ints
    assert params.get("stop_token_ids") == [1, 2, 3]

    # stop strings alias
    assert params.get("stop_strings") == "###"

    # invalid/unknown entries were ignored
    # the invalid max_new_tokens line should have been skipped, original valid line remains
    assert params.get("not_a_param") is None
