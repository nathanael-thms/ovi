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
    assert params.get("not_a_param") is None


def test_get_parameters_from_modelfile_alias_override_stress(tmp_path, monkeypatch):
    mf = tmp_path / "Modelfile"
    write_modelfile(mf, '''
    # temperature + alias overrides
    PARAMETER temperature 0.1
    PARAMETER temp 0.2
    PARAMETER temperature 0.3
    PARAMETER temp 0.4

    # top-k overrides
    PARAMETER top_k 10
    PARAMETER top_k 20
    PARAMETER top_k 30

    # top-p tests
    PARAMETER top_p 0.5
    PARAMETER top_p 0.6

    # beam aliases
    PARAMETER num_beams 2
    PARAMETER beam_width 3
    PARAMETER beam_size 4

    # presence/frequency aliases
    PARAMETER presence_penalty 0.01
    PARAMETER presence 0.02
    PARAMETER frequency_penalty 0.11
    PARAMETER frequency 0.22

    # stop strings
    PARAMETER stop_strings "END"
    PARAMETER stop_sequence "STOP"
    PARAMETER stop_string "HALT"

    # stop token ids merging
    PARAMETER stop_token_ids 5,6,7
    PARAMETER stop_token 8
    PARAMETER stop_tokens 9,10

    # booleans
    PARAMETER ignore_eos yes
    PARAMETER echo on

    # numerics
    PARAMETER max_length 4096
    PARAMETER min_new_tokens 12
    PARAMETER max_ngram_size 6
    PARAMETER min_p 0.15
    PARAMETER diversity_penalty 0.25
    PARAMETER length_penalty 0.75
    PARAMETER logprobs 7

    # malformed/unknown
    PARAMATER test 4056 tye
    PARAMETER no_value
    PARAMETER top_k 10 20
    PARAMETER unknown_param 123
    ''')

    monkeypatch.setattr(parse_modelfile, "get_model_file_path", lambda model_name: str(mf))
    params = parse_modelfile.get_parameters_from_modelfile("unused")

    assert params["temperature"] == 0.4
    assert params["top_k"] == 30
    assert params["top_p"] == 0.6
    assert params["num_beams"] == 4
    assert params["presence_penalty"] == 0.02
    assert params["frequency_penalty"] == 0.22
    assert params["stop_strings"] == "HALT"
    assert params["stop_token_ids"] == [5, 6, 7, 8, 9, 10]
    assert params["ignore_eos"] is True
    assert params["echo"] is True
    assert params["max_length"] == 4096
    assert params["min_new_tokens"] == 12
    assert params["max_ngram_size"] == 6
    assert params["min_p"] == 0.15
    assert params["diversity_penalty"] == 0.25
    assert params["length_penalty"] == 0.75
    assert params["logprobs"] == 7

    # malformed/unknown ignored
    assert "unknown_param" not in params
    assert "test" not in params
    assert params["top_k"] == 30  # malformed override ignored
