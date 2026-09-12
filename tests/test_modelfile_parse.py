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


def test_get_device_from_modelfile_defaults_to_cpu_on_missing_or_invalid(tmp_path, monkeypatch):
    missing = tmp_path / "nope" / "Modelfile"
    monkeypatch.setattr(parse_modelfile, "get_model_file_path", lambda model_name: str(missing))
    assert parse_modelfile.get_device_from_modelfile("m") == "CPU"

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
    PARAMETER temp "0.5"
    PARAMETER top_p 0.9

    # booleans
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

    assert params["max_new_tokens"] == 128
    assert params["temperature"] == 0.5
    assert params["top_p"] == 0.9
    assert params["ignore_eos"] is True
    assert params["echo"] is True
    assert params["stop_token_ids"] == {1, 2, 3}
    assert params["stop_strings"] == "###"
    assert params.get("not_a_param") is None


def test_get_parameters_from_modelfile_alias_override_stress(tmp_path, monkeypatch):
    mf = tmp_path / "Modelfile"
    write_modelfile(mf, '''
    PARAMETER temperature 0.1
    PARAMETER temp 0.2
    PARAMETER temperature 0.3
    PARAMETER temp 1.0

    PARAMETER num_beams 2
    PARAMETER beam_width 3
    PARAMETER beam_size 4

    PARAMETER stop_strings "END"
    PARAMETER stop_sequence "STOP"
    PARAMETER stop_string "HALT"

    PARAMETER stop_token_ids 5,6,7
    PARAMETER stop_token 8
    PARAMETER stop_tokens 9,10

    PARAMETER ignore_eos yes
    PARAMETER echo on

    PARAMETER max_length 4096
    PARAMETER min_new_tokens 12
    PARAMETER max_ngram_size 0
    PARAMETER length_penalty 0.75
    PARAMETER logprobs 7
    ''')

    monkeypatch.setattr(parse_modelfile, "get_model_file_path", lambda model_name: str(mf))
    params = parse_modelfile.get_parameters_from_modelfile("unused")

    assert params["temperature"] == 1.0
    assert params["num_beams"] == 4
    assert params["stop_strings"] == "HALT"
    assert params["stop_token_ids"] == {5, 6, 7, 8, 9, 10}
    assert params["ignore_eos"] is True
    assert params["echo"] is True
    assert params["max_length"] == 4096
    assert params["min_new_tokens"] == 12
    assert params["max_ngram_size"] == 0
    assert params["length_penalty"] == 0.75
    assert params["logprobs"] == 7


@pytest.mark.parametrize(
    "content",
    [
        "PARAMETER num_beams 0\n",
        "PARAMETER length_penalty -0.1\n",
        "PARAMETER logprobs 11\n",
        "PARAMETER max_length 5\nPARAMETER min_new_tokens 10\n",
        "PARAMETER max_length 12\nPARAMETER max_new_tokens 20\n",
    ],
)
def test_get_parameters_from_modelfile_exits_on_invalid_values(tmp_path, monkeypatch, content):
    mf = tmp_path / "Modelfile"
    write_modelfile(mf, content)
    monkeypatch.setattr(parse_modelfile, "get_model_file_path", lambda model_name: str(mf))

    with pytest.raises(SystemExit) as excinfo:
        parse_modelfile.get_parameters_from_modelfile("unused")

    assert excinfo.value.code == 1
