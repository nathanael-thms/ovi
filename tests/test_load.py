import importlib
import sys

import pytest


def _load_load_module():
    sys.modules.pop("ovi_core.load", None)
    return importlib.import_module("ovi_core.load")


def test_get_pipeline_loads_model_and_caches_pipeline(monkeypatch, tmp_path, fake_openvino):
    load_module = _load_load_module()
    model_dir = tmp_path / "demo-model"
    model_dir.mkdir()
    (model_dir / "openvino_model.xml").write_text("<model/>")

    monkeypatch.setattr(load_module, "get_model_path", lambda model_name: str(model_dir))
    load_module.OviEngine._pipeline_instance = None

    pipeline = load_module.OviEngine.get_pipeline("demo-model", device="GPU")

    assert isinstance(pipeline, fake_openvino)
    assert pipeline.model_dir == str(model_dir)
    assert pipeline.device == "GPU"
    assert load_module.OviEngine._pipeline_instance is pipeline


def test_get_pipeline_exits_when_openvino_model_is_missing(monkeypatch, tmp_path, fake_openvino):
    load_module = _load_load_module()
    model_dir = tmp_path / "demo-model"
    model_dir.mkdir()

    monkeypatch.setattr(load_module, "get_model_path", lambda model_name: str(model_dir))
    load_module.OviEngine._pipeline_instance = None

    with pytest.raises(SystemExit) as excinfo:
        load_module.OviEngine.get_pipeline("demo-model", device="CPU")

    assert excinfo.value.code == 1
