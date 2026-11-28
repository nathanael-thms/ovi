# Copyright 2026 Nathanael Thomas
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 
# ovi_core/manager.py
import os
from dataclasses import dataclass
import asyncio
import time
import gc

import openvino_genai as ov_genai

from ovi_core.exceptions import ModelNotFoundError, ModelCompilationError
from ovi_core.parse_modelfile import get_device_from_modelfile
from ovi_core.parse_modelfile import get_parameters_from_modelfile
from ovi_core.parse_modelfile import get_system_prompt_from_modelfile
from ovi_core.path import get_model_path

@dataclass
class LoadedModel:
    name: str
    device: str
    pipeline: ov_genai.LLMPipeline
    system_prompt: str
    parameters: dict
    expires_at: float
    in_flight: int = 0


class OviEngine:
    _models: dict[str, LoadedModel] = {}
    _locks: dict[str, asyncio.Lock] = {}

    @classmethod
    def _get_lock(cls, model_name: str) -> asyncio.Lock:
        """Get or create an asyncio lock for a specific model."""
        if model_name not in cls._locks:
            cls._locks[model_name] = asyncio.Lock()
        return cls._locks[model_name]

    @classmethod
    async def acquire(
            cls,
            model_name: str,
            device: str | None = None,
            keep_alive: float = 300.0,
    ) -> LoadedModel:
        resolved_device = cls._resolve_device(model_name, device)

        async with cls._get_lock(model_name):
            entry = cls._models.get(model_name)
            if entry is not None and entry.device != resolved_device:
                cls._unload_unlocked(model_name)
                entry = None
            if entry is None:
                entry = await asyncio.to_thread(
                    cls._load_model_entry, model_name, resolved_device, keep_alive
                )
                cls._models[model_name] = entry

            entry.in_flight += 1
            return entry

    @classmethod
    def release(cls, model_name: str, keep_alive: float = 300.0) -> None:
        entry = cls._models.get(model_name)
        if entry is None:
            return

        entry.in_flight = max(0, entry.in_flight - 1)
        if keep_alive < 0:
            entry.expires_at = float("inf")
        else:
            entry.expires_at = time.monotonic() + keep_alive

    @classmethod
    def _unload_unlocked(cls, model_name: str) -> None:
        if model_name in cls._models:
            del cls._models[model_name]
            gc.collect()

    @classmethod
    async def reaper_loop(cls, interval: float = 5.0) -> None:
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()

            expired = [
                name
                for name, entry in cls._models.items()
                if entry.in_flight == 0 and now >= entry.expires_at
            ]
            for name in expired:
                async with cls._get_lock(name):
                    entry = cls._models.get(name)
                    if entry and entry.in_flight == 0 and time.monotonic() >= entry.expires_at:
                        cls._unload_unlocked(name)

    @classmethod
    def _resolve_device(cls, model_name: str, device: str | None = None) -> str:
        """Resolve the target OpenVINO device from an explicit override or the Modelfile."""
        if device is not None:
            normalized = str(device).upper()
            if normalized in {"CPU", "GPU", "NPU", "AUTO"}:
                return normalized
        return get_device_from_modelfile(model_name)

    @classmethod
    def _load_model_entry(
        cls,
        model_name: str,
        device: str,
        keep_alive: float = 300.0,
    ) -> LoadedModel:
        """Compile the OpenVINO pipeline from disk and return a LoadedModel entry."""
        parameters = get_parameters_from_modelfile(model_name)
        if "stop_token_ids" in parameters and isinstance(parameters["stop_token_ids"], list):
            parameters["stop_token_ids"] = set(parameters["stop_token_ids"])

        system_prompt = get_system_prompt_from_modelfile(model_name)
        model_dir = get_model_path(model_name)

        required_xml = os.path.join(model_dir, "openvino_model.xml")
        if not os.path.exists(required_xml):
            raise ModelNotFoundError(model_name, model_dir)

        print(f"Loading model '{model_name}' onto target device: {device}...")

        try:
            pipeline = ov_genai.LLMPipeline(model_dir, device)
            if hasattr(ov_genai, "GenerationConfig") and hasattr(pipeline, "set_generation_config"):
                config = ov_genai.GenerationConfig(**parameters)
                pipeline.set_generation_config(config)

            expires_at = float("inf") if keep_alive < 0 else time.monotonic() + keep_alive

            return LoadedModel(
                name=model_name,
                device=device,
                pipeline=pipeline,
                system_prompt=system_prompt,
                parameters=parameters,
                expires_at=expires_at,
            )
        except Exception as e:
            print(f"Native OpenVINO Compilation Failed: {e}")
            raise ModelCompilationError(model_name, device, e) from e

    @classmethod
    def get_pipeline(cls, model_name: str, device: str | None = None) -> dict:
        """Temporary synchronous bridge for chat.py until it migrates to client.py."""
        resolved_device = cls._resolve_device(model_name, device)
        entry = cls._models.get(model_name)

        if entry is None or entry.device != resolved_device:
            entry = cls._load_model_entry(model_name, resolved_device)
            cls._models[model_name] = entry

        return {
            "pipeline": entry.pipeline,
            "system_prompt": entry.system_prompt,
        }