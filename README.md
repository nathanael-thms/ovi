# ovi

ovi is a lightweight local model server that delivers an ollama‑style developer experience, built natively on Intel’s OpenVINO runtime for efficient CPU and GPU inference.

ovi is in early development. Core functionality is usable, but the project is evolving rapidly as the runtime and CLI mature.

## Documentation

Documentation is available at [https://ovi.nattho.com](https://ovi.nattho.com).

## Current Features
- **Local Execution** — Run raw OpenVINO‑format models directly from your `project_root/models/` directory.

- **Familiar CLI** — A clean, drop‑in alternative to ollama using intuitive commands like ovi run.

- **Interactive Chat Shell** — Navigate previous prompts using ↑/↓ for a smooth REPL‑style workflow.

- **Device Selection** — Load models onto CPU, iGPU, dGPU, or other OpenVINO‑supported targets.

## Upcoming Features
- **Custom Modelfiles** — Define execution devices, context length, warm‑duration, and other runtime parameters.

- **Internal /x Command Interface** — Inspect and adjust runtime parameters from within the chat shell.

- **OpenAI‑Compatible API** — Native endpoints for seamless integration with tools like Open WebUI.

ovi is fully free, open‑source, and licensed under the Apache License 2.0.

---

> [!WARNING]
> **Disclaimer**  
>
> ovi is an independent, community‑driven project and is not affiliated with Intel Corporation.
> Intel and OpenVINO are trademarks of Intel Corporation or its subsidiaries.
