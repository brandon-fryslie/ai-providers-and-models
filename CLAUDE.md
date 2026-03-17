# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multi-language data package providing a regularly updated, structured catalog of AI providers and models (OpenAI, Gemini, Claude). The canonical data source is `models.yaml` at the repo root, with Node.js and Python SDKs that consume it.

## Architecture

- `models.yaml` — single source of truth for all provider/model data. Both SDKs copy this file into their own `data/` directories at build/test time.
- `modules/node/` — TypeScript Node.js package (`@dwmkerr/ai-providers-and-models`). Parses `models.yaml` and exposes typed provider/model objects.
- `modules/python/` — Python package (`ai_providers_and_models`). Same data, Python dataclasses.
- `modules/validate-schema/` — YAML schema validation against `models-schema.yaml`.
- `modules/integration-tests/` — Tests that hit live APIs (require API keys).
- Versioning is managed by release-please (`# x-release-please-version` comment in `models.yaml`).

## Common Commands

All makefiles support `make help`.

### Node module (`modules/node/`)
```bash
make init          # install deps, copy models.yaml
make test          # jest with coverage
make build         # compile TypeScript
make lint          # eslint
make lint-fix      # eslint --fix
npm test -- --testPathPattern=<pattern>  # run a single test
```

### Python module (`modules/python/`)
```bash
make init          # pip install deps + editable install
make test          # pytest + coverage
make build         # python -m build
make lint          # flake8
make lint-fix      # black
pytest tests/test_providers_file.py -k <test_name>  # run a single test
```

### All modules (`modules/`)
```bash
make test          # runs node + python + schema validation
make build         # builds node + python
```

### Schema validation (`modules/validate-schema/`)
```bash
make validate-schema
```

## Key Conventions

- When updating model data, edit only `models.yaml`. The SDKs copy it at build/test time.
- YAML anchors (`&ref_0`, `*ref_1`) are used to share data between providers (e.g., `gemini` and `gemini_openai` share the same models).
- Model schema fields: `id`, `name`, `status`, `context_window`, `max_output_tokens`, `pricing` (input/output per million tokens), `modalities` (input/output), `endpoints`, `versions`.
- There is an inconsistency in the YAML: some models use `knowledgeCutoff` (camelCase) and others use `knowledge_cutoff` (snake_case).
