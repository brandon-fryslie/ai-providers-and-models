# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A regularly updated, structured catalog of AI providers and models (OpenAI, Gemini, Claude). The canonical data source is `models.yaml` at the repo root.

## Architecture

- `models.yaml` — single source of truth for all provider/model data.
- `modules/validate-schema/` — YAML schema validation against `models-schema.yaml`.
- Versioning is managed by release-please (`# x-release-please-version` comment in `models.yaml`).

## Common Commands

### Schema validation (`modules/validate-schema/`)
```bash
make validate-schema
```

## Key Conventions

- When updating model data, edit only `models.yaml`.
- YAML anchors (`&ref_0`, `*ref_1`) are used to share data between providers (e.g., `gemini` and `gemini_openai` share the same models).
- Model schema fields: `id`, `name`, `status`, `context_window`, `max_output_tokens`, `pricing` (input/output per million tokens), `modalities` (input/output), `endpoints`, `versions`.
- There is an inconsistency in the YAML: some models use `knowledgeCutoff` (camelCase) and others use `knowledge_cutoff` (snake_case).
