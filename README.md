# ai-providers-and-models

Automatically updated catalog of AI providers, models, and the APIs they support. Data is refreshed daily from provider APIs and documentation.

**Live JSON endpoint:** [brandon-fryslie.github.io/ai-providers-and-models/models.json](https://brandon-fryslie.github.io/ai-providers-and-models/models.json)

## Providers

| Provider | Models | Data Source |
|----------|--------|-------------|
| **OpenAI** | GPT-3.5 through GPT-5.4, o1, o3, o4 | `/v1/models` API + docs scraping |
| **Google Gemini** | Gemini 2.0 through 3.1 | Gemini API + pricing page scraping |
| **Anthropic Claude** | Claude Haiku 4.5, Sonnet 4.6, Opus 4.6 | Docs page scraping |

## What's in the data

For each model:
- **Pricing** (input/output per million tokens)
- **Context window** and **max output tokens**
- **Supported API endpoints** (chat completions, responses, assistants, batch, fine-tuning, etc.)
- **Input/output modalities** (text, image, audio, video)
- **Knowledge cutoff date**
- **Model versions** with deprecation status

## Usage

### JSON (recommended)

Fetch the latest data directly — no dependencies needed:

```
https://brandon-fryslie.github.io/ai-providers-and-models/models.json
```

### YAML

The raw [`models.yaml`](./models.yaml) file is also available:

```
https://raw.githubusercontent.com/brandon-fryslie/ai-providers-and-models/main/models.yaml
```

### Node.js / Python SDKs

This repo includes Node.js and Python packages from the original project. See [modules/node/README.md](./modules/node/README.md) and [modules/python/README.md](./modules/python/README.md).

## How it works

A [GitHub Action](./.github/workflows/update-openai-models.yml) runs daily at 9am UTC:

1. **OpenAI** — queries the `/v1/models` API for the model list, then scrapes each model's docs page at `developers.openai.com` for pricing, context windows, endpoint support, and modalities
2. **Gemini** — queries the Gemini API (`v1beta/models`) for model IDs, context windows, and output limits, then scrapes the pricing page for costs
3. **Anthropic** — scrapes the Claude models docs page for everything (no API key needed)
4. Updates `models.yaml`, commits to main, converts to JSON, and deploys to GitHub Pages

## Credits

Originally created by [dwmkerr](https://github.com/dwmkerr/ai-providers-and-models). Forked and automated by [brandon-fryslie](https://github.com/brandon-fryslie).
