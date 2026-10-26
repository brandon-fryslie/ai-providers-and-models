# ai-providers-and-models

A catalog of AI models from OpenAI, Google Gemini, and Anthropic Claude: pricing, context windows, endpoints, and modalities. It is rebuilt every day from the providers' own APIs and documentation.

**Live JSON:** [brandon-fryslie.github.io/ai-providers-and-models/models.json](https://brandon-fryslie.github.io/ai-providers-and-models/models.json)

## Usage

Fetch the JSON directly; no package needed:

```
https://brandon-fryslie.github.io/ai-providers-and-models/models.json
```

The same data as YAML is the source file, [`models.yaml`](./models.yaml):

```
https://raw.githubusercontent.com/brandon-fryslie/ai-providers-and-models/main/models.yaml
```

The data's shape is defined by [`modules/models-schema.yaml`](./modules/models-schema.yaml).

## What's in the data

| Provider key | Covers | Source |
|---|---|---|
| `openai` | GPT and o-series models | `/v1/models` API, plus each model's page on `developers.openai.com` |
| `gemini` | Gemini models | Gemini API, plus the Gemini pricing page |
| `gemini_openai` | The same Gemini models, for Gemini's OpenAI-compatible endpoint | Shared with `gemini` |
| `claude` | Claude models | Anthropic's models docs page |

Which models are listed follows the providers: a new model appears the day after its provider publishes it. See `models.json` for the current list rather than this README.

For each model:
- Pricing (input / output, USD per million tokens)
- Context window and max output tokens
- Supported API endpoints (chat completions, responses, batch, fine-tuning, etc.)
- Input and output modalities (text, image, audio, video)
- Knowledge cutoff
- Dated versions, with deprecation status

## How it updates

[A GitHub Action](./.github/workflows/update-openai-models.yml) runs daily at 09:00 UTC. It runs one update script per provider (`scripts/update_*_models.py`), validates `models.yaml` against the schema, commits the change to `main`, and publishes `models.json` to GitHub Pages.

Providers change their pages without notice, so failures are contained:
- If one model can't be read, it keeps its previous entry and the rest of its provider still updates.
- If a provider fails entirely, its section is left as it was and the other providers still update.
- If the result fails schema validation, nothing is committed or published.

In each case the run is marked failed and names what broke, so the data is never silently stale. Check the [Actions tab](https://github.com/brandon-fryslie/ai-providers-and-models/actions) to see when each provider last updated cleanly.

## Running the updates locally

```bash
pip install -r scripts/requirements.txt
OPENAI_API_KEY=... python scripts/update_openai_models.py --dry-run
GEMINI_API_KEY=... python scripts/update_gemini_models.py --dry-run
python scripts/update_anthropic_models.py --dry-run   # no key needed
```

`--dry-run` prints the updated YAML instead of writing `models.yaml`.

## Credits

Originally created by [dwmkerr](https://github.com/dwmkerr/ai-providers-and-models). Forked and automated by [brandon-fryslie](https://github.com/brandon-fryslie).
