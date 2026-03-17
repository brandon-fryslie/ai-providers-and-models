#!/usr/bin/env python3
"""Update Gemini models in models.yaml from API + pricing page scraping.

Usage:
    GEMINI_API_KEY=... python scripts/update_gemini_models.py [--dry-run]
"""

import argparse
import os
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_YAML = REPO_ROOT / "models.yaml"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"
DOCS_BASE_URL = "https://ai.google.dev/gemini-api/docs/models"

# Regex for version-specific suffixes like -001, -002
VERSION_SUFFIX_RE = re.compile(r"^(models/gemini-.+?)-(\d{3})$")
# Regex for dated preview variants like -preview-09-2025
DATED_PREVIEW_RE = re.compile(r"^(models/gemini-.+?-preview)-\d{2}-\d{4}$")


def fetch_api_models(api_key: str) -> list[dict]:
    """Fetch all models from Gemini API."""
    resp = requests.get(
        GEMINI_API_URL,
        params={"key": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["models"]


def is_tracked(model_name: str) -> bool:
    """Return True for gemini text/chat models we want to track."""
    if not model_name.startswith("models/gemini-"):
        return False
    if "-tts" in model_name:
        return False
    if "-image" in model_name:
        return False
    if "-latest" in model_name:
        return False
    if "-embedding" in model_name:
        return False
    return True


def extract_base_id(model_name: str) -> str:
    """Strip version suffix to get the base model ID.

    models/gemini-2.0-flash-001 -> models/gemini-2.0-flash
    models/gemini-2.5-flash-lite-preview-09-2025 -> skip (handled by is_dated_preview)
    models/gemini-2.0-flash -> models/gemini-2.0-flash (no change)
    """
    # Strip -NNN version suffixes
    m = VERSION_SUFFIX_RE.match(model_name)
    if m:
        return m.group(1)
    return model_name


def is_dated_preview(model_name: str) -> bool:
    """Return True for dated preview variants like gemini-2.5-flash-lite-preview-09-2025."""
    return bool(DATED_PREVIEW_RE.match(model_name))


def group_models(api_models: list[dict]) -> dict[str, list[dict]]:
    """Group tracked models by base ID. Returns {base_id: [model_dicts]}."""
    groups: dict[str, list[dict]] = {}
    for m in api_models:
        name = m["name"]
        if not is_tracked(name):
            continue
        if is_dated_preview(name):
            continue
        base = extract_base_id(name)
        groups.setdefault(base, []).append(m)
    return groups


def scrape_pricing() -> dict[str, dict]:
    """Scrape pricing from the Gemini pricing page.

    Returns {model_id: {"input": float, "output": float}} where
    model_id is the gemini model ID (e.g., "gemini-2.5-pro").
    """
    try:
        resp = requests.get(
            PRICING_URL,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; model-updater/1.0)"},
        )
        if resp.status_code != 200:
            print(f"  Warning: pricing page returned {resp.status_code}", file=sys.stderr)
            return {}
    except requests.RequestException as e:
        print(f"  Warning: failed to fetch pricing page: {e}", file=sys.stderr)
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text("\n", strip=True)
    lines = text.split("\n")

    pricing: dict[str, dict] = {}

    # The pricing page has sections per model. Each section has:
    #   "Gemini X.Y Name"     (display name)
    #   "gemini-x.y-name"     (model ID on next line)
    #   ...
    #   "Input price"
    #   "Free of charge"  or  "$X.XX"
    #   "$X.XX, prompts <= 200k tokens"   (paid tier price)
    #   ...
    #   "Output price (including thinking tokens)"
    #   "Free of charge"  or  "$X.XX"
    #   "$X.XX, prompts <= 200k tokens"
    current_model = None
    for i, line in enumerate(lines):
        stripped = line.strip()

        # Detect model ID lines (e.g., "gemini-2.5-pro")
        if re.match(r"^gemini-\d", stripped) and not stripped.startswith("gemini-embedding"):
            current_model = stripped

        if not current_model:
            continue

        # Input price: look for "Input price" then find first $X.XX in next few lines
        if stripped.lower() == "input price":
            for j in range(i + 1, min(len(lines), i + 5)):
                m = re.search(r"\$([0-9.]+)", lines[j])
                if m:
                    pricing.setdefault(current_model, {})["input"] = float(m.group(1))
                    break

        # Output price
        if "output price" in stripped.lower():
            for j in range(i + 1, min(len(lines), i + 5)):
                m = re.search(r"\$([0-9.]+)", lines[j])
                if m:
                    pricing.setdefault(current_model, {})["output"] = float(m.group(1))
                    break

    return pricing


def find_pricing_for_model(model_name: str, pricing_data: dict[str, dict]) -> dict:
    """Find the best pricing match for a model name from scraped data."""
    # Normalize: "models/gemini-2.0-flash" -> "gemini-2.0-flash"
    clean = model_name.replace("models/", "")
    for key, vals in pricing_data.items():
        if clean in key or key in clean:
            return vals
    return {}


def _make_modalities() -> CommentedMap:
    """Gemini models support text+image+audio+video input, text output by default."""
    inp = CommentedMap({"text": True, "image": True, "audio": True, "video": True})
    out = CommentedMap({"text": True, "image": False, "audio": False, "video": False})
    return CommentedMap({"input": inp, "output": out})


def _make_endpoints(supported_methods: list[str] | None = None) -> CommentedMap:
    """Map supportedGenerationMethods to endpoint booleans."""
    methods = set(supported_methods or [])

    has_generate = "generateContent" in methods
    has_count = "countTokens" in methods
    has_cached = "createCachedContent" in methods
    has_batch = "batchGenerateContent" in methods

    return CommentedMap({
        "assistants": False,
        "batch": has_batch or has_cached,
        "chat_completions": has_generate,
        "completions_legacy": False,
        "embeddings": False,
        "fine_tuning": "createTunedModel" in methods,
        "image_generation": False,
        "moderation": False,
        "realtime": False,
        "responses": has_generate,
        "speech_generation": False,
        "transcription": False,
        "translation": False,
    })


def build_model_entry(
    base_id: str,
    versions: list[dict],
    pricing_match: dict,
    existing: dict | None,
) -> CommentedMap:
    """Build a model entry for models.yaml."""
    entry = CommentedMap()
    ex = existing or {}

    # Use the first (or canonical) version's metadata
    canonical = versions[0]
    for v in versions:
        if v["name"] == base_id:
            canonical = v
            break

    display_name = canonical.get("displayName", base_id.replace("models/", ""))
    description = canonical.get("description", "")
    input_limit = canonical.get("inputTokenLimit", 0)
    output_limit = canonical.get("outputTokenLimit", 0)
    supported_methods = canonical.get("supportedGenerationMethods", [])

    # Status based on name
    status = "preview" if "preview" in base_id else "general-availability"

    entry["id"] = base_id
    entry["name"] = ex.get("name", display_name)
    entry["documentation_url"] = ex.get(
        "documentation_url",
        f"{DOCS_BASE_URL}#{ base_id.replace('models/', '') }",
    )
    entry["description_short"] = ex.get("description_short", description[:120] if description else "")
    entry["description"] = ex.get("description", description)
    entry["status"] = status
    entry["knowledge_cutoff"] = ex.get("knowledge_cutoff", "")
    entry["context_window"] = input_limit or ex.get("context_window", 0)
    entry["max_output_tokens"] = output_limit or ex.get("max_output_tokens", 0)
    entry["validated"] = ex.get("validated", False)

    # Pricing
    pricing = CommentedMap()
    ex_pricing = ex.get("pricing", {})
    pricing["input_per_million"] = pricing_match.get(
        "input", ex_pricing.get("input_per_million", 0)
    )
    pricing["output_per_million"] = pricing_match.get(
        "output", ex_pricing.get("output_per_million", 0)
    )
    entry["pricing"] = pricing

    # Modalities
    entry["modalities"] = _make_modalities()

    # Endpoints
    entry["endpoints"] = _make_endpoints(supported_methods)

    # Versions - list non-base version IDs
    version_entries = CommentedSeq()
    for v in versions:
        vid = v["name"]
        if vid == base_id:
            continue
        ve = CommentedMap()
        ve["id"] = vid
        ve["version"] = v.get("version", "")
        ve["isDefault"] = False
        ve["isDeprecated"] = False
        # Preserve existing version metadata
        if "versions" in ex:
            for ev in ex["versions"]:
                if ev.get("id") == vid:
                    ve["isDeprecated"] = ev.get("isDeprecated", False)
                    if ev.get("description"):
                        ve["description"] = ev["description"]
                    break
        version_entries.append(ve)

    if version_entries:
        entry["versions"] = version_entries

    return entry


def main():
    parser = argparse.ArgumentParser(description="Update Gemini models in models.yaml")
    parser.add_argument(
        "--yaml-path",
        type=Path,
        default=MODELS_YAML,
        help="Path to models.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print updated YAML to stdout instead of writing file",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable required", file=sys.stderr)
        sys.exit(1)

    # 1. Fetch model list from API
    print("Fetching models from Gemini API...", file=sys.stderr)
    api_models = fetch_api_models(api_key)
    print(f"  {len(api_models)} total models from API", file=sys.stderr)

    # 2. Group by base model
    groups = group_models(api_models)
    print(
        f"  {len(groups)} tracked model families: {sorted(groups.keys())}",
        file=sys.stderr,
    )

    # 3. Scrape pricing
    print("Scraping pricing from Gemini pricing page...", file=sys.stderr)
    pricing_data = scrape_pricing()
    print(f"  Found pricing for {len(pricing_data)} model patterns", file=sys.stderr)

    # 4. Load existing YAML
    yml = YAML()
    yml.preserve_quotes = True
    yml.width = 4096  # prevent unwanted line wrapping
    with open(args.yaml_path) as f:
        data = yml.load(f)

    existing_models = {}
    gemini_provider = data["providers"]["gemini"]
    for k, v in gemini_provider.get("models", {}).items():
        existing_models[k] = dict(v) if v else {}

    # 5. Build new model entries
    new_models = CommentedMap()
    for base_id in sorted(groups.keys()):
        print(f"  Processing {base_id}...", file=sys.stderr)
        pricing_match = find_pricing_for_model(base_id, pricing_data)
        existing = existing_models.get(base_id)
        entry = build_model_entry(base_id, groups[base_id], pricing_match, existing)
        new_models[base_id] = entry

    # 6. Update YAML - replace gemini.models with anchor preserved
    # The gemini_openai provider uses *ref_1 which points to &ref_1 on gemini.models.
    # ruamel.yaml preserves anchors/aliases automatically when we replace the value
    # on the anchor side, as long as we set the anchor on the new object.
    new_models.yaml_set_anchor("ref_1", always_dump=True)
    gemini_provider["models"] = new_models

    if args.dry_run:
        yml.dump(data, sys.stdout)
        print(f"\n[dry-run] Would update {len(new_models)} models", file=sys.stderr)
    else:
        with open(args.yaml_path, "w") as f:
            yml.dump(data, f)
        print(f"Updated {len(new_models)} Gemini models in {args.yaml_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
