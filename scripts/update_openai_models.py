#!/usr/bin/env python3
"""Update OpenAI models in models.yaml from API + docs scraping.

Usage:
    OPENAI_API_KEY=sk-... python scripts/update_openai_models.py [--dry-run]
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_YAML = REPO_ROOT / "models.yaml"
OPENAI_API_URL = "https://api.openai.com/v1/models"
DOCS_BASE_URL = "https://developers.openai.com/docs/models"

# Prefixes for text/chat models we track
MODEL_PREFIXES = ("gpt-3", "gpt-4", "gpt-5", "o1", "o3", "o4", "o5")

# Model IDs containing these substrings are skipped
SKIP_PATTERNS = (
    "embed", "dall-e", "tts", "whisper", "moderation",
    "babbage", "davinci", "curie", "ada",
    "gpt-image", "gpt-audio", "gpt-realtime",
    "sora", "computer-use", "search",
    "-audio-preview", "-realtime-preview", "-transcribe",
    "-chat-latest", "-codex",
)

# Regex to strip date-based version suffixes
VERSION_SUFFIX_RE = re.compile(r"^(.+?)(-\d{4}-\d{2}-\d{2})$")
# Also handle older YYMM format like gpt-4-0613
VERSION_SUFFIX_SHORT_RE = re.compile(r"^(.+?)(-\d{4})$")

SCRAPE_DELAY = 0.5  # seconds between doc page requests


def fetch_api_models(api_key: str) -> list[dict]:
    """Fetch all models from OpenAI API."""
    resp = requests.get(
        OPENAI_API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"]


def is_tracked(model_id: str) -> bool:
    """Return True for text/chat models we want to track."""
    if model_id.startswith("ft:"):
        return False
    if model_id.startswith("chatgpt-"):
        return False
    if any(skip in model_id for skip in SKIP_PATTERNS):
        return False
    return any(model_id.startswith(p) for p in MODEL_PREFIXES)


def extract_base_id(model_id: str) -> str:
    """Strip version suffix to get the base model ID.

    gpt-4o-2024-08-06 -> gpt-4o
    gpt-4-0613 -> gpt-4
    gpt-4-0125-preview -> gpt-4
    gpt-4-1106-preview -> gpt-4
    gpt-4-turbo-2024-04-09 -> gpt-4-turbo
    o3-mini-2025-01-31 -> o3-mini
    gpt-4o -> gpt-4o (no change)
    """
    # Try YYYY-MM-DD suffix first
    m = VERSION_SUFFIX_RE.match(model_id)
    if m:
        return m.group(1)
    # Handle gpt-4-MMDD-preview pattern (e.g., gpt-4-0125-preview)
    m = re.match(r"^(.+?)-\d{4}-preview$", model_id)
    if m:
        return m.group(1)
    # Try YYMM suffix (e.g., gpt-4-0613, gpt-4-0314)
    m = VERSION_SUFFIX_SHORT_RE.match(model_id)
    if m:
        base = m.group(1)
        suffix = m.group(2).lstrip("-")
        if len(suffix) == 4 and suffix.isdigit():
            return base
    return model_id


def group_models(api_models: list[dict]) -> dict[str, list[dict]]:
    """Group tracked models by base ID. Returns {base_id: [model_dicts]}."""
    groups: dict[str, list[dict]] = {}
    for m in api_models:
        mid = m["id"]
        if not is_tracked(mid):
            continue
        base = extract_base_id(mid)
        groups.setdefault(base, []).append(m)

    # Sort versions by created timestamp (oldest first)
    for versions in groups.values():
        versions.sort(key=lambda m: m.get("created", 0))
    return groups


def scrape_model_page(base_id: str) -> dict:
    """Scrape model metadata from OpenAI docs. Returns extracted fields or empty dict."""
    url = f"{DOCS_BASE_URL}/{base_id}"
    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; model-updater/1.0)"},
        )
        if resp.status_code != 200:
            print(f"  Warning: {url} returned {resp.status_code}", file=sys.stderr)
            return {}
    except requests.RequestException as e:
        print(f"  Warning: failed to fetch {url}: {e}", file=sys.stderr)
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract text-based fields (pricing, context window, etc.)
    text = soup.get_text("\n", strip=True)
    result = _extract_from_text(text)

    # Extract endpoint support from HTML structure (CSS classes indicate support)
    result["endpoints"] = _extract_endpoints(soup)

    return result


def _extract_from_text(text: str) -> dict:
    """Extract model data from page text content using pattern matching.

    The OpenAI docs pages render data across separate lines, e.g.:
        128,000
        context window
        16,384
        max output tokens
        Oct 01, 2023 knowledge cutoff
    And pricing as:
        Input
        $2.50
        Cached input
        $1.25
        Output
        $10.00
    """
    result = {}
    lines = text.split("\n")

    # Scan line-by-line with lookahead to next line
    for i, line in enumerate(lines):
        line_s = line.strip()
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""

        # Context window: number on this line, "context window" on next
        if re.match(r"^[0-9,]+$", line_s) and "context" in next_line.lower():
            result["context_window"] = int(line_s.replace(",", ""))

        # Max output tokens: number on this line, "max output" on next
        if re.match(r"^[0-9,]+$", line_s) and "output" in next_line.lower() and "max" in next_line.lower():
            result["max_output_tokens"] = int(line_s.replace(",", ""))

        # Knowledge cutoff: "MMM DD, YYYY knowledge cutoff" or similar
        m = re.match(r"^(.+?)\s+knowledge\s+cutoff", line_s, re.IGNORECASE)
        if m:
            result["knowledge_cutoff"] = m.group(1).strip()

        # Pricing: "Input" on this line, "$X.XX" on next line
        if line_s == "Input" and re.match(r"^\$[0-9.]+$", next_line):
            result.setdefault("input_price", float(next_line.lstrip("$")))

        # Cached input pricing
        if line_s == "Cached input" and re.match(r"^\$[0-9.]+$", next_line):
            result["cached_input_price"] = float(next_line.lstrip("$"))

        # Output pricing
        if line_s == "Output" and re.match(r"^\$[0-9.]+$", next_line):
            result.setdefault("output_price", float(next_line.lstrip("$")))

    # Modalities - check for "Supported" near input types
    # The page shows input/output modalities with Supported/Not supported
    for i, line_s in enumerate(lines):
        next_l = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if line_s == "Image" and next_l == "Supported":
            result["image_input"] = True
        if line_s == "Audio" and next_l == "Supported":
            result["audio_input"] = True
        if line_s == "Video" and next_l == "Supported":
            result["video_input"] = True

    # Also try single-line patterns as fallback
    for pattern in [
        r"[Cc]ontext\s+window\s*[:\s]*([0-9,]+)",
        r"([0-9,]+)\s*(?:token)?\s*context\s*window",
    ]:
        if "context_window" not in result:
            m = re.search(pattern, text)
            if m:
                result["context_window"] = int(m.group(1).replace(",", ""))

    for pattern in [
        r"[Mm]ax\s+output\s*[:\s]*([0-9,]+)",
        r"([0-9,]+)\s*max\s*output",
    ]:
        if "max_output_tokens" not in result:
            m = re.search(pattern, text)
            if m:
                result["max_output_tokens"] = int(m.group(1).replace(",", ""))

    # Description - first long quoted string or sentence about the model
    m = re.search(r"\u201c([^\u201d]{20,300})\u201d", text)  # smart quotes
    if not m:
        m = re.search(r'"([^"]{20,300})"', text)
    if m:
        result["description"] = m.group(1)

    # Status
    text_lower = text.lower()
    if "deprecated" in text_lower:
        result["status"] = "deprecated"
    elif "preview" in text_lower and "deprecated" not in text_lower:
        result["status"] = "preview"

    return result


# Map from doc page endpoint names to our YAML keys
_ENDPOINT_NAME_MAP = {
    "Chat Completions": "chat_completions",
    "Responses": "responses",
    "Realtime": "realtime",
    "Assistants": "assistants",
    "Batch": "batch",
    "Fine-tuning": "fine_tuning",
    "Embeddings": "embeddings",
    "Image generation": "image_generation",
    "Speech generation": "speech_generation",
    "Transcription": "transcription",
    "Translation": "translation",
    "Moderation": "moderation",
    "Completions (legacy)": "completions_legacy",
}


def _extract_endpoints(soup: BeautifulSoup) -> dict:
    """Extract endpoint support from HTML structure.

    Supported endpoints have: <div class="text-sm font-semibold">Name</div>
    Unsupported endpoints have: <div class="text-sm font-semibold text-gray-400">Name</div>
    """
    endpoints = {}
    for name_div in soup.find_all(
        "div",
        class_=lambda c: c and "text-sm" in c and "font-semibold" in c,
    ):
        name = name_div.get_text(strip=True)
        if name not in _ENDPOINT_NAME_MAP:
            continue
        key = _ENDPOINT_NAME_MAP[name]
        # Already found this endpoint (first occurrence is the real one from
        # the model spec section, later ones may be from nav/comparison)
        if key in endpoints:
            continue
        supported = "text-gray-400" not in name_div.get("class", [])
        endpoints[key] = supported
    return endpoints


def _make_modalities(
    image_input: bool = False,
    audio_input: bool = False,
    video_input: bool = False,
) -> CommentedMap:
    inp = CommentedMap({"text": True, "image": image_input, "audio": audio_input, "video": video_input})
    out = CommentedMap({"text": True, "image": False, "audio": False, "video": False})
    return CommentedMap({"input": inp, "output": out})


def _make_endpoints(scraped_endpoints: dict | None = None) -> CommentedMap:
    defaults = {
        "assistants": False,
        "batch": False,
        "chat_completions": True,
        "completions_legacy": False,
        "embeddings": False,
        "fine_tuning": False,
        "image_generation": False,
        "moderation": False,
        "realtime": False,
        "responses": True,
        "speech_generation": False,
        "transcription": False,
        "translation": False,
    }
    if scraped_endpoints:
        for k, v in scraped_endpoints.items():
            if k in defaults:
                defaults[k] = v
    return CommentedMap(defaults)


def build_model_entry(
    base_id: str,
    versions: list[dict],
    scraped: dict,
    existing: dict | None,
) -> CommentedMap:
    """Build a model entry for models.yaml."""
    entry = CommentedMap()

    # Use existing data as fallback where scraped data is missing
    ex = existing or {}

    entry["id"] = base_id
    entry["name"] = ex.get("name", base_id)
    entry["documentation_url"] = f"{DOCS_BASE_URL}/{base_id}"
    entry["description_short"] = scraped.get(
        "description_short", ex.get("description_short", "")
    )
    entry["description"] = scraped.get("description", ex.get("description", ""))
    entry["status"] = scraped.get("status", ex.get("status", "general-availability"))
    entry["knowledgeCutoff"] = scraped.get(
        "knowledge_cutoff", ex.get("knowledgeCutoff", ex.get("knowledge_cutoff", ""))
    )
    entry["context_window"] = scraped.get(
        "context_window", ex.get("context_window", 0)
    )
    entry["max_output_tokens"] = scraped.get(
        "max_output_tokens", ex.get("max_output_tokens", 0)
    )
    entry["validated"] = ex.get("validated", False)

    # Pricing
    pricing = CommentedMap()
    ex_pricing = ex.get("pricing", {})
    pricing["input_per_million"] = scraped.get(
        "input_price", ex_pricing.get("input_per_million", 0)
    )
    pricing["output_per_million"] = scraped.get(
        "output_price", ex_pricing.get("output_per_million", 0)
    )
    entry["pricing"] = pricing

    # Modalities
    entry["modalities"] = _make_modalities(
        image_input=scraped.get("image_input", False),
        audio_input=scraped.get("audio_input", False),
        video_input=scraped.get("video_input", False),
    )

    # Endpoints
    entry["endpoints"] = _make_endpoints(scraped.get("endpoints"))

    # Versions - built from API data
    version_entries = CommentedSeq()
    for v in versions:
        vid = v["id"]
        if vid == base_id:
            continue  # Skip the alias entry
        ve = CommentedMap()
        ve["id"] = vid
        date_m = re.search(r"(\d{4}-\d{2}-\d{2})", vid)
        ve["release_date"] = date_m.group(1) if date_m else ""
        # Latest version (last in sorted-by-created list) is default
        ve["isDefault"] = v is versions[-1]
        ve["isDeprecated"] = False
        ve["description"] = ""
        # Preserve existing version metadata
        if "versions" in ex:
            for ev in ex["versions"]:
                if ev.get("id") == vid:
                    ve["isDeprecated"] = ev.get("isDeprecated", False)
                    ve["description"] = ev.get("description", "")
                    break
        version_entries.append(ve)

    if version_entries:
        entry["versions"] = version_entries

    return entry


def main():
    parser = argparse.ArgumentParser(description="Update OpenAI models in models.yaml")
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

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable required", file=sys.stderr)
        sys.exit(1)

    # 1. Fetch model list from API
    print("Fetching models from OpenAI API...", file=sys.stderr)
    api_models = fetch_api_models(api_key)
    print(f"  {len(api_models)} total models from API", file=sys.stderr)

    # 2. Group by base model
    groups = group_models(api_models)
    print(
        f"  {len(groups)} tracked model families: {sorted(groups.keys())}",
        file=sys.stderr,
    )

    # 3. Load existing YAML
    yml = YAML()
    yml.preserve_quotes = True
    yml.width = 4096  # prevent unwanted line wrapping
    with open(args.yaml_path) as f:
        data = yml.load(f)

    existing_models = dict(data["providers"]["openai"].get("models", {}))

    # 4. Scrape docs + build entries
    new_models = CommentedMap()
    for base_id in sorted(groups.keys()):
        print(f"  Processing {base_id}...", file=sys.stderr)
        scraped = scrape_model_page(base_id)
        existing = dict(existing_models[base_id]) if base_id in existing_models else None
        entry = build_model_entry(base_id, groups[base_id], scraped, existing)
        new_models[base_id] = entry
        time.sleep(SCRAPE_DELAY)

    # 5. Update YAML
    data["providers"]["openai"]["models"] = new_models

    if args.dry_run:
        yml.dump(data, sys.stdout)
        print(f"\n[dry-run] Would update {len(new_models)} models", file=sys.stderr)
    else:
        with open(args.yaml_path, "w") as f:
            yml.dump(data, f)
        print(f"Updated {len(new_models)} OpenAI models in {args.yaml_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
