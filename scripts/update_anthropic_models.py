#!/usr/bin/env python3
"""Update Anthropic Claude models in models.yaml by scraping docs.

Usage:
    python scripts/update_anthropic_models.py [--dry-run]
"""

import argparse
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_YAML = REPO_ROOT / "models.yaml"
DOCS_URL = "https://platform.claude.com/docs/en/docs/about-claude/models"
DOCUMENTATION_URL = "https://docs.anthropic.com/en/docs/about-claude/models"

# Maps human-readable size tokens to integers
_SIZE_MAP = {
    "1M": 1_000_000,
    "200k": 200_000,
    "128k": 128_000,
    "64k": 64_000,
    "32k": 32_000,
    "16k": 16_000,
    "8k": 8_000,
    "4k": 4_096,
}

# Version suffix pattern: -YYYYMMDD at end of id
_VERSION_SUFFIX_RE = re.compile(r"^(.+?)-(\d{8})$")


def _parse_token_count(text: str) -> int:
    """Parse '1M tokens', '200k tokens', '128,000', etc. to int."""
    text = text.strip().lower().replace(",", "").replace("tokens", "").strip()
    for suffix, val in _SIZE_MAP.items():
        if text == suffix.lower():
            return val
    # Try numeric
    m = re.match(r"^(\d+)$", text)
    if m:
        return int(m.group(1))
    # Try 1m, 200k patterns
    m = re.match(r"^(\d+(?:\.\d+)?)\s*m$", text)
    if m:
        return int(float(m.group(1)) * 1_000_000)
    m = re.match(r"^(\d+(?:\.\d+)?)\s*k$", text)
    if m:
        return int(float(m.group(1)) * 1_000)
    return 0


def _parse_pricing(text: str) -> tuple[float, float]:
    """Parse pricing like '$5 / input MTok$25 / output MTok' returning (input, output) per MTok."""
    input_price = 0.0
    output_price = 0.0
    # Match "$X / input MTok"
    m = re.search(r"\$([0-9.]+)\s*/\s*input", text)
    if m:
        input_price = float(m.group(1))
    # Match "$X / output MTok"
    m = re.search(r"\$([0-9.]+)\s*/\s*output", text)
    if m:
        output_price = float(m.group(1))
    # Fallback: $X / $Y pattern
    if not input_price and not output_price:
        m = re.search(r"\$([0-9.]+)\s*/\s*\$([0-9.]+)", text)
        if m:
            input_price, output_price = float(m.group(1)), float(m.group(2))
    return input_price, output_price


def _extract_base_id(model_id: str) -> str:
    """Strip version suffix to get the alias/base model ID.

    claude-opus-4-5-20250929 -> claude-opus-4-5
    claude-haiku-4-5-20251001 -> claude-haiku-4-5
    claude-opus-4-6 -> claude-opus-4-6 (no change, already an alias)
    """
    m = _VERSION_SUFFIX_RE.match(model_id)
    if m:
        return m.group(1)
    return model_id


def fetch_models_page() -> BeautifulSoup:
    """Fetch and parse the Anthropic models documentation page."""
    resp = requests.get(
        DOCS_URL,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (compatible; model-updater/1.0)"},
    )
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def _parse_table(table) -> list[dict]:
    """Parse a transposed HTML table (features as rows, models as columns).

    The Anthropic docs table has:
    - Row 0 (th): Feature | Model1 | Model2 | ...
    - Row N (td): FeatureName | Value1 | Value2 | ...

    Returns a list of dicts, one per model column.
    """
    rows = table.find_all("tr")
    if not rows:
        return []

    # First row has headers: Feature, Model1Name, Model2Name, ...
    header_cells = rows[0].find_all(["th", "td"])
    model_names = [c.get_text(strip=True) for c in header_cells[1:]]  # skip "Feature"
    num_models = len(model_names)

    # Initialize one dict per model
    models = [{} for _ in range(num_models)]

    # Each subsequent row: FeatureName | Value1 | Value2 | ...
    for tr in rows[1:]:
        cells = tr.find_all(["th", "td"])
        if len(cells) < 2:
            continue
        feature = cells[0].get_text(strip=True)
        for i, cell in enumerate(cells[1:]):
            if i < num_models:
                models[i][feature] = cell.get_text(strip=True)

    return models


def _find_column(row: dict, *candidates: str) -> str:
    """Find a value in a row dict by trying multiple possible column names."""
    for c in candidates:
        for key, val in row.items():
            if c.lower() in key.lower():
                return val
    return ""


def parse_models(soup: BeautifulSoup) -> list[dict]:
    """Extract model data from all tables on the page."""
    tables = soup.find_all("table")
    models = []

    for table in tables:
        model_dicts = _parse_table(table)
        for row in model_dicts:
            # Need at minimum a Claude API ID
            model_id = _find_column(row, "Claude API ID", "API ID")
            if not model_id or not model_id.startswith("claude"):
                continue

            alias = _find_column(row, "Claude API alias", "API alias")
            description = _find_column(row, "Description")
            pricing_text = _find_column(row, "Pricing")
            context_text = _find_column(row, "Context window")
            output_text = _find_column(row, "Max output")
            knowledge_cutoff_raw = _find_column(row, "Reliable knowledge cutoff", "knowledge cutoff")
            # Strip trailing footnote numbers (e.g., "May 20252" -> "May 2025")
            knowledge_cutoff = re.sub(r"(\d{4})\d+$", r"\1", knowledge_cutoff_raw)
            training_cutoff = _find_column(row, "Training data cutoff")

            input_price, output_price = _parse_pricing(pricing_text)
            context_window = _parse_token_count(context_text)
            max_output = _parse_token_count(output_text)

            # Use alias as the key if available, otherwise derive from id
            base_id = alias if alias and alias != "N/A" else _extract_base_id(model_id)
            # The full snapshot ID is the version
            snapshot_id = model_id if model_id != base_id else None

            models.append({
                "base_id": base_id,
                "snapshot_id": snapshot_id,
                "description": description,
                "input_price": input_price,
                "output_price": output_price,
                "context_window": context_window,
                "max_output": max_output,
                "knowledge_cutoff": knowledge_cutoff,
                "training_cutoff": training_cutoff,
            })

    return models


def _detect_status(model: dict) -> str:
    """Determine model status from description and context."""
    desc = model.get("description", "").lower()
    if "deprecated" in desc:
        return "deprecated"
    # Legacy models typically have snapshot IDs with dates
    base = model["base_id"]
    snapshot = model.get("snapshot_id")
    if snapshot and snapshot != base:
        # Has a dated snapshot — check if there's a newer base
        return "legacy"
    return "general-availability"


def _make_modalities() -> CommentedMap:
    """Claude models support text+image input, text output."""
    inp = CommentedMap({"text": True, "image": True, "audio": False, "video": False})
    out = CommentedMap({"text": True, "image": False, "audio": False, "video": False})
    return CommentedMap({"input": inp, "output": out})


def _make_endpoints() -> CommentedMap:
    return CommentedMap({
        "chat_completions": True,
        "responses": True,
    })


def build_model_entry(
    base_id: str,
    models: list[dict],
    existing: dict | None,
) -> CommentedMap:
    """Build a model entry for models.yaml from parsed data."""
    # Use the first (primary/current) model's data
    primary = models[0]
    ex = existing or {}

    entry = CommentedMap()
    entry["id"] = base_id
    entry["name"] = ex.get("name", base_id)
    entry["documentation_url"] = DOCUMENTATION_URL
    entry["description_short"] = ex.get("description_short", primary.get("description", ""))
    entry["description"] = ex.get("description", primary.get("description", ""))
    entry["status"] = ex.get("status", primary.get("status", "general-availability"))
    entry["knowledge_cutoff"] = primary.get("knowledge_cutoff", ex.get("knowledge_cutoff", ""))
    entry["context_window"] = primary.get("context_window", ex.get("context_window", 0))
    entry["max_output_tokens"] = primary.get("max_output", ex.get("max_output_tokens", 0))
    entry["validated"] = ex.get("validated", False)

    pricing = CommentedMap()
    ex_pricing = ex.get("pricing", {})
    pricing["input_per_million"] = primary.get("input_price", ex_pricing.get("input_per_million", 0))
    pricing["output_per_million"] = primary.get("output_price", ex_pricing.get("output_per_million", 0))
    entry["pricing"] = pricing

    entry["modalities"] = _make_modalities()
    entry["endpoints"] = _make_endpoints()

    # Versions from snapshot IDs
    version_entries = CommentedSeq()
    for m in models:
        sid = m.get("snapshot_id")
        if not sid or sid == base_id:
            continue
        ve = CommentedMap()
        ve["id"] = sid
        date_m = re.search(r"(\d{8})", sid)
        if date_m:
            d = date_m.group(1)
            ve["release_date"] = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        else:
            ve["release_date"] = ""
        ve["isDefault"] = True
        ve["isDeprecated"] = False
        version_entries.append(ve)

    if version_entries:
        entry["versions"] = version_entries

    return entry


def main():
    parser = argparse.ArgumentParser(description="Update Anthropic Claude models in models.yaml")
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

    # 1. Scrape models page
    print("Fetching Anthropic models page...", file=sys.stderr)
    soup = fetch_models_page()

    # 2. Parse model data from tables
    raw_models = parse_models(soup)
    print(f"  Found {len(raw_models)} model entries on page", file=sys.stderr)

    if not raw_models:
        print("Error: no models found on page. Page structure may have changed.", file=sys.stderr)
        sys.exit(1)

    # 3. Group by base_id, determine status
    # Current models (alias == base_id, no dated suffix) come first;
    # legacy models group under their base_id
    groups: dict[str, list[dict]] = {}
    current_ids: set[str] = set()

    for m in raw_models:
        base = m["base_id"]
        groups.setdefault(base, []).append(m)
        # Track which base_ids appear as aliases (current models)
        if not m.get("snapshot_id") or m["snapshot_id"] == base:
            current_ids.add(base)

    # Set status on grouped models
    for base_id, model_list in groups.items():
        is_current = base_id in current_ids
        for m in model_list:
            desc = m.get("description", "").lower()
            if "deprecated" in desc:
                m["status"] = "deprecated"
            elif is_current:
                m["status"] = "general-availability"
            else:
                m["status"] = "legacy"

    # 4. Load existing YAML
    yml = YAML()
    yml.preserve_quotes = True
    yml.width = 4096
    with open(args.yaml_path) as f:
        data = yml.load(f)

    existing_models = dict(data["providers"]["claude"].get("models", {}) or {})

    # 5. Build entries
    new_models = CommentedMap()
    for base_id in sorted(groups.keys()):
        print(f"  Processing {base_id}...", file=sys.stderr)
        existing = dict(existing_models[base_id]) if base_id in existing_models else None
        entry = build_model_entry(base_id, groups[base_id], existing)
        new_models[base_id] = entry

    # 6. Update YAML
    data["providers"]["claude"]["models"] = new_models

    if args.dry_run:
        yml.dump(data, sys.stdout)
        print(f"\n[dry-run] Would update {len(new_models)} models", file=sys.stderr)
    else:
        with open(args.yaml_path, "w") as f:
            yml.dump(data, f)
        print(f"Updated {len(new_models)} Claude models in {args.yaml_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
