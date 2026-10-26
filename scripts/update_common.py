"""Failure isolation shared by the provider update scripts.

One model failing must not cost the rest of its provider's update, and must
not silently drop or corrupt that model's data: it keeps its last good entry,
and the script exits non-zero so the scheduled run shows red.
"""

import sys
import traceback
from typing import Callable

from ruamel.yaml.comments import CommentedMap


def build_entries(
    base_ids: list[str],
    existing_models: dict,
    build: Callable[[str], dict],
) -> tuple[CommentedMap, list[str]]:
    """Build an entry per model id. Returns (entries, ids that failed)."""
    entries = CommentedMap()
    failed = []
    for base_id in base_ids:
        print(f"  Processing {base_id}...", file=sys.stderr)
        try:
            entries[base_id] = build(base_id)
        except Exception:
            failed.append(base_id)
            print(f"  ERROR: {base_id} failed; keeping its previous entry", file=sys.stderr)
            traceback.print_exc()
            if base_id in existing_models:
                entries[base_id] = existing_models[base_id]
    return entries, failed


def exit_if_failed(provider: str, failed: list[str]) -> None:
    if failed:
        print(f"::error::{provider}: {len(failed)} model(s) failed to update: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)
