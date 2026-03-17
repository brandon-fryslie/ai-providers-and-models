#!/usr/bin/env python3
"""Convert models.yaml to models.json for GitHub Pages."""

import json
import sys
from pathlib import Path

from ruamel.yaml import YAML


def to_plain(obj):
    """Convert ruamel.yaml objects to plain Python types."""
    if hasattr(obj, "items"):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain(i) for i in obj]
    return obj


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.yaml> <output.json>", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    yml = YAML()
    data = yml.load(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(to_plain(data), indent=2))
    print(f"Wrote {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
