"""asr_bench compare — delta/matrix comparison across results JSON sidecars.

Standalone and pure: reads schema_version 1 sidecars (plain dicts) and renders a
markdown comparison. Imports nothing from asr_bench.py so it stays independently
testable and torch-free.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

SCHEMA_VERSION = 1

# metric key -> rendering + direction-of-good. pct metrics are stored as fractions
# (0.089) and rendered x100 (8.9). lower_better drives the delta ✓/✗ mark.
METRIC_META = {
    "wer":  {"label": "WER%", "pct": True,  "lower_better": True},
    "mer":  {"label": "MER%", "pct": True,  "lower_better": True},
    "wil":  {"label": "WIL%", "pct": True,  "lower_better": True},
    "rtfx": {"label": "RTFx", "pct": False, "lower_better": False},
    "der":  {"label": "DER%", "pct": True,  "lower_better": True},
}

# non-der metric key -> the aggregates field it comes from
_AGG_KEYS = {"wer": "avg_wer", "mer": "avg_mer", "wil": "avg_wil",
             "rtfx": "aggregate_rtfx"}


def load_results_json(path) -> Optional[dict]:
    """Read a results sidecar; return the dict if schema_version == 1, else None.
    Tags the dict with `_source_label` (the file stem) for display. Unreadable
    files, invalid JSON, or a different schema_version warn to stderr -> None."""
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"warning: skipping {path}: {e}", file=sys.stderr)
        return None
    if doc.get("schema_version") != SCHEMA_VERSION:
        print(f"warning: skipping {path}: schema_version "
              f"{doc.get('schema_version')!r} != {SCHEMA_VERSION}", file=sys.stderr)
        return None
    doc["_source_label"] = path.stem
    return doc
