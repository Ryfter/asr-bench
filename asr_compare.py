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


def _model_der(model_doc: dict) -> Optional[float]:
    """Per-model DER = mean of non-null clip `der` values (DER is per-clip, not in
    aggregates). None if no clip has a der."""
    ders = [c.get("der") for c in model_doc.get("clips", [])
            if c.get("der") is not None]
    return round(sum(ders) / len(ders), 10) if ders else None


def _model_value(model_doc: dict, metric: str) -> Optional[float]:
    if metric == "der":
        return _model_der(model_doc)
    return model_doc.get("aggregates", {}).get(_AGG_KEYS[metric])


def _has_any_der(docs: List[dict]) -> bool:
    return any(c.get("der") is not None
               for d in docs for m in d.get("models", [])
               for c in m.get("clips", []))


def compare_runs(docs: List[dict], *, mode: str) -> dict:
    """Pure builder. Joins per-model headline metrics on model_id across `docs`
    (input order; docs[0] is the baseline in delta mode). Returns a report dict."""
    runs = [{"label": d.get("_source_label", "?"),
             "corpus": d.get("run", {}).get("corpus"),
             "config": d.get("run", {}).get("config", {})} for d in docs]
    metrics = ["wer", "mer", "wil", "rtfx"] + (["der"] if _has_any_der(docs) else [])

    # union of model_ids in first-seen order; remember display + per-run lookup
    order: List[str] = []
    display: Dict[str, str] = {}
    per_run: List[Dict[str, dict]] = []
    for d in docs:
        idx: Dict[str, dict] = {}
        for m in d.get("models", []):
            mid = m.get("model_id")
            idx[mid] = m
            if mid not in display:
                order.append(mid)
                display[mid] = m.get("display", mid)
        per_run.append(idx)

    models: List[dict] = []
    for mid in order:
        present_in = [i for i, idx in enumerate(per_run) if mid in idx]
        values: Dict[str, List[Optional[float]]] = {k: [] for k in metrics}
        for idx in per_run:
            m = idx.get(mid)
            for k in metrics:
                values[k].append(_model_value(m, k) if m is not None else None)
        entry: dict = {"model_id": mid, "display": display[mid],
                       "present_in": present_in, "values": values}
        if mode == "delta":
            entry["status"] = ("both" if set(present_in) >= {0, 1}
                               else "removed" if 0 in present_in else "added")
        models.append(entry)

    report: dict = {"mode": mode, "runs": runs, "metrics": metrics, "models": models}
    return report


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
