# `asr_bench compare` Subcommand Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `compare` subcommand that reads 2+ `schema_version 1` results JSON sidecars and renders a markdown delta (2 files) or matrix (3+) comparison of per-model metrics.

**Architecture:** A standalone, pure, torch-free module `asr_compare.py` (reads plain JSON dicts, imports nothing from `asr_bench.py`) plus a tiny first-positional-keyword pre-dispatch at the top of `asr_bench.py`'s `main()`. Builder/renderer are pure functions over dicts; `compare_main` handles CLI + I/O.

**Tech Stack:** Python 3.14 stdlib only (`argparse`, `json`, `pathlib`, `datetime`). pytest. No torch, no third-party deps.

**Reference:** spec at `docs/superpowers/specs/2026-06-06-asr-compare-subcommand-design.md`.

---

## File Structure

- **Create `asr_compare.py`** — the whole feature: `SCHEMA_VERSION`, `METRIC_META`, `_AGG_KEYS`, `load_results_json`, per-model helpers, `compare_runs`, formatting helpers, `render_comparison_markdown`, `_resolve_files`, `compare_main`.
- **Modify `asr_bench.py`** — only `main()` gets a 3-line pre-dispatch at its very top (current `main()` starts at line 2273; entry point `sys.exit(main())` at line 2586). Nothing else changes.
- **Create `tests/test_compare.py`** — all tests, with a `_doc(...)` synthetic-sidecar helper.
- **Modify `README.md`, `CLAUDE.md`, `SPEC.md`** — docs (Task 8).

The sidecar shape these consume (from the shipped JSON sidecar, `schema_version 1`):
```json
{
  "schema_version": 1,
  "run": {"corpus": "...", "config": {"device": "...", "compute_type": "...",
          "beam_size": 5, "vad_filter": true, "batch_size": 1, ...}},
  "models": [{"model_id": "large-v3-turbo", "display": "Whisper Large V3 Turbo",
              "aggregates": {"avg_wer": 0.089, "avg_mer": 0.08, "avg_wil": 0.10,
                             "aggregate_rtfx": 64.8, ...},
              "clips": [{"audio": "lec.mp4", "wer": 0.089, "der": 0.138,
                         "num_speakers": 2, ...}]}]
}
```

---

## Task 1: Module skeleton, constants, and `load_results_json`

**Files:**
- Create: `asr_compare.py`
- Test: `tests/test_compare.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_compare.py`:

```python
import json
from pathlib import Path

import asr_compare


def _doc(stem="run", *, models=None, corpus="test-corpus", config=None,
         schema_version=1):
    """Build a minimal schema_version-1 sidecar dict (already labeled)."""
    doc = {
        "schema_version": schema_version,
        "run": {
            "corpus": corpus,
            "config": config or {"device": "cuda", "compute_type": "float16",
                                 "beam_size": 5, "vad_filter": True, "batch_size": 1},
        },
        "models": models if models is not None else [],
        "_source_label": stem,
    }
    return doc


def _model(model_id="m", display=None, *, wer=0.10, mer=0.09, wil=0.12,
           rtfx=60.0, clips=None):
    return {
        "model_id": model_id,
        "display": display or model_id,
        "aggregates": {"avg_wer": wer, "avg_mer": mer, "avg_wil": wil,
                       "aggregate_rtfx": rtfx, "peak_vram_bytes": None},
        "clips": clips if clips is not None else [],
    }


def test_load_valid_v1(tmp_path):
    p = tmp_path / "20260606-120000.json"
    p.write_text(json.dumps(_doc()), encoding="utf-8")
    doc = asr_compare.load_results_json(p)
    assert doc is not None
    assert doc["schema_version"] == 1
    assert doc["_source_label"] == "20260606-120000"


def test_load_wrong_schema_version_returns_none(tmp_path, capsys):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(_doc(schema_version=2)), encoding="utf-8")
    assert asr_compare.load_results_json(p) is None
    assert "schema_version" in capsys.readouterr().err


def test_load_missing_file_returns_none(tmp_path, capsys):
    assert asr_compare.load_results_json(tmp_path / "nope.json") is None
    assert "skipping" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_compare.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'asr_compare'`.

- [ ] **Step 3: Create `asr_compare.py` with constants and `load_results_json`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_compare.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add asr_compare.py tests/test_compare.py
git commit -m "feat(compare): module skeleton + load_results_json with schema guard"
```

---

## Task 2: `compare_runs` — join, values, metric set, status

**Files:**
- Modify: `asr_compare.py`
- Test: `tests/test_compare.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_compare.py`:

```python
def test_join_shared_model_collects_both_values():
    a = _doc("a", models=[_model("big", "Big", wer=0.10, rtfx=60.0)])
    b = _doc("b", models=[_model("big", "Big", wer=0.08, rtfx=70.0)])
    rep = asr_compare.compare_runs([a, b], mode="delta")
    row = [m for m in rep["models"] if m["model_id"] == "big"][0]
    assert row["values"]["wer"] == [0.10, 0.08]
    assert row["values"]["rtfx"] == [60.0, 70.0]
    assert row["present_in"] == [0, 1]
    assert row["status"] == "both"


def test_model_only_in_baseline_is_removed():
    a = _doc("a", models=[_model("old", "Old"), _model("keep", "Keep")])
    b = _doc("b", models=[_model("keep", "Keep")])
    rep = asr_compare.compare_runs([a, b], mode="delta")
    old = [m for m in rep["models"] if m["model_id"] == "old"][0]
    assert old["status"] == "removed"
    assert old["values"]["wer"] == [0.10, None]


def test_model_only_in_candidate_is_added():
    a = _doc("a", models=[_model("keep", "Keep")])
    b = _doc("b", models=[_model("keep", "Keep"), _model("new", "New")])
    rep = asr_compare.compare_runs([a, b], mode="delta")
    new = [m for m in rep["models"] if m["model_id"] == "new"][0]
    assert new["status"] == "added"
    assert new["values"]["wer"] == [None, 0.10]


def test_der_metric_absent_when_no_clip_der():
    a = _doc("a", models=[_model("m")])
    b = _doc("b", models=[_model("m")])
    rep = asr_compare.compare_runs([a, b], mode="delta")
    assert "der" not in rep["metrics"]


def test_der_metric_present_and_averaged_from_clips():
    clips = [{"audio": "x.mp4", "der": 0.10, "num_speakers": 2},
             {"audio": "y.mp4", "der": 0.20, "num_speakers": 2}]
    a = _doc("a", models=[_model("m", clips=clips)])
    b = _doc("b", models=[_model("m", clips=[{"audio": "x.mp4", "der": None,
                                              "num_speakers": None}])])
    rep = asr_compare.compare_runs([a, b], mode="matrix")
    assert "der" in rep["metrics"]
    row = rep["models"][0]
    assert row["values"]["der"][0] == 0.15      # (0.10 + 0.20) / 2
    assert row["values"]["der"][1] is None      # all-null clips -> None


def test_model_union_preserves_first_seen_order():
    a = _doc("a", models=[_model("z"), _model("a")])
    b = _doc("b", models=[_model("a"), _model("q")])
    rep = asr_compare.compare_runs([a, b], mode="matrix")
    assert [m["model_id"] for m in rep["models"]] == ["z", "a", "q"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_compare.py -k "join or removed or added or der or union" -v`
Expected: FAIL — `AttributeError: module 'asr_compare' has no attribute 'compare_runs'`.

- [ ] **Step 3: Implement `compare_runs` (join/values/metrics/status; deltas/warnings come in Task 3)**

Append to `asr_compare.py`:

```python
def _model_der(model_doc: dict) -> Optional[float]:
    """Per-model DER = mean of non-null clip `der` values (DER is per-clip, not in
    aggregates). None if no clip has a der."""
    ders = [c.get("der") for c in model_doc.get("clips", [])
            if c.get("der") is not None]
    return sum(ders) / len(ders) if ders else None


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
            entry["status"] = ("both" if present_in == [0, 1] or set(present_in) >= {0, 1}
                               else "removed" if 0 in present_in else "added")
        models.append(entry)

    report: dict = {"mode": mode, "runs": runs, "metrics": metrics, "models": models}
    return report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_compare.py -v`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
git add asr_compare.py tests/test_compare.py
git commit -m "feat(compare): compare_runs join, metric set, added/removed status"
```

---

## Task 3: Deltas and mismatch warnings

**Files:**
- Modify: `asr_compare.py`
- Test: `tests/test_compare.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_compare.py`:

```python
def test_deltas_candidate_minus_baseline():
    a = _doc("a", models=[_model("m", wer=0.10, rtfx=60.0)])
    b = _doc("b", models=[_model("m", wer=0.08, rtfx=70.0)])
    rep = asr_compare.compare_runs([a, b], mode="delta")
    d = rep["models"][0]["deltas"]
    assert round(d["wer"], 4) == -0.02
    assert round(d["rtfx"], 4) == 10.0


def test_delta_none_when_value_missing():
    a = _doc("a", models=[_model("m", wer=0.10)])
    b = _doc("b", models=[_model("other", wer=0.08)])
    rep = asr_compare.compare_runs([a, b], mode="delta")
    m = [x for x in rep["models"] if x["model_id"] == "m"][0]
    assert m["deltas"]["wer"] is None


def test_warning_on_corpus_mismatch():
    a = _doc("a", corpus="corpus-A", models=[_model("m")])
    b = _doc("b", corpus="corpus-B", models=[_model("m")])
    rep = asr_compare.compare_runs([a, b], mode="delta")
    assert any("corpus differs" in w for w in rep["warnings"])


def test_warning_on_beam_size_mismatch():
    a = _doc("a", models=[_model("m")],
             config={"device": "cuda", "compute_type": "float16",
                     "beam_size": 5, "vad_filter": True, "batch_size": 1})
    b = _doc("b", models=[_model("m")],
             config={"device": "cuda", "compute_type": "float16",
                     "beam_size": 1, "vad_filter": True, "batch_size": 1})
    rep = asr_compare.compare_runs([a, b], mode="delta")
    assert any("beam_size differs" in w for w in rep["warnings"])


def test_no_warnings_when_runs_match():
    a = _doc("a", models=[_model("m")])
    b = _doc("b", models=[_model("m")])
    rep = asr_compare.compare_runs([a, b], mode="delta")
    assert rep["warnings"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_compare.py -k "delta or warning or match" -v`
Expected: FAIL — `KeyError: 'deltas'` / `KeyError: 'warnings'`.

- [ ] **Step 3: Add deltas to `compare_runs` and a `_mismatch_warnings` helper**

In `asr_compare.py`, add the config-keys constant near the top constants (after `_AGG_KEYS`):

```python
# config fields whose mismatch makes a cross-run comparison suspect
_CONFIG_KEYS = ["device", "compute_type", "beam_size", "vad_filter", "batch_size"]
```

Add the helper (place it above `compare_runs`):

```python
def _mismatch_warnings(docs: List[dict]) -> List[str]:
    """Warn when later runs differ from docs[0] in corpus or key config."""
    out: List[str] = []
    base = docs[0].get("run", {})
    base_corpus = base.get("corpus")
    base_cfg = base.get("config", {})
    base_label = docs[0].get("_source_label", "run[0]")
    for i in range(1, len(docs)):
        run = docs[i].get("run", {})
        label = docs[i].get("_source_label", f"run[{i}]")
        if run.get("corpus") != base_corpus:
            out.append(f"corpus differs: {label} used {run.get('corpus')!r} "
                       f"(baseline {base_label} used {base_corpus!r})")
        cfg = run.get("config", {})
        for key in _CONFIG_KEYS:
            if cfg.get(key) != base_cfg.get(key):
                out.append(f"{key} differs: {label}={cfg.get(key)!r} "
                           f"vs baseline {base_label}={base_cfg.get(key)!r}")
    return out
```

In `compare_runs`, inside the `if mode == "delta":` block add delta computation, and set `report["warnings"]` before returning. Replace the delta block and the report assembly with:

```python
        if mode == "delta":
            entry["status"] = ("both" if set(present_in) >= {0, 1}
                               else "removed" if 0 in present_in else "added")
            deltas: Dict[str, Optional[float]] = {}
            for k in metrics:
                base_v, cand_v = values[k][0], values[k][1]
                deltas[k] = ((cand_v - base_v)
                             if base_v is not None and cand_v is not None else None)
            entry["deltas"] = deltas
        models.append(entry)

    report: dict = {"mode": mode, "runs": runs, "metrics": metrics,
                    "models": models, "warnings": _mismatch_warnings(docs)}
    return report
```

(Note: in matrix mode `values[k]` may have >2 entries and there is no `deltas`/`status` key — that is intended; matrix render never reads them.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_compare.py -v`
Expected: PASS (all tests so far).

- [ ] **Step 5: Commit**

```bash
git add asr_compare.py tests/test_compare.py
git commit -m "feat(compare): per-model deltas + corpus/config mismatch warnings"
```

---

## Task 4: Formatting helpers + `render_comparison_markdown` (delta + matrix)

**Files:**
- Modify: `asr_compare.py`
- Test: `tests/test_compare.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_compare.py`:

```python
def test_fmt_pct_and_rtfx_and_none():
    assert asr_compare._fmt("wer", 0.089) == "8.9"
    assert asr_compare._fmt("rtfx", 64.8) == "64.8"
    assert asr_compare._fmt("wer", None) == "—"


def test_delta_mark_direction():
    # WER lower is better: negative delta -> improvement
    assert asr_compare._delta_mark("wer", -0.02) == "✓"
    assert asr_compare._delta_mark("wer", 0.02) == "✗"
    # RTFx higher is better: positive delta -> improvement
    assert asr_compare._delta_mark("rtfx", 10.0) == "✓"
    assert asr_compare._delta_mark("rtfx", -10.0) == "✗"
    # zero -> no mark
    assert asr_compare._delta_mark("wer", 0.0) == ""


def test_render_delta_has_models_metrics_and_marks():
    a = _doc("runA", models=[_model("big", "Big Model", wer=0.10, rtfx=60.0)])
    b = _doc("runB", models=[_model("big", "Big Model", wer=0.08, rtfx=70.0)])
    rep = asr_compare.compare_runs([a, b], mode="delta")
    md = asr_compare.render_comparison_markdown(rep)
    assert "# ASR Run Comparison" in md
    assert "Big Model" in md
    assert "WER%" in md and "RTFx" in md
    assert "✓" in md                      # both metrics improved
    assert "8.0" in md                     # candidate WER 0.08 -> 8.0


def test_render_matrix_has_one_column_per_run():
    a = _doc("runA", models=[_model("m", "M", wer=0.10)])
    b = _doc("runB", models=[_model("m", "M", wer=0.09)])
    c = _doc("runC", models=[_model("m", "M", wer=0.08)])
    rep = asr_compare.compare_runs([a, b, c], mode="matrix")
    md = asr_compare.render_comparison_markdown(rep)
    assert "`runA`" in md and "`runB`" in md and "`runC`" in md
    assert "10.0" in md and "9.0" in md and "8.0" in md


def test_render_warnings_as_blockquotes():
    a = _doc("a", corpus="A", models=[_model("m")])
    b = _doc("b", corpus="B", models=[_model("m")])
    md = asr_compare.render_comparison_markdown(
        asr_compare.compare_runs([a, b], mode="delta"))
    assert "> ⚠️" in md
    assert "corpus differs" in md


def test_render_added_removed_show_dash():
    a = _doc("a", models=[_model("old", "Old", wer=0.10)])
    b = _doc("b", models=[_model("new", "New", wer=0.08)])
    md = asr_compare.render_comparison_markdown(
        asr_compare.compare_runs([a, b], mode="delta"))
    assert "removed" in md and "added" in md
    assert "—" in md
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_compare.py -k "fmt or mark or render" -v`
Expected: FAIL — `AttributeError: ... has no attribute '_fmt'`.

- [ ] **Step 3: Implement formatting + render**

Append to `asr_compare.py`:

```python
def _fmt(metric: str, value: Optional[float]) -> str:
    if value is None:
        return "—"
    if METRIC_META[metric]["pct"]:
        return f"{value * 100:.1f}"
    return f"{value:.1f}"


def _delta_mark(metric: str, delta: Optional[float]) -> str:
    if delta is None or delta == 0:
        return ""
    improved = (delta < 0) == METRIC_META[metric]["lower_better"]
    return "✓" if improved else "✗"


def _fmt_delta(metric: str, delta: Optional[float]) -> str:
    if delta is None:
        return ""
    scale = 100 if METRIC_META[metric]["pct"] else 1
    return f"{delta * scale:+.1f}"


def _render_delta(report: dict) -> List[str]:
    metrics = report["metrics"]
    head = "| Model | Status | " + " | ".join(METRIC_META[m]["label"]
                                              for m in metrics) + " |"
    sep = "|" + "---|" * (2 + len(metrics))
    rows = [head, sep]
    for m in report["models"]:
        cells = [m["display"], m.get("status", "")]
        for k in metrics:
            base_v, cand_v = m["values"][k][0], m["values"][k][1]
            if base_v is None and cand_v is None:
                cells.append("—")
                continue
            txt = f"{_fmt(k, base_v)} → {_fmt(k, cand_v)}"
            d = m.get("deltas", {}).get(k)
            if d is not None:
                mark = _delta_mark(k, d)
                txt += f" ({_fmt_delta(k, d)}{(' ' + mark) if mark else ''})"
            cells.append(txt)
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def _render_matrix(report: dict) -> List[str]:
    runs = report["runs"]
    metrics = report["metrics"]
    head = "| Model | Metric | " + " | ".join(f"`{r['label']}`" for r in runs) + " |"
    sep = "|" + "---|" * (2 + len(runs))
    rows = [head, sep]
    for m in report["models"]:
        for k in metrics:
            cells = [m["display"], METRIC_META[k]["label"]]
            cells += [_fmt(k, m["values"][k][i]) for i in range(len(runs))]
            rows.append("| " + " | ".join(cells) + " |")
    return rows


def render_comparison_markdown(report: dict) -> str:
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines: List[str] = ["# ASR Run Comparison", "", f"_Generated {now}_", "",
                        "Runs compared:"]
    for i, r in enumerate(report["runs"]):
        tag = " (baseline)" if report["mode"] == "delta" and i == 0 else ""
        lines.append(f"- `{r['label']}`{tag} — corpus `{r['corpus']}`")
    lines.append("")
    for w in report.get("warnings", []):
        lines.append(f"> ⚠️ {w}")
    if report.get("warnings"):
        lines.append("")
    lines += (_render_delta(report) if report["mode"] == "delta"
              else _render_matrix(report))
    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_compare.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add asr_compare.py tests/test_compare.py
git commit -m "feat(compare): markdown render for delta + matrix views"
```

---

## Task 5: `--per-clip` detail

**Files:**
- Modify: `asr_compare.py`
- Test: `tests/test_compare.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_compare.py`:

```python
def test_per_clip_joins_by_audio_basename():
    ca = [{"audio": "wk1.mp4", "wer": 0.10, "der": None, "num_speakers": None}]
    cb = [{"audio": "wk1.mp4", "wer": 0.08, "der": None, "num_speakers": None}]
    a = _doc("a", models=[_model("m", "M", clips=ca)])
    b = _doc("b", models=[_model("m", "M", clips=cb)])
    rep = asr_compare.compare_runs([a, b], mode="delta", per_clip=True)
    assert rep["per_clip"] is True
    mrow = rep["models"][0]
    assert mrow["clip_order"] == ["wk1.mp4"]
    clip = mrow["clips"]["wk1.mp4"]
    assert clip["values"]["wer"] == [0.10, 0.08]
    assert round(clip["deltas"]["wer"], 4) == -0.02


def test_per_clip_render_has_clip_section():
    ca = [{"audio": "wk1.mp4", "wer": 0.10, "der": None, "num_speakers": None}]
    cb = [{"audio": "wk1.mp4", "wer": 0.08, "der": None, "num_speakers": None}]
    a = _doc("a", models=[_model("m", "Model M", clips=ca)])
    b = _doc("b", models=[_model("m", "Model M", clips=cb)])
    rep = asr_compare.compare_runs([a, b], mode="delta", per_clip=True)
    md = asr_compare.render_comparison_markdown(rep)
    assert "Per-clip: Model M" in md
    assert "wk1.mp4" in md


def test_per_clip_default_off():
    a = _doc("a", models=[_model("m")])
    b = _doc("b", models=[_model("m")])
    rep = asr_compare.compare_runs([a, b], mode="delta")
    assert rep["per_clip"] is False
    assert "clips" not in rep["models"][0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_compare.py -k "per_clip" -v`
Expected: FAIL — `TypeError: compare_runs() got an unexpected keyword argument 'per_clip'`.

- [ ] **Step 3: Add `per_clip` to `compare_runs` and a per-clip render section**

In `asr_compare.py`, change the `compare_runs` signature and, when `per_clip`, attach per-clip data to each model entry. Update the signature line and add the per-clip block just before `models.append(entry)`:

```python
def compare_runs(docs: List[dict], *, mode: str, per_clip: bool = False) -> dict:
```

Add this helper above `compare_runs`:

```python
# per-clip comparison covers these (per-clip der/wer are stored on each clip)
_CLIP_METRICS = ["wer", "der"]


def _clip_value(clip: dict, metric: str) -> Optional[float]:
    return clip.get(metric)


def _build_clip_table(per_run_clips: List[Dict[str, dict]], *, mode: str) -> dict:
    """per_run_clips[i] maps clip-basename -> clip dict for run i. Returns
    {clip_order: [...], clips: {name: {present_in, values, deltas?}}}."""
    order: List[str] = []
    seen = set()
    for idx in per_run_clips:
        for name in idx:
            if name not in seen:
                seen.add(name)
                order.append(name)
    clips: Dict[str, dict] = {}
    for name in order:
        present_in = [i for i, idx in enumerate(per_run_clips) if name in idx]
        values: Dict[str, List[Optional[float]]] = {k: [] for k in _CLIP_METRICS}
        for idx in per_run_clips:
            c = idx.get(name)
            for k in _CLIP_METRICS:
                values[k].append(_clip_value(c, k) if c is not None else None)
        cell: dict = {"present_in": present_in, "values": values}
        if mode == "delta":
            cell["deltas"] = {
                k: ((values[k][1] - values[k][0])
                    if values[k][0] is not None and values[k][1] is not None else None)
                for k in _CLIP_METRICS}
        clips[name] = cell
    return {"clip_order": order, "clips": clips}
```

In the model loop, before `models.append(entry)`, add:

```python
        if per_clip:
            per_run_clips: List[Dict[str, dict]] = []
            for idx in per_run:
                m = idx.get(mid)
                cmap = {}
                if m is not None:
                    for c in m.get("clips", []):
                        cmap[Path(c.get("audio", "")).name] = c
                per_run_clips.append(cmap)
            entry.update(_build_clip_table(per_run_clips, mode=mode))
```

Set `report["per_clip"] = per_clip` in the report dict assembly:

```python
    report: dict = {"mode": mode, "runs": runs, "metrics": metrics,
                    "models": models, "warnings": _mismatch_warnings(docs),
                    "per_clip": per_clip}
    return report
```

Add a per-clip render block. In `render_comparison_markdown`, after the main table (`lines += (...)`) and before the trailing `lines.append("")`, insert:

```python
    if report.get("per_clip"):
        lines += _render_per_clip(report)
```

And add the renderer:

```python
def _render_per_clip(report: dict) -> List[str]:
    runs = report["runs"]
    out: List[str] = []
    for m in report["models"]:
        if not m.get("clip_order"):
            continue
        out += ["", f"### Per-clip: {m['display']}", ""]
        if report["mode"] == "delta":
            head = ("| Clip | " + " | ".join(METRIC_META[k]["label"]
                                            for k in _CLIP_METRICS) + " |")
            sep = "|" + "---|" * (1 + len(_CLIP_METRICS))
            out += [head, sep]
            for name in m["clip_order"]:
                cell = m["clips"][name]
                cells = [name]
                for k in _CLIP_METRICS:
                    base_v, cand_v = cell["values"][k][0], cell["values"][k][1]
                    if base_v is None and cand_v is None:
                        cells.append("—")
                        continue
                    txt = f"{_fmt(k, base_v)} → {_fmt(k, cand_v)}"
                    d = cell.get("deltas", {}).get(k)
                    if d is not None:
                        mark = _delta_mark(k, d)
                        txt += f" ({_fmt_delta(k, d)}{(' ' + mark) if mark else ''})"
                    cells.append(txt)
                out.append("| " + " | ".join(cells) + " |")
        else:
            head = ("| Clip | Metric | " + " | ".join(f"`{r['label']}`"
                                                      for r in runs) + " |")
            sep = "|" + "---|" * (2 + len(runs))
            out += [head, sep]
            for name in m["clip_order"]:
                cell = m["clips"][name]
                for k in _CLIP_METRICS:
                    row = [name, METRIC_META[k]["label"]]
                    row += [_fmt(k, cell["values"][k][i]) for i in range(len(runs))]
                    out.append("| " + " | ".join(row) + " |")
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_compare.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add asr_compare.py tests/test_compare.py
git commit -m "feat(compare): --per-clip detail (join clips by basename)"
```

---

## Task 6: `compare_main` — CLI, file resolution, flow

**Files:**
- Modify: `asr_compare.py`
- Test: `tests/test_compare.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_compare.py`:

```python
def _write(tmp_path, name, doc):
    p = tmp_path / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def test_compare_main_two_files_prints_table(tmp_path, capsys):
    a = _write(tmp_path, "a.json", _doc("a", models=[_model("m", "M", wer=0.10)]))
    b = _write(tmp_path, "b.json", _doc("b", models=[_model("m", "M", wer=0.08)]))
    rc = asr_compare.compare_main([str(a), str(b)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ASR Run Comparison" in out and "M" in out


def test_compare_main_one_file_errors(tmp_path, capsys):
    a = _write(tmp_path, "a.json", _doc("a", models=[_model("m")]))
    rc = asr_compare.compare_main([str(a)])
    assert rc != 0
    assert "at least 2" in capsys.readouterr().err


def test_compare_main_output_writes_file(tmp_path):
    a = _write(tmp_path, "a.json", _doc("a", models=[_model("m", "M")]))
    b = _write(tmp_path, "b.json", _doc("b", models=[_model("m", "M")]))
    out = tmp_path / "cmp.md"
    rc = asr_compare.compare_main([str(a), str(b), "--output", str(out)])
    assert rc == 0
    assert "ASR Run Comparison" in out.read_text(encoding="utf-8")


def test_compare_main_delta_force_with_three_files_errors(tmp_path, capsys):
    files = [str(_write(tmp_path, f"{n}.json", _doc(n, models=[_model("m")])))
             for n in ("a", "b", "c")]
    rc = asr_compare.compare_main(files + ["--delta"])
    assert rc != 0
    assert "--delta requires exactly 2" in capsys.readouterr().err


def test_compare_main_three_files_default_matrix(tmp_path, capsys):
    files = [str(_write(tmp_path, f"{n}.json",
                        _doc(n, models=[_model("m", "M", wer=0.10)])))
             for n in ("a", "b", "c")]
    asr_compare.compare_main(files)
    out = capsys.readouterr().out
    assert "| Model | Metric |" in out      # matrix header


def test_compare_main_matrix_force_with_two_files(tmp_path, capsys):
    a = _write(tmp_path, "a.json", _doc("a", models=[_model("m", "M")]))
    b = _write(tmp_path, "b.json", _doc("b", models=[_model("m", "M")]))
    asr_compare.compare_main([str(a), str(b), "--matrix"])
    assert "| Model | Metric |" in capsys.readouterr().out


def test_compare_main_last_n_selects_recent(tmp_path, capsys):
    rdir = tmp_path / "results"
    rdir.mkdir()
    for n in ("20260601-000000", "20260602-000000", "20260603-000000"):
        _write(rdir, f"{n}.json", _doc(n, models=[_model("m", "M")]))
    rc = asr_compare.compare_main(["--last", "2", "--results-dir", str(rdir)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "20260602-000000" in out and "20260603-000000" in out
    assert "20260601-000000" not in out


def test_compare_main_directory_arg_globs_json(tmp_path, capsys):
    d = tmp_path / "runs"
    d.mkdir()
    _write(d, "a.json", _doc("a", models=[_model("m", "M")]))
    _write(d, "b.json", _doc("b", models=[_model("m", "M")]))
    rc = asr_compare.compare_main([str(d)])
    assert rc == 0
    assert "ASR Run Comparison" in capsys.readouterr().out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_compare.py -k "compare_main" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'compare_main'`.

- [ ] **Step 3: Implement `_resolve_files` and `compare_main`**

Append to `asr_compare.py`:

```python
def _resolve_files(files: List[str], last: Optional[int],
                   results_dir: Path) -> List[Path]:
    """Build the ordered file list: directory args expand to their sorted *.json;
    --last prepends the N most-recent results_dir/*.json (ascending, so the older
    of the pair is the delta baseline), de-duplicated against explicit paths."""
    paths: List[Path] = []
    for f in files:
        p = Path(f)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.json")))
        else:
            paths.append(p)
    if last is not None and last > 0:
        recent = sorted(results_dir.glob("*.json"))[-last:] if results_dir.is_dir() else []
        existing = set(paths)
        paths = [r for r in recent if r not in existing] + paths
    return paths


def compare_main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="asr_bench.py compare",
        description="Compare 2+ asr-bench results JSON sidecars (schema_version 1).",
    )
    ap.add_argument("files", nargs="*",
                    help="Results JSON files, or a directory of them.")
    ap.add_argument("--last", type=int, default=None,
                    help="Use the N most-recent results/*.json (by filename).")
    ap.add_argument("--results-dir", default="results",
                    help="Directory that --last reads from.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--delta", action="store_true",
                   help="Force delta view (requires exactly 2 files).")
    g.add_argument("--matrix", action="store_true", help="Force matrix view.")
    ap.add_argument("--per-clip", action="store_true",
                    help="Include per-clip detail.")
    ap.add_argument("--output", default=None,
                    help="Write markdown to this path instead of stdout.")
    ns = ap.parse_args(argv)

    paths = _resolve_files(ns.files, ns.last, Path(ns.results_dir))
    docs = [d for d in (load_results_json(p) for p in paths) if d is not None]
    if len(docs) < 2:
        print(f"error: need at least 2 valid result files to compare "
              f"(got {len(docs)}).", file=sys.stderr)
        return 2
    if ns.delta and len(docs) != 2:
        print("error: --delta requires exactly 2 files.", file=sys.stderr)
        return 2

    if ns.matrix:
        mode = "matrix"
    elif ns.delta:
        mode = "delta"
    else:
        mode = "delta" if len(docs) == 2 else "matrix"

    report = compare_runs(docs, mode=mode, per_clip=ns.per_clip)
    md = render_comparison_markdown(report)
    if ns.output:
        out = Path(ns.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"Wrote comparison to {out}")
    else:
        print(md)
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_compare.py -v`
Expected: PASS (all compare tests).

- [ ] **Step 5: Commit**

```bash
git add asr_compare.py tests/test_compare.py
git commit -m "feat(compare): compare_main CLI, --last, directory globbing, file resolution"
```

---

## Task 7: Pre-dispatch in `asr_bench.py`

**Files:**
- Modify: `asr_bench.py:2273` (top of `main()`)
- Test: `tests/test_compare.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_compare.py`:

```python
import sys
import asr_bench
import asr_compare as _ac


def test_dispatch_routes_compare_to_compare_main(monkeypatch):
    captured = {}

    def fake_compare_main(argv):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(_ac, "compare_main", fake_compare_main)
    monkeypatch.setattr(sys, "argv", ["asr_bench.py", "compare", "x.json", "y.json"])
    rc = asr_bench.main()
    assert rc == 0
    assert captured["argv"] == ["x.json", "y.json"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_compare.py::test_dispatch_routes_compare_to_compare_main -v`
Expected: FAIL — `main()` builds the bench parser and (with argv `compare x.json y.json`) errors on unknown args / does not call `compare_main`.

- [ ] **Step 3: Add the pre-dispatch at the top of `main()`**

In `asr_bench.py`, the function currently begins:

```python
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark local Whisper variants on your own audio.",
```

Insert the dispatch as the first statements of `main()`:

```python
def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "compare":
        from asr_compare import compare_main
        return compare_main(argv[1:])
    ap = argparse.ArgumentParser(
        description="Benchmark local Whisper variants on your own audio.",
```

(The `from asr_compare import compare_main` runs each call, so the test's
`monkeypatch.setattr(asr_compare, "compare_main", ...)` is picked up.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_compare.py -v`
Expected: PASS.

Then the full suite (no regressions):
Run: `python -m pytest -q`
Expected: all prior tests still pass (137 + the new compare tests), 2 skipped.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_compare.py
git commit -m "feat(compare): route 'asr_bench.py compare ...' to compare_main"
```

---

## Task 8: Documentation + decision log

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `SPEC.md`

- [ ] **Step 1: README — add a "Comparing runs" section**

Add after the section that documents the JSON results sidecar (search README for
`results/` or "JSON"). Insert:

```markdown
### Comparing runs

Every run writes a `results/<timestamp>.json` sidecar. `compare` reads 2+ of them
and prints a markdown comparison:

```powershell
# 2 files -> delta view (baseline -> candidate, signed Δ with ✓/✗)
python asr_bench.py compare results/20260605-190913.json results/20260606-101500.json

# 3+ files -> matrix (one column per run)
python asr_bench.py compare results/a.json results/b.json results/c.json

# the two most recent runs, with per-clip detail
python asr_bench.py compare --last 2 --per-clip

# write to a file instead of stdout
python asr_bench.py compare results/a.json results/b.json --output report/compare.md
```

Force a layout with `--delta` (exactly 2 files) or `--matrix`. WER/MER/WIL/DER are
lower-is-better (improvement marked ✓); RTFx is higher-is-better. Differing corpus
or key config between runs is flagged as a ⚠️ warning (comparing WER across
different corpora is not meaningful).
```

- [ ] **Step 2: CLAUDE.md — Status note + workflow entry**

In the "What's new in v0.3" area (or a new "v0.3+" note), add a bullet:

```markdown
- **`compare` subcommand** — `python asr_bench.py compare a.json b.json` reads 2+
  `results/*.json` sidecars and renders a delta (2 files) or matrix (3+) markdown
  comparison of per-model WER/MER/WIL/RTFx/DER, with corpus/config mismatch
  warnings and optional `--per-clip` detail. Standalone `asr_compare.py`; bench CLI
  unchanged (first-positional `compare` keyword pre-dispatch).
```

Under "## Common workflows", add:

```markdown
### Compare runs across the JSON sidecars
```powershell
python asr_bench.py compare results/<old>.json results/<new>.json          # delta
python asr_bench.py compare --last 3                                        # matrix of 3 newest
python asr_bench.py compare results/a.json results/b.json --per-clip        # + per-clip
```
Reads `schema_version 1` sidecars. 2 files → delta view; 3+ → matrix; `--delta`/`--matrix` force.
```

- [ ] **Step 3: SPEC.md — mark `compare` shipped + decision log**

Find the line listing `asr_bench compare` as a future/planned item and move it to
shipped (mirror however the JSON sidecar line was marked shipped). Add a
decision-log entry:

```markdown
- **2026-06-06** — `compare` subcommand shipped: reads 2+ JSON sidecars, delta
  (2) / matrix (3+) markdown, joins per-model aggregates on model_id, warns on
  corpus/config drift. Implemented as a first-positional `compare` keyword
  pre-dispatch into a standalone, pure `asr_compare.py` — the existing bench CLI
  is byte-for-byte unchanged. Per-model DER averaged from per-clip `der`.
```

- [ ] **Step 4: Verify docs reference real flags**

Run: `python asr_bench.py compare --help`
Expected: usage shows `files`, `--last`, `--results-dir`, `--delta`, `--matrix`,
`--per-clip`, `--output`. Confirm the README/CLAUDE.md examples match.

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md SPEC.md
git commit -m "docs(compare): README + CLAUDE.md workflows + SPEC.md shipped"
```

---

## Final verification (after all tasks)

- [ ] `python -m pytest -q` — full suite green (prior 137 + new compare tests, 2 skipped).
- [ ] `python asr_bench.py --help` — bench CLI unchanged (no `compare` noise in it).
- [ ] `python asr_bench.py compare --help` — compare CLI present.
- [ ] Smoke: run a tiny bench twice (`--llm fake` or `--limit 1` if needed) to get two
      `results/*.json`, then `python asr_bench.py compare --last 2` renders a table.
