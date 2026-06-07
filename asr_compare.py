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

# config fields whose mismatch makes a cross-run comparison suspect
_CONFIG_KEYS = ["device", "compute_type", "beam_size", "vad_filter", "batch_size"]


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


# per-clip comparison covers these (per-clip der/wer are stored on each clip)
_CLIP_METRICS = ["wer", "der"]


def _clip_value(clip: dict, metric: str) -> Optional[float]:
    return clip.get(metric)


def _build_clip_table(per_run_clips: List[Dict[str, dict]], *, mode: str) -> dict:
    """per_run_clips[i] maps clip-basename -> clip dict for run i. Returns
    {clip_order: [...], clips: {name: {present_in, values, deltas?}}}.
    In delta mode, deltas are candidate-minus-baseline over per_run_clips[1] and
    [0]; the caller (compare_runs) guarantees exactly 2 runs in delta mode."""
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


def compare_runs(docs: List[dict], *, mode: str, per_clip: bool = False) -> dict:
    """Pure builder. Joins per-model headline metrics on model_id across `docs`
    (input order; docs[0] is the baseline in delta mode). Returns a report dict."""
    if mode == "delta" and len(docs) != 2:
        raise ValueError(f"delta mode requires exactly 2 docs, got {len(docs)}")
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
            deltas: Dict[str, Optional[float]] = {}
            for k in metrics:
                base_v, cand_v = values[k][0], values[k][1]
                deltas[k] = ((cand_v - base_v)
                             if base_v is not None and cand_v is not None else None)
            entry["deltas"] = deltas
        if per_clip:
            per_run_clips: List[Dict[str, dict]] = []
            for ri, idx in enumerate(per_run):
                cm = idx.get(mid)
                cmap: Dict[str, dict] = {}
                if cm is not None:
                    for c in cm.get("clips", []):
                        name = Path(c.get("audio", "")).name
                        if name in cmap:
                            print(f"warning: duplicate clip basename {name!r} in "
                                  f"run {runs[ri]['label']!r}; per-clip view keeps "
                                  f"the last.", file=sys.stderr)
                        cmap[name] = c
                per_run_clips.append(cmap)
            entry.update(_build_clip_table(per_run_clips, mode=mode))
        models.append(entry)

    report: dict = {"mode": mode, "runs": runs, "metrics": metrics,
                    "models": models, "warnings": _mismatch_warnings(docs),
                    "per_clip": per_clip}
    return report


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
    if report.get("per_clip"):
        lines += _render_per_clip(report)
    lines.append("")
    return "\n".join(lines)


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
        if not results_dir.is_dir():
            print(f"warning: --results-dir {str(results_dir)!r} not found; "
                  f"--last ignored.", file=sys.stderr)
            recent: List[Path] = []
        else:
            recent = sorted(results_dir.glob("*.json"))[-last:]
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

    if ns.last is not None and ns.last < 1:
        print("error: --last must be a positive integer.", file=sys.stderr)
        return 2

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
        print(f"Wrote comparison to {out}", file=sys.stderr)
    else:
        print(md)
    return 0


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
