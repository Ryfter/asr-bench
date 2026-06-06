# JSON Results Sidecar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit a machine-readable `results/<timestamp>.json` sidecar alongside every markdown report, capturing the full run (config, per-model + per-clip metrics, transcripts, speaker/DER data) as NaN-safe JSON for cross-run aggregation.

**Architecture:** Two pure-ish functions added to `asr_bench.py` — `build_results_document(...)` (walks `ModelResult`/`ClipResult` → plain dict, sanitizing NaN/Inf → null and redacting secrets) and `write_results_json(...)` (serializes with `allow_nan=False` as a loud guard). A `--json/--no-json` flag (default on) and one call site in `main()` after the report is saved. The reproducibility command string is extracted from `render_markdown` into a shared helper so markdown + JSON agree.

**Tech Stack:** Python 3.14 (stdlib `json`, `dataclasses`, `pathlib`), pytest. No new dependencies. All tasks torch-free.

**Spec:** `docs/superpowers/specs/2026-06-05-json-results-sidecar-design.md`

---

## File Structure

- **`asr_bench.py`** (modify) — new `# ---- Results JSON sidecar ----` section (place it just before `render_markdown`, ~line 1875) containing:
  - `_json_sanitize(obj)` — recursive NaN/Inf → None, tuple → list.
  - `_reproducibility_command(args, results) -> str` — the `python asr_bench.py ...` string (extracted from `render_markdown`; both callers share it).
  - `build_results_document(...) -> dict`.
  - `write_results_json(document, json_path) -> Path`.
  - `render_markdown` edited to call `_reproducibility_command`.
  - `main()` edited: add `--json/--no-json`, write the JSON after the markdown save.
- **`tests/test_results_json.py`** (create) — unit tests for the four helpers + a `main()` integration test, reusing fixtures imported from `tests/test_render.py`.
- **`README.md`, `CLAUDE.md`, `SPEC.md`** (modify) — document the sidecar.

Tests import via `import asr_bench` (repo root on `sys.path` via `tests/conftest.py`).

**Security note carried through every task:** `RunConfig.hf_token` and `RunConfig.nim_api_key` are secrets and MUST NEVER appear in the JSON. The config dict is built field-by-field and simply omits them.

---

## Task 1: `_json_sanitize` helper

**Files:**
- Modify: `asr_bench.py` — new section before `render_markdown` (~line 1875)
- Test: `tests/test_results_json.py` (create)

- [ ] **Step 1: Write the failing test**

```python
import math
import asr_bench


def test_json_sanitize_nan_and_inf_to_none():
    out = asr_bench._json_sanitize(
        {"a": float("nan"), "b": float("inf"), "c": float("-inf"), "d": 1.5}
    )
    assert out["a"] is None and out["b"] is None and out["c"] is None
    assert out["d"] == 1.5


def test_json_sanitize_tuples_become_lists_recursively():
    out = asr_bench._json_sanitize({"segs": [(0.0, 1.0, "S0"), (1.0, 2.0, "S1")]})
    assert out["segs"] == [[0.0, 1.0, "S0"], [1.0, 2.0, "S1"]]
    assert isinstance(out["segs"][0], list)


def test_json_sanitize_passes_clean_values():
    val = {"x": 1, "y": "str", "z": [1, 2, {"w": None}], "b": True}
    assert asr_bench._json_sanitize(val) == val
```

- [ ] **Step 2: Run** `python -m pytest tests/test_results_json.py -v` → FAIL (`_json_sanitize` missing).

- [ ] **Step 3: Implement** — add the section header + function:

```python
# ---- Results JSON sidecar ---------------------------------------------------
def _json_sanitize(obj):
    """Make a value strictly-JSON-safe: NaN/Inf floats -> None, tuples -> lists,
    recursing through dicts and lists. (json allows NaN by default but emits the
    literal token `NaN`, which is invalid JSON for strict parsers.)"""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return obj
```

(`math` is already imported at the top of `asr_bench.py`.)

- [ ] **Step 4: Run** `python -m pytest tests/test_results_json.py -v` → PASS (3). Then `python -m pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_results_json.py
git commit -m "feat: _json_sanitize helper (NaN/Inf -> null, tuples -> lists)"
```

---

## Task 2: Extract `_reproducibility_command` helper

**Files:**
- Modify: `asr_bench.py` — new helper in the Results JSON section; edit `render_markdown` (~line 2116-2124) to call it
- Test: `tests/test_results_json.py` (append)

**Context:** `render_markdown` currently builds the reproducibility command inline (lines ~2116-2124). Extract it verbatim so the JSON `command` field and the markdown share one source of truth. The markdown wraps the string in backticks; the helper returns the raw string (no backticks).

- [ ] **Step 1: Write the failing test**

```python
import types


def _cmd_args():
    return types.SimpleNamespace(
        models=["small", "large-v3-turbo"], device="cuda", compute_type="float16",
        batch_size=1, beam_size=5, vad_filter=True,
        nim_url="localhost:50051", nim_model="", nim_language="en-US",
    )


def test_reproducibility_command_basic():
    cmd = asr_bench._reproducibility_command(_cmd_args(), Path("/corpus"), [_whisper_result()])
    assert cmd.startswith("python asr_bench.py --corpus '/corpus' --models small,large-v3-turbo")
    assert "--device cuda" in cmd and "--compute-type float16" in cmd
    # default beam/batch/vad produce no extra flags
    assert "--batch-size" not in cmd and "--beam-size" not in cmd and "--no-vad-filter" not in cmd


def test_reproducibility_command_nondefault_flags_and_nim():
    args = _cmd_args()
    args.batch_size = 8; args.beam_size = 3; args.vad_filter = False
    args.nim_model = "canary"
    cmd = asr_bench._reproducibility_command(args, Path("/c"), [_nim_result()])
    assert "--batch-size 8" in cmd and "--beam-size 3" in cmd and "--no-vad-filter" in cmd
    assert "--nim-url localhost:50051" in cmd and "--nim-model canary" in cmd
```

Add this import line near the top of `tests/test_results_json.py` (reuse render fixtures):

```python
from pathlib import Path
from tests.test_render import _whisper_result, _nim_result
```

- [ ] **Step 2: Run** `python -m pytest tests/test_results_json.py -k reproducibility -v` → FAIL (`_reproducibility_command` missing).

- [ ] **Step 3a: Implement** the helper in the Results JSON section:

```python
def _reproducibility_command(args, corpus_path: Path, results: List["ModelResult"]) -> str:
    """The `python asr_bench.py ...` command that reproduces this run (no backticks).
    Shared by the markdown reproducibility footnote and the JSON sidecar."""
    batch_flag = f" --batch-size {args.batch_size}" if args.batch_size > 1 else ""
    beam_flag = f" --beam-size {args.beam_size}" if args.beam_size != 5 else ""
    vad_flag = "" if args.vad_filter else " --no-vad-filter"
    nim_flag = ""
    if any(r.engine == "nim" for r in results):
        nim_flag = f" --nim-url {args.nim_url} --nim-language {args.nim_language}"
        if args.nim_model:
            nim_flag += f" --nim-model {args.nim_model}"
    return (f"python asr_bench.py --corpus '{corpus_path}' --models {','.join(args.models)} "
            f"--device {args.device} --compute-type {args.compute_type}"
            f"{batch_flag}{beam_flag}{vad_flag}{nim_flag}")
```

- [ ] **Step 3b: Refactor** `render_markdown` — replace the inline block (lines ~2116-2124) with:

```python
    lines.append(f"- Command: `{_reproducibility_command(args, corpus_path, results)}`")
```

(Delete the now-dead `batch_flag`/`beam_flag`/`vad_flag`/`nim_flag` locals that preceded that `lines.append`.)

- [ ] **Step 4: Run** `python -m pytest tests/test_results_json.py -k reproducibility -v` → PASS (2). Then `python -m pytest tests/test_render.py -q` and `python -m pytest -q` → green (the markdown command line is byte-identical to before).

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_results_json.py
git commit -m "refactor: extract _reproducibility_command shared by markdown + JSON"
```

---

## Task 3: `build_results_document`

**Files:**
- Modify: `asr_bench.py` — Results JSON section
- Test: `tests/test_results_json.py` (append)

**Context:** Pure builder. Signature:

```
build_results_document(results, *, corpus, cfg, args, gold_label, pairs,
                       report_path, generated_at) -> dict
```

- `cfg: RunConfig` → `run.config` (field-by-field; **omits `hf_token` and `nim_api_key`**).
- `gold_label: str` → `run.reference_quality` ("gold"/"proxy") + `run.reference_quality_label` (raw).
- `pairs: List[Pair]` → fusion output paths when `args.fuse`.
- Run-level totals come from the first model's clip set (every model runs the same clips, so summing across models would double-count).
- Whole document passed through `_json_sanitize` before return.

- [ ] **Step 1: Write the failing test**

```python
import math


def _wx_result_with_nan():
    """A whisperx ModelResult whose clip has der=NaN to exercise sanitization."""
    clip = asr_bench.ClipResult(
        audio="lec.mp4", audio_sec=600.0, transcribe_sec=20.0, rtfx=30.0,
        vram_peak_bytes=None, hypothesis="hello world",
        reference_normalized="hello world", hypothesis_normalized="hello world",
        wer=0.10, mer=0.09, wil=0.12, hits=90, substitutions=5, deletions=3,
        insertions=2, cue_count=40, num_speakers=2, der=float("nan"),
        speaker_segments=[(0.0, 300.0, "SPEAKER_00"), (300.0, 600.0, "SPEAKER_01")],
        reference_origin="unknown", reference_label="user-provided reference",
    )
    return asr_bench.ModelResult(
        model_id="large-v3-turbo+whisperx", display="Whisper Large V3 Turbo + WhisperX",
        fw_name="large-v3-turbo", params="809M", developer="OpenAI", languages="99",
        notes="x", disk_bytes=None, load_sec=0.0, engine="whisperx",
        vram_is_total=False, clips=[clip])


def _doc_args(**over):
    base = dict(models=["large-v3-turbo+whisperx"], device="cuda", compute_type="float16",
                batch_size=1, beam_size=5, vad_filter=True, nim_url="localhost:50051",
                nim_model="", nim_language="en-US", fuse=False, profile="both")
    base.update(over)
    return types.SimpleNamespace(**base)


def _doc_cfg():
    return asr_bench.RunConfig(
        device="cuda", compute_type="float16", diarize=True,
        hf_token="hf_SECRET", nim_api_key="nim_SECRET",
        min_speakers=2, max_speakers=2)


def test_build_document_top_level_shape():
    doc = asr_bench.build_results_document(
        [_wx_result_with_nan()], corpus=Path("/corpus"), cfg=_doc_cfg(),
        args=_doc_args(), gold_label="**proxy** (default: pass --gold ...)",
        pairs=[], report_path=Path("report/20260605-120000.md"),
        generated_at="2026-06-05T12:00:00-06:00")
    assert doc["schema_version"] == 1
    assert doc["generated_at"] == "2026-06-05T12:00:00-06:00"
    assert doc["report_markdown"].endswith("20260605-120000.md")
    assert doc["command"].startswith("python asr_bench.py")
    assert doc["run"]["device"] == "cuda"
    assert doc["run"]["reference_quality"] == "proxy"
    assert doc["run"]["clips_count"] == 1
    assert len(doc["models"]) == 1


def test_build_document_redacts_secrets():
    doc = asr_bench.build_results_document(
        [_wx_result_with_nan()], corpus=Path("/c"), cfg=_doc_cfg(), args=_doc_args(),
        gold_label="proxy", pairs=[], report_path=Path("r.md"),
        generated_at="t")
    cfg_out = doc["run"]["config"]
    assert "hf_token" not in cfg_out and "nim_api_key" not in cfg_out
    assert cfg_out["diarize"] is True and cfg_out["min_speakers"] == 2
    # and no secret value leaked anywhere in the serialized form
    import json as _json
    blob = _json.dumps(doc)
    assert "hf_SECRET" not in blob and "nim_SECRET" not in blob


def test_build_document_aggregates_and_clip_fields():
    m = _wx_result_with_nan()
    doc = asr_bench.build_results_document(
        [m], corpus=Path("/c"), cfg=_doc_cfg(), args=_doc_args(),
        gold_label="gold", pairs=[], report_path=Path("r.md"), generated_at="t")
    agg = doc["models"][0]["aggregates"]
    assert abs(agg["avg_wer"] - m.avg_wer) < 1e-9
    assert abs(agg["aggregate_rtfx"] - m.aggregate_rtfx) < 1e-9
    assert agg["peak_vram_bytes"] is None
    clip = doc["models"][0]["clips"][0]
    assert clip["der"] is None  # NaN -> null
    assert clip["num_speakers"] == 2
    assert clip["speaker_segments"][0] == {"start": 0.0, "end": 300.0, "speaker": "SPEAKER_00"}
    assert clip["hypothesis"] == "hello world"


def test_build_document_reference_quality_gold():
    doc = asr_bench.build_results_document(
        [_wx_result_with_nan()], corpus=Path("/c"), cfg=_doc_cfg(), args=_doc_args(),
        gold_label="**gold (hand-corrected, declared via --gold)**", pairs=[],
        report_path=Path("r.md"), generated_at="t")
    assert doc["run"]["reference_quality"] == "gold"


def test_build_document_fusion_stub_absent_and_present():
    off = asr_bench.build_results_document(
        [_wx_result_with_nan()], corpus=Path("/c"), cfg=_doc_cfg(),
        args=_doc_args(fuse=False), gold_label="proxy", pairs=[],
        report_path=Path("r.md"), generated_at="t")
    assert off["fusion"] == {"ran": False}
    on = asr_bench.build_results_document(
        [_wx_result_with_nan()], corpus=Path("/c"), cfg=_doc_cfg(),
        args=_doc_args(fuse=True, profile="verbatim"), pairs=[],
        gold_label="proxy", report_path=Path("r.md"), generated_at="t")
    assert on["fusion"]["ran"] is True
    assert on["fusion"]["profiles"] == ["verbatim"]


def test_build_document_fusion_outputs_listed():
    pair = asr_bench.Pair(audio=Path("/aud/Lec_default.mp4"), reference=Path("/aud/Lec.txt"))
    doc = asr_bench.build_results_document(
        [_wx_result_with_nan()], corpus=Path("/c"), cfg=_doc_cfg(),
        args=_doc_args(fuse=True, profile="both"), gold_label="proxy", pairs=[pair],
        report_path=Path("r.md"), generated_at="t")
    outs = doc["fusion"]["outputs"]
    assert any(o.endswith("Lec_Captions_Fused.vtt") for o in outs)
    assert any(o.endswith("Lec_KB_Fused.jsonl") for o in outs)
    assert any(o.endswith("Lec_KB_Fused.md") for o in outs)
```

- [ ] **Step 2: Run** `python -m pytest tests/test_results_json.py -k build_document -v` → FAIL.

- [ ] **Step 3: Implement** in the Results JSON section:

```python
def _clip_to_dict(c: "ClipResult") -> Dict:
    return {
        "audio": c.audio, "audio_sec": c.audio_sec, "transcribe_sec": c.transcribe_sec,
        "rtfx": c.rtfx, "vram_peak_bytes": c.vram_peak_bytes,
        "wer": c.wer, "mer": c.mer, "wil": c.wil,
        "hits": c.hits, "substitutions": c.substitutions,
        "deletions": c.deletions, "insertions": c.insertions,
        "cue_count": c.cue_count, "num_speakers": c.num_speakers, "der": c.der,
        "speaker_segments": [{"start": s, "end": e, "speaker": spk}
                             for (s, e, spk) in c.speaker_segments],
        "vtt_path": c.vtt_path,
        "reference_origin": c.reference_origin, "reference_label": c.reference_label,
        "hypothesis": c.hypothesis,
        "reference_normalized": c.reference_normalized,
        "hypothesis_normalized": c.hypothesis_normalized,
    }


def _model_to_dict(m: "ModelResult") -> Dict:
    return {
        "model_id": m.model_id, "display": m.display, "engine": m.engine,
        "fw_name": m.fw_name, "params": m.params, "developer": m.developer,
        "languages": m.languages, "disk_bytes": m.disk_bytes, "load_sec": m.load_sec,
        "vram_is_total": m.vram_is_total, "notes": m.notes,
        "aggregates": {
            "avg_wer": m.avg_wer, "avg_mer": m.avg_mer, "avg_wil": m.avg_wil,
            "total_audio_sec": m.total_audio_sec,
            "total_transcribe_sec": m.total_transcribe_sec,
            "aggregate_rtfx": m.aggregate_rtfx, "peak_vram_bytes": m.peak_vram_bytes,
        },
        "clips": [_clip_to_dict(c) for c in m.clips],
    }


def _fusion_output_paths(pairs: List["Pair"], profiles: List[str]) -> List[str]:
    """The fusion output files this run is expected to have written, per profile
    (verbatim -> <base>_Captions_Fused.vtt; kb -> <base>_KB_Fused.jsonl + .md).
    Reconstructed from the same `_fused_base` convention the writers use."""
    out: List[str] = []
    for p in pairs:
        base = p.audio.parent / _fused_base(p.audio)
        if "verbatim" in profiles:
            out.append(f"{base}_Captions_Fused.vtt")
        if "kb" in profiles:
            out.append(f"{base}_KB_Fused.jsonl")
            out.append(f"{base}_KB_Fused.md")
    return out


def _config_to_dict(cfg: "RunConfig") -> Dict:
    """RunConfig as a dict, OMITTING secrets (hf_token, nim_api_key)."""
    return {
        "device": cfg.device, "compute_type": cfg.compute_type,
        "batch_size": cfg.batch_size, "beam_size": cfg.beam_size,
        "vad_filter": cfg.vad_filter,
        "nim_url": cfg.nim_url, "nim_model": cfg.nim_model,
        "nim_language": cfg.nim_language, "nim_ssl": cfg.nim_ssl,
        "whisperx_python": cfg.whisperx_python, "diarize": cfg.diarize,
        "min_speakers": cfg.min_speakers, "max_speakers": cfg.max_speakers,
    }


def build_results_document(results: List["ModelResult"], *, corpus: Path,
                           cfg: "RunConfig", args, gold_label: str,
                           pairs: List["Pair"], report_path: Path,
                           generated_at: str) -> Dict:
    """Build the JSON sidecar document (a plain, strictly-JSON-safe dict) from a
    completed run. Secrets are omitted; NaN/Inf become null."""
    ref_quality = ("gold" if gold_label.replace("*", "").strip().lower().startswith("gold")
                   else "proxy")
    first = results[0] if results else None
    fusion = {"ran": bool(getattr(args, "fuse", False))}
    if fusion["ran"]:
        fusion["profiles"] = (["verbatim", "kb"] if args.profile == "both"
                              else [args.profile])
        fusion["outputs"] = _fusion_output_paths(pairs, fusion["profiles"])
    doc = {
        "schema_version": 1,
        "generated_at": generated_at,
        "report_markdown": str(report_path),
        "command": _reproducibility_command(args, corpus, results),
        "run": {
            "corpus": str(corpus),
            "device": cfg.device,
            "compute_type": cfg.compute_type,
            "reference_quality": ref_quality,
            "reference_quality_label": gold_label,
            "clips_count": len(first.clips) if first else 0,
            "total_audio_sec": first.total_audio_sec if first else 0.0,
            "vram_tracking": any(c.vram_peak_bytes is not None
                                 for r in results for c in r.clips),
            "config": _config_to_dict(cfg),
        },
        "models": [_model_to_dict(m) for m in results],
        "fusion": fusion,
    }
    return _json_sanitize(doc)
```

- [ ] **Step 4: Run** `python -m pytest tests/test_results_json.py -k build_document -v` → PASS (6). Full suite green.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_results_json.py
git commit -m "feat: build_results_document (full mirror, secrets redacted, NaN-safe)"
```

---

## Task 4: `write_results_json`

**Files:**
- Modify: `asr_bench.py` — Results JSON section
- Test: `tests/test_results_json.py` (append)

- [ ] **Step 1: Write the failing test**

```python
import json


def test_write_results_json_roundtrips(tmp_path):
    doc = {"schema_version": 1, "run": {"device": "cuda"}, "models": []}
    out = asr_bench.write_results_json(doc, tmp_path / "r.json")
    assert out == tmp_path / "r.json"
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == 1 and loaded["run"]["device"] == "cuda"


def test_write_results_json_no_nan_token(tmp_path):
    # A sanitized document never contains NaN; the written text must be valid JSON.
    doc = asr_bench._json_sanitize({"der": float("nan"), "wer": 0.1})
    out = asr_bench.write_results_json(doc, tmp_path / "r.json")
    text = out.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert json.loads(text)["der"] is None


def test_write_results_json_raises_on_stray_nan(tmp_path):
    # Belt-and-suspenders: an unsanitized NaN must fail loudly, not emit invalid JSON.
    import pytest
    with pytest.raises(ValueError):
        asr_bench.write_results_json({"der": float("nan")}, tmp_path / "r.json")
```

- [ ] **Step 2: Run** `python -m pytest tests/test_results_json.py -k write_results_json -v` → FAIL.

- [ ] **Step 3: Implement** in the Results JSON section:

```python
def write_results_json(document: Dict, json_path: Path) -> Path:
    """Serialize the results document to json_path. allow_nan=False is a guard:
    any NaN/Inf that escaped _json_sanitize raises ValueError rather than writing
    invalid JSON."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return json_path
```

(`json` and `Path` are already imported.)

- [ ] **Step 4: Run** `python -m pytest tests/test_results_json.py -k write_results_json -v` → PASS (3). Full suite green.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_results_json.py
git commit -m "feat: write_results_json with allow_nan=False guard"
```

---

## Task 5: CLI flag + `main()` integration

**Files:**
- Modify: `asr_bench.py` — argparse (near `--output`/`--gold`) + the save block (~line 2421-2432)
- Test: `tests/test_results_json.py` (append) + `tests/test_cli.py` (append a help check)

**Context:** Add `--json/--no-json` (default on). After the markdown is saved, build + write the JSON unless disabled. JSON path: sibling of `--output` if given, else `results/<same-timestamp>.json`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_results_json.py`)

```python
def _fake_whisper_run(monkeypatch):
    """Patch a fake whisperx adapter + known audio duration so main() runs torch-free."""
    canned = asr_bench.WhisperXResult.from_dict(
        {"segments": [{"start": 0, "end": 2, "text": "hello world", "speaker": "SPEAKER_00"}],
         "speakers": ["SPEAKER_00"], "der": None, "language": "en"})
    monkeypatch.setattr(asr_bench, "make_whisperx_adapter",
                        lambda cfg: asr_bench.FakeWhisperXAdapter(canned))
    monkeypatch.setattr(asr_bench, "_audio_duration_sec", lambda p: 2.0)


def test_main_writes_json_sidecar_with_output(tmp_path, monkeypatch):
    _fake_whisper_run(monkeypatch)
    audio = tmp_path / "Lec_default.mp4"; audio.write_bytes(b"x")
    (tmp_path / "Lec.txt").write_text("hello world", encoding="utf-8")
    md = tmp_path / "out" / "report.md"
    monkeypatch.setattr("sys.argv", [
        "asr_bench.py", "--corpus", str(tmp_path), "--models", "small+whisperx",
        "--device", "cpu", "--no-diarize", "--output", str(md)])
    assert asr_bench.main() == 0
    js = tmp_path / "out" / "report.json"
    assert js.is_file()
    import json
    doc = json.loads(js.read_text(encoding="utf-8"))
    assert doc["schema_version"] == 1
    assert doc["models"][0]["model_id"] == "small+whisperx"


def test_main_no_json_flag_skips_sidecar(tmp_path, monkeypatch):
    _fake_whisper_run(monkeypatch)
    audio = tmp_path / "Lec_default.mp4"; audio.write_bytes(b"x")
    (tmp_path / "Lec.txt").write_text("hello world", encoding="utf-8")
    md = tmp_path / "report.md"
    monkeypatch.setattr("sys.argv", [
        "asr_bench.py", "--corpus", str(tmp_path), "--models", "small+whisperx",
        "--device", "cpu", "--no-diarize", "--output", str(md), "--no-json"])
    assert asr_bench.main() == 0
    assert not (tmp_path / "report.json").exists()
```

- [ ] **Step 2: Run** `python -m pytest tests/test_results_json.py -k "main_writes_json or no_json_flag" -v` → FAIL (unknown `--no-json` / no sidecar).

- [ ] **Step 3a: Add the flag** — in `main()`'s argparse, immediately after the `--output` argument:

```python
    ap.add_argument("--json", action=argparse.BooleanOptionalAction, default=True,
                    help="Write a machine-readable results JSON sidecar next to the report "
                         "(default on). results/<timestamp>.json, or <output>.json with --output.")
```

- [ ] **Step 3b: Write the sidecar** — replace the save block (~lines 2421-2432) with:

```python
    # Save markdown report
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    output_path = Path(args.output) if args.output else None
    if output_path is None:
        report_dir = Path(__file__).resolve().parent / "report"
        report_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = report_dir / f"{ts}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    print(f"\nSaved report to {output_path}")

    # Save JSON results sidecar (default on; --no-json opts out)
    if args.json:
        if args.output:
            json_path = output_path.with_suffix(".json")
        else:
            results_dir = Path(__file__).resolve().parent / "results"
            results_dir.mkdir(exist_ok=True)
            json_path = results_dir / f"{output_path.stem}.json"
        document = build_results_document(
            results, corpus=corpus, cfg=cfg, args=args, gold_label=gold_label,
            pairs=pairs, report_path=output_path, generated_at=generated_at)
        write_results_json(document, json_path)
        print(f"Saved results JSON to {json_path}")

    return 0
```

- [ ] **Step 4a: Run** `python -m pytest tests/test_results_json.py -v` → all PASS. Full suite `python -m pytest -q` → green.

- [ ] **Step 4b: Add a help-text assertion** to `tests/test_cli.py`:

```python
def test_help_lists_json_flag():
    import subprocess, sys
    out = subprocess.run([sys.executable, "asr_bench.py", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert "--no-json" in out.stdout
```

Run: `python -m pytest tests/test_cli.py -k json -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_results_json.py tests/test_cli.py
git commit -m "feat: --json/--no-json flag + write results sidecar in main()"
```

---

## Task 6: Documentation

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `SPEC.md`

- [ ] **Step 1: `README.md`** — under the output section, add a short subsection:

```markdown
### JSON results sidecar

Every run also writes a machine-readable `results/<timestamp>.json` (same
timestamp as the markdown report; sibling `<output>.json` when you pass
`--output`). It mirrors the full run — run config, per-model and per-clip
metrics, transcripts, and speaker/DER data — for cross-run aggregation. NaN
values (e.g. DER on a non-diarized clip) serialize as `null`. Secrets
(`hf_token`, `nim_api_key`) are never written. Opt out with `--no-json`.
```

- [ ] **Step 2: `CLAUDE.md`** — add to the v0.3 status / "What's new" area:

```markdown
- **JSON results sidecar** — every run writes `results/<timestamp>.json` (or `<output>.json`) mirroring the full run (config, per-model/per-clip metrics, transcripts, speaker/DER) for cross-run aggregation. `schema_version: 1`. NaN→null; `hf_token`/`nim_api_key` redacted. `--no-json` opts out. Foundation for a future `asr_bench compare`.
```

And a decision-log entry:

```markdown
- **2026-06-05** — JSON results sidecar always emitted (no flag friction; aggregation needs the data to reliably exist) to `results/<ts>.json` or a `--output` sibling. Full mirror including transcripts (cheap text, expensive to regenerate). NaN→null with an `allow_nan=False` write guard so output is always valid JSON. Secrets (hf_token, nim_api_key) omitted from `run.config`. `compare` subcommand deferred to its own spec; schema is compare-ready via `schema_version`.
```

- [ ] **Step 3: `SPEC.md`** — in the metrics/output roadmap, move the v0.3 "JSON sidecar (`results/<timestamp>.json`)" line from planned to shipped, noting `compare` remains the follow-up.

- [ ] **Step 4: Verify** `python -m pytest -q` still green (docs only — no code change).

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md SPEC.md
git commit -m "docs: JSON results sidecar (README/CLAUDE/SPEC + decision log)"
```

---

## Final verification

- [ ] `python -m pytest -q` — all green (existing 120 + ~14 new; 2 pyannote DER tests still skip in the core 3.14 venv).
- [ ] `python asr_bench.py --help` exits 0 and lists `--no-json`.
- [ ] Spot-check a real sidecar: `python asr_bench.py --models small --include <clip> --limit 1` (or a fake-adapter smoke) writes `results/<ts>.json`; open it and confirm valid JSON, `schema_version: 1`, no `NaN`, no token strings.
- [ ] Use superpowers:requesting-code-review before merging `feat/json-results-sidecar`.
```
