# asr-bench v0.2 — MER/WIL Metrics + Multi-Transcript Fusion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add information-theoretic ASR metrics (MER, WIL, H/S/D/I) from the Morris/Maier/Green paper, and a pluggable-LLM, chunked/timing-anchored fusion stage that combines multiple transcripts + Panopto into a verbatim caption (also usable as a scoring reference) and a knowledge-base (RAG) artifact.

**Architecture:** All code lives in the single-file `asr_bench.py` (project rule; `engines/` split still deferred), organized into new `# ---- Word metrics ----`, `# ---- Caption cue parsing ----`, `# ---- Fusion ----`, and `# ---- LLM backends ----` sections. Part A swaps the standalone `jiwer.wer()` calls for one `jiwer.process_words()` call surfaced through a `WordMetrics` helper. Part B is a post-processing pass: after the normal benchmark produces per-model VTTs, a `--fuse` stage parses those VTTs + Panopto into timed cues, windows them (overlap = context carryover), asks an `LLMBackend` to fuse each window under a per-profile prompt, then writes a verbatim VTT and/or KB JSONL+MD, with a per-window drift guard.

**Tech Stack:** Python 3.14, `jiwer` (RapidFuzz Levenshtein), `argparse`, stdlib `subprocess`/`urllib`/`json`, `pytest`. LLM backends: Ollama (local HTTP, default), authenticated frontier CLI (subprocess), plus a deterministic `FakeLLMBackend` for tests.

---

## File Structure

- **`asr_bench.py`** (modify) — all production code. New sections:
  - `# ---- Word metrics ----`: `WordMetrics` dataclass + `compute_word_metrics()`.
  - `# ---- Caption cue parsing ----`: `Cue` dataclass + `parse_caption_cues()`.
  - `# ---- Fusion ----`: windowing, prompt building, orchestration, output writers, drift guard, re-scoring, context scaffolding.
  - `# ---- LLM backends ----`: `LLMBackend` ABC + `FakeLLMBackend`/`OllamaBackend`/`CliBackend` + `make_llm_backend()`.
  - Extend `ClipResult`/`ModelResult`; extend `render_markdown`; extend `main()` CLI.
- **`tests/test_metrics.py`** (create) — Part A: Table-1 oracle vectors + clip/model wiring.
- **`tests/test_cue_parsing.py`** (create) — VTT/SRT cue parsing.
- **`tests/test_windowing.py`** (create) — window tiling + source collection.
- **`tests/test_llm_backends.py`** (create) — backend factory + Fake/Cli(mock)/Ollama(mock).
- **`tests/test_fusion.py`** (create) — prompt construction, orchestration with FakeLLMBackend, drift guard, output writers, re-scoring.
- **`tests/test_context.py`** (create) — context loader + `--init-context` scaffolder.
- **`tests/test_render.py`** (modify) — assert new metric columns render.
- **`CLAUDE.md`**, **`SPEC.md`**, **`README.md`** (modify) — docs.

Tests import the module via `import asr_bench` (repo root is on `sys.path` via `tests/conftest.py`).

---

# PART A — Metrics

## Task 1: `WordMetrics` + `compute_word_metrics()`

**Files:**
- Modify: `asr_bench.py` (new `# ---- Word metrics ----` section, placed just after `normalize_for_wer` ~line 247)
- Test: `tests/test_metrics.py` (create)

- [ ] **Step 1: Write the failing test (paper Table-1 oracle vectors)**

Create `tests/test_metrics.py`:

```python
import math
import asr_bench


# Morris/Maier/Green Table 1. Tokens chosen so each row reproduces the paper's
# H,S,D,I and integer %WER/%MER/%WIL. (a,b,c are distinct words.)
TABLE_1 = [
    # ref,            hyp,                  WER%, MER%, WIL%
    ("a",             "a",                  0,    0,    0),
    ("a",             "a a b b",            300,  75,   75),
    ("a b a",         "a c",                67,   67,   83),
    ("a",             "b",                  100,  100,  100),
    ("a",             "b c",                200,  100,  100),
]


def test_table1_wer_mer_wil_match_paper():
    for ref, hyp, wer_pct, mer_pct, wil_pct in TABLE_1:
        m = asr_bench.compute_word_metrics(ref, hyp)
        assert round(m.wer * 100) == wer_pct, (ref, hyp, "wer", m.wer)
        assert round(m.mer * 100) == mer_pct, (ref, hyp, "mer", m.mer)
        assert round(m.wil * 100) == wil_pct, (ref, hyp, "wil", m.wil)


def test_table1_hsdi_counts():
    # Row 3: ref "a b a" vs hyp "a c"  -> H=1, S=1, D=1, I=0
    m = asr_bench.compute_word_metrics("a b a", "a c")
    assert (m.hits, m.substitutions, m.deletions, m.insertions) == (1, 1, 1, 0)


def test_empty_reference_is_nan_not_crash():
    m = asr_bench.compute_word_metrics("", "")
    assert math.isnan(m.wer) and math.isnan(m.mer) and math.isnan(m.wil)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL — `AttributeError: module 'asr_bench' has no attribute 'compute_word_metrics'`.

- [ ] **Step 3: Implement `WordMetrics` + `compute_word_metrics`**

In `asr_bench.py`, immediately after `normalize_for_wer` (~line 247), add:

```python
# ---- Word metrics -----------------------------------------------------------
@dataclass
class WordMetrics:
    """All word-level scores from a single jiwer alignment.

    WER  = (S+D+I)/N1                      (edit cost; can exceed 1.0)
    MER  = (S+D+I)/(H+S+D+I)               (Morris et al.; bounded [0,1])
    WIL  = 1 - H*H/(N1*N2)                 (Morris et al.; bounded [0,1])
    where N1 = ref words = H+S+D, N2 = hyp words = H+S+I.
    """
    wer: float
    mer: float
    wil: float
    hits: int
    substitutions: int
    deletions: int
    insertions: int


def compute_word_metrics(reference: str, hypothesis: str) -> WordMetrics:
    """One jiwer.process_words call -> WER, MER, WIL, and H/S/D/I counts.

    Inputs should already be normalized (see normalize_for_wer). Returns NaN
    metrics (not an exception) when alignment is impossible (e.g. empty ref).
    """
    from jiwer import process_words
    try:
        out = process_words(reference, hypothesis)
        return WordMetrics(
            wer=float(out.wer),
            mer=float(out.mer),
            wil=float(out.wil),
            hits=int(out.hits),
            substitutions=int(out.substitutions),
            deletions=int(out.deletions),
            insertions=int(out.insertions),
        )
    except Exception:
        nan = float("nan")
        return WordMetrics(nan, nan, nan, 0, 0, 0, 0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_metrics.py
git commit -m "feat: add WordMetrics (WER/MER/WIL + HSDI) via jiwer.process_words"
```

---

## Task 2: Extend `ClipResult`/`ModelResult` and wire both engines

**Files:**
- Modify: `asr_bench.py` — `ClipResult` (~477), `ModelResult` (~494), `FasterWhisperEngine.run` WER site (~801-833), `NimEngine.run` WER site (~937-959)
- Test: `tests/test_metrics.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_metrics.py`:

```python
def test_clipresult_has_metric_fields_with_defaults():
    # New fields must have defaults so existing positional constructors keep working.
    c = asr_bench.ClipResult(
        audio="x.mp4", audio_sec=1.0, transcribe_sec=1.0, rtfx=1.0,
        vram_peak_bytes=None, hypothesis="h", reference_normalized="r",
        hypothesis_normalized="h", wer=0.5,
    )
    assert c.mer != c.mer or c.mer == 0 or isinstance(c.mer, float)  # field exists
    assert hasattr(c, "wil") and hasattr(c, "substitutions")


def test_modelresult_avg_mer_wil():
    c1 = asr_bench.ClipResult(
        audio="a", audio_sec=1, transcribe_sec=1, rtfx=1, vram_peak_bytes=None,
        hypothesis="", reference_normalized="", hypothesis_normalized="",
        wer=0.2, mer=0.1, wil=0.3,
    )
    c2 = asr_bench.ClipResult(
        audio="b", audio_sec=1, transcribe_sec=1, rtfx=1, vram_peak_bytes=None,
        hypothesis="", reference_normalized="", hypothesis_normalized="",
        wer=0.4, mer=0.3, wil=0.5,
    )
    mr = asr_bench.ModelResult(
        model_id="m", display="M", fw_name="m", params="1", developer="d",
        languages="en", notes="", disk_bytes=None, load_sec=0.0, clips=[c1, c2],
    )
    assert abs(mr.avg_mer - 0.2) < 1e-9
    assert abs(mr.avg_wil - 0.4) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -k "metric_fields or avg_mer" -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'mer'`.

- [ ] **Step 3a: Add fields to `ClipResult`**

In `asr_bench.py`, in `ClipResult` (after `wer: float` ~line 486, before `cue_count`):

```python
    wer: float
    mer: float = float("nan")
    wil: float = float("nan")
    hits: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    cue_count: int = 0
```

- [ ] **Step 3b: Add aggregate props to `ModelResult`**

In `asr_bench.py`, after the `avg_wer` property (~line 512), add:

```python
    @property
    def avg_mer(self) -> float:
        if not self.clips:
            return 0.0
        return sum(c.mer for c in self.clips) / len(self.clips)

    @property
    def avg_wil(self) -> float:
        if not self.clips:
            return 0.0
        return sum(c.wil for c in self.clips) / len(self.clips)
```

- [ ] **Step 3c: Wire `FasterWhisperEngine.run`**

In `asr_bench.py`, replace the WER block (~801-806):

```python
            ref_norm = normalize_for_wer(ref_text)
            hyp_norm = normalize_for_wer(hypothesis)
            try:
                wer_val = jiwer_wer(ref_norm, hyp_norm)
            except Exception:
                wer_val = float("nan")
```

with:

```python
            ref_norm = normalize_for_wer(ref_text)
            hyp_norm = normalize_for_wer(hypothesis)
            metrics = compute_word_metrics(ref_norm, hyp_norm)
            wer_val = metrics.wer
```

Then in the `ClipResult(...)` constructor (~818-833) add, right after `wer=wer_val,`:

```python
                    wer=wer_val,
                    mer=metrics.mer,
                    wil=metrics.wil,
                    hits=metrics.hits,
                    substitutions=metrics.substitutions,
                    deletions=metrics.deletions,
                    insertions=metrics.insertions,
```

Delete the now-unused `from jiwer import wer as jiwer_wer` import (~722).

- [ ] **Step 3d: Wire `NimEngine.run`**

In `asr_bench.py`, replace the WER block (~937-942):

```python
            ref_norm = normalize_for_wer(ref_text)
            hyp_norm = normalize_for_wer(hypothesis)
            try:
                wer_val = jiwer_wer(ref_norm, hyp_norm)
            except Exception:
                wer_val = float("nan")
```

with:

```python
            ref_norm = normalize_for_wer(ref_text)
            hyp_norm = normalize_for_wer(hypothesis)
            metrics = compute_word_metrics(ref_norm, hyp_norm)
            wer_val = metrics.wer
```

Then in the NIM `ClipResult(...)` (~951-959) add after `wer=wer_val,`:

```python
                    wer=wer_val, mer=metrics.mer, wil=metrics.wil,
                    hits=metrics.hits, substitutions=metrics.substitutions,
                    deletions=metrics.deletions, insertions=metrics.insertions,
```

Delete the unused `from jiwer import wer as jiwer_wer` import in `NimEngine.run` (~866).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS (5 tests).
Run full suite to confirm no regressions: `python -m pytest -q`
Expected: all pass (existing 37 + new).

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_metrics.py
git commit -m "feat: record MER/WIL/HSDI per clip and aggregate per model"
```

---

## Task 3: Render MER%/WIL% columns + per-clip S/D/I

**Files:**
- Modify: `asr_bench.py` — `render_markdown` headline table (~1010-1020), per-clip table (~1055-1064), per-model table (~1075-1091)
- Test: `tests/test_render.py` (append), `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_render.py` (reuse its `_whisper_result`, `_args`):

```python
def test_headline_has_mer_and_wil_columns():
    md = asr_bench.render_markdown([_whisper_result()], Path("."), _args(), "proxy")
    header = [l for l in md.splitlines() if l.startswith("| Model | Params")][0]
    assert "MER%" in header and "WIL%" in header


def test_per_clip_table_has_sdi_columns():
    md = asr_bench.render_markdown([_whisper_result()], Path("."), _args(), "proxy")
    assert "| S | D | I |" in md
```

Also update `_whisper_result`/`_nim_result` in `tests/test_render.py` to pass the new metric fields so rows are realistic — add to each `ClipResult(...)`:

```python
        mer=0.09, wil=0.12, hits=90, substitutions=5, deletions=3, insertions=2,
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_render.py -k "mer_and_wil or sdi" -v`
Expected: FAIL — `"MER%"` not found in header.

- [ ] **Step 3a: Headline table** — replace lines ~1010-1020:

```python
    lines.append("| Model | Params | Disk | Overall WER% | MER% | WIL% | RTFx | Total time | Peak VRAM | Notes |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        wall_clock = f"{r.total_transcribe_sec:.1f}s"
        wer_pct = f"{r.avg_wer * 100:.1f}" if r.clips else "—"
        mer_pct = f"{r.avg_mer * 100:.1f}" if r.clips else "—"
        wil_pct = f"{r.avg_wil * 100:.1f}" if r.clips else "—"
        rtfx = f"{r.aggregate_rtfx:.2f}x" if r.clips else "—"
        vram = _vram_cell(r.peak_vram_bytes, r.vram_is_total)
        disk = _disk_cell(r)
        lines.append(
            f"| {r.display} | {r.params} | {disk} | {wer_pct} | {mer_pct} | {wil_pct} | {rtfx} | {wall_clock} | {vram} | {r.notes} |"
        )
```

- [ ] **Step 3b: Per-clip table** — replace lines ~1055-1064:

```python
            lines.append("| Model | WER% | MER% | WIL% | S | D | I | RTFx | Transcribe time | VRAM peak |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for r in results:
                if i < len(r.clips):
                    c = r.clips[i]
                    wer_pct = f"{c.wer * 100:.1f}"
                    mer_pct = f"{c.mer * 100:.1f}"
                    wil_pct = f"{c.wil * 100:.1f}"
                    vram = _vram_cell(c.vram_peak_bytes, r.vram_is_total)
                    lines.append(
                        f"| {r.display} | {wer_pct} | {mer_pct} | {wil_pct} | {c.substitutions} | {c.deletions} | {c.insertions} | {c.rtfx:.2f}x | {c.transcribe_sec:.1f}s | {vram} |"
                    )
```

- [ ] **Step 3c: Per-model table** — replace lines ~1075-1091:

```python
        lines.append("| Clip | Audio | WER% | MER% | WIL% | RTFx | Transcribe time | VRAM peak |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for c in r.clips:
            wer_pct = f"{c.wer * 100:.1f}"
            mer_pct = f"{c.mer * 100:.1f}"
            wil_pct = f"{c.wil * 100:.1f}"
            vram = _vram_cell(c.vram_peak_bytes, r.vram_is_total)
            audio_label = f"{c.audio_sec / 60:.1f} min"
            lines.append(
                f"| {c.audio} | {audio_label} | {wer_pct} | {mer_pct} | {wil_pct} | {c.rtfx:.2f}x | {c.transcribe_sec:.1f}s | {vram} |"
            )
        overall_audio = f"{r.total_audio_sec / 60:.1f} min"
        overall_wer = f"{r.avg_wer * 100:.1f}" if r.clips else "—"
        overall_mer = f"{r.avg_mer * 100:.1f}" if r.clips else "—"
        overall_wil = f"{r.avg_wil * 100:.1f}" if r.clips else "—"
        overall_rtfx = f"{r.aggregate_rtfx:.2f}x" if r.clips else "—"
        overall_vram = _vram_cell(r.peak_vram_bytes, r.vram_is_total)
        lines.append(
            f"| **OVERALL** | **{overall_audio}** | **{overall_wer}** | **{overall_mer}** | **{overall_wil}** | **{overall_rtfx}** | **{r.total_transcribe_sec:.1f}s** | **{overall_vram}** |"
        )
```

- [ ] **Step 3d: Update the metrics note** — after the `WER computed via jiwer` line (~1168) add:

```python
    lines.append("- **MER** (match error rate) and **WIL** (word information lost) are the bounded-[0,1] measures from Morris, Maier & Green (2004); both derive from the same H/S/D/I alignment as WER. S/D/I in the per-clip table are raw substitution/deletion/insertion counts.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_render.py -v`
Expected: PASS (existing + 2 new).
Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_render.py
git commit -m "feat: report MER%/WIL% columns and per-clip S/D/I counts"
```

---

## Task 4: `--show-alignment` flag + alignment section

**Files:**
- Modify: `asr_bench.py` — `main()` argparse (~1262, after `--include`), `render_markdown` signature + body, the `cfg`/call path so `args.show_alignment` exists
- Test: `tests/test_render.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_render.py`:

```python
def test_show_alignment_section_emitted_when_flag_on():
    args = _args()
    args.show_alignment = True
    r = _whisper_result()
    r.clips[0].reference_normalized = "the quick brown fox"
    r.clips[0].hypothesis_normalized = "the quick brown box"
    md = asr_bench.render_markdown([r], Path("."), args, "proxy")
    assert "## Alignment detail" in md


def test_no_alignment_section_by_default():
    md = asr_bench.render_markdown([_whisper_result()], Path("."), _args(), "proxy")
    assert "## Alignment detail" not in md
```

Also add `show_alignment=False` to the `_args()` SimpleNamespace in `tests/test_render.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_render.py -k alignment -v`
Expected: FAIL — `## Alignment detail` not present (and/or AttributeError on `show_alignment`).

- [ ] **Step 3a: Add the CLI flag** in `main()` after the `--include` arg (~1266):

```python
    ap.add_argument(
        "--show-alignment",
        action="store_true",
        help="Append a per-clip word-level alignment diff (jiwer) to the report. Verbose.",
    )
```

- [ ] **Step 3b: Render the section.** In `render_markdown`, just before the `## Reproducibility` block (~1154), add:

```python
    # ---- Optional alignment detail ----
    if getattr(args, "show_alignment", False) and results and results[0].clips:
        from jiwer import process_words, visualize_alignment
        lines.append("## Alignment detail")
        lines.append("")
        lines.append("Word-level reference→hypothesis alignment (S=substitution, D=deletion, I=insertion).")
        lines.append("")
        for r in results:
            for c in r.clips:
                if not c.reference_normalized or not c.hypothesis_normalized:
                    continue
                try:
                    viz = visualize_alignment(
                        process_words(c.reference_normalized, c.hypothesis_normalized),
                        show_measures=False,
                    )
                except Exception:
                    continue
                lines.append(f"### {r.display} — {c.audio}")
                lines.append("")
                lines.append("```")
                lines.append(viz.rstrip())
                lines.append("```")
                lines.append("")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_render.py -k alignment -v`
Expected: PASS (2).
Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_render.py
git commit -m "feat: --show-alignment renders per-clip jiwer alignment diff"
```

---

# PART B — Fusion

## Task 5: Caption cue parsing (`Cue` + `parse_caption_cues`)

**Files:**
- Modify: `asr_bench.py` — new `# ---- Caption cue parsing ----` section after `load_reference_text` (~234)
- Test: `tests/test_cue_parsing.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_cue_parsing.py`:

```python
import asr_bench


VTT = """WEBVTT

1
00:00:00.000 --> 00:00:02.500
Hello world

2
00:00:02.500 --> 00:00:05.000
this is a test
"""

SRT = """1
00:00:00,000 --> 00:00:02,500
Hello world

2
00:00:02,500 --> 00:00:05,000
this is a test
"""


def test_parse_vtt_cues(tmp_path):
    p = tmp_path / "cap.vtt"
    p.write_text(VTT, encoding="utf-8")
    cues = asr_bench.parse_caption_cues(p)
    assert len(cues) == 2
    assert cues[0].start == 0.0 and abs(cues[0].end - 2.5) < 1e-6
    assert cues[0].text == "Hello world"
    assert cues[1].text == "this is a test"


def test_parse_srt_cues(tmp_path):
    p = tmp_path / "cap.srt"
    p.write_text(SRT, encoding="utf-8")
    cues = asr_bench.parse_caption_cues(p)
    assert len(cues) == 2
    assert abs(cues[1].end - 5.0) < 1e-6


def test_parse_skips_panopto_header(tmp_path):
    p = tmp_path / "cap.vtt"
    p.write_text("WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\n[Auto-generated transcript.]\nreal text\n", encoding="utf-8")
    cues = asr_bench.parse_caption_cues(p)
    assert cues[0].text == "real text"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cue_parsing.py -v`
Expected: FAIL — `module 'asr_bench' has no attribute 'parse_caption_cues'`.

- [ ] **Step 3: Implement.** After `load_reference_text` (~234), add:

```python
# ---- Caption cue parsing ----------------------------------------------------
@dataclass
class Cue:
    start: float
    end: float
    text: str


_VTT_TS_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
)


def _ts_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_caption_cues(path: Path) -> List[Cue]:
    """Parse a VTT or SRT file into timed cues.

    Tolerant of both '.' (VTT) and ',' (SRT) millisecond separators. Drops the
    WEBVTT header, numeric cue indices, and bracketed editorial notes (Panopto's
    '[Auto-generated transcript...]'). Multi-line cue text is joined with spaces.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    cues: List[Cue] = []
    start = end = None
    buf: List[str] = []

    def flush() -> None:
        nonlocal start, end, buf
        if start is not None and buf:
            text = " ".join(buf).strip()
            if text:
                cues.append(Cue(start, end, text))
        start = end = None
        buf = []

    for line in raw.splitlines():
        s = line.strip()
        m = _VTT_TS_RE.search(s)
        if m:
            flush()
            start = _ts_to_seconds(m.group(1), m.group(2), m.group(3), m.group(4))
            end = _ts_to_seconds(m.group(5), m.group(6), m.group(7), m.group(8))
            continue
        if not s:
            flush()
            continue
        if s.upper() == "WEBVTT" or _CUE_NUM_RE.match(s) or _BRACKETED_RE.match(s):
            continue
        if start is not None:
            buf.append(s)
    flush()
    return cues
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cue_parsing.py -v`
Expected: PASS (3).

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_cue_parsing.py
git commit -m "feat: parse_caption_cues for VTT/SRT timed cues"
```

---

## Task 6: Windowing (`build_windows` + `collect_window_text`)

**Files:**
- Modify: `asr_bench.py` — new `# ---- Fusion ----` section after `ENGINES` (~969)
- Test: `tests/test_windowing.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_windowing.py`:

```python
import asr_bench
from asr_bench import Cue


def test_build_windows_tiles_with_overlap():
    # duration 60s, window 25s, overlap 5s -> stride 20s
    wins = asr_bench.build_windows(60.0, window=25.0, overlap=5.0)
    assert wins[0] == (0.0, 25.0)
    assert wins[1] == (20.0, 45.0)
    assert wins[2] == (40.0, 60.0)          # last clamped to duration
    assert wins[-1][1] == 60.0


def test_build_windows_short_clip_single_window():
    wins = asr_bench.build_windows(10.0, window=25.0, overlap=5.0)
    assert wins == [(0.0, 10.0)]


def test_collect_window_text_includes_overlapping_cues():
    cues = [Cue(0.0, 3.0, "alpha"), Cue(3.0, 22.0, "beta"), Cue(22.0, 40.0, "gamma")]
    # window [20,45] overlaps beta (ends 22>20) and gamma
    text = asr_bench.collect_window_text(cues, 20.0, 45.0)
    assert "beta" in text and "gamma" in text
    assert "alpha" not in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_windowing.py -v`
Expected: FAIL — `build_windows` not defined.

- [ ] **Step 3: Implement.** After the `ENGINES = {...}` block (~969), start the fusion section:

```python
# ---- Fusion -----------------------------------------------------------------
def build_windows(duration: float, window: float, overlap: float) -> List[Tuple[float, float]]:
    """Tile [0, duration] into (start, end) spans of length `window`, stepping by
    stride = window - overlap. The overlap is carried into prompts as context; the
    final window is clamped to `duration`. Returns a single full-span window when
    the clip is shorter than one window.
    """
    if duration <= window or window <= 0:
        return [(0.0, duration)]
    stride = max(window - overlap, 1.0)
    spans: List[Tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(start + window, duration)
        spans.append((round(start, 3), round(end, 3)))
        if end >= duration:
            break
        start += stride
    return spans


def collect_window_text(cues: List[Cue], start: float, end: float) -> str:
    """Concatenate the text of all cues that overlap [start, end)."""
    parts = [c.text for c in cues if c.end > start and c.start < end]
    return " ".join(parts).strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_windowing.py -v`
Expected: PASS (3).

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_windowing.py
git commit -m "feat: fusion windowing (build_windows + collect_window_text)"
```

---

## Task 7: LLM backends + factory

**Files:**
- Modify: `asr_bench.py` — new `# ---- LLM backends ----` section after the fusion windowing helpers
- Test: `tests/test_llm_backends.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_llm_backends.py`:

```python
import asr_bench


def test_fake_backend_returns_canned():
    b = asr_bench.FakeLLMBackend(lambda prompt: "FUSED:" + prompt[:3])
    assert b.generate("hello") == "FUSED:hel"


def test_make_llm_backend_fake():
    b = asr_bench.make_llm_backend("fake")
    assert isinstance(b, asr_bench.FakeLLMBackend)


def test_make_llm_backend_ollama_parses_model():
    b = asr_bench.make_llm_backend("ollama:qwen2.5")
    assert isinstance(b, asr_bench.OllamaBackend)
    assert b.model == "qwen2.5"


def test_make_llm_backend_cli_parses_command():
    b = asr_bench.make_llm_backend("cli:claude -p")
    assert isinstance(b, asr_bench.CliBackend)
    assert b.command == ["claude", "-p"]


def test_make_llm_backend_unknown_raises():
    try:
        asr_bench.make_llm_backend("nope:x")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_cli_backend_invokes_subprocess(monkeypatch):
    calls = {}

    class FakeCompleted:
        stdout = "fused text"
        returncode = 0

    def fake_run(cmd, input=None, capture_output=None, text=None, timeout=None, check=None):
        calls["cmd"] = cmd
        calls["input"] = input
        return FakeCompleted()

    monkeypatch.setattr(asr_bench.subprocess, "run", fake_run)
    b = asr_bench.CliBackend(["claude", "-p"])
    out = b.generate("my prompt")
    assert out == "fused text"
    assert calls["cmd"] == ["claude", "-p"]
    assert calls["input"] == "my prompt"


def test_ollama_backend_posts_prompt(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        import io, json
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))

        class Resp:
            def read(self_inner):
                return json.dumps({"response": "ollama fused"}).encode("utf-8")
            def __enter__(self_inner):
                return self_inner
            def __exit__(self_inner, *a):
                return False
        return Resp()

    monkeypatch.setattr(asr_bench.urllib.request, "urlopen", fake_urlopen)
    b = asr_bench.OllamaBackend(model="qwen2.5")
    out = b.generate("hi")
    assert out == "ollama fused"
    assert captured["body"]["model"] == "qwen2.5"
    assert captured["body"]["prompt"] == "hi"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm_backends.py -v`
Expected: FAIL — `FakeLLMBackend` not defined.

- [ ] **Step 3a: Ensure imports.** At the top of `asr_bench.py`, confirm/add to the import block:

```python
import subprocess
import urllib.request
```

(Both are stdlib; add only if not already imported.)

- [ ] **Step 3b: Implement backends.** Add a new section after the windowing helpers:

```python
# ---- LLM backends -----------------------------------------------------------
class LLMBackend(ABC):
    """Minimal contract: turn a prompt into text. Fusion builds the prompt; the
    backend only generates. Keeps profile/prompt logic in one place (DRY)."""
    name: str = ""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...


class FakeLLMBackend(LLMBackend):
    """Deterministic, dependency-free backend for tests."""
    name = "fake"

    def __init__(self, fn=None):
        # fn: Callable[[str], str]; default echoes the prompt's last source block.
        self._fn = fn or (lambda prompt: prompt)

    def generate(self, prompt: str) -> str:
        return self._fn(prompt)


class OllamaBackend(LLMBackend):
    """Local Ollama HTTP backend (default). Offline, free, no API key."""
    name = "ollama"

    def __init__(self, model: str = "qwen2.5", host: str = "http://localhost:11434", timeout: float = 300.0):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        import json
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("response") or "").strip()


class CliBackend(LLMBackend):
    """Shell out to an authenticated frontier CLI (e.g. `claude -p`, `gemini`).

    The prompt is passed on stdin to avoid arg-length limits. Uses the operator's
    existing subscription — no API key is stored in asr-bench.
    """
    name = "cli"

    def __init__(self, command: List[str], timeout: float = 300.0):
        self.command = command
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        proc = subprocess.run(
            self.command, input=prompt, capture_output=True, text=True,
            timeout=self.timeout, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"LLM CLI {self.command} exited {proc.returncode}: {proc.stderr[:500]}")
        return (proc.stdout or "").strip()


def make_llm_backend(spec: str) -> LLMBackend:
    """Parse a --llm spec into a backend.

    'fake'                 -> FakeLLMBackend (echo)
    'ollama:<model>'       -> OllamaBackend  (default model qwen2.5 if omitted)
    'cli:<command words>'  -> CliBackend     (command split on whitespace)
    """
    spec = (spec or "").strip()
    if spec == "fake":
        return FakeLLMBackend()
    kind, _, rest = spec.partition(":")
    kind = kind.strip().lower()
    rest = rest.strip()
    if kind == "ollama":
        return OllamaBackend(model=rest or "qwen2.5")
    if kind == "cli":
        if not rest:
            raise ValueError("cli backend needs a command, e.g. --llm cli:claude")
        return CliBackend(rest.split())
    raise ValueError(f"unknown --llm backend '{spec}' (use fake, ollama:<model>, or cli:<command>)")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_llm_backends.py -v`
Expected: PASS (7).

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_llm_backends.py
git commit -m "feat: pluggable LLMBackend (fake/ollama/cli) + make_llm_backend"
```

---

## Task 8: Per-profile prompt construction

**Files:**
- Modify: `asr_bench.py` — fusion section
- Test: `tests/test_fusion.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_fusion.py`:

```python
import asr_bench


def _payload():
    return asr_bench.WindowPayload(
        start=0.0, end=25.0,
        sources={
            "large-v3-turbo": "i think AI is great",
            "Panopto": "I think I is great",
        },
        prev_fused="Welcome back everyone.",
    )


def test_verbatim_prompt_forbids_rephrasing():
    p = asr_bench.build_fusion_prompt(_payload(), "verbatim", context="I teach 9-11am.", glossary="AI not I")
    low = p.lower()
    assert "verbatim" in low or "do not rephrase" in low or "actually" in low
    assert "AI not I" in p              # glossary injected
    assert "I teach 9-11am." in p       # context injected
    assert "large-v3-turbo" in p        # sources injected
    assert "Welcome back everyone." in p  # carryover context


def test_kb_prompt_allows_rewriting():
    p = asr_bench.build_fusion_prompt(_payload(), "kb", context="", glossary="")
    low = p.lower()
    assert "rewrite" in low or "clarity" in low or "readable" in low


def test_prompts_differ_by_profile():
    a = asr_bench.build_fusion_prompt(_payload(), "verbatim", "", "")
    b = asr_bench.build_fusion_prompt(_payload(), "kb", "", "")
    assert a != b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fusion.py -k prompt -v`
Expected: FAIL — `WindowPayload` not defined.

- [ ] **Step 3: Implement.** In the fusion section, add:

```python
@dataclass
class WindowPayload:
    start: float
    end: float
    sources: Dict[str, str]          # source label -> text in this window
    prev_fused: str = ""             # previous window's fused output (carryover)


_VERBATIM_INSTRUCTIONS = (
    "You are reconciling several speech-to-text transcripts of the SAME audio span "
    "into the single most accurate VERBATIM transcript of what was actually said.\n"
    "Rules:\n"
    "- Restore the actually-spoken words. When sources disagree (e.g. 'AI' vs 'I'), "
    "choose the reading that fits the context and glossary.\n"
    "- Do NOT rephrase, summarize, or clean up grammar. Preserve the speaker's wording "
    "and disfluencies.\n"
    "- Output ONLY the corrected transcript text for this span. No commentary, no labels."
)

_KB_INSTRUCTIONS = (
    "You are merging several speech-to-text transcripts of the SAME audio span into "
    "one clean, readable passage for a searchable knowledge base.\n"
    "Rules:\n"
    "- Rewrite for clarity and correct grammar. Normalize times, numbers and dates "
    "(e.g. '9 to 11' -> '9:00-11:00 am') using the context.\n"
    "- Fix mishearings and proper nouns using the glossary and context. Prefer meaning "
    "over literal wording, but never invent facts.\n"
    "- Output ONLY the cleaned passage text for this span. No commentary, no labels."
)


def build_fusion_prompt(payload: "WindowPayload", profile: str, context: str, glossary: str) -> str:
    instructions = _KB_INSTRUCTIONS if profile == "kb" else _VERBATIM_INSTRUCTIONS
    parts: List[str] = [instructions, ""]
    if context.strip():
        parts += ["## Context", context.strip(), ""]
    if glossary.strip():
        parts += ["## Glossary (canonical spellings / corrections)", glossary.strip(), ""]
    if payload.prev_fused.strip():
        parts += ["## Preceding text (already finalized — for continuity only, do not repeat)",
                  payload.prev_fused.strip(), ""]
    parts.append(f"## Transcripts for span {payload.start:.1f}s-{payload.end:.1f}s")
    for label, text in payload.sources.items():
        parts.append(f"### {label}")
        parts.append(text.strip() or "(empty)")
    parts.append("")
    parts.append("## Output")
    return "\n".join(parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fusion.py -k prompt -v`
Expected: PASS (3).

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_fusion.py
git commit -m "feat: per-profile fusion prompt construction (verbatim/kb)"
```

---

## Task 9: Context loader + `--init-context` scaffolder

**Files:**
- Modify: `asr_bench.py` — fusion section + `main()` early-exit handling
- Test: `tests/test_context.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_context.py`:

```python
import asr_bench


def test_init_context_template_has_all_sections():
    t = asr_bench.init_context_template()
    for needle in ["Schedule", "Glossary", "Jargon", "Names", "Style", "mishearings"]:
        assert needle.lower() in t.lower(), needle


def test_load_context_reads_file_and_glossary(tmp_path):
    ctx = tmp_path / "context.md"
    ctx.write_text("I teach 9-11am.\n\n## Glossary\nAI not I\n", encoding="utf-8")
    context_text, glossary_text = asr_bench.load_context(str(ctx), None)
    assert "I teach 9-11am." in context_text
    assert "AI not I" in glossary_text


def test_load_context_separate_glossary_file_overrides(tmp_path):
    ctx = tmp_path / "context.md"
    ctx.write_text("topic notes\n\n## Glossary\nin-file gloss\n", encoding="utf-8")
    gl = tmp_path / "gloss.txt"
    gl.write_text("override gloss", encoding="utf-8")
    _, glossary_text = asr_bench.load_context(str(ctx), str(gl))
    assert glossary_text.strip() == "override gloss"


def test_load_context_none_returns_empty():
    assert asr_bench.load_context(None, None) == ("", "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_context.py -v`
Expected: FAIL — `init_context_template` not defined.

- [ ] **Step 3: Implement.** In the fusion section, add:

```python
_CONTEXT_GLOSSARY_HEADER_RE = re.compile(r"^#+\s*glossary\b", re.IGNORECASE | re.MULTILINE)


def init_context_template() -> str:
    """A guided context.md the user fills in, then passes via --context."""
    return """# Fusion context

Fill in what the fusion LLM should know about this corpus. Everything here is
fed to the model for every clip. Delete sections you don't need.

## Topic / course
<!-- e.g. "Undergraduate intro statistics; lectures cover hypothesis testing." -->

## Schedule & recurring times
<!-- e.g. "I teach 9-11am; there are no evening sessions, so 'final at 9pm' is wrong." -->

## Names (people, places) — canonical spelling
<!-- e.g. "Dr. Nguyen; the dataset is called CIFAR-10." -->

## Jargon & acronyms
<!-- e.g. "Spell 'AI' (not 'I'); 'p-value' (not 'p value')." -->

## Known mishearings to watch for
<!-- e.g. "'their' vs 'there'; 'affect' vs 'effect'." -->

## Style preferences
<!-- e.g. captions: keep verbatim; KB: full sentences, normalize numbers. -->

## Glossary
<!-- One correction per line, e.g.:
AI not I
CIFAR-10 not cipher ten
-->
"""


def load_context(context_path: Optional[str], glossary_path: Optional[str]) -> Tuple[str, str]:
    """Return (context_text, glossary_text).

    The glossary is the '## Glossary' section of the context file, unless a
    separate --glossary file is given (which overrides it). Missing files -> "".
    """
    context_text = ""
    glossary_text = ""
    if context_path:
        raw = Path(context_path).read_text(encoding="utf-8", errors="replace")
        m = _CONTEXT_GLOSSARY_HEADER_RE.search(raw)
        if m:
            context_text = raw[: m.start()].strip()
            glossary_text = raw[m.end():].strip()
        else:
            context_text = raw.strip()
    if glossary_path:
        glossary_text = Path(glossary_path).read_text(encoding="utf-8", errors="replace").strip()
    return context_text, glossary_text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_context.py -v`
Expected: PASS (4).

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_context.py
git commit -m "feat: context loader + init_context_template scaffold"
```

---

## Task 10: Fusion orchestration + output writers + drift guard

**Files:**
- Modify: `asr_bench.py` — fusion section (orchestration + writers)
- Test: `tests/test_fusion.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fusion.py`:

```python
from pathlib import Path
from asr_bench import Cue


def _sources():
    # two "models" + Panopto, each one cue per ~5s over 30s
    turbo = [Cue(0, 10, "the AI model learns"), Cue(10, 20, "we meet nine to eleven"), Cue(20, 30, "no class tonight")]
    panopto = [Cue(0, 10, "the I model learns"), Cue(10, 20, "we meet 9 to 11"), Cue(20, 30, "no class tonight")]
    return {"large-v3-turbo": turbo, "Panopto": panopto}


def test_fuse_clip_verbatim_produces_cues():
    backend = asr_bench.FakeLLMBackend(lambda prompt: "FUSEDTEXT")
    res = asr_bench.fuse_clip(
        duration=30.0, base_label="large-v3-turbo", sources=_sources(),
        profiles=["verbatim"], backend=backend, context="", glossary="",
        window=25.0, overlap=5.0, drift_threshold=2.0,
    )
    assert res.verbatim_cues, "expected verbatim cues"
    assert all(c.text == "FUSEDTEXT" for c in res.verbatim_cues)
    # cues tile without overlap (each start >= previous end)
    for a, b in zip(res.verbatim_cues, res.verbatim_cues[1:]):
        assert b.start >= a.end - 1e-6


def test_fuse_clip_kb_chunks_retain_overlap_spans():
    backend = asr_bench.FakeLLMBackend(lambda prompt: "kb chunk")
    res = asr_bench.fuse_clip(
        duration=60.0, base_label="large-v3-turbo", sources=_sources(),
        profiles=["kb"], backend=backend, context="", glossary="",
        window=25.0, overlap=5.0, drift_threshold=2.0,
    )
    assert len(res.kb_chunks) >= 2
    assert res.kb_chunks[0]["text"] == "kb chunk"
    assert "start" in res.kb_chunks[0] and "end" in res.kb_chunks[0]


def test_drift_guard_flags_divergent_window():
    # LLM returns text totally unrelated to the base -> high WER -> flagged
    backend = asr_bench.FakeLLMBackend(lambda prompt: "zzz qqq xyz")
    res = asr_bench.fuse_clip(
        duration=10.0, base_label="large-v3-turbo", sources=_sources(),
        profiles=["verbatim"], backend=backend, context="", glossary="",
        window=25.0, overlap=5.0, drift_threshold=0.5,
    )
    assert res.flags, "expected a drift flag"


def test_write_fused_vtt(tmp_path):
    audio = tmp_path / "Lecture_default.mp4"
    audio.write_bytes(b"x")
    cues = [Cue(0.0, 5.0, "hello"), Cue(5.0, 10.0, "world")]
    out = asr_bench.write_fused_vtt(audio, cues)
    assert out.name == "Lecture_Captions_Fused.vtt"
    body = out.read_text(encoding="utf-8")
    assert "WEBVTT" in body and "hello" in body


def test_write_kb_jsonl(tmp_path):
    import json
    audio = tmp_path / "Lecture_default.mp4"
    audio.write_bytes(b"x")
    chunks = [{"start": 0.0, "end": 25.0, "text": "a"}, {"start": 20.0, "end": 45.0, "text": "b"}]
    out = asr_bench.write_kb_jsonl(audio, chunks)
    assert out.name == "Lecture_KB_Fused.jsonl"
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows[1]["text"] == "b"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fusion.py -k "fuse_clip or drift or write_" -v`
Expected: FAIL — `fuse_clip` not defined.

- [ ] **Step 3: Implement.** In the fusion section, add:

```python
@dataclass
class FusionResult:
    verbatim_cues: List[Cue] = field(default_factory=list)
    kb_chunks: List[Dict] = field(default_factory=list)   # {start, end, text}
    flags: List[str] = field(default_factory=list)        # drift warnings


def _fused_base(audio_path: Path) -> str:
    stem = audio_path.stem
    return stem[: -len("_default")] if stem.endswith("_default") else stem


def write_fused_vtt(audio_path: Path, cues: List[Cue]) -> Path:
    out = audio_path.parent / f"{_fused_base(audio_path)}_Captions_Fused.vtt"
    lines: List[str] = ["WEBVTT", ""]
    for i, c in enumerate(cues, start=1):
        text = c.text.strip()
        if not text:
            continue
        lines.append(str(i))
        lines.append(f"{_fmt_vtt_time(c.start)} --> {_fmt_vtt_time(c.end)}")
        lines.append(text)
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_kb_jsonl(audio_path: Path, chunks: List[Dict]) -> Path:
    import json
    out = audio_path.parent / f"{_fused_base(audio_path)}_KB_Fused.jsonl"
    lines = [json.dumps(c, ensure_ascii=False) for c in chunks]
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


def write_kb_md(audio_path: Path, chunks: List[Dict]) -> Path:
    out = audio_path.parent / f"{_fused_base(audio_path)}_KB_Fused.md"
    lines: List[str] = [f"# Knowledge base — {_fused_base(audio_path)}", ""]
    for c in chunks:
        lines.append(f"## {_fmt_vtt_time(c['start'])} – {_fmt_vtt_time(c['end'])}")
        lines.append("")
        lines.append(c["text"].strip())
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def fuse_clip(
    duration: float,
    base_label: str,
    sources: Dict[str, List[Cue]],
    profiles: List[str],
    backend: "LLMBackend",
    context: str,
    glossary: str,
    window: float,
    overlap: float,
    drift_threshold: float,
) -> FusionResult:
    """Window the timeline, fuse each window per requested profile, assemble.

    - verbatim cues tile without overlap: cue i spans [win_start_i, win_start_{i+1}]
      (last to `duration`), text = LLM fusion of that window.
    - kb chunks keep the full overlapping window span [start, end].
    - drift guard: per window, WER(fused vs base text); flagged if > threshold.
    """
    res = FusionResult()
    windows = build_windows(duration, window, overlap)
    base_cues = sources.get(base_label, [])

    verbatim_prev = ""
    kb_prev = ""
    fused_by_profile: Dict[str, List[Tuple[Tuple[float, float], str]]] = {p: [] for p in profiles}

    for (w_start, w_end) in windows:
        payload_sources = {
            label: collect_window_text(cues, w_start, w_end) for label, cues in sources.items()
        }
        base_text = collect_window_text(base_cues, w_start, w_end)
        for profile in profiles:
            prev = verbatim_prev if profile == "verbatim" else kb_prev
            payload = WindowPayload(w_start, w_end, payload_sources, prev_fused=prev)
            prompt = build_fusion_prompt(payload, profile, context, glossary)
            try:
                fused = backend.generate(prompt).strip()
            except Exception as e:
                fused = ""
                res.flags.append(f"[{w_start:.0f}-{w_end:.0f}s {profile}] backend error: {e}")
            fused_by_profile[profile].append(((w_start, w_end), fused))
            if profile == "verbatim":
                verbatim_prev = fused
            else:
                kb_prev = fused
            # drift guard against the base model's text for this window
            if base_text and fused:
                drift = compute_word_metrics(normalize_for_wer(base_text), normalize_for_wer(fused)).wer
                if drift > drift_threshold:
                    res.flags.append(
                        f"[{w_start:.0f}-{w_end:.0f}s {profile}] drift WER {drift*100:.0f}% vs base — review"
                    )

    # Assemble verbatim cues (non-overlapping tiling)
    if "verbatim" in profiles:
        items = fused_by_profile["verbatim"]
        for idx, ((w_start, w_end), text) in enumerate(items):
            cue_end = items[idx + 1][0][0] if idx + 1 < len(items) else w_end
            if text.strip():
                res.verbatim_cues.append(Cue(w_start, max(cue_end, w_start), text.strip()))

    # Assemble kb chunks (overlapping spans retained)
    if "kb" in profiles:
        for (w_start, w_end), text in fused_by_profile["kb"]:
            if text.strip():
                res.kb_chunks.append({"start": w_start, "end": w_end, "text": text.strip()})

    return res
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fusion.py -v`
Expected: PASS (all prompt + orchestration + writer tests).

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_fusion.py
git commit -m "feat: fuse_clip orchestration, VTT/KB writers, drift guard"
```

---

## Task 11: Re-scoring models against the verbatim fused reference

**Files:**
- Modify: `asr_bench.py` — fusion section (`rescore_against_reference`) + a render helper for the second table
- Test: `tests/test_fusion.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fusion.py`:

```python
def test_rescore_against_reference_recomputes_metrics():
    # Build two model results with known hypotheses; rescore against a reference.
    c1 = asr_bench.ClipResult(
        audio="L.mp4", audio_sec=10, transcribe_sec=1, rtfx=10, vram_peak_bytes=None,
        hypothesis="the cat sat", reference_normalized="x", hypothesis_normalized="the cat sat",
        wer=0.9,
    )
    m1 = asr_bench.ModelResult(
        model_id="a", display="A", fw_name="a", params="1", developer="d", languages="en",
        notes="", disk_bytes=None, load_sec=0, clips=[c1],
    )
    # reference cues -> "the cat sat" (perfect for m1)
    ref_cues_by_clip = {"L.mp4": [asr_bench.Cue(0, 10, "the cat sat")]}
    rescored = asr_bench.rescore_against_reference([m1], ref_cues_by_clip)
    assert abs(rescored[0].clips[0].wer) < 1e-9   # perfect match -> 0 WER


def test_render_fused_rescore_table_labeled_biased():
    c1 = asr_bench.ClipResult(
        audio="L.mp4", audio_sec=10, transcribe_sec=1, rtfx=10, vram_peak_bytes=None,
        hypothesis="hi", reference_normalized="hi", hypothesis_normalized="hi",
        wer=0.0, mer=0.0, wil=0.0,
    )
    m1 = asr_bench.ModelResult(
        model_id="a", display="A", fw_name="a", params="1", developer="d", languages="en",
        notes="", disk_bytes=None, load_sec=0, clips=[c1],
    )
    md = asr_bench.render_fused_rescore_table([m1])
    assert "fused verbatim consensus" in md.lower()
    assert "biased" in md.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fusion.py -k "rescore or fused_rescore" -v`
Expected: FAIL — `rescore_against_reference` not defined.

- [ ] **Step 3: Implement.** In the fusion section, add:

```python
def rescore_against_reference(
    results: List["ModelResult"],
    reference_cues_by_clip: Dict[str, List[Cue]],
) -> List["ModelResult"]:
    """Return deep-ish copies of `results` with each clip's metrics recomputed
    against the fused verbatim reference (keyed by clip audio filename).

    Models are scored on their stored `hypothesis`. Clips with no matching
    reference are left unscored (NaN).
    """
    import copy
    out: List[ModelResult] = []
    for r in results:
        r2 = copy.deepcopy(r)
        for c in r2.clips:
            ref_cues = reference_cues_by_clip.get(c.audio)
            if not ref_cues:
                c.wer = c.mer = c.wil = float("nan")
                continue
            ref_text = normalize_for_wer(" ".join(cu.text for cu in ref_cues))
            hyp_text = normalize_for_wer(c.hypothesis)
            m = compute_word_metrics(ref_text, hyp_text)
            c.wer, c.mer, c.wil = m.wer, m.mer, m.wil
            c.hits, c.substitutions, c.deletions, c.insertions = (
                m.hits, m.substitutions, m.deletions, m.insertions,
            )
        out.append(r2)
    return out


def render_fused_rescore_table(results: List["ModelResult"]) -> str:
    lines: List[str] = []
    lines.append("## Scores vs fused verbatim reference")
    lines.append("")
    lines.append(
        "> **Reference = fused verbatim consensus (agreement-biased).** This reference "
        "was built from the models below, so scores favor models that agreed with the "
        "majority. Treat these as *relative*, not absolute accuracy."
    )
    lines.append("")
    lines.append("| Model | WER% | MER% | WIL% |")
    lines.append("|---|---|---|---|")
    for r in results:
        wer = f"{r.avg_wer * 100:.1f}" if r.clips else "—"
        mer = f"{r.avg_mer * 100:.1f}" if r.clips else "—"
        wil = f"{r.avg_wil * 100:.1f}" if r.clips else "—"
        lines.append(f"| {r.display} | {wer} | {mer} | {wil} |")
    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fusion.py -k "rescore or fused_rescore" -v`
Expected: PASS (2).

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_fusion.py
git commit -m "feat: rescore models against fused verbatim reference (labeled biased)"
```

---

## Task 12: CLI wiring + `main()` integration

**Files:**
- Modify: `asr_bench.py` — `main()` argparse (~1262), `--init-context` early exit (~1268), fusion stage after `results` built (~1362), report assembly (~1364)
- Test: `tests/test_cli.py` (append) + `tests/test_fusion.py` (end-to-end)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fusion.py`:

```python
def test_run_fusion_stage_end_to_end(tmp_path, monkeypatch):
    # Build a model result whose VTT we will parse back, plus a Panopto ref file.
    audio = tmp_path / "Lecture_default.mp4"
    audio.write_bytes(b"x")
    # model VTT next to audio
    vtt = tmp_path / "Lecture_Captions_LargeV3Turbo.vtt"
    vtt.write_text("WEBVTT\n\n1\n00:00:00.000 --> 00:00:10.000\nthe AI model\n", encoding="utf-8")

    clip = asr_bench.ClipResult(
        audio="Lecture_default.mp4", audio_sec=10.0, transcribe_sec=1.0, rtfx=10.0,
        vram_peak_bytes=None, hypothesis="the AI model", reference_normalized="",
        hypothesis_normalized="the ai model", wer=0.0, vtt_path=str(vtt),
    )
    mr = asr_bench.ModelResult(
        model_id="large-v3-turbo", display="Whisper Large V3 Turbo", fw_name="large-v3-turbo",
        params="809M", developer="OpenAI", languages="99", notes="", disk_bytes=None,
        load_sec=0.0, clips=[clip],
    )
    pair = asr_bench.Pair(audio=audio, reference=vtt)  # reuse vtt as the "panopto" ref

    backend = asr_bench.FakeLLMBackend(lambda prompt: "the AI model")
    fusion_md, rescored = asr_bench.run_fusion_stage(
        results=[mr], pairs=[pair], backend=backend,
        profiles=["verbatim", "kb"], base_label="large-v3-turbo",
        context="", glossary="", window=25.0, overlap=5.0, drift_threshold=2.0,
        rescore=True,
    )
    assert (tmp_path / "Lecture_Captions_Fused.vtt").exists()
    assert (tmp_path / "Lecture_KB_Fused.jsonl").exists()
    assert "Fusion" in fusion_md
    assert rescored is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fusion.py -k end_to_end -v`
Expected: FAIL — `run_fusion_stage` not defined.

- [ ] **Step 3a: Implement `run_fusion_stage`** in the fusion section:

```python
def run_fusion_stage(
    results: List["ModelResult"],
    pairs: List["Pair"],
    backend: "LLMBackend",
    profiles: List[str],
    base_label: str,
    context: str,
    glossary: str,
    window: float,
    overlap: float,
    drift_threshold: float,
    rescore: bool,
) -> Tuple[str, Optional[List["ModelResult"]]]:
    """Fuse every clip and write outputs. Returns (markdown_section, rescored_or_None).

    Sources per clip = each model's written VTT (parsed back to timed cues) +
    the Panopto/reference caption file (if it parses as timed cues).
    """
    lines: List[str] = ["## Fusion", ""]
    lines.append(f"- Backend: `{backend.name}`  Profiles: `{', '.join(profiles)}`  "
                 f"Window: {window:.0f}s / overlap {overlap:.0f}s  Base: `{base_label}`")
    lines.append("")
    lines.append(
        "> **Accessibility note:** only the *verbatim* output targets ADA/WCAG caption "
        "fidelity. The *kb* output is rephrased and is **not** compliant captions."
    )
    lines.append("")

    verbatim_ref_by_clip: Dict[str, List[Cue]] = {}
    pair_by_audio = {p.audio.name: p for p in pairs}

    # map model_id -> display label for source naming
    for clip_idx in range(len(results[0].clips) if results else 0):
        audio_name = results[0].clips[clip_idx].audio
        pair = pair_by_audio.get(audio_name)
        if pair is None:
            continue
        audio_path = pair.audio

        sources: Dict[str, List[Cue]] = {}
        duration = results[0].clips[clip_idx].audio_sec or 1.0
        for r in results:
            if clip_idx < len(r.clips) and r.clips[clip_idx].vtt_path:
                vp = Path(r.clips[clip_idx].vtt_path)
                if vp.is_file():
                    sources[r.model_id] = parse_caption_cues(vp)
        # Panopto / reference as a source (best-effort timed parse)
        try:
            ref_cues = parse_caption_cues(pair.reference)
            if ref_cues:
                sources["Panopto"] = ref_cues
        except Exception:
            pass

        if not sources:
            lines.append(f"- {audio_name}: no parseable sources — skipped")
            continue

        res = fuse_clip(
            duration=duration, base_label=base_label, sources=sources,
            profiles=profiles, backend=backend, context=context, glossary=glossary,
            window=window, overlap=overlap, drift_threshold=drift_threshold,
        )
        written: List[str] = []
        if "verbatim" in profiles and res.verbatim_cues:
            vtt_out = write_fused_vtt(audio_path, res.verbatim_cues)
            verbatim_ref_by_clip[audio_name] = res.verbatim_cues
            written.append(vtt_out.name)
        if "kb" in profiles and res.kb_chunks:
            written.append(write_kb_jsonl(audio_path, res.kb_chunks).name)
            written.append(write_kb_md(audio_path, res.kb_chunks).name)
        lines.append(f"- **{audio_name}** → {', '.join(f'`{w}`' for w in written) or '(nothing written)'}")
        for flag in res.flags:
            lines.append(f"  - ⚠️ {flag}")
    lines.append("")

    rescored = None
    if rescore and verbatim_ref_by_clip:
        rescored = rescore_against_reference(results, verbatim_ref_by_clip)
    return "\n".join(lines), rescored
```

- [ ] **Step 3b: Add CLI flags** in `main()` after `--show-alignment`:

```python
    ap.add_argument("--fuse", action="store_true",
                    help="After benchmarking, fuse all models + reference into a best transcript.")
    ap.add_argument("--profile", default="both", choices=["verbatim", "kb", "both"],
                    help="Fusion profile(s). verbatim=captions/reference, kb=RAG knowledge base.")
    ap.add_argument("--fuse-base", default="large-v3-turbo",
                    help="Model whose cue timing anchors the fusion windows.")
    ap.add_argument("--llm", default="ollama:qwen2.5",
                    help="Fusion LLM backend: fake | ollama:<model> | cli:<command>.")
    ap.add_argument("--context", default=None, help="Path to a fusion context file (see --init-context).")
    ap.add_argument("--glossary", default=None, help="Optional separate glossary file (overrides in-context glossary).")
    ap.add_argument("--window", type=float, default=25.0, help="Fusion window length in seconds.")
    ap.add_argument("--overlap", type=float, default=5.0, help="Fusion window overlap in seconds (context carryover).")
    ap.add_argument("--drift-threshold", type=float, default=1.0,
                    help="Flag a fused window whose WER vs the base model exceeds this (1.0 = 100%).")
    ap.add_argument("--rescore-against-fused", action="store_true",
                    help="Re-score every model against the verbatim fused reference (agreement-biased; labeled as such).")
    ap.add_argument("--init-context", nargs="?", const="context.md", default=None,
                    metavar="PATH", help="Write a context.md template to PATH (default context.md) and exit.")
```

- [ ] **Step 3c: Handle `--init-context` early exit.** Immediately after `args = ap.parse_args()` (~1267):

```python
    if args.init_context is not None:
        dest = Path(args.init_context)
        if dest.exists():
            print(f"ERROR: {dest} already exists — refusing to overwrite", file=sys.stderr)
            return 2
        dest.write_text(init_context_template(), encoding="utf-8")
        print(f"Wrote fusion context template to {dest}. Edit it, then pass --context {dest} --fuse.")
        return 0
```

- [ ] **Step 3d: Run the fusion stage + render.** Replace the report block (~1364-1366):

```python
    md = render_markdown(results, corpus, args, gold_label)

    if args.fuse:
        profiles = ["verbatim", "kb"] if args.profile == "both" else [args.profile]
        try:
            backend = make_llm_backend(args.llm)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        context_text, glossary_text = load_context(args.context, args.glossary)
        fusion_md, rescored = run_fusion_stage(
            results=results, pairs=pairs, backend=backend, profiles=profiles,
            base_label=args.fuse_base, context=context_text, glossary=glossary_text,
            window=args.window, overlap=args.overlap, drift_threshold=args.drift_threshold,
            rescore=args.rescore_against_fused,
        )
        md = md + "\n" + fusion_md
        if rescored is not None:
            md = md + "\n" + render_fused_rescore_table(rescored)

    print()
    print(md)
```

- [ ] **Step 3e: CLI smoke test.** Append to `tests/test_cli.py`:

```python
def test_init_context_writes_and_exits(tmp_path, monkeypatch, capsys):
    import asr_bench
    dest = tmp_path / "context.md"
    monkeypatch.setattr("sys.argv", ["asr_bench.py", "--init-context", str(dest)])
    rc = asr_bench.main()
    assert rc == 0
    assert dest.is_file()
    assert "Glossary" in dest.read_text(encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fusion.py tests/test_cli.py -v`
Expected: PASS.
Run full suite: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_fusion.py tests/test_cli.py
git commit -m "feat: wire --fuse/--profile/--llm/--init-context into main()"
```

---

## Task 13: Documentation + memory

**Files:**
- Modify: `CLAUDE.md`, `SPEC.md`, `README.md`
- Create: `C:\Users\krank\.claude\projects\D--Dev-asr-bench\memory\fusion-llm-validation.md` + index line in `MEMORY.md`

- [ ] **Step 1: Update `CLAUDE.md`**
  - Bump Status heading to **v0.2** and add a "What's new in v0.2" bullet list: MER/WIL/HSDI metrics, `--show-alignment`, fusion stage (verbatim + kb profiles), pluggable LLM backend, `--init-context`.
  - Add workflows: `--init-context`, a `--fuse` example with `--llm ollama:qwen2.5` and one with `--llm cli:claude`, and `--rescore-against-fused`.
  - Add decision-log entries dated **2026-06-01**: (a) MER/WIL adopted from Morris et al.; (b) fusion is a post-pass, not an Engine; (c) two profiles (verbatim=captions/reference, kb=RAG); (d) LLM is pluggable, local-first default, CLI-subprocess for frontier without API keys; (e) verbatim-only is ADA/WCAG-eligible; rescore reference is agreement-biased.

- [ ] **Step 2: Update `SPEC.md`** — mark metrics + fusion as the shipped v0.2 line (ahead of the prior WhisperX slot, which moves to v0.3); reference `docs/superpowers/specs/2026-06-01-metrics-and-fusion-design.md`.

- [ ] **Step 3: Update `README.md`** — add a "Metrics" subsection (WER vs MER vs WIL, one line each, cite the paper) and a "Fusion" subsection (the two profiles, the accessibility caveat, the consensus-bias caveat, example commands). Note Ollama as an optional dependency for local fusion.

- [ ] **Step 4: Write the memory note**

Create `C:\Users\krank\.claude\projects\D--Dev-asr-bench\memory\fusion-llm-validation.md`:

```markdown
---
name: fusion-llm-validation
description: Fusion stage shipped with FakeLLMBackend tests; verbatim/kb output never validated against a live Ollama or CLI backend on real lecture audio.
metadata:
  type: project
---

v0.2 fusion (verbatim + kb profiles) is fully unit-tested via FakeLLMBackend but
has NOT been run end-to-end against a live LLM (Ollama `qwen2.5` or `cli:claude`)
on real lecture audio. Pending: a real `--fuse` run to validate prompt quality,
drift-guard threshold tuning, and VTT/KB output usefulness. Mirrors the
[[validate-live-nim]] deferral pattern.
```

Add to `MEMORY.md`:

```markdown
- [Fusion LLM validation](fusion-llm-validation.md) — pending: real --fuse run vs live Ollama/CLI on lecture audio; only FakeLLMBackend-tested so far.
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md SPEC.md README.md
git commit -m "docs: v0.2 metrics + fusion — usage, caveats, decision log"
```

(The memory files live outside the repo and are not committed.)

---

## Final verification

- [ ] Run the whole suite: `python -m pytest -q` — expected: all green (37 existing + ~30 new).
- [ ] Smoke the metrics path on real audio (if available):
  `python asr_bench.py --models small --limit 1 --show-alignment`
  Expect MER%/WIL% columns and an Alignment detail section in the report.
- [ ] Smoke fusion with the deterministic backend (no LLM needed):
  `python asr_bench.py --models small,medium --limit 1 --fuse --llm fake`
  Expect `*_Captions_Fused.vtt` + `*_KB_Fused.jsonl/.md` next to the audio and a Fusion section in the report.
- [ ] (Optional, needs Ollama) `python asr_bench.py --models small,medium --limit 1 --fuse --llm ollama:qwen2.5 --context context.md` — validates the live path; update `fusion-llm-validation.md` with findings.
- [ ] Use superpowers:requesting-code-review before merging `feat/metrics-and-fusion`.
