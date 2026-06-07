# CER + Median Latency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-clip CER (Character Error Rate) and a robust per-model median speed (median RTFx + median seconds-per-audio-minute) to asr-bench's metrics, report tables, JSON sidecar, and the `compare` subcommand.

**Architecture:** CER is computed in `WordMetrics`/`compute_word_metrics` (one place → all three engines get it). Median speed is two new `ModelResult` properties next to `aggregate_rtfx`. Rendering, the JSON sidecar, and `asr_compare.py`'s metric set are extended. `schema_version` stays 1 (additive fields).

**Tech Stack:** Python 3.14 stdlib + jiwer (already a dependency). pytest. No torch.

**Reference:** spec at `docs/superpowers/specs/2026-06-06-cer-median-latency-design.md`.

---

## File Structure

- **Modify `asr_bench.py`**: `WordMetrics` (+`cer` field), `compute_word_metrics` (compute CER + update the two NaN-construction sites), `ClipResult` (+`cer`), three `ClipResult(...)` build sites (+`cer=metrics.cer`), `ModelResult` (+`avg_cer`/`median_rtfx`/`median_sec_per_audio_min`, +`import statistics`), `render_markdown` (3 tables), `_clip_to_dict`/`_model_to_dict` (sidecar).
- **Modify `asr_compare.py`**: `METRIC_META` + `_AGG_KEYS` (+`cer`).
- **Modify tests**: `tests/test_metrics.py` (or wherever WER metrics are tested — see Task 1), `tests/test_render.py`, `tests/test_results_json.py`, `tests/test_compare.py`.
- **Modify docs**: README.md, CLAUDE.md, SPEC.md.

---

## Task 1: CER in `WordMetrics` + `compute_word_metrics`

**Files:**
- Modify: `asr_bench.py` (`WordMetrics` ~line 336, `compute_word_metrics` ~line 354)
- Test: `tests/test_metrics.py` (create if absent; otherwise add to the existing metrics test file)

- [ ] **Step 1: Find where word metrics are tested**

Run: `python -m pytest --collect-only -q 2>$null | Select-String -Pattern "metric"` (PowerShell) — or look for a test that calls `compute_word_metrics` / `process_words`. If a metrics test file exists, append there; otherwise create `tests/test_metrics.py` with:

```python
import math
import asr_bench
```

- [ ] **Step 2: Write the failing tests**

Append:

```python
def test_compute_word_metrics_has_cer():
    m = asr_bench.compute_word_metrics("the cat", "the bat")
    # one character substitution ('c'->'b') over 7 reference chars
    assert abs(m.cer - 1.0 / 7.0) < 1e-6


def test_compute_word_metrics_empty_ref_cer_is_nan():
    m = asr_bench.compute_word_metrics("", "anything")
    assert math.isnan(m.cer)


def test_compute_word_metrics_perfect_match_cer_zero():
    m = asr_bench.compute_word_metrics("hello world", "hello world")
    assert m.cer == 0.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL — `AttributeError: 'WordMetrics' object has no attribute 'cer'`.

- [ ] **Step 4: Add the `cer` field to `WordMetrics`**

In `asr_bench.py`, the dataclass currently is:

```python
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
```

Change the docstring's first line and add `cer` right after `wil`:

```python
@dataclass
class WordMetrics:
    """All alignment scores from a single jiwer alignment (word-level + CER).

    WER  = (S+D+I)/N1                      (edit cost; can exceed 1.0)
    MER  = (S+D+I)/(H+S+D+I)               (Morris et al.; bounded [0,1])
    WIL  = 1 - H*H/(N1*N2)                 (Morris et al.; bounded [0,1])
    CER  = char-level edit rate            (same family as WER, char units)
    where N1 = ref words = H+S+D, N2 = hyp words = H+S+I.
    """
    wer: float
    mer: float
    wil: float
    cer: float
    hits: int
    substitutions: int
    deletions: int
    insertions: int
```

- [ ] **Step 5: Compute CER and fix the two NaN-construction sites in `compute_word_metrics`**

The function currently is:

```python
def compute_word_metrics(reference: str, hypothesis: str) -> WordMetrics:
    """One jiwer.process_words call -> WER, MER, WIL, and H/S/D/I counts.

    Inputs should already be normalized (see normalize_for_wer). Returns NaN
    metrics (not an exception) when alignment is impossible (e.g. empty ref).
    """
    nan = float("nan")
    if not reference.strip():
        return WordMetrics(nan, nan, nan, 0, 0, 0, 0)
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
        return WordMetrics(nan, nan, nan, 0, 0, 0, 0)
```

Replace it with (CER added; both positional NaN sites get a 4th `nan`):

```python
def compute_word_metrics(reference: str, hypothesis: str) -> WordMetrics:
    """One jiwer alignment -> WER, MER, WIL, CER, and H/S/D/I counts.

    Inputs should already be normalized (see normalize_for_wer). Returns NaN
    metrics (not an exception) when alignment is impossible (e.g. empty ref).
    """
    nan = float("nan")
    if not reference.strip():
        return WordMetrics(nan, nan, nan, nan, 0, 0, 0, 0)
    from jiwer import process_words, cer as jiwer_cer
    try:
        out = process_words(reference, hypothesis)
        return WordMetrics(
            wer=float(out.wer),
            mer=float(out.mer),
            wil=float(out.wil),
            cer=float(jiwer_cer(reference, hypothesis)),
            hits=int(out.hits),
            substitutions=int(out.substitutions),
            deletions=int(out.deletions),
            insertions=int(out.insertions),
        )
    except Exception:
        return WordMetrics(nan, nan, nan, nan, 0, 0, 0, 0)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS. Then `python -m pytest -q` — confirm no regressions (any existing test that constructs `WordMetrics` positionally would break; if one does, it is in the test suite and must be updated to the 8-arg form — search tests for `WordMetrics(`).

- [ ] **Step 7: Commit**

```bash
git add asr_bench.py tests/test_metrics.py
git commit -m "feat(metrics): compute per-clip CER in WordMetrics"
```

---

## Task 2: `ClipResult.cer` field + three construction sites

**Files:**
- Modify: `asr_bench.py` (`ClipResult` ~line 614; build sites ~990, ~1120, ~1361)
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_metrics.py`:

```python
def test_clipresult_has_cer_field_default_nan():
    c = asr_bench.ClipResult(
        audio="x.mp4", audio_sec=10.0, transcribe_sec=1.0, rtfx=10.0,
        vram_peak_bytes=None, hypothesis="h", reference_normalized="h",
        hypothesis_normalized="h", wer=0.1,
    )
    assert math.isnan(c.cer)  # default
    c2 = asr_bench.ClipResult(
        audio="x.mp4", audio_sec=10.0, transcribe_sec=1.0, rtfx=10.0,
        vram_peak_bytes=None, hypothesis="h", reference_normalized="h",
        hypothesis_normalized="h", wer=0.1, cer=0.05,
    )
    assert c2.cer == 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py::test_clipresult_has_cer_field_default_nan -v`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'cer'`.

- [ ] **Step 3: Add the field**

In `ClipResult`, the fields currently are:

```python
    wer: float
    mer: float = float("nan")
    wil: float = float("nan")
    hits: int = 0
```

Insert `cer` after `wil`:

```python
    wer: float
    mer: float = float("nan")
    wil: float = float("nan")
    cer: float = float("nan")
    hits: int = 0
```

- [ ] **Step 4: Thread `cer=metrics.cer` into the three build sites**

Site A — faster-whisper (~line 990), currently has these lines:
```python
                    wer=wer_val,
                    mer=metrics.mer,
                    wil=metrics.wil,
```
Change to:
```python
                    wer=wer_val,
                    mer=metrics.mer,
                    wil=metrics.wil,
                    cer=metrics.cer,
```

Site B — NIM (~line 1120), currently:
```python
                    wer=wer_val, mer=metrics.mer, wil=metrics.wil,
```
Change to:
```python
                    wer=wer_val, mer=metrics.mer, wil=metrics.wil, cer=metrics.cer,
```

Site C — WhisperX (~line 1361), currently:
```python
                wer=metrics.wer, mer=metrics.mer, wil=metrics.wil, hits=metrics.hits,
```
Change to:
```python
                wer=metrics.wer, mer=metrics.mer, wil=metrics.wil, cer=metrics.cer, hits=metrics.hits,
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -v` then `python -m pytest -q`
Expected: PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add asr_bench.py tests/test_metrics.py
git commit -m "feat(metrics): ClipResult.cer field threaded through all three engines"
```

---

## Task 3: `ModelResult` median/avg properties

**Files:**
- Modify: `asr_bench.py` (`import statistics` near the top imports; `ModelResult` properties ~line 654-689)
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_metrics.py`:

```python
def _clip(rtfx=10.0, audio_sec=600.0, transcribe_sec=60.0, cer=0.10):
    return asr_bench.ClipResult(
        audio="c.mp4", audio_sec=audio_sec, transcribe_sec=transcribe_sec,
        rtfx=rtfx, vram_peak_bytes=None, hypothesis="h",
        reference_normalized="h", hypothesis_normalized="h", wer=0.1, cer=cer,
    )


def _model(clips):
    return asr_bench.ModelResult(
        model_id="m", display="M", fw_name="m", params="1", developer="x",
        languages="en", notes="", disk_bytes=None, load_sec=0.0, clips=clips,
    )


def test_avg_cer():
    m = _model([_clip(cer=0.10), _clip(cer=0.20)])
    assert abs(m.avg_cer - 0.15) < 1e-9


def test_median_rtfx_resists_outlier():
    # two fast clips + one lockup clip; median ignores the outlier,
    # but the totals-based aggregate_rtfx is dragged down by it.
    clips = [_clip(rtfx=60.0, audio_sec=600.0, transcribe_sec=10.0),
             _clip(rtfx=62.0, audio_sec=600.0, transcribe_sec=9.7),
             _clip(rtfx=3.0, audio_sec=600.0, transcribe_sec=200.0)]
    m = _model(clips)
    assert m.median_rtfx == 60.0
    assert m.median_rtfx > m.aggregate_rtfx  # outlier resistance


def test_median_sec_per_audio_min():
    # audio 600s (10 min), transcribe 10s -> 1.0 s per audio-minute
    m = _model([_clip(audio_sec=600.0, transcribe_sec=10.0)])
    assert abs(m.median_sec_per_audio_min - 1.0) < 1e-9


def test_median_sec_per_audio_min_skips_zero_audio():
    m = _model([_clip(audio_sec=0.0, transcribe_sec=5.0),
                _clip(audio_sec=600.0, transcribe_sec=10.0)])
    assert abs(m.median_sec_per_audio_min - 1.0) < 1e-9  # only the valid clip


def test_median_properties_empty_model():
    m = _model([])
    assert m.avg_cer == 0.0
    assert m.median_rtfx == 0.0
    assert m.median_sec_per_audio_min == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py -k "cer or median" -v`
Expected: FAIL — `AttributeError: 'ModelResult' object has no attribute 'avg_cer'`.

- [ ] **Step 3: Add `import statistics`**

Near the top stdlib imports of `asr_bench.py` (alphabetically, after `import shutil`/with the others), add:
```python
import statistics
```

- [ ] **Step 4: Add the properties**

In `ModelResult`, right after the `avg_wil` property (~line 670) add:

```python
    @property
    def avg_cer(self) -> float:
        if not self.clips:
            return 0.0
        return sum(c.cer for c in self.clips) / len(self.clips)
```

And right after the `aggregate_rtfx` property (~line 684) add:

```python
    @property
    def median_rtfx(self) -> float:
        """Median per-clip RTFx — robust to a single decoder-lockup outlier that
        would drag down the totals-based aggregate_rtfx."""
        if not self.clips:
            return 0.0
        return statistics.median(c.rtfx for c in self.clips)

    @property
    def median_sec_per_audio_min(self) -> float:
        """Median per-clip compute-seconds per minute of audio (lower = faster)."""
        vals = [c.transcribe_sec * 60.0 / c.audio_sec
                for c in self.clips if c.audio_sec > 0]
        return statistics.median(vals) if vals else 0.0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -v` then `python -m pytest -q`
Expected: PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add asr_bench.py tests/test_metrics.py
git commit -m "feat(metrics): avg_cer, median_rtfx, median_sec_per_audio_min on ModelResult"
```

---

## Task 4: Report rendering (3 tables)

**Files:**
- Modify: `asr_bench.py` `render_markdown` (Headline ~2061-2078; Per-clip view ~2120-2131; Per-model breakdown ~2142-2161)
- Test: `tests/test_render.py`

Note: `tests/test_render.py` builds `ModelResult`/`ClipResult` via `_whisper_result()`/`_whisperx_result()` helpers. Those clips don't set `cer`, so it defaults to NaN — fine (renders as a dash). For the positive assertions below, the tests set `cer` explicitly.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_render.py`:

```python
def test_headline_has_cer_and_median_rtfx_columns():
    r = _whisper_result()
    r.clips[0].cer = 0.05
    md = asr_bench.render_markdown([r], Path("."), _args(), "proxy")
    header = [l for l in md.splitlines() if l.startswith("| Model | Params")][0]
    assert "CER%" in header
    assert "RTFx (med)" in header


def test_headline_renders_median_rtfx_value():
    r = _whisper_result()  # one clip, rtfx 60.0 -> median 60.00x
    md = asr_bench.render_markdown([r], Path("."), _args(), "proxy")
    assert "60.00x" in md


def test_per_clip_view_has_cer_column():
    r = _whisper_result()
    r.clips[0].cer = 0.05
    md = asr_bench.render_markdown([r], Path("."), _args(), "proxy")
    # the per-clip view header row
    assert any("| Model | WER% | MER% | WIL% | CER% |" in l for l in md.splitlines())


def test_per_model_breakdown_has_cer_column():
    r = _whisper_result()
    r.clips[0].cer = 0.05
    md = asr_bench.render_markdown([r], Path("."), _args(), "proxy")
    assert any("| Clip | Audio | WER% | MER% | WIL% | CER% |" in l for l in md.splitlines())


def test_cer_nan_renders_as_dash_not_nan():
    r = _whisper_result()
    r.clips[0].cer = float("nan")
    md = asr_bench.render_markdown([r], Path("."), _args(), "proxy")
    # no literal 'nan' in the tables (before the reproducibility footer)
    assert "nan" not in md.lower().split("reproducibility")[0]
```

Check `_whisper_result()` in `tests/test_render.py`: its clip has `rtfx=60.0`. If it does not, adjust the `60.00x` assertion to match the helper's rtfx. (Read the helper first.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_render.py -k "cer or median" -v`
Expected: FAIL — columns not present.

- [ ] **Step 3: Headline table — add CER% and RTFx (med)**

Replace the header + separator lines (currently):
```python
    lines.append("| Model | Params | Disk | Overall WER% | MER% | WIL% | RTFx | Total time | Peak VRAM |" + diar_hdr + " Notes |")
    lines.append("|---|---|---|---|---|---|---|---|---|" + diar_sep + "---|")
```
with:
```python
    lines.append("| Model | Params | Disk | Overall WER% | MER% | WIL% | CER% | RTFx | RTFx (med) | Total time | Peak VRAM |" + diar_hdr + " Notes |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|" + diar_sep + "---|")
```

In the per-model row loop, after `wil_pct = ...` add a CER cell and a median-RTFx cell. Currently:
```python
        wil_pct = _fmt_pct(r.avg_wil) if r.clips else "—"
        rtfx = f"{r.aggregate_rtfx:.2f}x" if r.clips else "—"
```
Change to:
```python
        wil_pct = _fmt_pct(r.avg_wil) if r.clips else "—"
        cer_pct = _fmt_pct(r.avg_cer) if r.clips else "—"
        rtfx = f"{r.aggregate_rtfx:.2f}x" if r.clips else "—"
        rtfx_med = f"{r.median_rtfx:.2f}x" if r.clips else "—"
```

And the row append, currently:
```python
        lines.append(
            f"| {r.display} | {r.params} | {disk} | {wer_pct} | {mer_pct} | {wil_pct} | {rtfx} | {wall_clock} | {vram} |{diar_cells} {r.notes} |"
        )
```
Change to:
```python
        lines.append(
            f"| {r.display} | {r.params} | {disk} | {wer_pct} | {mer_pct} | {wil_pct} | {cer_pct} | {rtfx} | {rtfx_med} | {wall_clock} | {vram} |{diar_cells} {r.notes} |"
        )
```

- [ ] **Step 4: Per-clip view — add CER%**

Header + separator (currently):
```python
            lines.append("| Model | WER% | MER% | WIL% | S | D | I | RTFx | Transcribe time | VRAM peak |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
```
Change to:
```python
            lines.append("| Model | WER% | MER% | WIL% | CER% | S | D | I | RTFx | Transcribe time | VRAM peak |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
```

In the row loop, after `wil_pct = _fmt_pct(c.wil)` add `cer_pct = _fmt_pct(c.cer)`, and insert `{cer_pct}` into the row after `{wil_pct}`:
```python
                    wer_pct = _fmt_pct(c.wer)
                    mer_pct = _fmt_pct(c.mer)
                    wil_pct = _fmt_pct(c.wil)
                    cer_pct = _fmt_pct(c.cer)
                    vram = _vram_cell(c.vram_peak_bytes, r.vram_is_total)
                    lines.append(
                        f"| {r.display} | {wer_pct} | {mer_pct} | {wil_pct} | {cer_pct} | {c.substitutions} | {c.deletions} | {c.insertions} | {c.rtfx:.2f}x | {c.transcribe_sec:.1f}s | {vram} |"
                    )
```

- [ ] **Step 5: Per-model breakdown — add CER% (per-clip rows + OVERALL)**

Header + separator (currently):
```python
        lines.append("| Clip | Audio | WER% | MER% | WIL% | RTFx | Transcribe time | VRAM peak |")
        lines.append("|---|---|---|---|---|---|---|---|")
```
Change to:
```python
        lines.append("| Clip | Audio | WER% | MER% | WIL% | CER% | RTFx | Transcribe time | VRAM peak |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
```

Per-clip row loop, add `cer_pct = _fmt_pct(c.cer)` and insert into the row:
```python
        for c in r.clips:
            wer_pct = _fmt_pct(c.wer)
            mer_pct = _fmt_pct(c.mer)
            wil_pct = _fmt_pct(c.wil)
            cer_pct = _fmt_pct(c.cer)
            vram = _vram_cell(c.vram_peak_bytes, r.vram_is_total)
            audio_label = f"{c.audio_sec / 60:.1f} min"
            lines.append(
                f"| {c.audio} | {audio_label} | {wer_pct} | {mer_pct} | {wil_pct} | {cer_pct} | {c.rtfx:.2f}x | {c.transcribe_sec:.1f}s | {vram} |"
            )
```

OVERALL row — add `overall_cer` and insert it. Currently:
```python
        overall_wer = _fmt_pct(r.avg_wer) if r.clips else "—"
        overall_mer = _fmt_pct(r.avg_mer) if r.clips else "—"
        overall_wil = _fmt_pct(r.avg_wil) if r.clips else "—"
        overall_rtfx = f"{r.aggregate_rtfx:.2f}x" if r.clips else "—"
        overall_vram = _vram_cell(r.peak_vram_bytes, r.vram_is_total)
        lines.append(
            f"| **OVERALL** | **{overall_audio}** | **{overall_wer}** | **{overall_mer}** | **{overall_wil}** | **{overall_rtfx}** | **{r.total_transcribe_sec:.1f}s** | **{overall_vram}** |"
        )
```
Change to:
```python
        overall_wer = _fmt_pct(r.avg_wer) if r.clips else "—"
        overall_mer = _fmt_pct(r.avg_mer) if r.clips else "—"
        overall_wil = _fmt_pct(r.avg_wil) if r.clips else "—"
        overall_cer = _fmt_pct(r.avg_cer) if r.clips else "—"
        overall_rtfx = f"{r.aggregate_rtfx:.2f}x" if r.clips else "—"
        overall_vram = _vram_cell(r.peak_vram_bytes, r.vram_is_total)
        lines.append(
            f"| **OVERALL** | **{overall_audio}** | **{overall_wer}** | **{overall_mer}** | **{overall_wil}** | **{overall_cer}** | **{overall_rtfx}** | **{r.total_transcribe_sec:.1f}s** | **{overall_vram}** |"
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_render.py -v` then `python -m pytest -q`
Expected: PASS, no regressions. (Existing render tests that count columns or match header substrings may need updating if they asserted the OLD header exactly — check `test_headline_has_mer_and_wil_columns` and any DER column-position test; the DER columns are appended AFTER Notes-adjacent cells so they are unaffected, but verify.)

- [ ] **Step 7: Commit**

```bash
git add asr_bench.py tests/test_render.py
git commit -m "feat(report): CER% in all metric tables + RTFx (med) in headline"
```

---

## Task 5: JSON sidecar fields

**Files:**
- Modify: `asr_bench.py` (`_clip_to_dict` ~line 1920; `_model_to_dict` aggregates ~line 1944)
- Test: `tests/test_results_json.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_results_json.py` (reuse its existing result fixtures; if it imports helpers from test_render, mirror that). Use a locally-built result if simpler:

```python
def test_sidecar_clip_has_cer(_results_doc_helpers=None):
    import asr_bench
    clip = asr_bench.ClipResult(
        audio="c.mp4", audio_sec=600.0, transcribe_sec=10.0, rtfx=60.0,
        vram_peak_bytes=None, hypothesis="h", reference_normalized="h",
        hypothesis_normalized="h", wer=0.1, cer=0.05)
    d = asr_bench._clip_to_dict(clip)
    assert d["cer"] == 0.05


def test_sidecar_model_aggregates_have_new_speed_and_cer():
    import asr_bench
    clip = asr_bench.ClipResult(
        audio="c.mp4", audio_sec=600.0, transcribe_sec=10.0, rtfx=60.0,
        vram_peak_bytes=None, hypothesis="h", reference_normalized="h",
        hypothesis_normalized="h", wer=0.1, cer=0.05)
    m = asr_bench.ModelResult(
        model_id="m", display="M", fw_name="m", params="1", developer="x",
        languages="en", notes="", disk_bytes=None, load_sec=0.0, clips=[clip])
    agg = asr_bench._model_to_dict(m)["aggregates"]
    assert agg["avg_cer"] == 0.05
    assert agg["median_rtfx"] == 60.0
    assert abs(agg["median_sec_per_audio_min"] - 1.0) < 1e-9
```

(If `tests/test_results_json.py` already has helper fixtures that build a `ModelResult`, prefer those over the inline build — read the file first and match its style. Either is acceptable.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_results_json.py -k "cer or speed" -v`
Expected: FAIL — `KeyError: 'cer'` / `KeyError: 'avg_cer'`.

- [ ] **Step 3: Add the fields**

`_clip_to_dict` currently has:
```python
        "wer": c.wer, "mer": c.mer, "wil": c.wil,
```
Change to:
```python
        "wer": c.wer, "mer": c.mer, "wil": c.wil, "cer": c.cer,
```

`_model_to_dict` `aggregates` currently:
```python
        "aggregates": {
            "avg_wer": m.avg_wer, "avg_mer": m.avg_mer, "avg_wil": m.avg_wil,
            "total_audio_sec": m.total_audio_sec,
            "total_transcribe_sec": m.total_transcribe_sec,
            "aggregate_rtfx": m.aggregate_rtfx, "peak_vram_bytes": m.peak_vram_bytes,
        },
```
Change to:
```python
        "aggregates": {
            "avg_wer": m.avg_wer, "avg_mer": m.avg_mer, "avg_wil": m.avg_wil,
            "avg_cer": m.avg_cer,
            "total_audio_sec": m.total_audio_sec,
            "total_transcribe_sec": m.total_transcribe_sec,
            "aggregate_rtfx": m.aggregate_rtfx,
            "median_rtfx": m.median_rtfx,
            "median_sec_per_audio_min": m.median_sec_per_audio_min,
            "peak_vram_bytes": m.peak_vram_bytes,
        },
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_results_json.py -v` then `python -m pytest -q`
Expected: PASS, no regressions. `schema_version` is unchanged (still 1) — confirm no test asserts the old aggregates dict shape exactly; if one does, update it to include the new keys.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_results_json.py
git commit -m "feat(sidecar): add cer + median_rtfx + median_sec_per_audio_min (schema_version stays 1)"
```

---

## Task 6: CER in the `compare` subcommand

**Files:**
- Modify: `asr_compare.py` (`METRIC_META`, `_AGG_KEYS`)
- Test: `tests/test_compare.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_compare.py` (reuse the `_doc`/`_model` helpers; note `_model` builds `aggregates` — it must include `avg_cer` for this test; if the helper doesn't, pass it through). First check `_model` in that file: it builds `aggregates` with avg_wer/avg_mer/avg_wil/aggregate_rtfx. Add `avg_cer` support to the helper:

Update the `_model` helper's `aggregates` dict to include `"avg_cer": cer` and add a `cer=0.05` kwarg to its signature (default 0.05). Then append:

```python
def test_compare_surfaces_cer_metric():
    a = _doc("a", models=[_model("m", "M", wer=0.10, cer=0.05)])
    b = _doc("b", models=[_model("m", "M", wer=0.08, cer=0.04)])
    rep = asr_compare.compare_runs([a, b], mode="delta")
    assert "cer" in rep["metrics"]
    md = asr_compare.render_comparison_markdown(rep)
    assert "CER%" in md


def test_compare_cer_absent_renders_dash_for_old_sidecar():
    # an aggregates dict WITHOUT avg_cer (older sidecar) -> CER cell is —
    old_model = {"model_id": "m", "display": "M",
                 "aggregates": {"avg_wer": 0.10, "avg_mer": 0.09, "avg_wil": 0.12,
                                "aggregate_rtfx": 60.0, "peak_vram_bytes": None},
                 "clips": []}
    a = _doc("a", models=[old_model])
    b = _doc("b", models=[old_model])
    rep = asr_compare.compare_runs([a, b], mode="delta")
    row = rep["models"][0]
    assert row["values"]["cer"] == [None, None]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_compare.py -k "cer" -v`
Expected: FAIL — `"cer" not in rep["metrics"]`.

- [ ] **Step 3: Add CER to the metric set**

In `asr_compare.py`, `METRIC_META` currently:
```python
METRIC_META = {
    "wer":  {"label": "WER%", "pct": True,  "lower_better": True},
    "mer":  {"label": "MER%", "pct": True,  "lower_better": True},
    "wil":  {"label": "WIL%", "pct": True,  "lower_better": True},
    "rtfx": {"label": "RTFx", "pct": False, "lower_better": False},
    "der":  {"label": "DER%", "pct": True,  "lower_better": True},
}
```
Add `cer` after `wil`:
```python
METRIC_META = {
    "wer":  {"label": "WER%", "pct": True,  "lower_better": True},
    "mer":  {"label": "MER%", "pct": True,  "lower_better": True},
    "wil":  {"label": "WIL%", "pct": True,  "lower_better": True},
    "cer":  {"label": "CER%", "pct": True,  "lower_better": True},
    "rtfx": {"label": "RTFx", "pct": False, "lower_better": False},
    "der":  {"label": "DER%", "pct": True,  "lower_better": True},
}
```

`_AGG_KEYS` currently:
```python
_AGG_KEYS = {"wer": "avg_wer", "mer": "avg_mer", "wil": "avg_wil",
             "rtfx": "aggregate_rtfx"}
```
Add `cer`:
```python
_AGG_KEYS = {"wer": "avg_wer", "mer": "avg_mer", "wil": "avg_wil",
             "cer": "avg_cer", "rtfx": "aggregate_rtfx"}
```

Note: `compare_runs` builds the base metric list as `["wer", "mer", "wil", "rtfx"]` (+ der). It must now include `cer`. Find that line:
```python
    metrics = ["wer", "mer", "wil", "rtfx"] + (["der"] if _has_any_der(docs) else [])
```
Change to:
```python
    metrics = ["wer", "mer", "wil", "cer", "rtfx"] + (["der"] if _has_any_der(docs) else [])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_compare.py -v` then `python -m pytest -q`
Expected: PASS. Existing compare tests that asserted a specific `metrics` list or column layout may need the `cer` addition — update any that break (e.g. a test counting metric columns).

- [ ] **Step 5: Commit**

```bash
git add asr_compare.py tests/test_compare.py
git commit -m "feat(compare): surface CER% in delta/matrix views"
```

---

## Task 7: Documentation + decision log

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `SPEC.md`

- [ ] **Step 1: README — metrics description**

Find where README lists the metrics (search "WER", "MER", "RTFx"). Add CER and median RTFx, e.g.:
```markdown
- **CER%** — Character Error Rate (jiwer), the finer-grained companion to WER.
- **RTFx (med)** — median per-clip RTFx, robust to a single slow/locked clip that
  skews the totals-based aggregate RTFx.
```

- [ ] **Step 2: CLAUDE.md — Status bullet**

In the "What's new" area, add:
```markdown
- **CER% + median latency** — per-clip Character Error Rate (jiwer) alongside
  WER/MER/WIL; per-model `median_rtfx` (+ `median_sec_per_audio_min` in the
  sidecar) as an outlier-robust counterpart to the totals-based aggregate RTFx.
  CER also surfaced in `compare`. JSON sidecar fields are additive — schema_version
  stays 1.
```

- [ ] **Step 3: SPEC.md — mark shipped**

Find "CER" / "median latency" in the roadmap and mark it shipped (mirror the JSON-sidecar/compare style). Add a decision-log entry:
```markdown
- **2026-06-06** — Shipped CER (char-level, via jiwer on the same normalized text
  as WER, added to WordMetrics so all three engines get it from one call) and a
  robust median speed pair (`median_rtfx`, `median_sec_per_audio_min`) beside the
  totals-based `aggregate_rtfx`. Sidecar fields additive within schema_version 1;
  CER added to `compare`.
```

- [ ] **Step 4: Verify**

Run: `python -m pytest -q` — full suite green.
Optionally inspect a rendered report header to confirm the new columns read well.

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md SPEC.md
git commit -m "docs: CER + median latency (README/CLAUDE/SPEC)"
```

---

## Final verification (after all tasks)

- [ ] `python -m pytest -q` — full suite green (prior + new metrics/render/sidecar/compare tests).
- [ ] `python asr_bench.py --help` — unchanged.
- [ ] Headline table shows WER%/MER%/WIL%/CER% and RTFx + RTFx (med).
- [ ] A results JSON has `clips[].cer` and `aggregates.{avg_cer,median_rtfx,median_sec_per_audio_min}`.
- [ ] `python asr_bench.py compare` over two such sidecars shows a CER% column.
