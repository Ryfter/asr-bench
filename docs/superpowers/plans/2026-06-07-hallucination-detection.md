# Hallucination-rate Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reference-free per-clip hallucination signals (repeated-4-gram coverage + gzip compression ratio) with a per-clip flag, per-model hallucination_rate, a report section, and additive JSON sidecar fields.

**Architecture:** A pure `compute_hallucination_signals` helper computes both signals from the hypothesis; all three engines call it when building `ClipResult` (same pattern as metrics). A `ClipResult.is_hallucination_suspect` property and `ModelResult.hallucination_rate` property derive from stored signals + module-constant thresholds. Rendering + sidecar are extended. schema_version stays 1 (additive).

**Tech Stack:** Python 3.14 stdlib (`gzip`). pytest. No torch.

**Reference:** spec at `docs/superpowers/specs/2026-06-07-hallucination-detection-design.md`.

---

## File Structure

- **Modify `asr_bench.py`**: hallucination constants + `_repeat_coverage`/`_compression_ratio`/`compute_hallucination_signals` (near `compute_word_metrics`); `ClipResult` (+2 fields, +property); three `ClipResult` build sites; `ModelResult` (+`hallucination_rate`); `render_markdown` (new section after cue-density); `_clip_to_dict`/`_model_to_dict`.
- **Modify tests**: `tests/test_metrics.py` (signals + properties), `tests/test_render.py` (section), `tests/test_results_json.py` (sidecar).
- **Modify docs**: README.md, CLAUDE.md, SPEC.md.

---

## Task 1: Signals helper + constants

**Files:**
- Modify: `asr_bench.py` (add near `compute_word_metrics`, ~line 354)
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_metrics.py`:

```python
def test_repeat_coverage_high_on_loop():
    text = "thank you so much " * 6  # 24 words, heavy 4-gram repetition
    cov = asr_bench._repeat_coverage(text.strip())
    assert cov > 0.30


def test_repeat_coverage_zero_on_clean_prose():
    text = "the quick brown fox jumps over the lazy dog near twelve silent owls"
    assert asr_bench._repeat_coverage(text) == 0.0


def test_repeat_coverage_short_text_guard():
    assert asr_bench._repeat_coverage("one two three") == 0.0  # < 8 words


def test_compression_ratio_high_on_repetition():
    text = "thank you so much for watching this video. " * 20  # > 200 chars, repetitive
    assert asr_bench._compression_ratio(text) > 2.4


def test_compression_ratio_short_text_guard():
    assert asr_bench._compression_ratio("short text") == 1.0  # < 200 chars


def test_compute_hallucination_signals_returns_pair():
    loop = "thank you so much " * 20
    cov, ratio = asr_bench.compute_hallucination_signals(loop, loop.strip())
    assert cov > 0.30
    assert ratio > 2.4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py -k "repeat or compression or hallucination_signals" -v`
Expected: FAIL — `AttributeError: module 'asr_bench' has no attribute '_repeat_coverage'`.

- [ ] **Step 3: Add constants + helpers**

In `asr_bench.py`, just above `def compute_word_metrics` (~line 354), add:

```python
# ---- Hallucination signals (reference-free) ---------------------------------
HALLUCINATION_NGRAM = 4
HALLUCINATION_MIN_WORDS = 8          # below this, repeat_coverage is unreliable -> 0.0
HALLUCINATION_MIN_CHARS = 200        # below this, compression_ratio is meaningless -> 1.0
HALLUCINATION_REPEAT_COVERAGE = 0.30  # flag threshold
HALLUCINATION_COMPRESSION_RATIO = 2.4  # flag threshold (Whisper's own default)
HALLUCINATION_INSERTION_RATE = 0.5   # report annotation threshold (reference-based)


def _repeat_coverage(normalized_hypothesis: str, n: int = HALLUCINATION_NGRAM) -> float:
    """Fraction of word positions covered by an n-gram that occurs >= 2 times.
    0.0 when there are fewer than HALLUCINATION_MIN_WORDS words."""
    words = normalized_hypothesis.split()
    if len(words) < HALLUCINATION_MIN_WORDS:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    counts: Dict[tuple, int] = {}
    for g in ngrams:
        counts[g] = counts.get(g, 0) + 1
    covered = [False] * len(words)
    for i, g in enumerate(ngrams):
        if counts[g] >= 2:
            for j in range(i, i + n):
                covered[j] = True
    return sum(covered) / len(words)


def _compression_ratio(text: str) -> float:
    """len(utf8 bytes) / len(gzip(bytes)). ~1.5-2.2 normal prose, >2.4 repetitive.
    Returns 1.0 for text shorter than HALLUCINATION_MIN_CHARS (gzip overhead makes
    tiny inputs meaningless)."""
    raw = text.encode("utf-8")
    if len(raw) < HALLUCINATION_MIN_CHARS:
        return 1.0
    import gzip
    compressed = gzip.compress(raw)
    return len(raw) / len(compressed) if compressed else 1.0


def compute_hallucination_signals(hypothesis: str, hypothesis_normalized: str) -> Tuple[float, float]:
    """(repeat_coverage, compression_ratio) for a clip. Reference-free."""
    return _repeat_coverage(hypothesis_normalized), _compression_ratio(hypothesis)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -k "repeat or compression or hallucination_signals" -v` then `python -m pytest -q`
Expected: PASS, no regressions.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_metrics.py
git commit -m "feat(hallucination): reference-free repeat_coverage + compression_ratio signals"
```

---

## Task 2: `ClipResult` fields + property + three build sites

**Files:**
- Modify: `asr_bench.py` (`ClipResult` ~line 614; build sites — find by `cer=metrics.cer`)
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_metrics.py`:

```python
def test_clipresult_hallucination_fields_default():
    c = asr_bench.ClipResult(
        audio="x.mp4", audio_sec=10.0, transcribe_sec=1.0, rtfx=10.0,
        vram_peak_bytes=None, hypothesis="h", reference_normalized="h",
        hypothesis_normalized="h", wer=0.1)
    assert c.repeat_coverage == 0.0
    assert c.compression_ratio == 1.0
    assert c.is_hallucination_suspect is False


def test_is_hallucination_suspect_on_repeat_coverage():
    c = asr_bench.ClipResult(
        audio="x.mp4", audio_sec=10.0, transcribe_sec=1.0, rtfx=10.0,
        vram_peak_bytes=None, hypothesis="h", reference_normalized="h",
        hypothesis_normalized="h", wer=0.1, repeat_coverage=0.5,
        compression_ratio=1.5)
    assert c.is_hallucination_suspect is True


def test_is_hallucination_suspect_on_compression():
    c = asr_bench.ClipResult(
        audio="x.mp4", audio_sec=10.0, transcribe_sec=1.0, rtfx=10.0,
        vram_peak_bytes=None, hypothesis="h", reference_normalized="h",
        hypothesis_normalized="h", wer=0.1, repeat_coverage=0.0,
        compression_ratio=3.0)
    assert c.is_hallucination_suspect is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py -k "clipresult_hallucination or is_hallucination" -v`
Expected: FAIL — unexpected keyword `repeat_coverage`.

- [ ] **Step 3: Add the fields + property**

In `ClipResult`, the fields currently include (after the CER task):
```python
    wer: float
    mer: float = float("nan")
    wil: float = float("nan")
    cer: float = float("nan")
    hits: int = 0
```
Insert the two new fields after `cer`:
```python
    cer: float = float("nan")
    repeat_coverage: float = 0.0
    compression_ratio: float = 1.0
    hits: int = 0
```

Add the property to `ClipResult` (after the last field / anywhere in the class body, e.g. at the end of the dataclass):
```python
    @property
    def is_hallucination_suspect(self) -> bool:
        return (self.repeat_coverage > HALLUCINATION_REPEAT_COVERAGE
                or self.compression_ratio > HALLUCINATION_COMPRESSION_RATIO)
```

- [ ] **Step 4: Thread signals into the three build sites**

Find the three `ClipResult(` construction sites (grep for `cer=metrics.cer`). At EACH site, immediately before the `ClipResult(...)` (or `result.clips.append(ClipResult(`) construction, compute the signals from the in-scope `hypothesis` and `hyp_norm` locals:
```python
        rep_cov, comp_ratio = compute_hallucination_signals(hypothesis, hyp_norm)
```
and add to that ClipResult's kwargs (next to `cer=...`):
```python
                    repeat_coverage=rep_cov, compression_ratio=comp_ratio,
```

Concretely:

Site A (faster-whisper, the one with one-kwarg-per-line, `cer=metrics.cer` on its own line): add `repeat_coverage=rep_cov,` and `compression_ratio=comp_ratio,` lines after the `cer=metrics.cer,` line; add the `rep_cov, comp_ratio = ...` computation just before the `ClipResult(` call (the `hypothesis` and `hyp_norm` locals are already defined above it).

Site B (NIM, packed kwargs with `cer=metrics.cer,`): add `repeat_coverage=rep_cov, compression_ratio=comp_ratio,` to the kwargs; add the computation line before the construction.

Site C (WhisperX, `result_model.clips.append(ClipResult(` with `cer=metrics.cer,`): add `repeat_coverage=rep_cov, compression_ratio=comp_ratio,` to the kwargs; add the computation line before the `.append(` call.

VERIFY all three sites compute and pass the signals (grep `repeat_coverage=rep_cov` → 3 hits).

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -v` then `python -m pytest -q`
Expected: PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add asr_bench.py tests/test_metrics.py
git commit -m "feat(hallucination): ClipResult signals + suspect flag, wired through all engines"
```

---

## Task 3: `ModelResult.hallucination_rate`

**Files:**
- Modify: `asr_bench.py` (`ModelResult`, near the other properties)
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_metrics.py` (reuse the `_model` helper added in the CER tasks; it builds a ModelResult from a clips list):

```python
def test_hallucination_rate_half():
    clean = asr_bench.ClipResult(
        audio="a.mp4", audio_sec=10.0, transcribe_sec=1.0, rtfx=10.0,
        vram_peak_bytes=None, hypothesis="h", reference_normalized="h",
        hypothesis_normalized="h", wer=0.1)  # defaults -> not suspect
    suspect = asr_bench.ClipResult(
        audio="b.mp4", audio_sec=10.0, transcribe_sec=1.0, rtfx=10.0,
        vram_peak_bytes=None, hypothesis="h", reference_normalized="h",
        hypothesis_normalized="h", wer=0.1, repeat_coverage=0.6)
    m = _model([clean, suspect])
    assert m.hallucination_rate == 0.5


def test_hallucination_rate_empty_model():
    assert _model([]).hallucination_rate == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py -k "hallucination_rate" -v`
Expected: FAIL — no attribute `hallucination_rate`.

- [ ] **Step 3: Add the property**

In `ModelResult`, after the `peak_vram_bytes` property (or alongside the other aggregate properties), add:
```python
    @property
    def hallucination_rate(self) -> float:
        """Fraction of this model's clips flagged as hallucination-suspect."""
        if not self.clips:
            return 0.0
        return sum(1 for c in self.clips if c.is_hallucination_suspect) / len(self.clips)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -v` then `python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_metrics.py
git commit -m "feat(hallucination): ModelResult.hallucination_rate"
```

---

## Task 4: Report section

**Files:**
- Modify: `asr_bench.py` `render_markdown` (insert after the cue-density anomalies section, ~line 2241, before the "Generated VTT outputs" section)
- Test: `tests/test_render.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_render.py`:

```python
def test_hallucination_section_appears_when_flagged():
    r = _whisper_result()
    r.clips[0].repeat_coverage = 0.6   # trips the flag
    md = asr_bench.render_markdown([r], Path("."), _args(), "proxy")
    assert "Hallucination signals" in md
    assert r.clips[0].audio in md
    assert "1/1 clip" in md or "1/1 clips" in md  # per-model summary


def test_no_hallucination_section_when_clean():
    r = _whisper_result()  # defaults -> not suspect
    md = asr_bench.render_markdown([r], Path("."), _args(), "proxy")
    assert "Hallucination signals" not in md


def test_hallucination_insertion_burst_note():
    r = _whisper_result()
    c = r.clips[0]
    c.compression_ratio = 3.0           # trips the flag (compression)
    c.hits, c.substitutions, c.deletions, c.insertions = 10, 0, 0, 20  # insertion rate 2.0
    md = asr_bench.render_markdown([r], Path("."), _args(), "proxy")
    assert "Hallucination signals" in md
    assert "insertion burst" in md.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_render.py -k "hallucination" -v`
Expected: FAIL — section not present.

- [ ] **Step 3: Add the section**

In `render_markdown`, after the cue-density anomalies block (which ends with its `lines.append("")` after the anomalies table) and before the "Generated VTT outputs" block, insert:

```python
    # ---- Hallucination signals (reference-free, per clip) ----
    flagged: List[Tuple[str, "ClipResult"]] = [
        (r.display, c) for r in results for c in r.clips if c.is_hallucination_suspect
    ]
    if flagged:
        lines.append("## ⚠️ Hallucination signals")
        lines.append("")
        lines.append(
            "Reference-free heuristics that flag a clip for **manual inspection** "
            "(not a definitive error): **repeat coverage** = fraction of the "
            "transcript made of repeated 4-grams; **compression** = gzip ratio of "
            "the text (Whisper's own internal hallucination signal — normal prose "
            "is ~1.5–2.2, looped output is higher). A clip is flagged when repeat "
            f"coverage > {HALLUCINATION_REPEAT_COVERAGE:.0%} or compression > "
            f"{HALLUCINATION_COMPRESSION_RATIO:.1f}."
        )
        lines.append("")
        # per-model summary
        for r in results:
            n_flag = sum(1 for c in r.clips if c.is_hallucination_suspect)
            if n_flag:
                lines.append(f"- **{r.display}:** {n_flag}/{len(r.clips)} clips flagged")
        lines.append("")
        lines.append("| Model | Clip | Repeat cov % | Compression | Insertion rate | Note |")
        lines.append("|---|---|---|---|---|---|")
        for model_display, c in flagged:
            ref_words = c.hits + c.substitutions + c.deletions
            ins_rate = (c.insertions / ref_words) if ref_words > 0 else None
            ins_cell = f"{ins_rate * 100:.0f}%" if ins_rate is not None else "—"
            reasons: List[str] = []
            if c.repeat_coverage > HALLUCINATION_REPEAT_COVERAGE:
                reasons.append("repetition")
            if c.compression_ratio > HALLUCINATION_COMPRESSION_RATIO:
                reasons.append("high compression")
            if ins_rate is not None and ins_rate > HALLUCINATION_INSERTION_RATE:
                reasons.append("insertion burst")
            note = ", ".join(reasons)
            lines.append(
                f"| {model_display} | {c.audio} | {c.repeat_coverage * 100:.1f} | "
                f"{c.compression_ratio:.2f} | {ins_cell} | {note} |"
            )
        lines.append("")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_render.py -v` then `python -m pytest -q`
Expected: PASS, no regressions.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_render.py
git commit -m "feat(report): hallucination signals section"
```

---

## Task 5: JSON sidecar

**Files:**
- Modify: `asr_bench.py` (`_clip_to_dict`, `_model_to_dict`)
- Test: `tests/test_results_json.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_results_json.py`:

```python
def test_sidecar_clip_has_hallucination_fields():
    import asr_bench
    clip = asr_bench.ClipResult(
        audio="c.mp4", audio_sec=600.0, transcribe_sec=10.0, rtfx=60.0,
        vram_peak_bytes=None, hypothesis="h", reference_normalized="h",
        hypothesis_normalized="h", wer=0.1, repeat_coverage=0.6,
        compression_ratio=3.0)
    d = asr_bench._clip_to_dict(clip)
    assert d["repeat_coverage"] == 0.6
    assert d["compression_ratio"] == 3.0
    assert d["hallucination_suspect"] is True


def test_sidecar_model_has_hallucination_rate():
    import asr_bench
    clip = asr_bench.ClipResult(
        audio="c.mp4", audio_sec=600.0, transcribe_sec=10.0, rtfx=60.0,
        vram_peak_bytes=None, hypothesis="h", reference_normalized="h",
        hypothesis_normalized="h", wer=0.1, repeat_coverage=0.6)
    m = asr_bench.ModelResult(
        model_id="m", display="M", fw_name="m", params="1", developer="x",
        languages="en", notes="", disk_bytes=None, load_sec=0.0, clips=[clip])
    agg = asr_bench._model_to_dict(m)["aggregates"]
    assert agg["hallucination_rate"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_results_json.py -k "hallucination" -v`
Expected: FAIL — KeyError.

- [ ] **Step 3: Add the fields**

`_clip_to_dict` — after the `"cer": c.cer,` entry add:
```python
        "repeat_coverage": c.repeat_coverage,
        "compression_ratio": c.compression_ratio,
        "hallucination_suspect": c.is_hallucination_suspect,
```

`_model_to_dict` aggregates — after `"avg_cer": m.avg_cer,` (or anywhere in the aggregates dict) add:
```python
            "hallucination_rate": m.hallucination_rate,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_results_json.py -v` then `python -m pytest -q`
Expected: PASS. schema_version stays 1 — confirm unchanged.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_results_json.py
git commit -m "feat(sidecar): hallucination signals + rate (schema_version stays 1)"
```

---

## Task 6: Documentation

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `SPEC.md`

- [ ] **Step 1: README** — add a short note where report sections / quality features are described:
```markdown
- **Hallucination signals** — a report section flags clips with looping or
  fabricated output using reference-free heuristics: repeated-4-gram coverage and
  gzip compression ratio (Whisper's own internal hallucination signal). Per-model
  `hallucination_rate` is recorded in the JSON sidecar.
```

- [ ] **Step 2: CLAUDE.md** — "What's new" bullet:
```markdown
- **Hallucination-rate detection** — reference-free per-clip signals
  (`repeat_coverage` = repeated-4-gram fraction; `compression_ratio` = gzip ratio,
  Whisper's own signal) flag looping/fabricated output in a dedicated report
  section (works without a reference and in single-model runs, unlike cue-density);
  per-model `hallucination_rate`; additive sidecar fields (`schema_version` stays 1).
```

- [ ] **Step 3: SPEC.md** — move "hallucination-rate detection" from planned to shipped (mirror prior shipped lines). Add a decision-log entry:
```markdown
- **2026-06-07** — Shipped hallucination detection: reference-free repeat-coverage
  (repeated 4-grams) + gzip compression ratio (Whisper's own internal signal),
  flag = coverage>0.30 OR compression>2.4, per-model hallucination_rate, dedicated
  report section + additive sidecar fields (schema_version stays 1). Detection
  only; compare integration deferred.
```

- [ ] **Step 4: Verify** — `python -m pytest -q` green.

- [ ] **Step 5: Commit**
```bash
git add README.md CLAUDE.md SPEC.md
git commit -m "docs: hallucination-rate detection (README/CLAUDE/SPEC)"
```

---

## Final verification (after all tasks)

- [ ] `python -m pytest -q` — full suite green.
- [ ] A result with a looping clip renders the "⚠️ Hallucination signals" section; a clean run does not.
- [ ] A results JSON has `clips[].{repeat_coverage,compression_ratio,hallucination_suspect}` and `aggregates.hallucination_rate`.
- [ ] schema_version still 1.
