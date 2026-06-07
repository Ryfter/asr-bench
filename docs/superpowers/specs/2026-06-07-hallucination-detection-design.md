# asr-bench — Hallucination-rate detection

**Status:** Approved design (2026-06-07)
**Author:** Kevin Rank (Ryfter) + Claude Code
**Extends:** the core metrics path and the existing cue-density anomaly detector.

## Motivation

Whisper-family models hallucinate in characteristic ways: **repetition loops**
("thank you. thank you. thank you."), trailing boilerplate ("thanks for
watching"), and **insertion bursts** of content not present in the audio. The
existing **cue-density anomaly** detector catches the *structural* lockup
(cross-model cue-count outliers) but not *content* repetition within a single
model's output, and it is silent when only one model is run. This feature adds a
**per-clip, reference-free hallucination signal** so a suspect transcript is
flagged even without a gold reference and even in a single-model run.

## Scope

**In:** two per-clip reference-free signals (`repeat_coverage`,
`compression_ratio`), a derived per-clip flag, a per-model `hallucination_rate`,
a report section, and additive JSON sidecar fields. A reference-based
**insertion-rate** annotation in the report (derived from existing S/D/I — no new
storage) for the "insertion burst" case.

**Out:** changing the flag to depend on a reference (it stays reference-free);
`compare` integration (hallucination is a within-report diagnostic, like
cue-density, which `compare` also does not surface — possible future follow-up);
a headline-table column (the dedicated section carries it); auto-suppression or
"fixing" hallucinations (detection only).

## Design

### Signals (`compute_hallucination_signals`)

A new pure helper in `asr_bench.py`:

```python
HALLUCINATION_NGRAM = 4
HALLUCINATION_MIN_WORDS = 8          # below this, repeat_coverage is unreliable -> 0.0
HALLUCINATION_MIN_CHARS = 200        # below this, compression_ratio is meaningless -> 1.0
HALLUCINATION_REPEAT_COVERAGE = 0.30 # flag threshold
HALLUCINATION_COMPRESSION_RATIO = 2.4  # flag threshold (Whisper's own default)
HALLUCINATION_INSERTION_RATE = 0.5   # report annotation threshold (ref-based)


def _repeat_coverage(normalized_hypothesis: str, n: int = HALLUCINATION_NGRAM) -> float:
    """Fraction of word positions covered by an n-gram that occurs >= 2 times.
    0.0 when there are fewer than HALLUCINATION_MIN_WORDS words."""
    words = normalized_hypothesis.split()
    if len(words) < HALLUCINATION_MIN_WORDS:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
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

`gzip` is stdlib (imported lazily inside `_compression_ratio`, or at module top —
implementer's choice; lazy keeps the top imports lean).

### `ClipResult`

Two new fields (after `cer`):
```python
    repeat_coverage: float = 0.0
    compression_ratio: float = 1.0
```

A derived property (thresholds in one place):
```python
    @property
    def is_hallucination_suspect(self) -> bool:
        return (self.repeat_coverage > HALLUCINATION_REPEAT_COVERAGE
                or self.compression_ratio > HALLUCINATION_COMPRESSION_RATIO)
```

Each of the three `ClipResult` build sites computes and passes the signals:
```python
        rep_cov, comp_ratio = compute_hallucination_signals(hypothesis, hyp_norm)
        # ... ClipResult(..., repeat_coverage=rep_cov, compression_ratio=comp_ratio)
```
(Variable names for the hypothesis/normalized differ slightly per site — the
implementer matches the locals already in scope: `hypothesis`/`hyp_norm`.)

### `ModelResult`

```python
    @property
    def hallucination_rate(self) -> float:
        """Fraction of this model's clips flagged as hallucination-suspect."""
        if not self.clips:
            return 0.0
        return sum(1 for c in self.clips if c.is_hallucination_suspect) / len(self.clips)
```

### Report section

A new "## ⚠️ Hallucination signals" section in `render_markdown`, placed **after**
the cue-density anomalies section, rendered **only if at least one clip across all
models is flagged**. For each flagged (model, clip):

- columns: `Model | Clip | Repeat cov % | Compression | Insertion rate | Note`
- `Repeat cov %` = `repeat_coverage * 100` one decimal.
- `Compression` = `compression_ratio` two decimals.
- `Insertion rate` = `insertions / ref_words` (ref_words = hits+sub+del) as a
  percentage when ref_words > 0, else `—`. (Derived; no new field.)
- `Note` = a short reason: "repetition" if coverage over threshold, "high
  compression" if ratio over threshold, "+ insertion burst" appended when
  insertion_rate > HALLUCINATION_INSERTION_RATE.

Above the table, an explanatory paragraph: these are reference-free heuristics
(repeated n-grams + gzip compressibility, the latter being Whisper's own internal
hallucination signal); a flag means *inspect the transcript*, not a definitive
error. A per-model summary line lists "N/M clips flagged" for each model with ≥1
flag.

### JSON sidecar (additive, schema_version stays 1)

- `_clip_to_dict`: add `"repeat_coverage"`, `"compression_ratio"`,
  `"hallucination_suspect": c.is_hallucination_suspect`.
- `_model_to_dict` `aggregates`: add `"hallucination_rate": m.hallucination_rate`.

Additive only — `schema_version` stays 1; `compare`'s `.get()`-based reader is
unaffected (it simply won't surface these unless a future change adds them to its
metric set).

## Testing

All torch-free.

- **`_repeat_coverage`:** a looped string ("a b c d " repeated) → high coverage
  (> 0.30); clean varied prose (≥ 8 distinct words, no repeats) → 0.0; a < 8-word
  string → 0.0 (guard).
- **`_compression_ratio`:** a long repetitive string (one phrase × 50) → ratio
  > 2.4; long varied prose → between ~1.5 and ~2.4; a short string (< 200 chars)
  → exactly 1.0 (guard).
- **`compute_hallucination_signals`:** returns the pair; a clean clip → low/safe;
  a looped clip → high.
- **`is_hallucination_suspect`:** trips when repeat_coverage > 0.30 (compression
  normal); trips when compression_ratio > 2.4 (coverage normal); False when both
  under threshold.
- **`hallucination_rate`:** 1 of 2 clips flagged → 0.5; empty model → 0.0.
- **report:** a result with a flagged clip renders the "Hallucination signals"
  section, the clip name, and the per-model "N/M clips flagged" line; a result
  with no flagged clip does NOT render the section; the insertion-burst note
  appears when a flagged clip has a high insertion rate (build a clip with
  high insertions vs ref).
- **sidecar:** `_clip_to_dict` carries `repeat_coverage`, `compression_ratio`,
  `hallucination_suspect`; `_model_to_dict` aggregates carries
  `hallucination_rate`; a flagged clip serializes `hallucination_suspect: true`.

## Documentation

- **README.md** — note the hallucination-signals section + the two heuristics.
- **CLAUDE.md** — "What's new" bullet + decision-log entry.
- **SPEC.md** — move "hallucination-rate detection" from planned to shipped.

## Decision log (to add)

- **2026-06-07** — Hallucination detection is **reference-free** (repeated-4-gram
  coverage + gzip compression ratio, the latter Whisper's own internal signal),
  so it flags suspect clips even without a gold reference and in single-model
  runs — unlike the cross-model cue-density detector. Flag = coverage > 0.30 OR
  compression > 2.4. A reference-based insertion-rate annotation is shown when a
  reference exists, derived from existing S/D/I (no new storage).
- **2026-06-07** — Detection only, no auto-fix; surfaced in a dedicated report
  section (not a headline column) + additive sidecar fields (schema_version stays
  1). `compare` integration deferred (within-report diagnostic, like cue-density).
