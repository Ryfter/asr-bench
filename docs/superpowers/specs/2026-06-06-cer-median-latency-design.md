# asr-bench — CER + median latency metrics

**Status:** Approved design (2026-06-06)
**Author:** Kevin Rank (Ryfter) + Claude Code
**Extends:** the core metrics path (WER/MER/WIL + RTFx) and the JSON results sidecar.

## Motivation

asr-bench reports WER/MER/WIL (word/information accuracy) and RTFx (aggregate
speed). Two gaps:

1. **No character-level error rate.** WER is unforgiving of small morphological
   slips (plurals, tense) that barely change meaning; **CER** (Character Error
   Rate) is the standard finer-grained companion and is expected in ASR
   benchmarks. It is cheap to add (same jiwer alignment infrastructure).
2. **Aggregate RTFx hides outliers.** `aggregate_rtfx` is `total_audio /
   total_transcribe`, so a single pathological clip (the observed Large-V3
   1-second-cue decoder lockup that turned 12% WER into 14.2% and tanked its
   speed) drags the whole number down. A **median per-clip** speed metric is
   robust to one bad clip and tells you the typical experience.

## Scope

**In:** per-clip CER via jiwer; per-model `avg_cer`, `median_rtfx`, and
`median_sec_per_audio_min`; CER% + RTFx(med) columns in the report; CER% in the
per-clip table; the three new fields in the JSON sidecar; CER in the `compare`
subcommand's metric set.

**Out:** CER-based fusion/rescoring; per-clip median (median is inherently a
multi-clip aggregate); changing schema_version (additive only — see below);
weighting CER by clip length (simple mean of per-clip CER, consistent with how
`avg_wer` already works).

## Design

### CER computation (`compute_word_metrics` + `WordMetrics`)

`WordMetrics` (the dataclass returned by one jiwer alignment) gains a `cer: float`
field. Its docstring is updated from "word-level scores" to "alignment scores
(word-level + CER)". `compute_word_metrics(reference, hypothesis)` computes CER on
the **same normalized strings** it already uses for WER, via `jiwer.cer`:

```python
cer=float("nan") if not reference.strip() else float(jiwer.cer(reference, hypothesis))
```

The existing empty-reference guard already returns an all-NaN `WordMetrics`; CER
joins it as NaN there. On a jiwer exception the existing `except` path returns
NaN metrics — CER is NaN too. CER, like WER, can exceed 1.0 and is rendered ×100.

**Why extend `WordMetrics` rather than a separate function:** all three engines
(faster-whisper, NIM, WhisperX) construct `ClipResult` from one
`compute_word_metrics` call. Adding `cer` to that dataclass means each
construction site adds a single `cer=metrics.cer` and every engine gets CER with
no per-engine logic. (whisperx_runner.py is untouched — it returns transcripts;
asr_bench.py computes all metrics.)

### `ClipResult`

New field `cer: float = float("nan")` (placed next to `wil`).

Each of the three `ClipResult(...)` construction sites adds `cer=metrics.cer`.

### `ModelResult` properties

```python
@property
def avg_cer(self) -> float:
    if not self.clips:
        return 0.0
    return sum(c.cer for c in self.clips) / len(self.clips)

@property
def median_rtfx(self) -> float:
    if not self.clips:
        return 0.0
    return statistics.median(c.rtfx for c in self.clips)

@property
def median_sec_per_audio_min(self) -> float:
    vals = [c.transcribe_sec * 60.0 / c.audio_sec
            for c in self.clips if c.audio_sec > 0]
    return statistics.median(vals) if vals else 0.0
```

`statistics` is added to the stdlib imports. `median_rtfx` is the robust sibling
of `aggregate_rtfx`; `median_sec_per_audio_min` is the same robustness expressed
as compute-seconds per minute of audio (lower = faster), stored for the sidecar.

### Report rendering (`render_markdown`)

- **Headline table:** insert **CER%** immediately after **WIL%**, and **RTFx (med)**
  immediately after the aggregate RTFx column. CER% renders `avg_cer * 100` with
  one decimal (NaN → the existing dash convention). RTFx(med) renders
  `median_rtfx` as `NN.NNx`.
- **Per-clip table:** insert a **CER%** column alongside the existing WER/S/D/I
  columns, rendered `cer * 100` one decimal (NaN → dash).
- The headline gains two columns; the per-clip table gains one. Acceptable width.

### JSON sidecar

- `_clip_to_dict`: add `"cer": c.cer`.
- `_model_to_dict` `aggregates`: add `"avg_cer": m.avg_cer`, `"median_rtfx":
  m.median_rtfx`, `"median_sec_per_audio_min": m.median_sec_per_audio_min`.
- **`schema_version` stays 1.** These are purely additive keys. The `compare`
  reader (just shipped) uses `.get()` throughout and tolerates missing/extra
  keys, so old sidecars and new sidecars both parse. **Decision: bump
  `schema_version` only on a breaking change (removed/renamed/retyped field), not
  on additive fields.** NaN values are still sanitized to null by the existing
  `_json_sanitize` + `allow_nan=False` write guard.

### `compare` subcommand (`asr_compare.py`)

- `METRIC_META`: add `"cer": {"label": "CER%", "pct": True, "lower_better": True}`,
  placed after `wil` so the metric order is wer, mer, wil, cer, rtfx (+der).
- `_AGG_KEYS`: add `"cer": "avg_cer"`.

That is all `compare` needs — its builder iterates `METRIC_META`/`metrics` and
reads `aggregates[_AGG_KEYS[k]]`, so CER flows into both delta and matrix views
automatically. Median RTFx stays report-only (a within-report robustness aid;
cross-run comparison keeps using aggregate RTFx, already present). An older
sidecar lacking `avg_cer` yields `None` → renders `—`, which is correct.

## Testing

All torch-free, extending `tests/` (metrics tests near the existing WER tests;
render tests in `tests/test_render.py`; sidecar in `tests/test_results_json.py`;
compare in `tests/test_compare.py`).

- **CER value:** `compute_word_metrics("the cat", "the bat").cer` ≈ 1/7 (one char
  substitution over 7 reference chars); empty reference → `cer` is NaN.
- **avg_cer:** two clips with cer 0.10 and 0.20 → `avg_cer == 0.15`.
- **median_rtfx robustness:** clips with rtfx [60.0, 62.0, 3.0] → `median_rtfx ==
  60.0` while `aggregate_rtfx` (driven by totals) is much lower — the test asserts
  median > aggregate, proving outlier resistance.
- **median_sec_per_audio_min:** clip audio_sec=600, transcribe_sec=10 → 1.0
  s/min; a clip with audio_sec=0 is skipped (no ZeroDivision); empty → 0.0.
- **headline render:** report contains `CER%` and `RTFx (med)` headers and a
  numeric median value.
- **per-clip render:** per-clip table contains a `CER%` column.
- **NaN render:** a clip with `cer=nan` renders as a dash, not "nan".
- **sidecar:** `build_results_document` output has `clips[].cer` and
  `aggregates.{avg_cer,median_rtfx,median_sec_per_audio_min}`; a `cer=nan` clip
  serializes to `null`; `schema_version` is still 1.
- **compare surfaces CER:** a delta over two sidecars with `avg_cer` shows a
  `CER%` column with a delta and ✓/✗; an older sidecar missing `avg_cer` shows
  `—` for CER (no crash).

## Documentation

- **README.md** — note CER% and RTFx(med) in the metrics/output description.
- **CLAUDE.md** — Status bullet (CER + median latency) + decision-log entry.
- **SPEC.md** — move "CER + median latency" from planned to shipped.

## Decision log (to add)

- **2026-06-06** — Added **CER** (char-level, via jiwer on the same normalized
  text as WER) and a **robust median speed** pair (`median_rtfx`,
  `median_sec_per_audio_min`) alongside the totals-based `aggregate_rtfx`.
  Rationale: aggregate RTFx hides single-clip outliers (the Large-V3 lockup);
  median reflects the typical clip. CER added to `WordMetrics` so all three
  engines get it from one `compute_word_metrics` call.
- **2026-06-06** — JSON sidecar fields are **additive within schema_version 1**;
  bump the version only on a breaking change. The `compare` reader tolerates
  missing/extra keys (`.get`), so additive growth needs no version bump and old
  sidecars stay readable. CER added to `compare`'s metric set; median RTFx stays
  report-only.
