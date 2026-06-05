# asr-bench — JSON results sidecar

**Status:** Approved design (2026-06-05)
**Author:** Kevin Rank (Ryfter) + Claude Code
**Extends:** v0.3 (WhisperX + diarization + DER). Realises the SPEC.md "v0.3 — JSON
sidecar (`results/<timestamp>.json`) for cross-run aggregation" line.

## Motivation

asr-bench writes one human-readable markdown report per run. There is no
machine-readable record, so comparing runs over time (after a model swap, a
setting change, new audio) means re-reading markdown by eye. This spec adds a
structured **JSON sidecar** emitted alongside every report: the full run —
config, per-model and per-clip metrics, transcripts, speaker/DER data — captured
as data. It is the foundation for a future `asr_bench compare` subcommand.

Guiding philosophy (user): disk is cheap for text; capture the whole run
(including the expensive-to-regenerate transcripts) so later needs — anticipated
or not — already have everything.

## Scope

**In:** a JSON document mirroring the in-memory result objects (`run → models[] →
clips[]`), always written next to/with the markdown report; NaN-safe
serialization; a `--no-json` opt-out; a light `fusion` stub when `--fuse` ran.

**Out (separate later specs):**
- The `asr_bench compare` subcommand (this spec only makes the schema
  compare-ready: a top-level `schema_version` + stable keys).
- Full fusion serialization (per-window drift, rescore tables). v1 records only
  that fusion ran + its output paths.
- Schema migration tooling (we are at v1; nothing to migrate from).

## Architecture

A small new section in `asr_bench.py` (consistent with the single-file core — no
venv boundary forces a split as it did for `whisperx_runner.py`). Two functions,
each with one job:

### `build_results_document(...) -> dict`
```
build_results_document(
    results: List[ModelResult], *, corpus: Path, args, gold_label: str,
    report_path: Path, generated_at: str
) -> dict
```
Pure builder: walks `ModelResult`/`ClipResult`, pulls run metadata + `RunConfig`
fields, materializes each model's aggregates, returns a plain dict. No I/O. NaN
is sanitized here (see below). Pure → directly unit-testable.

### `write_results_json(document: dict, json_path: Path) -> Path`
Serializes with `json.dumps(document, indent=2, ensure_ascii=False,
allow_nan=False)`. `allow_nan=False` is a belt-and-suspenders guard: if any NaN
slips past the sanitizer the write fails loudly rather than emitting invalid
JSON. Returns the written path.

### Helpers
- `_json_sanitize(obj)` — recursive: `float` NaN/Inf → `None`; tuples → lists;
  recurses dicts/lists. Applied inside `build_results_document` so the returned
  dict is already clean.
- Reproducibility command string — factor the existing markdown-header
  "Command:" construction into a small shared helper so the markdown and the
  JSON `command` field have one source of truth (small, in-scope cleanup).

## Schema (schema_version 1)

```json
{
  "schema_version": 1,
  "generated_at": "2026-06-05T12:23:00-06:00",
  "report_markdown": "report/20260605-122300.md",
  "command": "python asr_bench.py --models large-v3-turbo+whisperx --diarize ...",
  "run": {
    "corpus": "D:\\Dev\\asr-bench\\test-corpus",
    "device": "cuda",
    "compute_type": "float16",
    "reference_quality": "proxy",
    "clips_count": 1,
    "total_audio_sec": 4945.2,
    "vram_tracking": true,
    "config": {
      "batch_size": 1, "beam_size": 5, "vad_filter": true,
      "diarize": true, "min_speakers": 2, "max_speakers": 2
      /* all RunConfig fields */
    }
  },
  "models": [
    {
      "model_id": "large-v3-turbo+whisperx", "display": "Whisper Large V3 Turbo + WhisperX",
      "engine": "whisperx", "fw_name": "large-v3-turbo", "params": "809M",
      "developer": "OpenAI", "languages": "99", "disk_bytes": null,
      "load_sec": 0.0, "vram_is_total": false, "notes": "...",
      "aggregates": {
        "avg_wer": 0.08, "avg_mer": 0.079, "avg_wil": 0.095,
        "total_audio_sec": 4945.2, "total_transcribe_sec": 134.4,
        "aggregate_rtfx": 36.8, "peak_vram_bytes": null
      },
      "clips": [
        {
          "audio": "GMT...mp4", "audio_sec": 4945.2, "transcribe_sec": 134.4,
          "rtfx": 36.8, "vram_peak_bytes": null,
          "wer": 0.08, "mer": 0.079, "wil": 0.095,
          "hits": 11000, "substitutions": 177, "deletions": 566, "insertions": 196,
          "cue_count": 763, "num_speakers": 2, "der": 0.138,
          "speaker_segments": [{"start": 0.0, "end": 300.0, "speaker": "SPEAKER_00"}],
          "vtt_path": "...", "reference_origin": "unknown", "reference_label": "...",
          "hypothesis": "...", "reference_normalized": "...", "hypothesis_normalized": "..."
        }
      ]
    }
  ],
  "fusion": { "ran": false }
}
```

Field decisions:
- **`schema_version: 1`** — top-level int the future `compare` dispatches on.
- **NaN → `null`** — `ClipResult` defaults `mer`/`wil`/`der` to `float("nan")`.
  Raw `json.dumps` emits literal `NaN` (invalid JSON). The sanitizer turns every
  NaN/Inf into `null`; a non-diarized clip's `der` is `null`.
- **`aggregates` materialized** — `ModelResult`'s `@property` values
  (`avg_wer`, `avg_mer`, `avg_wil`, `total_audio_sec`, `total_transcribe_sec`,
  `aggregate_rtfx`, `peak_vram_bytes`) written explicitly so a consumer never
  recomputes run totals.
- **`speaker_segments` as objects** `{start, end, speaker}` (in-memory tuples) —
  self-describing for consumers.
- **`reference_quality`** — `"gold"` or `"proxy"`, derived from the same signal
  the markdown header uses (`--gold`).
- **`fusion`** — `{"ran": false}` normally; when `--fuse` ran,
  `{"ran": true, "profiles": [...], "outputs": [<paths>]}`. No fusion internals.

## File location & CLI

- **No `--output`:** markdown stays at `report/<ts>.md`; JSON →
  **`results/<ts>.json`** with the *same* timestamp (trivially correlated).
  `results/` is gitignored like `report/`.
- **With `--output some/r.md`:** JSON is the sibling `some/r.json`.
- **`--no-json`:** skip JSON (default: write). `argparse.BooleanOptionalAction`
  (`--json`/`--no-json`), default `True`.

## Data flow

In `main()`, immediately after the markdown report is saved (~line 2430): compute
`generated_at`, derive the JSON path from the report path, call
`build_results_document(...)`, then `write_results_json(...)` unless `--no-json`.
One call site, after all results (and any fusion) exist. Print a
`Saved results JSON to <path>` line mirroring the markdown save message.

## Testing

All torch-free, fits the existing `tests/` pytest suite. Reuse the
`_whisperx_result()` / `_whisper_result()` fixtures from `tests/test_render.py`.

- **Structure:** `build_results_document` over a synthetic `results` list →
  `schema_version == 1`; `generated_at`, `report_markdown`, `command`, `run`,
  `models` present; `run.config` carries RunConfig fields; `run.reference_quality`
  reflects gold vs proxy.
- **Aggregates:** per-model `aggregates` equal the `@property` values
  (avg_wer, aggregate_rtfx, peak_vram_bytes).
- **Per-clip round-trip:** metrics, S/D/I, `speaker_segments` as `{start,end,speaker}`
  objects, der, transcript fields present.
- **NaN handling:** a clip with `mer=nan`/`der=nan` serializes to `null`; the
  written string contains no `NaN`; `json.loads` of the output succeeds.
- **`allow_nan=False` guard:** writing a document containing a stray NaN raises
  (sanitizer regression guard).
- **File placement:** no `--output` → `results/<ts>.json`; `--output a/b.md` →
  `a/b.json`; `--no-json` writes nothing.
- **Fusion stub:** a `--fuse` run records `fusion.ran == true` + output paths; a
  non-fusion run records `fusion.ran == false`.
- **main() integration:** a fake-adapter run writes both `.md` and the JSON; the
  JSON parses and has the expected `model_id`.

## Documentation

- **README.md** — a short "JSON results sidecar" note under output: always
  written (unless `--no-json`), location rules, schema_version, intended for
  cross-run aggregation.
- **CLAUDE.md** — Status + a one-line workflow; `--json`/`--no-json` flag.
- **SPEC.md** — move "JSON sidecar" from Planned-v0.3 to Shipped; note `compare`
  remains the follow-up. Decision-log entry.

## Decision log (to add)

- **2026-06-05** — JSON sidecar always emitted (no flag friction — cross-run
  aggregation only works if the data reliably exists), to `results/<ts>.json`
  (separate from `report/`), or a sibling of `--output`. `--no-json` opts out.
- **2026-06-05** — Full mirror (transcripts included), not lean metrics-only:
  transcripts are the expensive thing to regenerate; capturing them makes the
  sidecar useful for transcript diffing / re-scoring later. Disk is cheap for text.
- **2026-06-05** — NaN → `null` with an `allow_nan=False` write guard, so the
  output is always valid JSON for strict parsers and the future `compare`.
- **2026-06-05** — `compare` subcommand and full fusion serialization deferred to
  their own specs; this spec only makes the schema compare-ready (`schema_version`).
