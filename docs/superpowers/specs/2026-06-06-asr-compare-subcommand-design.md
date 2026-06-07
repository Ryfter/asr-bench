# asr-bench — `compare` subcommand

**Status:** Approved design (2026-06-06)
**Author:** Kevin Rank (Ryfter) + Claude Code
**Extends:** the v0.3 JSON results sidecar (`results/<timestamp>.json`, `schema_version 1`).
Realises the SPEC.md "future `asr_bench compare` subcommand" follow-up.

## Motivation

Every run now emits a machine-readable `results/<ts>.json` mirroring the full run
(config, per-model/per-clip metrics, transcripts, speaker/DER). There is still no
way to *compare* runs: after a model swap, a setting change, or new audio, you
re-read two markdown reports side by side by eye. `compare` reads N sidecars and
emits a delta/comparison report — the payoff the sidecar foundation was built for.

## Scope

**In:** a `compare` subcommand that loads 2+ `schema_version 1` JSON sidecars,
joins per-model headline metrics on `model_id`, and renders either a **delta view**
(2 files: baseline → candidate, signed Δ) or an **N-column matrix** (3+ files),
auto-selected by file count with `--delta`/`--matrix` overrides. Mismatch warnings
for differing corpus/config. Optional `--per-clip` detail. Markdown to stdout or
`--output`.

**Out (not now):**
- Plotting/charts. Markdown tables only.
- Comparing across schema versions (we are at v1; mismatches warn-and-skip).
- Statistical significance testing of deltas (just raw numbers + direction).
- Transcript diffing between runs (the data is in the sidecar; a separate later
  feature can diff `hypothesis` fields).

## Architecture

A new standalone module **`asr_compare.py`** — pure, torch-free, importing nothing
from `asr_bench.py` (it reads plain JSON dicts, not in-memory result objects). This
keeps it independently testable and keeps `asr_bench.py` from growing a second
large entry path. `asr_bench.py` gains only a tiny **pre-dispatch**.

### Pre-dispatch (in `asr_bench.py`)

At the top of `main()`, before building the bench `ArgumentParser`:

```python
def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "compare":
        from asr_compare import compare_main
        return compare_main(argv[1:])
    # ... existing bench parser unchanged ...
```

Rationale: every existing invocation (`python asr_bench.py --models …`) is
untouched — `compare` is an opt-in first-positional keyword, not an argparse
subparser that would reshape the bench CLI. The import is lazy so a plain bench
run never imports the compare module.

### `asr_compare.py` units

Each has one job and a clear interface:

#### `load_results_json(path: Path) -> dict | None`
Reads and `json.loads` the file; returns the dict if `schema_version == 1`.
On unreadable file, invalid JSON, or a different `schema_version`: prints a
warning to stderr and returns `None` (caller filters out `None`). A run is
identified for display by its filename stem (e.g. `20260606-122300`).

#### `compare_runs(docs: list[dict], *, mode: str) -> dict`
Pure builder. `docs` is the list of loaded sidecars (already filtered of `None`),
in the order the user supplied them. Returns a plain report dict:

```python
{
  "mode": "delta" | "matrix",
  "runs": [{"label": "20260606-122300", "corpus": "...", "config": {...}}, ...],
  "metrics": ["wer", "mer", "wil", "rtfx", "der"],   # der only if any run has it
  "models": [                                         # union of model_ids, input order
    {
      "model_id": "large-v3-turbo",
      "display": "Whisper Large V3 Turbo",
      "present_in": [0, 1],                            # indices into runs
      "values": {                                      # metric -> [per-run value or None]
        "wer": [0.089, 0.081], "rtfx": [64.8, 71.2], ...
      },
      # delta mode only:
      "deltas": {"wer": -0.008, "rtfx": +6.4, ...},    # candidate - baseline
      "status": "both" | "added" | "removed",          # vs baseline (delta mode)
    },
  ],
  "warnings": ["corpus differs: run[1] used test-corpus-v2", "beam_size differs: 5 vs 1", ...],
}
```

- **Join key:** `model_id`. Model union preserves first-seen order.
- **Per-model values** pulled from each run's `models[].aggregates`
  (`avg_wer`→`wer`, `avg_mer`→`mer`, `avg_wil`→`wil`, `aggregate_rtfx`→`rtfx`).
  A model absent from a run gets `None` for every metric in that run's slot.
- **DER:** per-model DER is a per-clip value, not in `aggregates`. Compute a
  per-model average DER from `models[].clips[].der` (ignoring null), and a
  representative `num_speakers` (the modal/last non-null). Include the `der`
  metric only if at least one run has at least one non-null clip `der`.
- **Deltas (delta mode):** `candidate - baseline` where `runs[0]` is baseline and
  `runs[1]` is candidate. Only computed when both values are present.
- **Mismatch warnings:** compare every run against `runs[0]`; warn on differing
  `run.corpus` and on differing `config` keys among `device`, `compute_type`,
  `beam_size`, `vad_filter`, `batch_size`.

#### Metric direction-of-good
A module constant:

```python
METRIC_META = {
    "wer": {"label": "WER%", "pct": True, "lower_better": True},
    "mer": {"label": "MER%", "pct": True, "lower_better": True},
    "wil": {"label": "WIL%", "pct": True, "lower_better": True},
    "rtfx": {"label": "RTFx", "pct": False, "lower_better": False},
    "der": {"label": "DER%", "pct": True, "lower_better": True},
}
```
`pct` metrics are stored as fractions (0.089) and rendered ×100 with one decimal
(`8.9`). A delta is an **improvement** when `(delta < 0) == lower_better` → mark ✓;
otherwise ✗; a zero delta gets no mark.

#### `render_comparison_markdown(report: dict) -> str`
- Header: `# ASR Run Comparison`, generated-at line, the list of runs with their
  stem labels + corpus.
- **Warnings block** (if any): a `> ⚠️ …` blockquote per warning, before the table.
- **Delta mode table:** columns `Model | <Metric> base → cand (Δ) | …` — one
  composite cell per metric showing `base → cand` and a second line `Δ ±x ✓/✗`.
  Rows flagged `added`/`removed` render the present side and `—` for the missing
  side (no Δ). Pct metrics ×100.
- **Matrix mode table:** columns `Model | Metric | run0 | run1 | … | runN`; one row
  per (model, metric). Missing → `—`.
- `--per-clip` appends a per-model section joining `clips[].audio` basenames across
  runs (same delta/matrix logic on `wer`/`der` per clip).

#### `compare_main(argv: list[str]) -> int`
`ArgumentParser(prog="asr_bench.py compare")`:
- positional `files` (`nargs="*"`) — JSON paths; a directory arg expands to its
  `*.json` (sorted).
- `--last N` — instead of/in addition to positionals, take the N most-recent
  `results/*.json` (by filename, which is timestamp-sorted).
- `--delta` / `--matrix` — force the layout (default: auto by count, 2→delta else
  matrix). Forcing `--delta` with ≠2 files is an error.
- `--per-clip` — include per-clip detail.
- `--output PATH` — write markdown to PATH instead of stdout.

Flow: resolve file list (`--last` + positionals + directory expansion) →
`load_results_json` each (filter `None`) → require ≥2 survivors (else error) →
pick mode → `compare_runs` → `render_comparison_markdown` → print or write.
Returns 0 on success, non-zero on usage error / too few valid files.

## File location & CLI examples

```powershell
# Auto: 2 files -> delta view
python asr_bench.py compare results/20260605-190913.json results/20260606-101500.json

# 3+ files -> matrix
python asr_bench.py compare results/a.json results/b.json results/c.json

# The two most recent runs, forced delta, with per-clip detail
python asr_bench.py compare --last 2 --delta --per-clip

# Write to a file instead of stdout
python asr_bench.py compare results/a.json results/b.json --output report/compare.md
```

## Error handling

- **< 2 valid files:** print an error, return non-zero. (One file is not a
  comparison.)
- **Unreadable / bad JSON / wrong schema_version:** warn to stderr, skip that file,
  continue; if too few remain, the < 2 rule fires.
- **`--delta` with ≠ 2 files:** usage error.
- **No models in common (delta mode):** still render — every model shows as
  added/removed; that *is* the comparison (a full model swap).
- **NaN:** sidecar values are already `null` (the writer guarantees it); treat
  `None` as missing → `—`.

## Testing

All torch-free, in `tests/test_compare.py`. Use small synthetic v1 dicts (a helper
`_doc(stem, models=…, corpus=…, config=…)` building the minimal sidecar shape).

- **load:** valid v1 → dict; wrong `schema_version` → `None` + stderr warning;
  missing file → `None`.
- **join:** two runs sharing `model_id` → one model row with both values; a model
  only in run 1 → `status == "removed"`; only in run 2 → `"added"`.
- **delta sign/marks:** WER 0.10→0.08 → Δ −0.02 marked ✓ (lower better); RTFx
  60→70 → Δ +10 ✓ (higher better); RTFx 70→60 → ✗; equal → no mark.
- **der inclusion:** runs with no clip `der` → `"der"` not in `report["metrics"]`;
  one run with a clip `der` → included; per-model avg der ignores nulls.
- **mode auto + force:** 2 files → `"delta"`; 3 files → `"matrix"`; `--matrix` on
  2 files → matrix; `--delta` on 3 files → usage error.
- **warnings:** differing corpus → warning; differing `beam_size` → warning; all
  same → no warnings.
- **--last N:** given a temp `results/` of timestamp-named files, selects the N
  newest; combined with positionals dedups.
- **per-clip:** two runs with a shared `clips[].audio` → joined per-clip row.
- **render smoke:** delta + matrix render to markdown containing the model display
  names, metric labels, and (delta) a ✓ or ✗; warnings render as blockquotes.
- **compare_main integration:** 2 temp JSON files → exit 0, stdout (captured) has
  the table; `--output` writes a file that contains it; 1 file → non-zero exit.
- **dispatch:** `asr_bench.main()` with `sys.argv == ["asr_bench.py", "compare", …]`
  routes to `compare_main` (monkeypatch `compare_main`, assert called) without
  building the bench parser.

## Documentation

- **README.md** — a "Comparing runs" section with the example invocations.
- **CLAUDE.md** — Status note + a `compare` workflow entry under Common workflows.
- **SPEC.md** — move `compare` from follow-up to shipped; decision-log entry.

## Decision log (to add)

- **2026-06-06** — `compare` is a first-positional keyword pre-dispatch
  (`asr_bench.py compare …`), not an argparse subparser, so every existing bench
  invocation stays byte-for-byte valid. Lazy import of `asr_compare`.
- **2026-06-06** — `compare` lives in a standalone `asr_compare.py` (pure, reads
  JSON dicts, no `asr_bench` import), not inline — keeps the comparison path
  independently testable and out of the already-large `asr_bench.py`.
- **2026-06-06** — Layout auto-selects by file count (2→delta, 3+→matrix) with
  `--delta`/`--matrix` overrides. Delta is the common "what moved after I changed
  X" case; matrix is for surveying many runs.
- **2026-06-06** — Corpus/config mismatches are loud warnings, not hard errors:
  comparing WER across different corpora is meaningless, but the user may have a
  legitimate reason (e.g. comparing RTFx across machines), so we inform, not block.
- **2026-06-06** — Per-model DER is averaged from per-clip `der` (it is not in
  `aggregates`); the `der` metric appears only when at least one run carries it.
