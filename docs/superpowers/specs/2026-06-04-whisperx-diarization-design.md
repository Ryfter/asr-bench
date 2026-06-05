# asr-bench v0.3 — WhisperX alignment + speaker diarization + DER

**Status:** Approved design (2026-06-04)
**Author:** Kevin Rank (Ryfter) + Claude Code
**Extends:** v0.2 (metrics + fusion). Realises the SPEC.md "v0.3 — WhisperX + diarization" line.

## Motivation

[WhisperX](https://github.com/m-bain/whisperX) adds two capabilities on top of a
Whisper transcription:

1. **Forced alignment** (wav2vec2) → accurate **word-level timestamps** (no auth).
2. **Speaker diarization** (pyannote.audio) → **speaker labels** per word/segment
   (needs a free HuggingFace token + accepting a gated model).

This lets asr-bench cover multi-speaker conversational content, not just
single-speaker lecture/dictation, and adds the **DER (Diarization Error Rate)**
metric when speaker ground truth is available.

## Scope

**In:** `WhisperXEngine` (alignment + diarization), speaker-labeled output
(VTT + transcript), and the **DER** metric (gated on ground-truth presence).

**Out (separate later specs):** CER, hallucination-rate, median-latency metric,
JSON results sidecar, pip-packaging — the remaining items SPEC.md lumps under
"v0.3". The `engines/` package split is also deferred one more step (see Code
structure).

## The environment constraint (decisive)

- `python` = **3.14.0** — has `faster_whisper` (ctranslate2), **cannot install
  torch** (no 3.14 wheels). This is asr-bench's normal interpreter.
- `python3` = **3.12.10** — clean; torch *does* have 3.12 CUDA wheels.

WhisperX (torch + pyannote) therefore **cannot run in asr-bench's 3.14
interpreter**. The design supports **both** execution paths, auto-selected.

## Architecture

### `WhisperXEngine(Engine)`
- Registered in `ENGINES` as `"whisperx"`.
- Model IDs use the `<size>+whisperx` form (per SPEC): `small+whisperx`,
  `medium+whisperx`, `large-v3+whisperx`, `large-v3-turbo+whisperx`. Resolution
  maps the prefix to a Whisper size and routes to the whisperx engine. `MODELS`
  gains these entries (`"engine": "whisperx"`, plus the underlying whisper size).
- `run(entry, pairs, cfg) -> ModelResult` exactly like the other engines, so the
  report/metrics infrastructure is unchanged.

### `WhisperXAdapter` (mirrors the `LLMBackend` pattern)
Abstract interface: `transcribe(audio_path, opts) -> WhisperXResult`. Two impls:

- **`InProcessWhisperX`** — imports `whisperx` in the current interpreter.
  Selected iff `importlib.util.find_spec("torch")` is present *and* the whisperx
  import succeeds.
- **`SubprocessWhisperX`** — runs `whisperx_runner.py` under a configured 3.12
  venv python (`--whisperx-python <path>`), passing options as args and reading
  a single JSON document from stdout. Selected when in-process isn't available
  and a venv python is configured (flag or auto-detected default path).

**Selection order:** in-process → subprocess (`--whisperx-python` or a default
venv path like `./.venv-whisperx/Scripts/python.exe`) → clear error explaining
setup. The executable is resolved via `shutil.which`-style robustness for
Windows.

### `whisperx_runner.py` (new standalone file)
The one place that imports `torch`/`whisperx`/`pyannote`. It is BOTH the
subprocess entry point AND the shared core the in-process adapter calls (so the
WhisperX logic exists once). Responsibilities:

```
args: --audio --model --device --language [--diarize --hf-token
       --min-speakers --max-speakers --rttm --beam-size]
steps: load whisper model → transcribe → load wav2vec2 align model → align
       → if --diarize: pyannote diarize + assign speakers to words/segments
       → if --rttm: compute DER vs the ground-truth RTTM (pyannote.metrics)
output: one JSON document on stdout:
  {
    "segments": [{"start","end","text","speaker"}],
    "words":    [{"word","start","end","score","speaker"}],
    "speakers": ["SPEAKER_00", ...],
    "der":      <float|null>,
    "language": "en"
  }
```

### `WhisperXResult` (dataclass)
Holds the parsed JSON: `segments`, `words`, `speakers`, `der` (Optional),
`language`. `WhisperXEngine` converts it into a `ClipResult` (text from joined
segments for WER/MER/WIL; speaker data for the report; a speaker-labeled VTT).

## Diarization auth

- `--hf-token` flag; falls back to `HF_TOKEN` / `HUGGINGFACE_TOKEN` env.
- **Default: `--diarize` ON** for whisperx models. **Missing token does NOT hard
  fail** — the run **warns and falls back to alignment-only** (still produces a
  word-timestamped, single-"speaker" VTT), with a clear one-line message on how
  to enable diarization (get a free token + accept the gated
  `pyannote/speaker-diarization-3.1` terms). This keeps whisperx usable
  out-of-the-box without auth.
- `--no-diarize` explicitly skips diarization (no token lookup at all).
- A genuine pyannote auth/runtime failure *after* a token is supplied surfaces a
  clear error (the token was present but rejected / model not accepted).
- `--min-speakers` / `--max-speakers` pass through to pyannote as hints.

## Output & metrics

- **Speaker-labeled VTT:** cues are prefixed `SPEAKER_00: text` (a new
  `write_whisperx_vtt` or an extended writer). Named `<base>_Captions_<Model>.vtt`
  consistent with the existing convention. **Default: prefix style**, not WebVTT
  `<v Speaker>` voice spans.
- Word-level timestamps are written to a sidecar JSON
  (`<base>_Words_<Model>.json`), not stored on `ClipResult`, to keep the
  dataclass lean.
- **`ClipResult`** gains defaulted fields: `speaker_segments: List[Tuple[float,
  float, str]] = []`, `num_speakers: int = 0`, `der: float = float("nan")`.
- **WER / MER / WIL** are unchanged — computed on the joined transcript text,
  speaker-agnostic.
- **DER:** computed in `whisperx_runner.py` (where pyannote lives) via
  `pyannote.metrics.diarization.DiarizationErrorRate`, **only when an
  `<audio-base>.rttm` sidecar is present** for the clip. Gated and labeled like
  proxy/gold WER. The report renders a **DER%** column + speaker count **only
  when at least one clip has a DER value**.

## RTTM ground truth

- Discovery: an `<audio-base>.rttm` file next to the audio (NIST RTTM format).
- Parsed into a pyannote `Annotation`; the diarization hypothesis is built from
  the model's `speaker_segments`; DER (with default collar/skip settings,
  documented) is computed by `pyannote.metrics`.

## Report rendering

- Headline + per-clip tables gain a **DER%** column and a **Speakers** count,
  shown only when diarization ran / DER exists (otherwise omitted to avoid
  empty columns — same conditional approach as the NIM "Engines in this run"
  note).
- A short "Diarization" note explains DER gating and that speaker labels are
  hypotheses unless an RTTM is present.

## Code structure

- **`whisperx_runner.py`** — new standalone module (required by the venv split;
  also the shared core).
- **`WhisperXEngine` + `WhisperXAdapter` impls + `WhisperXResult`** live in
  `asr_bench.py` for now. The full `engines/` package split (which SPEC.md
  anticipated "when WhisperX lands") is deliberately deferred to its **own**
  follow-up so this spec stays focused on behavior, not a large file-move
  refactor.

## Testing

**Unit (no torch required):**
- `FakeWhisperXAdapter` (like `FakeLLMBackend`) returns canned `WhisperXResult`s
  to drive end-to-end `WhisperXEngine.run` + report tests without torch.
- `WhisperXResult` JSON parsing (incl. missing/`null` `der`, missing speakers).
- `<size>+whisperx` model-id resolution (valid sizes, bad sizes error).
- Adapter auto-select logic: mock `importlib.util.find_spec` and
  `subprocess.run` to assert in-process-preferred / subprocess-fallback / clear
  error when neither is available.
- Speaker-VTT writer: `SPEAKER_00: text` formatting; multi-speaker ordering.
- RTTM parsing → `Annotation`.
- DER unit test: a small hand-authored hypothesis + reference with a **known**
  DER value (computed by hand) to pin the metric wiring. (If `pyannote.metrics`
  isn't importable in the test env, this test is skipped with a clear marker and
  covered by the live run instead.)

**Live (full integration, this session):**
- Create the 3.12 venv: `py -3.12 -m venv .venv-whisperx` then
  `.venv-whisperx\Scripts\pip install whisperx`.
- Set the HF token; accept the gated `pyannote/speaker-diarization-3.1` terms.
- Run `large-v3-turbo+whisperx --diarize` on a lecture clip (expect ~1 speaker,
  clean word timestamps, speaker-labeled VTT).
- Run DER end-to-end on a provided/synthesized short **multi-speaker** clip + an
  RTTM ground-truth file; confirm a DER% renders in the report.

## CLI additions

```
--whisperx-python <path>   # venv python for the subprocess adapter (auto-detected if omitted)
--diarize / --no-diarize   # default: diarize on; missing token → warn + alignment-only fallback (no hard fail)
--hf-token <token>         # else HF_TOKEN / HUGGINGFACE_TOKEN env
--min-speakers <n>         # pyannote hint
--max-speakers <n>         # pyannote hint
```
(`--device`, `--beam-size`, `--language` already exist and pass through.)

## Setup docs

README + CLAUDE.md get a "WhisperX setup" section: the 3.12 venv, `pip install
whisperx`, the HF token + gated-model acceptance step, `--whisperx-python`, and
the auto-detect behavior. SPEC.md decision log updated; the WhisperX line moves
from "planned" to "shipped (experimental until broadly validated)".

## Documented caveats

1. **Two interpreters:** WhisperX runs on 3.12+torch; asr-bench core on 3.14.
   The subprocess bridge is the portable path; in-process only when torch is
   importable.
2. **Diarization quality** depends on pyannote + audio conditions; speaker
   labels are hypotheses. DER is only meaningful against real RTTM ground truth.
3. **DER settings** (collar, overlap handling) follow pyannote.metrics defaults
   and are stated in the report note for reproducibility.

## Decision log (to add)

- **2026-06-04** — WhisperX integrated as a third engine family. Auto-detects
  in-process (torch present) vs a 3.12 venv subprocess bridge, because torch has
  no Python 3.14 wheels and asr-bench's core runs on 3.14.
- **2026-06-04** — Diarization output ships for any audio; **DER** is gated on an
  `<base>.rttm` sidecar (labeled like proxy/gold WER). HF token required only for
  diarization; alignment-only needs no auth.
- **2026-06-04** — `engines/` package split deferred again; only the necessary
  `whisperx_runner.py` is broken out (forced by the venv split).
