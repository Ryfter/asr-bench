# Design — NVIDIA NIM (Canary) engine support

- **Date:** 2026-05-30
- **Status:** approved for planning
- **Author:** Kevin Rank (with Claude Code)
- **Topic:** Add a second ASR engine family — NVIDIA NIM ASR (Riva gRPC) — to asr-bench, starting with a self-hosted Canary NIM, with a configurable endpoint URL so the same engine reaches a hosted endpoint later.

## 1. Goal & scope

asr-bench v0.1 benchmarks four local Whisper variants, all loaded into the
Python process via `faster-whisper`. This adds asr-bench's **first
service-based engine**: an ASR model served over **gRPC** by an NVIDIA NIM
microservice (NIM ASR is NVIDIA Riva under the hood).

In scope:

- A `nim` engine family that benchmarks a NIM ASR endpoint and reports the same
  core metrics (WER%, RTFx, wall clock) into the same report as Whisper.
- A pre-seeded `canary-nim` registry entry (validated against Kevin's live
  self-hosted Canary NIM).
- An ad-hoc `nim:<riva-model-name>` model id so anyone can benchmark an
  unregistered NIM without editing code.
- A configurable endpoint (`--nim-url`, plus SSL/API-key flags) so the **same**
  client path serves both self-hosted and hosted endpoints.

Out of scope (this change):

- Hosted-endpoint validation (the design supports it; we validate self-hosted).
- A standalone unit-test suite (separate roadmap item; new logic is written to
  be testable).
- Diarization / DER (v0.2 WhisperX item, unrelated).

### Relationship to the documented roadmap

SPEC.md slots NIM under v0.3 and both SPEC.md and CLAUDE.md carry a hard rule:
*"Local engines only in v0.1. No cloud API comparisons until at least v0.5."*
This change pulls NIM forward but **stays inside that rule**: a self-hosted NIM
container runs local inference on the user's own GPU, behind an HTTP/gRPC port —
it is not a cloud ASR API. The endpoint URL flag *permits* a hosted endpoint,
but the shipped/validated path and defaults are local. The hard rule's intent
(no audio leaving the box by default) is preserved. **CLAUDE.md / SPEC.md will
be updated** to record this nuance in the decision log.

## 2. Architecture — Approach C (hybrid registry + ad-hoc)

CLAUDE.md's dev guide already prescribes the shape: *"Add a new engine family:
write a sibling to the `run_model` loop that handles the engine's API; share the
metrics infrastructure (`ClipResult`, `ModelResult`, `render_markdown`)."* This
design follows it exactly.

### 2.1 Engine field on the registry

- Add `"engine"` to every `MODELS` entry. The existing four become
  `"engine": "faster-whisper"`. Code treats a missing `engine` as
  `"faster-whisper"` for safety.
- Seed one NIM entry:

  ```python
  "canary-nim": {
      "display": "Canary (NIM)", "developer": "NVIDIA",
      "params": "—", "languages": "en (+multi)",
      "engine": "nim",
      "riva_model": "",   # "" => server default model
      "notes": "NVIDIA NIM ASR via Riva gRPC. Endpoint set by --nim-url.",
  },
  ```

### 2.2 Ad-hoc `nim:<name>` ids

In model resolution, any `--models` token matching `^nim:(.+)$` synthesizes an
in-memory entry: `engine="nim"`, `riva_model="<name>"`,
`display="NIM (<name>)"`, generic metadata. Pure helper
`resolve_model_entry(model_id) -> dict` centralizes this (registry lookup OR
ad-hoc synthesis) and is unit-testable.

`main()` validation accepts an id if it is in `MODELS` **or** matches `nim:`.

### 2.3 The `Engine` contract

Rather than branch on an engine string with ad-hoc sibling functions, introduce
a small abstraction now — formalizing what `run_model` already is — derived from
**two** concrete implementations (faster-whisper + NIM) so the contract is
tested by real diversity, not guessed from one case. Files stay in
`asr_bench.py`; the package split is deferred (see §9).

```python
@dataclass
class RunConfig:
    # shared
    device: str
    compute_type: str
    # whisper-only (ignored by NIM)
    batch_size: int = 1
    beam_size: int = 5
    vad_filter: bool = True
    # nim-only (ignored by whisper)
    nim_url: str = "localhost:50051"
    nim_model: str = ""
    nim_language: str = "en-US"
    nim_api_key: Optional[str] = None
    nim_ssl: bool = False

class Engine(ABC):
    name: str                      # "faster-whisper" | "nim"
    @abstractmethod
    def run(self, entry: dict, pairs: List[Pair], cfg: RunConfig) -> ModelResult: ...

ENGINES: Dict[str, type[Engine]] = {
    "faster-whisper": FasterWhisperEngine,
    "nim": NimEngine,
}
```

- The existing `run_model` body is refactored, behavior-preserving, into
  `FasterWhisperEngine.run` (same logic, same outputs — verified by an
  unchanged Whisper-only report on the reference corpus).
- `NimEngine.run` is the new implementation (§3).
- `main()` resolves each id to an `entry`, looks up `ENGINES[entry["engine"]]`,
  and calls `.run(entry, pairs, cfg)`. Both return `ModelResult`, so
  `render_markdown` is structurally unchanged.
- A missing/unknown `engine` value raises a clear error listing valid engines.

## 3. `NimEngine.run(...) -> ModelResult`

Mirrors `FasterWhisperEngine`'s structure and failure handling.

- **Lazy import:** `import riva.client` inside the function (mirrors the lazy
  `faster_whisper` import) so `--help` and Whisper-only runs never require the
  dependency. Package missing or endpoint unreachable → return a `ModelResult`
  with `notes="LOAD FAILED: …"` (same pattern as the Whisper load-failure path),
  so the report shows a failure **row** instead of crashing the run.
- **Auth selection** (pure helper `build_nim_auth_kwargs(url, api_key, ssl)`):
  - No API key → `Auth(uri=url)` (insecure; local default).
  - API key set → SSL + `metadata_args=[["authorization", f"Bearer {key}"]]`
    (auto-enables SSL). This is the same path a hosted endpoint needs.
- **Connect/warmup timing:** `load_sec` measures `Auth` + `ASRService`
  construction plus one warmup `offline_recognize` on a short silence buffer.
  Labeled "connect/warmup" in the report, not "load."

### 3.1 Audio decode — `decode_to_pcm16(path) -> (bytes, n_samples)`

Riva `offline_recognize` wants raw PCM; the corpus is mp4/mp3/etc.

- **Primary: pyav** (already installed as a faster-whisper dependency — no PATH
  assumptions, cross-platform). Decode → resample to **16 kHz, mono, s16le**.
- **Fallback: ffmpeg subprocess** (`ffmpeg -i <in> -ar 16000 -ac 1 -f s16le -`)
  if pyav import/decoding fails. Clear error if neither is available.
- **Audio duration** = `n_samples / 16000`, computed locally and used for RTFx
  and the report — independent of anything Riva returns.

### 3.2 Transcribe & parse

- `RecognitionConfig(language_code=<--nim-language, default "en-US">,
  enable_automatic_punctuation=True, enable_word_time_offsets=True,
  max_alternatives=1, encoding=LINEAR_PCM, sample_rate_hertz=16000,
  audio_channel_count=1)`; set `model=<riva_model>` when non-empty.
- One `offline_recognize(audio_bytes, config)` RPC, wall-clocked → RTFx.
- **Hypothesis:** concat `results[].alternatives[0].transcript`. Pure helper
  `nim_response_to_hypothesis(response) -> str`.
- **Cues for VTT:** pure helper `group_words_into_cues(words) ->
  List[(start, end, text)]`, breaking on sentence-final punctuation, or ~12
  words, or ~6 s span. `cue_count = len(cues)` → **cue-density anomaly
  detection works for NIM rows too**. Reuses existing `write_whisper_vtt`.
- WER via the existing `normalize_for_wer` + `jiwer` path (unchanged).

### 3.3 Known degradation: no streaming progress

`offline_recognize` is a single blocking RPC, so NIM clips cannot emit the
per-10% streaming progress lines the Whisper loop produces. NIM clips print one
`transcribing…` line, then the result line. Documented; acceptable.

## 4. Metrics mapping (honest, never faked)

NIM runs behind a service, so it exposes **less** than the in-process Whisper
engine. We report what is measurable and clearly mark the rest. **The report
states plainly that Whisper returns the fuller, more directly-comparable set of
readings, and that NIM's numbers are indicative — run it and see how it does.**

| Metric | Whisper | NIM | Notes |
|---|---|---|---|
| WER% | ✅ | ✅ | client-measurable |
| RTFx / wall clock | ✅ | ✅ | wall-clock around the RPC; duration from local decode |
| Peak VRAM | NVML **delta** (per-clip) | NVML **total used** (best-effort, `*`-marked) | NIM model is pre-resident in the container; we report peak *total* GPU-used during the clip, **not** comparable to Whisper's delta |
| Disk size | HF cache dir | **n/a** | it's a container image, not an HF cache dir |
| load_sec | weights load | **connect/warmup** | service already up |
| batch / beam / VAD | ✅ client-controlled | n/a | server-side config |
| cue_count / VTT | from segments | from word offsets | feeds anomaly detection |

### 4.1 Best-effort VRAM for NIM

Because the single RPC has no per-segment loop to sample in, `NimEngine.run`
spawns a **background sampler thread** polling `gpu_used_bytes()` every ~100 ms
during the RPC and records the peak **total** GPU-used. Stored in
`ClipResult.vram_peak_bytes`.

To prevent a NIM total being mistaken for a Whisper delta:

- Add `ModelResult.vram_is_total: bool` (True for NIM rows).
- `render_markdown` appends a `*` to VRAM cells where `vram_is_total` is True,
  and a footnote: *"`*` = total GPU memory in use during the clip (model
  pre-resident in the NIM container), not the per-clip allocation delta that
  Whisper rows report."*

## 5. CLI flags

All default to a working self-hosted local setup; no flag needed for the happy
path beyond `--models canary-nim`.

| Flag | Default | Purpose |
|---|---|---|
| `--nim-url` | `localhost:50051` | Riva/NIM gRPC endpoint (self-hosted or hosted) |
| `--nim-model` | `""` | Override `riva_model` for the `canary-nim` entry |
| `--nim-language` | `en-US` | Riva language code (note: Whisper uses `en`) |
| `--nim-api-key` | none | Bearer token; presence auto-enables SSL |
| `--nim-ssl` | off | Force SSL without a key (e.g. self-signed local TLS) |

These flags only affect `nim`-engine models; they are ignored for Whisper runs.

## 6. Report / `render_markdown` changes

Structurally unchanged. Additions:

- An **"Engines in this run"** note when any NIM row is present, stating the
  metric-fidelity difference (§4) and the `*` VRAM footnote.
- VRAM cell `*` marker for `vram_is_total` rows.
- The reproducibility command line echoes the `--nim-*` flags when a NIM model
  ran.
- NIM `disk` cell renders `n/a` (today `fmt_bytes(None)` → `?`; we special-case
  NIM to `n/a` for clarity).

## 7. Dependencies

- New optional runtime dep: **`nvidia-riva-client`** (provides `riva.client`).
  Lazy-imported; only required when a `nim` model is requested. Document in
  README/CLAUDE.md install notes.
- **pyav** (`av`) already present via faster-whisper; used for decode. ffmpeg
  (already on Kevin's PATH) is the documented fallback.

## 8. Validation plan

Against the live self-hosted Canary NIM:

1. **Smoke:** `python asr_bench.py --models canary-nim --include "<one clip>" --limit 1`
   — confirm connect, decode, transcribe, WER, VTT, total-VRAM sampling.
2. **Head-to-head:** `python asr_bench.py --models large-v3-turbo,canary-nim --limit 2`
   — confirm one report ranks both engines with correct metric markings.
3. **Ad-hoc id:** `--models nim:<some-model>` resolves and runs without code edits.
4. **Graceful failure:** wrong `--nim-url` produces a `LOAD FAILED` row, not a crash.

New pure helpers (`resolve_model_entry`, `build_nim_auth_kwargs`,
`decode_to_pcm16`, `nim_response_to_hypothesis`, `group_words_into_cues`) are
written to be unit-testable without a GPU or a live endpoint, ahead of the
planned test-suite work.

## 9. Single-file constraint & the structure decision

CLAUDE.md says keep asr-bench a single file until v0.2's multi-engine support
forces a split. This *is* the first multi-engine step. Decision (chosen over a
bare sibling function and over a full package split):

**Introduce the `Engine` contract now (§2.3) but keep everything in
`asr_bench.py`.** Rationale:

- The *interface* is the valuable, hard-to-change artifact; the *file layout* is
  cheap and mechanical to change later. Define the interface now, while two
  genuinely different engines (in-process delta-VRAM-segments vs. service
  total-VRAM-words) are available to factor it from.
- Defer the `engines/` package split until a **third** family (WhisperX in v0.2,
  then NeMo in v0.3) actually lands — at which point the split is driven by real
  code volume and the contract is already proven. WhisperX is in-process and
  *wraps* faster-whisper, so it will compose against `FasterWhisperEngine`
  cleanly when that day comes.
- Note for later: Canary overlaps two families (NIM now, NeMo in-process later);
  the `Engine` contract is what makes a future "same model, two serving paths"
  comparison clean.

This respects CLAUDE.md's "resist breaking it up" guidance (no new files) while
still leaving the codebase ready for later engines at the contract level.

## 10. Open items deferred (not blocking)

- Hosted-endpoint validation.
- `docker images` introspection to fill NIM disk size.
- Concurrency/throughput benchmarking (server-side batching) — single-request
  offline only for now.
