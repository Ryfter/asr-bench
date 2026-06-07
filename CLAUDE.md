# asr-bench — Claude Code Instructions

This file is loaded automatically when Claude Code is opened at this path. Future sessions: read this first.

## Project purpose

Independent benchmarking tool for local speech recognition models. CLI, markdown output. Built for educators and accessibility teams who need to pick an ASR engine for their actual content — not whatever was on a public leaderboard last year.

**Split from canvas-toolchain on 2026-05-30** because the audience is broader than Canvas LMS users. canvas-toolchain's `compare_transcripts` workflow is the production consumer; asr-bench is the engine-comparison tool that informs which model to plug in.

## Repository

- **Code:** https://github.com/Ryfter/asr-bench (private)
- **Owner:** Ryfter (Kevin Rank)
- **License:** MIT

## Status — v0.3 on feat/whisperx-diarization (not yet merged to main)

| Model ID | Params | Disk | Run via |
|---|---|---|---|
| `small` | 244M | ~470MB | faster-whisper |
| `medium` | 769M | ~1.5GB | faster-whisper |
| `large-v3` | 1550M | ~3.1GB | faster-whisper |
| `large-v3-turbo` | 809M | ~1.6GB | faster-whisper |
| `<size>+whisperx` | same | same | WhisperX (align + diarize) |

### What works in v0.1
- Multi-model benchmark with one report-per-run
- Metrics: WER%, RTFx, wall clock, peak VRAM, disk size, params
- Three corpus layouts (flat name-matched, Panopto export shape, manifest.json)
- Per-clip + per-model + headline tables in the report
- VTT outputs as `<base>_Captions_<Model>.vtt` next to the source audio
- Auto-detect Panopto/ASR-generated captions; report labels reference origin
- **VAD filter** (Silero) on by default — prevents Whisper-Large 1-second-cue lockup
- **GPU-aware `batch_size auto`** — probes free VRAM via NVML, recommends a fit
- **Streaming `[N/M] xx.x% progress`** lines per clip so the user can see it's alive
- **Cue-density anomaly detection** — flags any (model, clip) whose cue count is ≥ 1.5× the median of other models on the same clip
- Reports under `./report/`, gitignored

### What's new in v0.2
- **MER% and WIL%** — Match Error Rate and Word Information Lost (Morris, Maier & Green 2004) alongside WER%; bounded [0,1], information-theoretic, better reflect information communicated on lecture content
- **Per-clip S/D/I counts** — substitution, deletion, and insertion counts per clip via `jiwer.process_words`
- **`--show-alignment`** — prints per-clip alignment diffs to stdout for manual inspection
- **Fusion stage (`--fuse`)** — post-benchmark pass that parses each model's VTT + the Panopto reference into timed cues, windows them (default 25 s / 5 s overlap), and feeds each window to a pluggable LLM; produces:
  - **verbatim profile** → `<base>_Captions_Fused.vtt` (accessibility captions; also usable as a scoring reference)
  - **kb profile** → `<base>_KB_Fused.jsonl` + `.md` (RAG knowledge base, overlapping time-tagged chunks)
  - `--profile verbatim|kb|both` (default `both`)
- **Pluggable LLM backend (`--llm`)** — `ollama:<model>` (default `ollama:qwen2.5`, local/offline), `cli:<command>` (shell out to an authenticated frontier CLI like `claude`/`gemini` — uses existing subscription, no API key), `fake` (tests/dry-runs)
- **`--init-context [PATH]`** — writes a guided `context.md` template (schedule, names, jargon, mishearings, style, glossary) to fill in; pass result via `--context` (+ optional `--glossary`)
- **Drift guard** — per-window WER(fused vs base model) flags potential hallucination or omission
- **`--rescore-against-fused`** — re-scores every model against the verbatim fused VTT as reference; emits a second table explicitly labeled "fused verbatim consensus (agreement-biased)"

### What's new in v0.3 (feat/whisperx-diarization — 119 tests pass, 2 skipped)
- **`<size>+whisperx` model IDs** — e.g. `large-v3-turbo+whisperx`; pairs any Whisper size with WhisperX word alignment + pyannote speaker diarization
- **Two execution paths, auto-selected** — in-process when `torch` is importable in the running interpreter; otherwise a subprocess to a Python ≤ 3.13 venv (`--whisperx-python`, auto-detects `./.venv-whisperx`). Rationale: torch has no Python 3.14 wheels and asr-bench's core runs on 3.14.
- **`whisperx_runner.py`** — standalone script: transcribe → align → diarize (pyannote) → DER; runs in the WhisperX venv, communicates via JSON
- **Speaker-labeled VTT** — cues prefixed `SPEAKER_00: text`; `<base>_Words_<Model>.json` word-timestamp sidecar
- **DER (Diarization Error Rate)** — computed via pyannote.metrics; gated on a `<base>.rttm` ground-truth sidecar next to the audio. Report shows DER% + Speakers columns only when diarization data is present.
- **Auth** — diarization needs a free HuggingFace token (`--hf-token` or `HF_TOKEN`/`HUGGINGFACE_TOKEN` env) + accepting the gated `pyannote/speaker-diarization-community-1` model (pyannote-audio 4.x default; self-contained — bundles segmentation + embedding). Missing token warns and falls back to alignment-only; `--no-diarize` skips diarization entirely. `--diarize-model` overrides (e.g. `pyannote/speaker-diarization-3.1` on a pyannote 3.x install).
- **New CLI flags** — `--diarize`/`--no-diarize`, `--hf-token`, `--min-speakers`, `--max-speakers`, `--whisperx-python`
- **JSON results sidecar** — every run writes `results/<timestamp>.json` (or `<output>.json`) mirroring the full run (config, per-model/per-clip metrics, transcripts, speaker/DER) for cross-run aggregation. `schema_version: 1`. NaN→null; `hf_token`/`nim_api_key` redacted. `--no-json` opts out. Consumed by the `compare` subcommand (below).
- **`compare` subcommand** — `python asr_bench.py compare a.json b.json` reads 2+
  `results/*.json` sidecars and renders a delta (2 files) or matrix (3+) markdown
  comparison of per-model WER/MER/WIL/RTFx/DER, with corpus/config mismatch
  warnings and optional `--per-clip` detail. Standalone `asr_compare.py`; bench CLI
  unchanged (first-positional `compare` keyword pre-dispatch).
- **CER% + median latency** — per-clip Character Error Rate (jiwer) alongside
  WER/MER/WIL in every report table and in `compare`; per-model `median_rtfx`
  (a robust counterpart to the totals-based aggregate RTFx) shown as "RTFx (med)",
  plus `median_sec_per_audio_min` in the sidecar. JSON sidecar fields are additive
  — `schema_version` stays 1.
- **Hallucination-rate detection** — reference-free per-clip signals
  (`repeat_coverage` = repeated-4-gram fraction; `compression_ratio` = gzip ratio,
  Whisper's own signal) flag looping/fabricated output in a dedicated "⚠️
  Hallucination signals" report section (works without a reference and in
  single-model runs, unlike cue-density); per-model `hallucination_rate`; additive
  sidecar fields (`schema_version` stays 1).

Branch `feat/whisperx-diarization` is pushed to GitHub. **Not yet merged to main** (live WhisperX + diarization run pending — no pyannote venv set up on reference machine yet).

See [`SPEC.md`](./SPEC.md) for the v0.4 (NVIDIA NeMo / Canary-Qwen) and v0.5 (community models) roadmap.

## Reference benchmark — sample lecture corpus

The first real run on 2026-05-30 used 12 lectures, 614 minutes of audio, from a single-speaker university lecture course (two sections, three weeks each, ~50 min per lecture). Default settings, batch_size=1 (before the auto-batching landed):

| Model | Overall WER% | RTFx | Total time | Peak VRAM |
|---|---|---|---|---|
| Whisper Small | 10.7 | 43.5× | 14 min | 372 MB |
| Whisper Medium | 11.8 | 29.1× | 21 min | 269 MB |
| Whisper Large V3 | 14.2 ⚠️ | 14.7× | 42 min | 1.2 GB |
| **Whisper Large V3 Turbo** | **8.9** ✅ | **64.8×** ✅ | **9.5 min** ✅ | 168 MB |

Report: `report/20260530-190913.md` (local only — gitignored).

**Key findings:**
- **Large V3 Turbo wins on both axes** (lowest WER + fastest) on this single-speaker lecture content. Unusual — typically Large V3 has the accuracy edge.
- **Medium > Small on WER** (Medium more literal where Panopto's editorial cleanup makes the proxy reference diverge — same pattern as the canvas-toolchain `compare_transcripts` runs)
- **Large V3 had a 1-second-cue decoder lockup ~25 min into one 54-min lecture** — fell into 1-cue-per-second for 33 min, inflating its WER from ~12% to 14.2%. Fixed by `vad_filter=True` (now default).

## Locally-discovered setup notes (reference machine)

These need to be true on any machine that runs asr-bench against GPU. Document equivalents for new machines.

- **GPU**: NVIDIA RTX 5090, 34 GB VRAM
- **Python**: 3.14.0 on PATH as `python` (has `faster_whisper`); 3.12.10 on PATH as `python3` (does NOT have faster_whisper). asr-bench's auto-detection picks the right one, but be aware.
- **ffmpeg**: at `C:\Users\krank\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe`. Not on bash's PATH; on PowerShell PATH. Not strictly needed for the bench itself (faster-whisper handles decoding internally via pyav).
- **CUDA runtime libs**: from pip wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`). asr-bench's `_add_nvidia_dll_directories()` auto-adds them to PATH at startup — both via `os.environ['PATH']` (universal) and `os.add_dll_directory()` (belt-and-suspenders for native loaders).
- **NVML DLL**: at `C:\Windows\System32\nvml.dll` (modern install). The deprecated `nvidia-ml-py3` package looks at `C:\Program Files\NVIDIA Corporation\NVSMI\` which is the old layout — use `nvidia-ml-py` (no `3`) instead.
- **nvidia-riva-client** (for NIM engine): `pip install nvidia-riva-client`. Lazy-imported — only required when a `nim`-engine model (e.g. `canary-nim`, `nim:<name>`) is requested. Whisper-only runs don't need it.

### WhisperX setup notes (reference machine)

- **Python 3.12 venv** — WhisperX and PyTorch have no Python 3.14 wheels. Use `./setup_whisperx_venv.ps1` (creates `.venv-whisperx`, installs CUDA torch + whisperx, verifies CUDA). asr-bench auto-detects this path; `--whisperx-python` overrides.
- **CUDA wheels — install torch FIRST** — live-validated 2026-06-05: a bare `pip install whisperx` pulls the **CPU-only** torch wheel on Windows (`torch.cuda.is_available()` == False, silent CPU fallback). Install `torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128` BEFORE whisperx (cu128 = Blackwell/RTX 5090/sm_120). The setup script does this. Always verify `cuda True` in the venv.
- **HuggingFace token** — diarization uses the gated `pyannote/speaker-diarization-community-1` model (pyannote-audio 4.x default; self-contained, bundles segmentation + embedding — accept just this one repo). Loading the old `3.1` id under pyannote 4.x pulls community-1 assets anyway. Accept terms at huggingface.co/pyannote/speaker-diarization-community-1, set `HF_TOKEN` (or `--hf-token`). Token passed to the subprocess via env, not argv. Override with `--diarize-model` for pyannote 3.x.
- **Missing token behavior** — pre-flight check in `main()` warns and falls back to alignment-only (no diarization, no DER). Does not hard-fail.
- **RTTM sidecars** — to get DER scoring, drop `<base>.rttm` next to the audio file. The `find_rttm` helper locates it by stem match.
- **Speaker hints for long audio** — pyannote over-clusters on long noisy recordings (an 82-min 2-speaker Zoom call estimated as 12 speakers, DER 27.4%). Pass `--min-speakers`/`--max-speakers` when the count is known — constraining to 2 gave 2 speakers, DER 13.8%.
- **Live validation (2026-06-05)** — full path exercised on RTX 5090 (torch 2.8.0+cu128, whisperx 3.8.6, pyannote-audio 4.0.4): transcribe+align (82 min → 763 segs / 11,274 words), diarization (community-1), DER end-to-end (13.8% @ 2-speaker hint). Runner fixes the live run required: pure-JSON stdout (lib chatter → stderr), `use_auth_token`→`token` kwarg, community-1 default + `--diarize-model`.

## Common workflows

### Run the full benchmark (default models, current defaults)
```powershell
cd D:\dev\asr-bench
python asr_bench.py
```
Defaults:
- `--corpus ./test-corpus`
- `--models small,medium,large-v3,large-v3-turbo`
- `--device auto` (CUDA if available)
- `--batch-size auto` (probes VRAM, picks a fit)
- `--vad-filter` (on)
- `--beam-size 5`

### Smoke run on a subset
```powershell
python asr_bench.py --models small,medium --include "Week 16 - Friday" --limit 2
```

### Force CPU
```powershell
python asr_bench.py --device cpu
```

### Disable VAD (for diagnosing pathological model behavior)
```powershell
python asr_bench.py --no-vad-filter
```

### Benchmark a self-hosted NIM against Whisper
```powershell
python asr_bench.py --models large-v3-turbo,canary-nim --nim-url localhost:50051
```
NIM rows report WER/RTFx/wall-clock normally; VRAM is shown as total-used (`*`) and disk as `n/a` (see the report's "Engines in this run" note). Ad-hoc unregistered NIM models: `--models nim:<riva-model-name>`.

### Run WhisperX (word alignment + speaker diarization)
```powershell
python asr_bench.py --models large-v3-turbo+whisperx --diarize --hf-token hf_...
```
Auto-detects `.venv-whisperx`; runs as subprocess if torch not importable in current interpreter. Produces speaker-labeled VTT and a `_Words_<Model>.json` word-timestamp sidecar. DER column appears if a `<base>.rttm` sidecar is present.

### WhisperX alignment-only (no auth required)
```powershell
python asr_bench.py --models large-v3-turbo+whisperx --no-diarize
```
Word timestamps and VTT without speaker labels; no HF token needed.

### WhisperX with a custom Python path
```powershell
python asr_bench.py --models large-v3-turbo+whisperx --whisperx-python C:\venvs\wx312\Scripts\python.exe
```

### WhisperX with speaker count hints
```powershell
python asr_bench.py --models large-v3-turbo+whisperx --diarize --min-speakers 2 --max-speakers 4 --hf-token hf_...
```

### Generate a context file template for fusion
```powershell
python asr_bench.py --init-context context.md
```
Fill in the generated `context.md` (course schedule, speaker names, domain jargon, common mishearings, style preferences, glossary). Pass it to a fusion run via `--context`.

### Run a full fusion pass (local Ollama)
```powershell
python asr_bench.py --models small,medium,large-v3-turbo --fuse --profile both --llm ollama:qwen2.5 --context context.md
```
Produces `_Captions_Fused.vtt` (verbatim, ADA/WCAG-eligible) and `_KB_Fused.jsonl` + `.md` (RAG knowledge base) next to each audio file. Requires Ollama running locally with the `qwen2.5` model pulled.

### Run fusion with a frontier CLI backend
```powershell
python asr_bench.py --models large-v3-turbo --fuse --profile verbatim --llm cli:claude --context context.md
```
`cli:claude` shells out to the `claude` CLI using your existing subscription — no API key needed. Substitute `cli:gemini` etc. for other authenticated CLIs.

### Re-score all models against the fused verbatim reference
```powershell
python asr_bench.py --models small,medium,large-v3-turbo --fuse --rescore-against-fused --context context.md
```
Emits a second metrics table labeled "fused verbatim consensus (agreement-biased)" — useful for tracking improvement after fusion, but note the reference was built from the same models being scored.

### Dry-run fusion without a live LLM
```powershell
python asr_bench.py --models small,medium --fuse --llm fake --limit 1
```
Uses `FakeLLMBackend` — no Ollama required. Good for verifying corpus layout and pipeline wiring before a real run.

### Watch live output from another shell
```powershell
Get-Content -Wait $(Get-ChildItem report\*.md | Sort LastWriteTime -Desc | Select -First 1).FullName
```
(After a fresh run starts and the report stub exists.)

### Compare runs across the JSON sidecars
```powershell
python asr_bench.py compare results/<old>.json results/<new>.json          # delta
python asr_bench.py compare --last 3                                        # 3 newest (matrix if ≥3 exist)
python asr_bench.py compare results/a.json results/b.json --per-clip        # + per-clip
```
Reads `schema_version 1` sidecars. 2 files → delta view; 3+ → matrix; `--delta`/`--matrix` force.

## Development workflow

- **Main script** — `asr_bench.py` is the core. `whisperx_runner.py` is the only file broken out so far (it must run inside a Python ≤ 3.13 venv). The `engines/` subpackage split remains deferred until a fourth engine family (NeMo/Canary-Qwen) lands.
- **Add a new engine family**: implement the `Engine` ABC (`run(entry, pairs, cfg) -> ModelResult`), register the class in `ENGINES`, and give its models `"engine": "<name>"` in `MODELS`. `FasterWhisperEngine`, `NimEngine`, and `WhisperXEngine` are the three reference implementations. Share the metrics infrastructure (`ClipResult`, `ModelResult`, `render_markdown`).
- **Add a new Whisper variant**: extend the `MODELS` dict (`"engine": "faster-whisper"`) + add an entry to `_MODEL_VRAM_COST` for batch sizing.
- **Add a new WhisperX variant**: extend `MODELS` with `"engine": "whisperx"` and ensure the base model name (before `+whisperx`) maps to a valid faster-whisper model key.
- **Tests**: pytest suite under `tests/`. Run `python -m pytest`. 119 pass, 2 skipped (pyannote not installed in core venv — WhisperX diarization tests are subprocess-gated).
- **Linting**: none yet — follow the style already in `asr_bench.py`

## Hard rules

- **No bundled audio in the repo.** `test-corpus/*` is gitignored. Distributing sample audio creates licensing headaches.
- **WER labels reflect the reference.** Output explicitly labels gold vs proxy in the headline. Never silently pass off proxy WER as accuracy.
- **VAD on by default**, but always toggleable via `--no-vad-filter`.
- **Local ASR engines only** (through v0.4). No cloud ASR API comparisons until at least v0.5. The fusion LLM backend (`--llm cli:...`) may call a frontier model CLI, but that is post-processing of local transcription output, not a cloud ASR engine comparison.
- **CLI only.** No GUI. Audience is technical users + faculty IT staff.

## Related projects

- **canvas-toolchain** (https://github.com/Ryfter/canvas-toolchain) — sibling project. Its `compare_transcripts` workflow is the production consumer of whichever engine asr-bench surfaces as the winner. The TranscriptionEngine swap-in pattern in canvas-toolchain mirrors asr-bench's `MODELS` registry — keeping the contract compatible would let asr-bench-recommended engines drop straight into the canvas-toolchain workflow.

## Decision log

- **2026-05-30** — Split from canvas-toolchain as its own private repo. Audience broader than Canvas LMS.
- **2026-05-30** — v0.1 ships Whisper-only via faster-whisper. WhisperX deferred (pyannote auth complexity), Canary-Qwen deferred (NeMo's heavy install needs its own venv discipline). Better narrow + working than broad + broken.
- **2026-05-30** — CLI + markdown only. No GUI.
- **2026-05-30** — VAD filter on by default after observing the Whisper-Large 1-second-cue decoder lock on Week 14 Wednesday.
- **2026-05-30** — Batch size defaults to `auto` (NVML-probed) for non-CPU runs. Improves GPU utilization 50% → 80%+.
- **2026-05-31** — Added NVIDIA NIM ASR (Riva gRPC) as the second engine family, ahead of its v0.3 roadmap slot. Implemented and *statically* verified against `nvidia-riva-client` 2.26.0, but **not yet run against a live NIM** (see the ship-as-is entry below). Stays within the "local engines only" rule: a self-hosted NIM is local inference behind a gRPC port, not a cloud ASR API. The `--nim-url` flag *permits* a hosted endpoint, but defaults are local. Introduced the `Engine` contract (`FasterWhisperEngine` + `NimEngine`) in-file; deferred the `engines/` package split until WhisperX/NeMo land.
- **2026-05-31** — Shipped the NIM engine **as-is, without a live-NIM test run**. asr-bench's core purpose is benchmarking local Whisper variants; NIM and other extra engines are nice-to-have, not critical. Live validation was deferred because the reference box has no container runtime (no Docker, no WSL distro) and NIM is **container-only** (NVIDIA ships it as an `nvcr.io` image, not a native binary). Tested: audio decode, `main()`→engine dispatch, mixed-engine report rendering, graceful failure. Untested: the **local self-hosted Docker path**; the **remote/hosted (`--nim-api-key`/`--nim-ssl`) path is implemented but not fully tested**. Intent remains local self-hosted NIM. Pending live validation tracked in `memory/validate-live-nim.md`.
- **2026-06-01** — Adopted MER (Match Error Rate) and WIL (Word Information Lost) from Morris, Maier & Green 2004 as primary information-quality metrics alongside WER. Both are bounded [0, 1] and information-theoretic; on lecture content they better reflect how much meaning was communicated vs lost, compared to raw WER which weights all errors equally.
- **2026-06-01** — Fusion implemented as a **post-processing pass**, NOT a new Engine. It consumes multiple engines' VTT outputs (+ the Panopto reference cues) rather than running transcription itself. Keeping it outside the Engine ABC avoids conflating transcription benchmarking with post-processing quality.
- **2026-06-01** — Two fusion profiles — **verbatim** (captions + scoring reference) and **kb** (RAG knowledge base) — share one chunked/timing-anchored, overlapping-window pipeline, differing only in prompt and output format. One pass per window; `--profile both` runs both prompts in the same pass rather than re-chunking.
- **2026-06-01** — LLM backend is pluggable, **local-first default** (`ollama:qwen2.5` — offline, no API key). The `cli:<command>` backend shells out to an authenticated frontier CLI (e.g. `claude`, `gemini`) and uses an existing subscription rather than requiring an API key — consistent with the local-first ethos (billing stays with the user's existing account, not asr-bench's infrastructure).
- **2026-06-01** — Only the **verbatim profile is ADA/WCAG caption-eligible**. The kb profile deliberately rephrases and condenses for retrieval quality; it is explicitly NOT compliant captions and is labeled as such in the report.
- **2026-06-01** — The `--rescore-against-fused` reference is **agreement-biased** and labeled as such. The fused verbatim VTT is built from the same model outputs being scored — it measures consensus, not ground truth. Useful for tracking post-fusion improvement, but the caveat is printed in the report header so it is never mistaken for an independent gold reference.
- **2026-06-04** — Added WhisperX as the third engine family (`WhisperXEngine` + `whisperx_runner.py`). Design doc: `docs/superpowers/specs/2026-06-04-whisperx-diarization-design.md`. Key decisions: (1) `<size>+whisperx` model IDs pair any Whisper size with the WhisperX pipeline so users see a direct comparison in the same report; (2) in-process vs subprocess auto-detection solves the torch/Python 3.14 incompatibility without forcing users to manage two Python versions manually; (3) `WhisperXAdapter` factory encapsulates mode selection, keeping `WhisperXEngine` and `whisperx_runner.py` testable independently.
- **2026-06-04** — DER scoring gated on an RTTM sidecar (`<base>.rttm` next to the audio). Rationale: ground-truth speaker boundaries are labor-intensive; not every run has them. Diarization still runs (speaker labels in VTT) when no RTTM is present — only DER% and Speakers columns are suppressed. This keeps the default behavior useful without requiring annotation work.
- **2026-06-04** — Diarization auth uses a missing-token-warns-and-falls-back strategy rather than a hard fail. Rationale: the core value of a `+whisperx` run (word timestamps, better alignment) is available without a HF token. Forcing auth for alignment-only would be hostile UX. Token is passed to the subprocess via `HF_TOKEN` env var, not argv (avoids token appearing in `ps` output).
- **2026-06-04** — `engines/` package split deferred again. `whisperx_runner.py` is the only file broken out (forced by the venv boundary). The full split waits for a fourth engine family (NeMo/Canary-Qwen) where the weight of three Engine subclasses in one file becomes unwieldy.
- **2026-06-05** — JSON results sidecar always emitted (no flag friction; aggregation needs the data to reliably exist) to `results/<ts>.json` or a `--output` sibling. Full mirror including transcripts (cheap text, expensive to regenerate). NaN→null with an `allow_nan=False` write guard so output is always valid JSON. Secrets (hf_token, nim_api_key) omitted from `run.config`. `compare` subcommand deferred to its own spec; schema is compare-ready via `schema_version`.
- **2026-06-05** — **Live-validated WhisperX end-to-end** (Task 13) on the RTX 5090; merged after. Three fixes the real run forced: (1) the runner now emits **pure JSON on stdout** — whisperx/torch/pyannote write progress + logging there, which broke `SubprocessWhisperX`'s `json.loads(stdout)`; main() redirects both `sys.stdout` and OS fd 1 to stderr during processing. (2) **pyannote-audio 4.x kwarg** `use_auth_token` → `token` (try-new-fall-back-to-old). (3) **Default diarization model is now `pyannote/speaker-diarization-community-1`**, not 3.1 — pyannote 4.x unified on this one self-contained gated repo (loading the 3.1 id under 4.x pulls community-1 assets anyway); `--diarize-model` overrides for 3.x. Also: a bare `pip install whisperx` pulls **CPU-only torch** on Windows — the setup script now installs the cu128 build first. And pyannote **over-clusters on long audio** (82-min 2-speaker call → 12 speakers / DER 27.4%); `--min/max-speakers 2` → 2 speakers / DER 13.8%.
- **2026-06-06** — Shipped the **`compare` subcommand** (merged to main `6417f42`). Reads 2+ `results/*.json` sidecars (schema_version 1) and renders a markdown **delta** (2 files: baseline→candidate, signed Δ with ✓/✗ by metric direction) or **matrix** (3+: one column per run), auto by count with `--delta`/`--matrix` overrides. Joins per-model headline metrics on `model_id`; per-model DER averaged from per-clip `der`; corpus/config drift surfaced as ⚠️ warnings (not hard errors — comparing across corpora is suspect but sometimes intentional). `--last N`, `--per-clip`, `--output` round it out. Implemented as a **first-positional `compare` keyword pre-dispatch** (3 lines at top of `main()`) into a standalone, pure, torch-free **`asr_compare.py`** — the bench CLI is byte-for-byte unchanged and a plain run never imports the module. Robustness: warn-and-skip on unreadable / non-object / wrong-schema sidecars. 175 tests pass. Spec/plan under `docs/superpowers/`.
- **2026-06-06** — Shipped CER (char-level, via jiwer on the same normalized text as WER, added to WordMetrics so all three engines get it from one call) and a robust median speed pair (`median_rtfx`, `median_sec_per_audio_min`) beside the totals-based `aggregate_rtfx`. Sidecar fields additive within schema_version 1; CER also surfaced in `compare`. Report columns: "CER%" (per-clip and per-model tables), "RTFx (med)" (headline table).
