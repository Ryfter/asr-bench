# asr-bench — Claude Code Instructions

This file is loaded automatically when Claude Code is opened at this path. Future sessions: read this first.

## Project purpose

Independent benchmarking tool for local speech recognition models. CLI, markdown output. Built for educators and accessibility teams who need to pick an ASR engine for their actual content — not whatever was on a public leaderboard last year.

**Split from canvas-toolchain on 2026-05-30** because the audience is broader than Canvas LMS users. canvas-toolchain's `compare_transcripts` workflow is the production consumer; asr-bench is the engine-comparison tool that informs which model to plug in.

## Repository

- **Code:** https://github.com/Ryfter/asr-bench (private)
- **Owner:** Ryfter (Kevin Rank)
- **License:** MIT

## Status — v0.2 shipped 2026-06-01

Local Whisper variants only:

| Model | Params | Disk | Run via |
|---|---|---|---|
| Whisper Small | 244M | ~470MB | faster-whisper |
| Whisper Medium | 769M | ~1.5GB | faster-whisper |
| Whisper Large V3 | 1550M | ~3.1GB | faster-whisper |
| Whisper Large V3 Turbo | 809M | ~1.6GB | faster-whisper |

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

See [`SPEC.md`](./SPEC.md) for the v0.3 (WhisperX + diarization), v0.4 (NVIDIA NeMo / Canary-Qwen), v0.5 (community models) roadmap.

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

## Development workflow

- **Single-file script** — `asr_bench.py` is the whole tool. Resist the urge to break it up until a third engine family (WhisperX/NeMo) lands and forces an `engines/` subpackage. The fusion stage and LLM backends live in-file for now; extract when the file becomes unwieldy.
- **Add a new engine family**: implement the `Engine` ABC (`run(entry, pairs, cfg) -> ModelResult`), register the class in `ENGINES`, and give its models `"engine": "<name>"` in `MODELS`. `FasterWhisperEngine` and `NimEngine` are the two reference implementations. Share the metrics infrastructure (`ClipResult`, `ModelResult`, `render_markdown`). The `engines/` package split is deferred until a third family (WhisperX/NeMo) lands.
- **Add a new Whisper variant**: extend the `MODELS` dict (`"engine": "faster-whisper"`) + add an entry to `_MODEL_VRAM_COST` for batch sizing.
- **Tests**: pytest suite under `tests/` (added with the NIM engine). Run `python -m pytest`.
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
