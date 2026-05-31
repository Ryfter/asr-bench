# asr-bench — Claude Code Instructions

This file is loaded automatically when Claude Code is opened at this path. Future sessions: read this first.

## Project purpose

Independent benchmarking tool for local speech recognition models. CLI, markdown output. Built for educators and accessibility teams who need to pick an ASR engine for their actual content — not whatever was on a public leaderboard last year.

**Split from canvas-toolchain on 2026-05-30** because the audience is broader than Canvas LMS users. canvas-toolchain's `compare_transcripts` workflow is the production consumer; asr-bench is the engine-comparison tool that informs which model to plug in.

## Repository

- **Code:** https://github.com/Ryfter/asr-bench (private)
- **Owner:** Ryfter (Kevin Rank)
- **License:** MIT

## Status — v0.1 shipped 2026-05-30

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

See [`SPEC.md`](./SPEC.md) for the v0.2 (WhisperX + diarization), v0.3 (NVIDIA NeMo / Canary-Qwen), v0.4 (community models) roadmap.

## Reference benchmark — Kevin's ITM310 corpus

The first real run on 2026-05-30 used 12 lectures, 614 minutes of audio, from Kevin's ITM310 Spring 2026 course (sections 002 + 003, weeks 14-16). Default settings, batch_size=1 (before the auto-batching landed):

| Model | Overall WER% | RTFx | Total time | Peak VRAM |
|---|---|---|---|---|
| Whisper Small | 10.7 | 43.5× | 14 min | 372 MB |
| Whisper Medium | 11.8 | 29.1× | 21 min | 269 MB |
| Whisper Large V3 | 14.2 ⚠️ | 14.7× | 42 min | 1.2 GB |
| **Whisper Large V3 Turbo** | **8.9** ✅ | **64.8×** ✅ | **9.5 min** ✅ | 168 MB |

Report: `report/20260530-190913.md` (local only — gitignored).

**Key findings:**
- **Large V3 Turbo wins on both axes** (lowest WER + fastest) on Kevin's voice + lecture content. Unusual — typically Large V3 has the accuracy edge.
- **Medium > Small on WER** (Medium more literal where Panopto's editorial cleanup makes the proxy reference diverge — same pattern as the canvas-toolchain `compare_transcripts` runs)
- **Large V3 had a 1-second-cue decoder lockup at 25:40 of Week 14 Wednesday section 002** — fell into 1-cue-per-second for 33 min, inflating its WER from ~12% to 14.2%. Fixed by `vad_filter=True` (now default).

## Locally-discovered setup notes (Kevin's box)

These need to be true on any machine that runs asr-bench against GPU. Document equivalents for new machines.

- **GPU**: NVIDIA RTX 5090, 34 GB VRAM
- **Python**: 3.14.0 on PATH as `python` (has `faster_whisper`); 3.12.10 on PATH as `python3` (does NOT have faster_whisper). asr-bench's auto-detection picks the right one, but be aware.
- **ffmpeg**: at `C:\Users\krank\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe`. Not on bash's PATH; on PowerShell PATH. Not strictly needed for the bench itself (faster-whisper handles decoding internally via pyav).
- **CUDA runtime libs**: from pip wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`). asr-bench's `_add_nvidia_dll_directories()` auto-adds them to PATH at startup — both via `os.environ['PATH']` (universal) and `os.add_dll_directory()` (belt-and-suspenders for native loaders).
- **NVML DLL**: at `C:\Windows\System32\nvml.dll` (modern install). The deprecated `nvidia-ml-py3` package looks at `C:\Program Files\NVIDIA Corporation\NVSMI\` which is the old layout — use `nvidia-ml-py` (no `3`) instead.

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

### Watch live output from another shell
```powershell
Get-Content -Wait $(Get-ChildItem report\*.md | Sort LastWriteTime -Desc | Select -First 1).FullName
```
(After a fresh run starts and the report stub exists.)

## Development workflow

- **Single-file script** — `asr_bench.py` is the whole tool. Resist the urge to break it up until v0.2 demands it (multi-engine support across NeMo + WhisperX will probably trigger a `engines/` subpackage).
- **Add a new Whisper variant**: extend the `MODELS` dict + add an entry to `_MODEL_VRAM_COST` for batch sizing.
- **Add a new engine family** (e.g., NeMo Canary): write a sibling to the `run_model` loop that handles the engine's API; share the metrics infrastructure (`ClipResult`, `ModelResult`, `render_markdown`).
- **Tests**: none yet (v0.2 plan)
- **Linting**: none yet — follow the style already in `asr_bench.py`

## Hard rules

- **No bundled audio in the repo.** `test-corpus/*` is gitignored. Distributing sample audio creates licensing headaches.
- **WER labels reflect the reference.** Output explicitly labels gold vs proxy in the headline. Never silently pass off proxy WER as accuracy.
- **VAD on by default**, but always toggleable via `--no-vad-filter`.
- **Local engines only in v0.1**. No cloud API comparisons until at least v0.5.
- **CLI only.** No GUI. Audience is technical users + faculty IT staff.

## Related projects

- **canvas-toolchain** (https://github.com/Ryfter/canvas-toolchain) — sibling project. Its `compare_transcripts` workflow is the production consumer of whichever engine asr-bench surfaces as the winner. The TranscriptionEngine swap-in pattern in canvas-toolchain mirrors asr-bench's `MODELS` registry — keeping the contract compatible would let asr-bench-recommended engines drop straight into the canvas-toolchain workflow.

## Decision log

- **2026-05-30** — Split from canvas-toolchain as its own private repo. Audience broader than Canvas LMS.
- **2026-05-30** — v0.1 ships Whisper-only via faster-whisper. WhisperX deferred (pyannote auth complexity), Canary-Qwen deferred (NeMo's heavy install needs its own venv discipline). Better narrow + working than broad + broken.
- **2026-05-30** — CLI + markdown only. No GUI.
- **2026-05-30** — VAD filter on by default after observing the Whisper-Large 1-second-cue decoder lock on Week 14 Wednesday.
- **2026-05-30** — Batch size defaults to `auto` (NVML-probed) for non-CPU runs. Improves GPU utilization 50% → 80%+.
