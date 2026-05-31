# asr-bench — full roadmap

This is the long-form ambition for asr-bench. README.md covers what ships in v0.1; this document covers everything intended to ship eventually.

## Goals

1. Reproducibly compare local ASR engines on your own audio + hardware.
2. Quantify both *accuracy* (WER) and *cost* (RTFx, VRAM, disk).
3. Cover single-speaker dictation/lecture AND multi-speaker conversational content (needs diarization).
4. Single markdown file output, paste-able anywhere.

## Non-goals

- A GUI. CLI tool for technically literate users.
- Cloud ASR comparisons (Google/AWS/Azure). v0.1 is local-only.
- Real-time streaming. Complete audio files only.
- Anything beyond ASR (no summarization, topic extraction, etc).

## Engines — full roadmap

### Shipped in v0.1 (Whisper family, faster-whisper backend)

| Engine | Variant | Notes |
|---|---|---|
| Whisper Small | `small` | 244M, ~470MB. Real-time on CPU. |
| Whisper Medium | `medium` | 769M, ~1.5GB. Production sweet spot. |
| Whisper Large V3 | `large-v3` | 1550M, ~3.1GB. State-of-art OpenAI accuracy. |
| Whisper Large V3 Turbo | `large-v3-turbo` | 809M, ~1.6GB. Distilled large-v3 — fast as medium, accuracy near large. |

All four run via [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper).

### Planned for v0.2 — WhisperX + diarization

[WhisperX](https://github.com/m-bain/whisperX) adds forced alignment (wav2vec2) and speaker diarization (pyannote.audio).

Blockers for immediate inclusion:
- `pyannote.audio` 3.x requires HuggingFace-gated diarization model + auth token (free).
- New metric **DER (Diarization Error Rate)** needs ground-truth speaker boundaries.

Once shipped, the runner pairs each Whisper size with WhisperX as a wrapper (`small+whisperx`, etc.) and emits DER only when diarization ground truth is present.

### Planned for v0.3 — NVIDIA NeMo (Canary-Qwen and family)

[NVIDIA NeMo](https://github.com/NVIDIA/NeMo) — separate ASR stack with strong English models. **Canary-Qwen-2.5B** (2025) is competitive with Whisper Large-V3 on English, faster on NVIDIA hardware.

Why v0.3 not v0.2:
- NeMo install is heavy, CUDA-pinned, benefits from its own venv.
- Different API — needs its own engine wrapper.
- Production deployments often use NVIDIA NIM (Inference Microservice) — benchmark could exercise via HTTP.

Other NeMo models to slot in: `Parakeet-CTC-1.1B`, `Parakeet-TDT`, older `Citrinet`.

### Planned for v0.4 — Conformer + community models

Stretch. Wav2vec2-large, conformer open models, distil-whisper community fine-tunes. Criteria: locally runnable, Python wrapper, timed segments output.

## Metrics — full roadmap

| Metric | Shipped | Notes |
|---|---|---|
| WER% | v0.1 | via `jiwer`. Case + punctuation normalized. |
| RTFx | v0.1 | audio_sec / wall_clock_sec. >1 = faster than realtime. |
| Wall clock | v0.1 | total transcribe time. |
| Peak VRAM | v0.1 | via `nvidia-ml-py3` polling. NVIDIA only. |
| Disk size | v0.1 | model file size after first download. |
| Params | v0.1 | static metadata. |
| **DER** | v0.2 | needs WhisperX + ground-truth speaker boundaries. |
| **CER** | v0.2 | for languages with noisy word boundaries. |
| **Hallucination rate** | v0.2 | engines invent text on silence/music; detect via cross-engine + silence detection. |
| **Median per-clip latency** | v0.2 | for batch-processing decisions. |
| **CPU watts/hour** | v0.3 | if power monitor available (Intel RAPL, asitop). |

## Ground-truth strategy

Three reference-set classes:

1. **Gold standard (hand-corrected).** Defensible absolute WER. Requires labor.
2. **Proxy (auto-caption derived).** Use Panopto/YouTube/Zoom captions. Cheap. Surfaces *relative* divergence. Output labels as `WER (proxy)` so it's never mistaken for gold.
3. **Public benchmark sets.** LibriSpeech test-clean, CommonVoice English, TED-LIUM 3. General ranking, not domain-specific.

Future v0.2: `asr_bench prepare-gold ./test-corpus` — walks user through hand-correcting Whisper output line-by-line to build incremental gold reference.

## Output roadmap

- v0.1: markdown table per run + per-clip detail. Stdout + `./results/<timestamp>.md`.
- v0.2: JSON sidecar (`results/<timestamp>.json`) for cross-run aggregation.
- v0.3: `asr_bench compare` subcommand — delta report between N result files.

## Corpus structure roadmap

v0.1: three layouts (flat name-matched, Panopto export, manifest.json).

v0.2 adds:
- Speaker labels in reference (DER scoring).
- Per-clip metadata (recording conditions, speaker counts, audio quality notes).
- Test/train splits (`asr_bench eval --split test`).

## Distribution roadmap

- v0.1: public GitHub repo, `python asr_bench.py ...` entry.
- v0.2: pip-installable (`pip install asr-bench`), `asr-bench` CLI.
- v0.3: prebuilt Windows/macOS binaries via PyInstaller. Audience: faculty IT staff making accessibility purchasing decisions.

## Anti-goals

- Cloud API comparisons. Local only.
- Streaming / real-time. Static files only.
- Synthetic TTS-generated audio. Real recordings only.
- Inference cost in dollars. Too hardware-dependent.
- Bundled audio dataset. License compliance is painful; users bring their own.

## Decision log

- **2026-05-30** — Split from canvas-toolchain as its own repo. Audience broader than Canvas LMS. canvas-toolchain's `compare_transcripts` workflow is the application; asr-bench is the engine-comparison tool that informs which model to plug in.
- **2026-05-30** — v0.1 ships Whisper-only. WhisperX deferred (pyannote auth complexity); Canary-Qwen deferred (needs own venv discipline). Better to ship a working narrow tool than a broken broad one.
- **2026-05-30** — CLI only, markdown output. No GUI. Audience is technical.
- **2026-05-31** — Added NVIDIA NIM ASR (Riva gRPC) as the second engine family, ahead of its v0.3 roadmap slot, validated against a self-hosted Canary NIM. Stays within the "local engines only" rule: a self-hosted NIM is local inference behind a gRPC port, not a cloud ASR API. The `--nim-url` flag *permits* a hosted endpoint, but defaults and validation are local. Introduced the `Engine` contract (`FasterWhisperEngine` + `NimEngine`) in-file; deferred the `engines/` package split until WhisperX/NeMo land.
