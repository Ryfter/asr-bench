# asr-bench — full roadmap

This is the long-form ambition for asr-bench. README.md covers what ships through v0.3; this document covers everything intended to ship eventually.

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

### Shipped in v0.2 — metrics + fusion (2026-06-01)

Extended metrics and a post-benchmark fusion stage. Design doc: `docs/superpowers/specs/2026-06-01-metrics-and-fusion-design.md`. Plan: `docs/superpowers/plans/2026-06-01-metrics-and-fusion.md`.

**Metrics additions:**
- **MER%** (Match Error Rate) and **WIL%** (Word Information Lost) — Morris, Maier & Green 2004. Bounded [0, 1], information-theoretic. Sit alongside WER% in every per-clip and per-model table.
- **Per-clip S/D/I counts** — substitutions, deletions, insertions via `jiwer.process_words`.
- **`--show-alignment`** — per-clip alignment diffs to stdout.

**Fusion stage (`--fuse`):**
- Parses each model's VTT + Panopto reference into timed cues; windows them (default `--window 25` s / `--overlap 5` s).
- Pluggable LLM backend (`--llm`): `ollama:<model>` (default `ollama:qwen2.5`, local/offline), `cli:<command>` (frontier CLI via existing subscription), `fake` (unit tests).
- Two profiles via `--profile verbatim|kb|both` (default `both`):
  - **verbatim** → `<base>_Captions_Fused.vtt` (accessibility captions, ADA/WCAG-eligible).
  - **kb** → `<base>_KB_Fused.jsonl` + `.md` (RAG knowledge base, NOT compliant captions).
- **`--init-context [PATH]`** — guided context template (schedule, names, jargon, mishearings, glossary).
- **Drift guard** — per-window WER(fused vs base) flags hallucination/omission.
- **`--rescore-against-fused`** — re-scores all models against the verbatim fused VTT; emits a second table labeled "agreement-biased" (reference built from the models being scored).

Fusion is fully unit-tested via `FakeLLMBackend` but **not yet validated against a live Ollama or CLI backend** on real lecture audio. See `memory/fusion-llm-validation.md`.

### Shipped in v0.3 — WhisperX + diarization (2026-06-04, feat/whisperx-diarization)

[WhisperX](https://github.com/m-bain/whisperX) adds forced wav2vec2 alignment and pyannote speaker diarization. Design doc: `docs/superpowers/specs/2026-06-04-whisperx-diarization-design.md`.

**Model IDs:** `<size>+whisperx` (e.g. `small+whisperx`, `large-v3-turbo+whisperx`).

**What shipped:**
- Word-level forced alignment (wav2vec2) — `<base>_Words_<Model>.json` word-timestamp sidecar
- Speaker-labeled VTT — cues prefixed `SPEAKER_00: text`
- **DER (Diarization Error Rate)** — gated on a `<base>.rttm` ground-truth sidecar; DER% and Speakers columns appear in the headline table only when RTTM is present. Without RTTM, diarization still runs (speaker labels in VTT) but DER is not computed.
- Two execution paths auto-selected: in-process when `torch` is importable; subprocess to a Python ≤ 3.13 venv otherwise (torch has no 3.14 wheels)
- `whisperx_runner.py` — standalone subprocess script; communicates via JSON
- Auth: free HF token (`--hf-token`/`HF_TOKEN`) + accepting `pyannote/speaker-diarization-community-1` (pyannote-audio 4.x default; `--diarize-model` overrides). Missing token warns and falls back to alignment-only; `--no-diarize` skips diarization entirely.
- New flags: `--diarize`/`--no-diarize`, `--hf-token`, `--min-speakers`, `--max-speakers`, `--whisperx-python`
- 120 tests pass, 2 skipped in the core 3.14 venv (the 2 skips are the DER tests needing pyannote — they pass in the whisperx venv, 122 total); **live-validated 2026-06-05** on RTX 5090: full transcribe→align→diarize→DER (DER 13.8% @ 2-speaker hint over an 82-min 2-speaker recording)

**Not yet merged to main** — pending a live WhisperX + diarization run (no pyannote venv on reference machine yet). Branch is pushed to GitHub.

**Remaining v0.3 items (not yet implemented):**
- `pip install asr-bench` packaging + `asr-bench` CLI entry point
- CER (Character Error Rate) for noisy-word-boundary languages
- Hallucination-rate detection (cross-engine + silence detection)
- Median per-clip latency metric
- `asr_bench prepare-gold` hand-correction helper
- Speaker labels in reference sets for DER ground-truth preparation

### Planned for v0.4 — NVIDIA NeMo (Canary-Qwen and family)

[NVIDIA NeMo](https://github.com/NVIDIA/NeMo) — separate ASR stack with strong English models. **Canary-Qwen-2.5B** (2025) is competitive with Whisper Large-V3 on English, faster on NVIDIA hardware.

Why v0.4 not earlier:
- NeMo install is heavy, CUDA-pinned, benefits from its own venv.
- Different API — needs its own engine wrapper.
- Production deployments often use NVIDIA NIM (Inference Microservice) — benchmark could exercise via HTTP.

Other NeMo models to slot in: `Parakeet-CTC-1.1B`, `Parakeet-TDT`, older `Citrinet`.

### Planned for v0.5 — Conformer + community models

Stretch. Wav2vec2-large, conformer open models, distil-whisper community fine-tunes. Criteria: locally runnable, Python wrapper, timed segments output.

## Metrics — full roadmap

| Metric | Shipped | Notes |
|---|---|---|
| WER% | v0.1 | via `jiwer`. Case + punctuation normalized. |
| RTFx | v0.1 | audio_sec / wall_clock_sec. >1 = faster than realtime. |
| Wall clock | v0.1 | total transcribe time. |
| Peak VRAM | v0.1 | via `nvidia-ml-py` polling. NVIDIA only. |
| Disk size | v0.1 | model file size after first download. |
| Params | v0.1 | static metadata. |
| **MER%** | **v0.2** | Match Error Rate (Morris, Maier & Green 2004). Bounded [0,1]. |
| **WIL%** | **v0.2** | Word Information Lost (Morris, Maier & Green 2004). Bounded [0,1]. |
| **Per-clip S/D/I** | **v0.2** | Substitutions/deletions/insertions per clip via `jiwer.process_words`. |
| **Fusion drift** | **v0.2** | Per-window WER(fused vs base) as a hallucination/omission guard. |
| **DER** | **v0.3** | Diarization Error Rate via pyannote.metrics. Gated on `<base>.rttm` sidecar. |
| **Speakers** | **v0.3** | Detected speaker count from pyannote diarization. |
| **CER** | v0.3+ | for languages with noisy word boundaries. Not yet implemented. |
| **Hallucination rate** | v0.3+ | engines invent text on silence/music; detect via cross-engine + silence detection. Not yet implemented. |
| **Median per-clip latency** | v0.3+ | for batch-processing decisions. Not yet implemented. |
| **CPU watts/hour** | v0.4 | if power monitor available (Intel RAPL, asitop). |

## Ground-truth strategy

Three reference-set classes:

1. **Gold standard (hand-corrected).** Defensible absolute WER. Requires labor.
2. **Proxy (auto-caption derived).** Use Panopto/YouTube/Zoom captions. Cheap. Surfaces *relative* divergence. Output labels as `WER (proxy)` so it's never mistaken for gold.
3. **Public benchmark sets.** LibriSpeech test-clean, CommonVoice English, TED-LIUM 3. General ranking, not domain-specific.

Future v0.3+: `asr_bench prepare-gold ./test-corpus` — walks user through hand-correcting Whisper output line-by-line to build incremental gold reference. The v0.2 fusion verbatim VTT can serve as a better starting point for hand-correction than raw ASR output.

## Output roadmap

- v0.1: markdown table per run + per-clip detail. Stdout + `./results/<timestamp>.md`.
- v0.2: MER/WIL/S/D/I columns; `_Captions_Fused.vtt` and `_KB_Fused.jsonl`/`.md` from fusion stage.
- v0.3: Speaker-labeled VTT (`SPEAKER_XX: text` cues) + `_Words_<Model>.json` word-timestamp sidecar from WhisperX runs. JSON results sidecar (`results/<timestamp>.json`, `schema_version: 1`) for cross-run aggregation — shipped.
- v0.4: `asr_bench compare` subcommand — delta report between N result JSON files (not yet built; schema is compare-ready).

## Corpus structure roadmap

v0.1: three layouts (flat name-matched, Panopto export, manifest.json).

v0.2 adds:
- Context file (`--context`, `--init-context`) for domain jargon, speaker names, and fusion prompting.
- Fused VTT and JSONL/MD outputs sit alongside source audio.

v0.3 adds:
- `<base>.rttm` sidecar auto-detection for DER ground truth (shipped).
- Speaker labels in VTT output (shipped).
- Per-clip metadata (recording conditions, speaker counts, audio quality notes) — not yet implemented.
- Test/train splits (`asr_bench eval --split test`) — not yet implemented.

## Distribution roadmap

- v0.1: public GitHub repo, `python asr_bench.py ...` entry.
- v0.2: same single-file entry; optional `ollama` dependency for fusion.
- v0.3: pip-installable (`pip install asr-bench`), `asr-bench` CLI — not yet implemented (deferred to v0.3+).
- v0.4: prebuilt Windows/macOS binaries via PyInstaller. Audience: faculty IT staff making accessibility purchasing decisions.

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
- **2026-05-31** — Added NVIDIA NIM ASR (Riva gRPC) as the second engine family, ahead of its v0.3 roadmap slot. Implemented and *statically* verified against `nvidia-riva-client` 2.26.0, but **not yet run against a live NIM** (see the ship-as-is entry below). Stays within the "local engines only" rule: a self-hosted NIM is local inference behind a gRPC port, not a cloud ASR API. The `--nim-url` flag *permits* a hosted endpoint, but defaults are local. Introduced the `Engine` contract (`FasterWhisperEngine` + `NimEngine`) in-file; deferred the `engines/` package split until WhisperX/NeMo land.
- **2026-05-31** — Shipped the NIM engine **as-is, without a live-NIM test run**. asr-bench's core purpose is benchmarking local Whisper variants; NIM and other extra engines are nice-to-have, not critical. Live validation deferred: the reference box has no container runtime (no Docker, no WSL distro) and NIM is **container-only** (`nvcr.io` image, not a native binary). Tested: decode, engine dispatch, mixed-engine rendering, graceful failure. Untested: local self-hosted Docker path; the remote/hosted (`--nim-api-key`/`--nim-ssl`) path is implemented but not fully tested. Intent remains local self-hosted NIM.
- **2026-06-01** — Adopted MER/WIL (Morris, Maier & Green 2004) as bounded, information-theoretic companions to WER. Design doc: `docs/superpowers/specs/2026-06-01-metrics-and-fusion-design.md`.
- **2026-06-01** — Fusion is a post-processing pass, not a new ASR engine. Keeps benchmarking and post-processing concerns cleanly separated.
- **2026-06-01** — Two fusion profiles (verbatim + kb) share one windowed pipeline. Only verbatim is ADA/WCAG caption-eligible; kb rephrases for retrieval quality and is labeled accordingly.
- **2026-06-01** — LLM backend defaults to local Ollama (`qwen2.5`). The `cli:` backend is the escape hatch for frontier models without requiring an asr-bench-managed API key.
- **2026-06-01** — `--rescore-against-fused` table is agreement-biased (reference built from the same models being scored) and prominently labeled as such.
- **2026-06-04** — Added WhisperX as the third engine family. `<size>+whisperx` IDs pair any Whisper size with WhisperX alignment + diarization so users see a direct speed/accuracy/DER comparison in one report. Design doc: `docs/superpowers/specs/2026-06-04-whisperx-diarization-design.md`.
- **2026-06-04** — In-process vs subprocess auto-detection: when `torch` is importable in the running interpreter (a 3.12 venv that has WhisperX), asr-bench runs WhisperX in-process. Otherwise it spawns `whisperx_runner.py` in a separate venv. Rationale: torch has no Python 3.14 wheels and asr-bench's core runs on 3.14; the subprocess bridge avoids forcing users to install asr-bench itself into a 3.12 venv.
- **2026-06-04** — DER gated on RTTM sidecar. Ground-truth speaker boundaries are labor-intensive; not every run has them. Diarization still runs (and labels VTT cues) without RTTM — only the DER% and Speakers columns are suppressed. This keeps the default useful without requiring annotation work.
- **2026-06-04** — `engines/` package split deferred again. `whisperx_runner.py` is the only file broken out (forced by the venv/Python version boundary). Full split waits for NeMo/Canary-Qwen where the weight of three Engine subclasses in `asr_bench.py` becomes unwieldy.
