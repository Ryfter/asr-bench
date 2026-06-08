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

**v0.3.5 items (shipped 2026-06-07):**
- ✅ `pip install asr-bench` packaging + `asr-bench` CLI entry point (`pyproject.toml`, flat `py-modules`, GPU/NIM extras)
- ✅ `asr-bench prepare-gold` helper — converts VTT/SRT captions → plain `.txt` references (timing stripped); proxy sources stay flagged
- ✅ `compare` surfaces `Halluc%`; headline gains `s/aud-min (med)`; markdown pipe-escaping in table cells

**Remaining v0.3-era item (not yet implemented):**
- Speaker labels in reference sets for DER ground-truth preparation

### v0.4 — NVIDIA NeMo (Canary-Qwen + Parakeet) — **live-validated on `feat/v0.4-nemo`, ready to merge**

> **Status: live-validated on the RTX 5090 (2026-06-07); NOT yet merged to main.** Both models run end-to-end through the full asr_bench pipeline. Validated version triple: **Python 3.12 · torch 2.11.0+cu128 · nemo_toolkit 2.7.3**. Peak VRAM: **Parakeet ~8.3 GB, Canary-Qwen ~9.8 GB**. First-run results (6.8-min lecture, proxy ref): Parakeet WER 6.2% / RTFx 201.8× / VTT written; Canary-Qwen WER 7.2% / RTFx 5.57× / WER-only; alongside Whisper Large V3 Turbo WER 6.7% / RTFx 111×. 263 tests pass, 2 skipped. Four live-run fixes (PS-script ASCII, pyarrow pre-import segfault, SALM `audio_lens` API, ffmpeg mp4 decode) — see decision log. Spec: `docs/superpowers/specs/2026-06-08-v0.4-nemo-engine-design.md`; plan: `docs/superpowers/plans/2026-06-08-v0.4-nemo-engine.md`.

[NVIDIA NeMo](https://github.com/NVIDIA/NeMo) — separate ASR stack with strong English models, added as the fourth `Engine` family (`NeMoEngine` + standalone `nemo_runner.py`).

**Model IDs (registered):**
- `parakeet-tdt-0.6b-v2` (Parakeet TDT 0.6B v2) — native word/segment timestamps → full VTT + `_Words_*.json`.
- `canary-qwen-2.5b` (Canary-Qwen 2.5B) — best-in-class English WER, **WER-only / no VTT** (no native timestamps; text-only row like NIM's `n/a`). Needs 40 s non-overlapping chunked inference (no native long-form); raised `max_new_tokens` per chunk.
- `nemo:<model>` ad-hoc IDs (mirrors `nim:<name>`).

**What's implemented:**
- Runs as a **subprocess** into a dedicated **Python 3.12 `.venv-nemo`** (torch has no 3.14 wheels) via `nemo_runner.py` — pure-JSON stdout, dual `stdout`→`stderr` redirect mirroring `whisperx_runner.py`. Provisioned by `setup_nemo_venv.ps1` (cu128 torch FIRST, then `nemo_toolkit[asr]`, verify CUDA). NeMo gets its **OWN** venv (not shared with `.venv-whisperx`) due to aggressive pins (numpy>=2.0, transformers, lightning).
- **Transcription-only** — no diarization/DER (NeMo diarization is a separate model family, out of scope for v0.4).
- Both venv installers are **optional & independent** — a Whisper/NIM run never needs `.venv-nemo`; a NeMo model with no venv is **skipped (warning), not a crash** (graceful pre-flight in `main()`).
- New CLI flag `--nemo-python` (auto-detects `./.venv-nemo`); `nemo_python` in the sidecar config (non-secret). Additive only — **`schema_version` stays 1**; `render_markdown` + JSON sidecar byte-stable; core stays torch-free.

Why v0.4 not earlier:
- NeMo install is heavy, CUDA-pinned, benefits from its own venv.
- Different API — needs its own engine wrapper.
- Production deployments often use NVIDIA NIM (Inference Microservice) — benchmark could exercise via HTTP.

Other NeMo models to slot in later: `Parakeet-CTC-1.1B`, older `Citrinet`.

**Still deferred (separate later milestone):** PyInstaller prebuilt binaries.

### v0.5 — Conformer + community models (CODE-COMPLETE on `feat/v0.5-community-models`; live-validation PENDING)

The `engines/` package split landed as **Phase 0** of v0.5 (`engines/base.py` holds the shared types + writers; each engine family is its own module under `engines/`; `asr_bench` re-exports for a byte-stable public surface). Phase 1 + 2 then add three community models, criteria met (locally runnable, Python wrapper, timed segments output):

- **`distil-large-v3.5`** (Phase 1) — Distil-Whisper Large V3.5, 756M, English. Runs through the **existing faster-whisper engine** (CT2 id `distil-large-v3.5`, resolved to `distil-whisper/distil-large-v3.5-ct2`); registry-only addition (`MODELS` + `_MODEL_VRAM_COST` sized like turbo).
- **`wav2vec2-large-960h`** + **`wav2vec2-conformer-large`** (Phase 2) — a new **HuggingFace-transformers engine** (`HFTransformersEngine` + `hf_runner.py`) running CTC models via the transformers ASR pipeline. Word-level timestamps (`return_timestamps="word"`) -> full VTT + `_Words_*.json`. Plus `hf:<model>` ad-hoc ids.
- Runs as a **subprocess into a dedicated `.venv-hf`** (Python 3.12; torch has no 3.14 wheels) via pure-JSON stdout, mirroring the NeMo engine. Own venv (not shared with `.venv-nemo`/`.venv-whisperx`) to avoid pin collisions. `--hf-python` overrides; `hf_python` is a non-secret sidecar config field. Graceful pre-flight skip when no venv is present (other engines still run). Additive only — `schema_version` stays 1, core stays torch-free.

**Live-validation gate (PENDING, RTX 5090):** the `run_hf` path (transformers pipeline, real CTC timestamps, CUDA) is exercised only on the GPU box; not yet merged to main. Other NeMo models to slot in later still apply.

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
| **CER%** | **v0.3** | Character Error Rate (jiwer, same normalized text as WER). Per-clip in all report tables and in `compare`. Additive sidecar field `cer` (per-clip) + `avg_cer` (aggregate); schema_version stays 1. |
| **RTFx (med)** | **v0.3** | Median per-clip RTFx — robust to a single slow/locked clip that skews the totals-based aggregate. Sidecar fields `median_rtfx` + `median_sec_per_audio_min`. |
| **Hallucination rate** | **v0.3** | Reference-free: `repeat_coverage` (repeated-4-gram fraction) + `compression_ratio` (gzip ratio — Whisper's own signal). Flag = coverage > 0.30 OR compression > 2.4. Per-model `hallucination_rate`; per-clip fields in JSON sidecar; dedicated "⚠️ Hallucination signals" report section. Works without a reference and in single-model runs. |
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
- v0.3 (shipped): `asr_bench compare` subcommand — delta (2 files) or matrix (3+) markdown report across N result JSON files. Joins per-model aggregates on `model_id`; warns on corpus/config drift. `--last N`, `--per-clip`, `--delta`/`--matrix`, `--output`. Implemented as `asr_compare.py` with first-positional `compare` keyword pre-dispatch in `asr_bench.py`.
- v0.3 (shipped): **CER% + median latency** — per-clip `CER%` (Character Error Rate, jiwer) in all report tables and `compare`; per-model `RTFx (med)` (median per-clip RTFx) in the headline as an outlier-robust counterpart to the totals-based aggregate. JSON sidecar gains `clips[].cer`, `aggregates.avg_cer`, `aggregates.median_rtfx`, `aggregates.median_sec_per_audio_min`; additive within `schema_version 1`.
- v0.3 (shipped): **Hallucination signals** — "⚠️ Hallucination signals" report section lists flagged (model, clip) pairs with `repeat_coverage`, `compression_ratio`, and an insertion-rate annotation when a reference is available. Per-model `hallucination_rate` line per model. JSON sidecar gains `clips[].repeat_coverage`, `clips[].compression_ratio`, `clips[].hallucination_suspect`, `aggregates.hallucination_rate`; additive within `schema_version 1`.

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
- v0.3.5 (shipped): pip-installable (`pip install asr-bench`), `asr-bench` CLI (flat `py-modules`, GPU/NIM extras).
- v0.4+ (deferred — NOT part of the v0.4 NeMo branch): prebuilt Windows/macOS binaries via PyInstaller, plus the `engines/` package split. Audience: faculty IT staff making accessibility purchasing decisions.

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
- **2026-06-06** — `compare` subcommand shipped: reads 2+ JSON sidecars, delta
  (2) / matrix (3+) markdown, joins per-model aggregates on model_id, warns on
  corpus/config drift. Implemented as a first-positional `compare` keyword
  pre-dispatch into a standalone, pure `asr_compare.py` — the existing bench CLI
  is byte-for-byte unchanged. Per-model DER averaged from per-clip `der`.
- **2026-06-06** — Shipped CER (char-level, via jiwer on the same normalized text
  as WER, added to WordMetrics so all three engines get it from one call) and a
  robust median speed pair (`median_rtfx`, `median_sec_per_audio_min`) beside the
  totals-based `aggregate_rtfx`. Sidecar fields additive within schema_version 1;
  CER also surfaced in `compare`.
- **2026-06-07** — Shipped hallucination detection: reference-free repeat-coverage
  (repeated 4-grams) + gzip compression ratio (Whisper's own internal signal),
  flag = coverage>0.30 OR compression>2.4, per-model hallucination_rate, dedicated
  report section + additive sidecar fields (schema_version stays 1). Detection
  only; compare integration deferred.
- **2026-06-07** — NIM transport policy made explicit (no code change — confirming
  intent): the NIM engine supports **two transports, local self-hosted preferred
  (default) and remote hosted NVCF a flag-gated fallback**. Local
  (`--nim-url localhost:<port>`) stays within "local engines only" (local inference
  behind a gRPC port); remote (`--nim-url <host>:443 --nim-api-key <key> --nim-ssl`)
  exists only for users without a local runtime and is never the default. Both
  implemented since 2026-05-31; neither live-validated (pending,
  `memory/validate-live-nim.md`), local-first when Docker Desktop + NGC are ready.
- **2026-06-07** — `AGENTS.md` (Codex handoff) is kept as a **faithful mirror of
  `CLAUDE.md`** — same body, differing only in the tool-name intro + a sync note.
  Rationale: a stale handoff (AGENTS.md had drifted to v0.1) is worse than none;
  mirroring guarantees Codex/Gemini/Claude start from identical project state.
  Substantive changes to one must be propagated to the other. Reference PDFs and
  the local `.claude/` tooling dir are gitignored (copyright / local-only).
- **2026-06-07 (v0.3.5)** — Shipped a polish batch (branch `feat/v0.3.5`, merged
  to main). **(B1)** packaging via `pyproject.toml` with an `asr-bench` console
  entry point — chose a **flat `py-modules` layout** over a package dir so
  `python asr_bench.py` and `import asr_bench` stay byte-for-byte unchanged; core
  deps torch-free (`faster-whisper`, `jiwer`), GPU/NIM as opt-in extras, WhisperX
  excluded (no 3.14 wheels). **(B2)** `prepare-gold` converts VTT/SRT → plain
  `.txt` references (user chose the format-converter shape over a model-bootstrapped
  draft). Correctness call: `load_reference_text` strips the `[Auto-generated
  transcript]` header, so a naive convert would launder a proxy into apparent
  gold — proxy sources keep that marker re-prepended (stripped at scoring, still
  flagged proxy); asr-bench's own `_Captions_*.vtt` outputs are excluded to avoid
  circular references. **(C1)** `Halluc%` in `compare`, gated on presence like DER.
  **(C2)** `s/aud-min (med)` headline column (the sidecar's `median_sec_per_audio_min`
  finally has a report consumer). **(C3)** `_md_escape` for pipe/newline-safe table
  cells. All additive; `schema_version` stays 1. 229 tests pass.
- **2026-06-08 (v0.4 — code-complete on `feat/v0.4-nemo`, NOT shipped/merged,
  live validation pending)** — Added **NVIDIA NeMo** as the fourth `Engine` family
  (`NeMoEngine` + standalone `nemo_runner.py`). Spec:
  `docs/superpowers/specs/2026-06-08-v0.4-nemo-engine-design.md`; plan:
  `docs/superpowers/plans/2026-06-08-v0.4-nemo-engine.md`. Six locked decisions:
  **(1)** register **both** `parakeet-tdt-0.6b-v2` and `canary-qwen-2.5b` (user
  wants a direct Parakeet-vs-Canary comparison in one report). **(2)** **Canary-Qwen
  is WER-only (no VTT)** — no native timestamps and asr-bench is a *benchmark*, so a
  text-only row (like NIM's `n/a`) is the honest shape; Parakeet has native
  word/segment timestamps → full VTT + `_Words_*.json`. **(3)** NeMo gets its **own
  dedicated `.venv-nemo`** (Python 3.12, not shared with `.venv-whisperx`) because
  its pins (numpy>=2.0, transformers, lightning) conflict; subprocess (torch has no
  3.14 wheels) via `nemo_runner.py` with pure-JSON stdout + dual `stdout`→`stderr`
  redirect, mirroring `whisperx_runner.py`. **(4)** **Transcription-only** — no
  diarization/DER (separate NeMo model family, out of scope). **(5)** Both venv
  installers (`setup_nemo_venv.ps1`, `setup_whisperx_venv.ps1`) are **optional &
  independent** — a Whisper/NIM run never needs `.venv-nemo`; a NeMo model with no
  venv is **skipped with a warning, not a crash** (graceful pre-flight in `main()`).
  **(6)** Support both **registered IDs and `nemo:<model>` ad-hoc IDs** (mirrors
  `nim:<name>`). New CLI flag `--nemo-python` (auto-detects `./.venv-nemo`);
  `nemo_python` in the sidecar config (non-secret). Additive only — `schema_version`
  stays 1, `render_markdown` + JSON sidecar byte-stable, core stays torch-free.
  **NeMo was added in-file; the `engines/` package split + PyInstaller binaries
  remain deferred** to separate later work. Status: implemented + unit-tested, then
  **live-validated on the RTX 5090 (2026-06-07)** — see the 2026-06-07 entry below.

- **2026-06-07 (v0.4 — LIVE-VALIDATED on RTX 5090, ready to merge)** — Ran the full
  setup + benchmark on the reference box. Validated version triple: **Python 3.12 ·
  torch 2.11.0+cu128 · nemo_toolkit 2.7.3**. Peak VRAM: **Parakeet ~8.3 GB,
  Canary-Qwen ~9.8 GB**. First-real-run results on a 6.8-min lecture (proxy
  reference): **Parakeet WER 6.2% / RTFx 201.8× / VTT written** (won this lecture on
  both WER and speed), **Canary-Qwen WER 7.2% / RTFx 5.57× / WER-only (no VTT)**, both
  alongside Whisper Large V3 Turbo (WER 6.7% / RTFx 111×) in one report + JSON
  sidecar. **Four bugs the live run forced (none caught by unit tests — exactly why
  the live gate exists):** **(a)** both `setup_*_venv.ps1` were UTF-8-no-BOM with
  em-dashes; under Windows PowerShell 5.1 the em-dash's `0x94` byte decodes as a curly
  quote PS treats as a string delimiter → cascading parse errors. Fixed to pure ASCII
  (`e9013d3`). **(b)** `import nemo.collections.asr` **segfaults (0xC0000005)** on
  Windows — NeMo's chain (sklearn→pandas→pyarrow) loads pyarrow's native libs after a
  conflicting native dep; pre-importing pyarrow at the top of `run_nemo` loads it
  cleanly first (guarded; keeps the module import torch-free *and* pyarrow-free for
  core-venv tests) (`cbcbc38`). **(c)** SALM (Canary) `generate()` needs a
  **torch.Tensor float32 (B, T)** + **`audio_lens`** (int64) — was passing numpy +
  `audio_lengths=` which silently fell into `**generation_kwargs`, leaving
  `audio_lens=None` and tripping perception's validation (`cbcbc38`). **(d)** NeMo/lhotse
  decode via **`torchaudio.io`, removed in torchaudio ≥ 2.11** → can't open mp4;
  `nemo_runner` now decodes any non-wav input to a temp 16 kHz mono WAV via the ffmpeg
  CLI up front, mirroring WhisperX's self-contained decode (`9696e8d`). 263 tests pass,
  2 skipped. Merge to main now unblocked.
