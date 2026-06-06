# asr-bench

A command-line benchmark for comparing automatic speech recognition (ASR) models on your own audio.

Built for professors, researchers, and accessibility teams who need to pick an
ASR engine for their actual content — not whatever was on a public leaderboard
last year. Outputs markdown tables you can paste into a doc.

## Summary

- **What it is** — a single-file Python CLI that benchmarks local speech-to-text engines on *your* audio and *your* hardware, writing one markdown report per run (WER, speed/RTFx, wall-clock, peak VRAM, disk size).
- **Why** — model rankings shift with your content and your GPU. Measure them yourself instead of trusting a public leaderboard.
- **Engines** — four Whisper variants via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (the stable core), plus an **experimental** NVIDIA NIM engine (Riva gRPC) that's implemented but not yet validated against a live server. See [NVIDIA NIM engine (optional)](#nvidia-nim-engine-optional).
- **Headline result** — on a 12-lecture, single-speaker corpus (RTX 5090), **Large V3 Turbo won on both accuracy (8.9% WER) and speed (~65× realtime)** — unusual, and exactly the kind of per-corpus surprise this tool surfaces. See [Example output](#example-output).
- **Bottom line** — run it on your own audio, or just read the numbers below to get a feel for the accuracy/speed/VRAM tradeoffs.
- **License** — [MIT](#license). Use it freely; keep the copyright notice.

## Status

**v0.3 — feat/whisperx-diarization branch. Local Whisper variants + optional fusion + WhisperX word alignment and speaker diarization:**
- `small` (244M, ~470MB)
- `medium` (769M, ~1.5GB)
- `large-v3` (1550M, ~3.1GB)
- `large-v3-turbo` (809M, ~1.6GB)
- `<size>+whisperx` — any of the above paired with WhisperX alignment + diarization (e.g. `large-v3-turbo+whisperx`)

New in v0.2: MER/WIL metrics, per-clip S/D/I counts, `--show-alignment`, and the
`--fuse` post-processing stage (verbatim captions + RAG knowledge base). See
[What it measures](#what-it-measures) and [Fusion](#fusion-optional) below.

New in v0.3: WhisperX word-level alignment, speaker-labeled VTT, DER scoring, and
word-timestamp sidecars. See [WhisperX (word alignment + diarization)](#whisperx-word-alignment--diarization-optional) below.

See [SPEC.md](./SPEC.md) for the full roadmap (Canary-Qwen + NVIDIA NeMo,
multi-language coverage, hand-corrected reference sets).

## What it measures

| Metric | What it means |
|---|---|
| WER% | Word Error Rate vs your reference transcript |
| MER% | Match Error Rate — fraction of reference+hypothesis words that are errors (bounded [0,1]) |
| WIL% | Word Information Lost — information-theoretic complement to WER (bounded [0,1]) |
| S/D/I | Per-clip substitution / deletion / insertion counts |
| RTFx | Audio seconds processed per wall-clock second (higher = faster than realtime) |
| Wall clock | Total processing time |
| Peak VRAM | NVIDIA GPU memory peak during transcription (requires `nvidia-ml-py`) |
| Disk size | Model file size after first download |
| Params | Model parameter count |

WER is the load-bearing metric and **requires a reference transcript**. If your
reference is hand-corrected gold standard, the numbers are defensible. If you
use auto-generated captions (like Panopto exports) as the reference, treat the
WER as a *relative* divergence rate rather than an absolute accuracy score.

**MER and WIL** (Morris, Maier & Green 2004) sit alongside WER in every table.
Both are bounded [0, 1] and information-theoretic: WIL in particular captures
how much information was *lost* rather than just counting token-level edits,
which makes it a better proxy for comprehension quality on lecture content. Use
`--show-alignment` to print per-clip alignment diffs when you want to see
exactly what was substituted, deleted, or inserted.

## Quick start

### Install

```bash
# Python 3.10+ recommended
python -m pip install -r requirements.txt
```

The `nvidia-ml-py` dep is optional but enables peak-VRAM tracking. Without it
the VRAM column shows `n/a`. For the fusion stage, [Ollama](https://ollama.ai)
is an additional optional dependency (only needed with `--llm ollama:...`).

### Prepare your corpus

Put audio files and their reference transcripts in matching pairs under one
folder. Three layouts are recognized:

**Layout A — flat folder, name-matched pairs:**
```
test-corpus/
  lecture-week-3.mp4
  lecture-week-3.txt          # reference transcript
  lecture-week-4.mp4
  lecture-week-4.txt
```

**Layout B — Panopto export shape (auto-detected):**
```
test-corpus/
  Lecture 12_default.mp4
  Lecture 12_Captions_English (United States).txt
```

**Layout C — explicit pairing via manifest.json:**
```json
{
  "clips": [
    {"audio": "wk03.mp4", "reference": "wk03-corrected.txt"},
    {"audio": "wk04.mp4", "reference": "wk04-corrected.txt"}
  ]
}
```

Reference files can be:
- Plain text (one transcript per file)
- SRT-shaped (Panopto exports use this — timestamps + cue numbers stripped automatically)
- WebVTT (`.vtt`)

### Run

```bash
python asr_bench.py --corpus ./test-corpus
```

With explicit model selection:

```bash
python asr_bench.py --corpus ./test-corpus --models small,medium,large-v3
```

Output goes to stdout (markdown) and `./report/<timestamp>.md`.

### CPU-only run

The Whisper variants will use CUDA if available; otherwise CPU. Force CPU
explicitly with `--device cpu`. Expect large-v3 to take 5-10× the audio
duration on CPU — start it overnight.

## Example output

A real run over a 12-lecture corpus (~614 min ≈ 10.2 hours of single-speaker lecture
audio) on an NVIDIA RTX 5090, default settings. The reference here was auto-generated
captions, so these WER numbers are *relative divergence between engines*, not absolute
accuracy — and lecture names are anonymized.

**You don't have to run a full benchmark yourself to get a feel for the tradeoffs** —
here's what these four models did on that ~10 hours of audio:

| Model | Disk | Overall WER% | Speed (RTFx) | Total time | Peak VRAM |
|---|---|---|---|---|---|
| Whisper Small | 1.4 GB | 10.7 | 43.5× | 14.1 min | 372 MB |
| Whisper Medium | 1.4 GB | 11.8 | 29.1× | 21.1 min | 269 MB |
| Whisper Large V3 | 8.6 GB | 14.2 | 14.7× | 41.9 min | 1.2 GB |
| **Whisper Large V3 Turbo** | 1.5 GB | **8.9** | **64.8×** | **9.5 min** | 168 MB |

*Total time* is wall-clock to transcribe the entire 10-hour corpus. *RTFx* (real-time
factor) is the hardware-portable way to estimate your own runtime:
**time ≈ audio length ÷ RTFx**. At 64.8× a one-hour lecture takes ~55 s; at 14.7× it
takes ~4 min. A faster GPU pushes RTFx higher; CPU pushes it far lower (CPU `large-v3`
can run 5–10× *slower* than realtime). So you can either run the tool on your own
audio, or just read these numbers off the run above.

Per-lecture WER%:

| Lecture | Audio | Small | Medium | Large V3 | Large V3 Turbo |
|---|---|---|---|---|---|
| Lecture 1 | 58.4 min | 12.2 | 14.4 | 13.8 | 8.3 |
| Lecture 2 | 54.5 min | 10.2 | 9.6 | 29.7 | 8.2 |
| Lecture 3 | 54.5 min | 13.8 | 14.4 | 15.9 | 12.7 |
| Lecture 4 | 68.9 min | 9.2 | 13.1 | 15.5 | 8.3 |
| Lecture 5 | 11.4 min | 13.9 | 19.3 | 18.0 | 13.4 |
| Lecture 6 | 56.2 min | 9.3 | 9.3 | 10.2 | 8.7 |
| Lecture 7 | 64.4 min | 9.1 | 12.9 | 10.9 | 9.2 |
| Lecture 8 | 58.3 min | 12.4 | 10.4 | 9.6 | 8.7 |
| Lecture 9 | 56.1 min | 11.6 | 11.7 | 14.7 | 9.8 |
| Lecture 10 | 66.2 min | 10.8 | 6.5 | 11.6 | 5.6 |
| Lecture 11 | 6.8 min | 7.6 | 10.6 | 7.5 | 6.8 |
| Lecture 12 | 58.5 min | 8.8 | 9.8 | 13.3 | 7.2 |
| **Overall** | **614.4 min** | **10.7** | **11.8** | **14.2** | **8.9** |

On this content, **Large V3 Turbo** had both the lowest overall WER (8.9%) and the
highest throughput (~65× realtime) — unusual, since Large V3 normally has the
accuracy edge. Lecture 2's Large V3 spike (29.7%) was a decoder lockup on one
clip; it's the failure mode the default VAD filter now prevents. This is exactly
the kind of per-corpus surprise the tool exists to surface — run it on *your*
audio rather than trusting a generic leaderboard.

## WhisperX (word alignment + diarization) (optional)

WhisperX adds forced wav2vec2 word-level alignment and pyannote speaker
diarization on top of any Whisper size. Use `<size>+whisperx` model IDs:

```bash
python asr_bench.py --models large-v3-turbo+whisperx --diarize
```

What it adds compared to plain Whisper:
- **Word-level timestamps** — `<base>_Words_<Model>.json` sidecar next to each audio file.
- **Speaker-labeled VTT** — cues prefixed `SPEAKER_00: text`, `SPEAKER_01: text`, etc.
- **DER column** — Diarization Error Rate appears in the headline table whenever a
  `<base>.rttm` ground-truth sidecar is present next to the audio. Without an RTTM
  file the DER column is omitted and diarization still runs (speaker labels in VTT).
- **Speakers column** — detected speaker count alongside DER.

### Setup

WhisperX requires a Python ≤ 3.13 environment (PyTorch has no 3.14 wheels). The
convenience script sets up the venv asr-bench auto-detects:

```powershell
./setup_whisperx_venv.ps1            # default CUDA build (cu128, RTX 50xx/Blackwell)
./setup_whisperx_venv.ps1 -CudaIndex cu124   # older CUDA toolkit
./setup_whisperx_venv.ps1 -CudaIndex cpu     # CPU-only
```

Or by hand:

```bash
py -3.12 -m venv .venv-whisperx
# Install the CUDA torch build FIRST — `pip install whisperx` otherwise pulls the
# CPU-only torch wheel on Windows (torch.cuda.is_available() == False, silent CPU).
.venv-whisperx\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv-whisperx\Scripts\pip install whisperx
.venv-whisperx\Scripts\python -c "import torch; print(torch.cuda.is_available())"   # expect True
```

asr-bench auto-detects `./.venv-whisperx` and uses it as a subprocess for
WhisperX runs. To use a different path:

```bash
python asr_bench.py --models large-v3-turbo+whisperx --whisperx-python path\to\python.exe
```

If `torch` is already importable in the running interpreter (e.g. you are running
asr-bench from a 3.12 venv that already has WhisperX installed), asr-bench runs
WhisperX in-process automatically — no subprocess overhead.

### Auth (diarization only)

Speaker diarization uses the gated `pyannote/speaker-diarization-community-1`
model from HuggingFace (pyannote-audio 4.x unified on this single self-contained
repo — it bundles segmentation + embedding, so you accept just one). To use it:

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Accept the model terms at [huggingface.co/pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
3. Generate a **read** token and pass it via `--hf-token` or the `HF_TOKEN` /
   `HUGGINGFACE_TOKEN` environment variable.

> On a pyannote 3.x install, pass `--diarize-model pyannote/speaker-diarization-3.1`
> (and accept that repo + `pyannote/segmentation-3.0` instead). The runner defaults
> to community-1 for pyannote 4.x.

```bash
python asr_bench.py --models large-v3-turbo+whisperx --diarize --hf-token hf_...
```

**Missing token:** asr-bench warns and falls back to alignment-only (word timestamps
+ VTT without speaker labels) — it does not hard-fail. To skip diarization entirely:

```bash
python asr_bench.py --models large-v3-turbo+whisperx --no-diarize
```

### DER scoring

Drop a `<base>.rttm` ground-truth annotation file next to the audio file. asr-bench
picks it up automatically and adds the DER% and Speakers columns to the report.
Without an RTTM file diarization still runs but DER is not computed.

**Tip — long recordings:** pyannote tends to over-cluster on long, noisy audio
(e.g. a 2-person call estimated as 12 speakers). If you know the speaker count,
pass `--min-speakers`/`--max-speakers` — on an 82-min 2-speaker validation clip
this took the detected count 12 → 2 and DER 27.4% → 13.8%.

### Speaker count hints

```bash
python asr_bench.py --models large-v3-turbo+whisperx --diarize \
  --min-speakers 2 --max-speakers 4
```

`--min-speakers` / `--max-speakers` are hints to pyannote, not hard limits.

### JSON results sidecar

Every run also writes a machine-readable `results/<timestamp>.json` (same
timestamp as the markdown report; sibling `<output>.json` when you pass
`--output`). It mirrors the full run — run config, per-model and per-clip
metrics, transcripts, and speaker/DER data — for cross-run aggregation. NaN
values (e.g. DER on a non-diarized clip) serialize as `null`. Secrets
(`hf_token`, `nim_api_key`) are never written. Opt out with `--no-json`.

## What's in the box

```
asr-bench/
├── README.md              ← this file
├── SPEC.md                ← full roadmap including Canary-Qwen + NeMo
├── requirements.txt
├── asr_bench.py           ← the script
├── whisperx_runner.py     ← standalone subprocess for WhisperX runs
├── test-corpus/           ← bring-your-own (gitignored except README)
└── report/                ← timestamped markdown outputs (gitignored)
```

## Ground-truth strategy — read this before trusting the WER numbers

Your reference transcript IS the ground truth. The WER score is only as good as
the reference.

Best: hand-correct 5-10 short clips (10-30 sec each) covering the kinds of
audio you actually deal with — single speaker, multi-speaker, technical vocab,
accents, your idiolect on numbers and dates. One evening of labor for
defensible numbers.

Acceptable: use existing auto-generated captions (Panopto, YouTube auto-caps,
Zoom transcripts) as reference. Treat the resulting WER as a *relative
divergence rate* — engine vs your existing pipeline. Useful for "does engine X
disagree with Panopto more than engine Y?" but not "what is engine X's
absolute accuracy?"

Worst: use one ASR model's output as the reference for benchmarking another
ASR model. This produces numbers that look real but measure nothing.

## Fusion (optional)

The `--fuse` flag adds a post-benchmark pass that combines each model's VTT output
with the Panopto reference into a consensus transcript. It works by windowing the
timed cues into overlapping chunks (default 25 s windows, 5 s overlap) and asking
a pluggable LLM to fuse each window into one of two profiles:

| Profile | Output | Use for |
|---|---|---|
| `verbatim` | `<base>_Captions_Fused.vtt` | Accessibility captions; optional rescoring reference |
| `kb` | `<base>_KB_Fused.jsonl` + `.md` | RAG / knowledge-base ingestion |

`--profile both` (the default) produces both from a single chunked pass.

**Two important caveats:**
1. **Only the verbatim profile is ADA/WCAG caption-eligible.** The kb profile
   deliberately rephrases and condenses for retrieval quality — it is explicitly
   NOT compliant captions and is labeled as such in the report.
2. **`--rescore-against-fused` is agreement-biased.** The fused verbatim VTT is
   built from the same model outputs it is then used to score — it measures
   consensus, not ground truth. The report labels this table clearly.

### LLM backend

Pass `--llm <backend>` to choose how the fusion prompt is served:

- **`ollama:<model>`** (default: `ollama:qwen2.5`) — calls a locally-running
  [Ollama](https://ollama.ai) server. Fully offline, no API key. Requires
  `ollama serve` and the model pulled (`ollama pull qwen2.5`).
- **`cli:<command>`** — shells out to an authenticated frontier CLI. Uses your
  existing subscription — no asr-bench API key required. The prompt is piped on
  **stdin** by default (e.g. `cli:claude -p`); if the command contains a
  `{prompt}` token it is substituted as an **argument** instead, which some CLIs
  require (e.g. `"cli:gemini -p {prompt}"`). The executable is resolved via
  `PATH` (Windows `.cmd`/`.bat` shims included). **Caveat:** agentic CLIs like
  `gemini` reload their whole harness per invocation (often 10s–minutes per
  call), so they are impractical for bulk per-window fusion — prefer **Ollama**
  for full runs and reserve `cli:` for one-off/small fusions or a fast headless
  completion CLI.
- **`fake`** — returns deterministic stub output. No LLM required. Good for
  testing pipeline wiring before you have Ollama set up.

### Context file

Provide domain context to improve fusion quality:

```bash
# Generate a guided template
python asr_bench.py --init-context context.md

# Fill in context.md (course schedule, speaker names, jargon, glossary, …)
# Then pass it to a fusion run:
python asr_bench.py --models large-v3-turbo --fuse --context context.md
```

### Example fusion commands

```bash
# Local Ollama — both profiles, with context
python asr_bench.py --models small,medium,large-v3-turbo \
  --fuse --profile both --llm ollama:qwen2.5 --context context.md

# Frontier CLI backend — verbatim captions only (prompt on stdin)
python asr_bench.py --models large-v3-turbo \
  --fuse --profile verbatim --llm "cli:claude -p" --context context.md

# Frontier CLI that needs the prompt as an argument (e.g. gemini)
python asr_bench.py --models large-v3-turbo \
  --fuse --profile verbatim --llm "cli:gemini -p {prompt}" --context context.md

# Re-score all models against the fused verbatim reference
python asr_bench.py --models small,medium,large-v3-turbo \
  --fuse --rescore-against-fused --context context.md

# Dry-run with no LLM (FakeLLMBackend — no Ollama required)
python asr_bench.py --models small --fuse --llm fake --limit 1
```

> **Note:** Fusion is fully unit-tested via `FakeLLMBackend` but has not yet
> been validated end-to-end against a live Ollama or `cli:` backend on real
> lecture audio. Expect to tune the drift-guard threshold and review output
> quality on your first real run.

## NVIDIA NIM engine (optional)

> **Status — experimental; not yet validated against a live NIM.** asr-bench's
> core purpose is benchmarking local **Whisper** variants. The NIM engine (and
> other extra models) are a nice-to-have, not critical. The engine is
> implemented and its `riva.client` API usage is verified *statically* against
> `nvidia-riva-client` 2.26.0, and the supporting paths — audio decode, engine
> dispatch, report rendering, and graceful failure — are tested. But it has
> **not been run end-to-end against a live NIM**: the intended deployment is a
> **self-hosted NIM container running locally**, which is untested here because
> NIM ships only as a container (`nvcr.io` image, no native binary) and no
> container runtime was set up on the reference machine. The remote/hosted path
> (`--nim-api-key` / `--nim-ssl`) is implemented but likewise not fully tested.
> Expect to debug the first real run.

To benchmark a NIM ASR endpoint (e.g. a self-hosted Canary NIM):

```bash
pip install nvidia-riva-client
```

This is only needed when you request a `nim`-engine model. Example:

```bash
python asr_bench.py --models large-v3-turbo,canary-nim --nim-url localhost:50051
```

NIM rows report WER, RTFx, and wall-clock like any engine. Because the model
runs behind a gRPC service, VRAM is reported as *total* GPU memory in use
(marked `*`) rather than a per-clip delta, and disk size shows `n/a`. Use
`--models nim:<riva-model-name>` to benchmark an unregistered NIM model.

## License

MIT — see [LICENSE](./LICENSE).

## Contributing

Issues + PRs welcome. The model registry in `asr_bench.py` is the natural place
to add new engines. Each engine needs a wrapper that exposes
`transcribe(audio_path) -> str` plus static metadata.

## Acknowledgments

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 port of OpenAI Whisper
- [WhisperX](https://github.com/m-bain/whisperX) — word-level alignment + speaker diarization
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — speaker diarization model
- [jiwer](https://github.com/jitsi/jiwer) — WER / MER / WIL computation
- [nvidia-ml-py](https://pypi.org/project/nvidia-ml-py/) — GPU memory tracking
- [Ollama](https://ollama.ai) — optional local LLM backend for fusion
- Morris, Maier & Green (2004) — MER and WIL metric definitions
