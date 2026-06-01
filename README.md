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

**v0.1 — released 2026-05-30. Local Whisper variants only:**
- `small` (244M, ~470MB)
- `medium` (769M, ~1.5GB)
- `large-v3` (1550M, ~3.1GB)
- `large-v3-turbo` (809M, ~1.6GB)

See [SPEC.md](./SPEC.md) for the full roadmap (WhisperX + diarization,
Canary-Qwen + NVIDIA NeMo, multi-language coverage, hand-corrected reference
sets).

## What it measures

| Metric | What it means |
|---|---|
| WER% | Word Error Rate vs your reference transcript |
| RTFx | Audio seconds processed per wall-clock second (higher = faster than realtime) |
| Wall clock | Total processing time |
| Peak VRAM | NVIDIA GPU memory peak during transcription (requires `nvidia-ml-py3`) |
| Disk size | Model file size after first download |
| Params | Model parameter count |

WER is the load-bearing metric and **requires a reference transcript**. If your
reference is hand-corrected gold standard, the numbers are defensible. If you
use auto-generated captions (like Panopto exports) as the reference, treat the
WER as a *relative* divergence rate rather than an absolute accuracy score.

## Quick start

### Install

```bash
# Python 3.10+ recommended
python -m pip install -r requirements.txt
```

The `nvidia-ml-py3` dep is optional but enables peak-VRAM tracking. Without it
the VRAM column shows `n/a`.

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

## What's in the box

```
asr-bench/
├── README.md            ← this file
├── SPEC.md              ← full roadmap including WhisperX + Canary-Qwen
├── requirements.txt
├── asr_bench.py         ← the script
├── test-corpus/         ← bring-your-own (gitignored except README)
└── report/              ← timestamped markdown outputs (gitignored)
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
- [jiwer](https://github.com/jitsi/jiwer) — WER computation
- [nvidia-ml-py3](https://pypi.org/project/nvidia-ml-py3/) — GPU memory tracking
