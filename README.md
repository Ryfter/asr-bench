# asr-bench

A command-line benchmark for comparing speech recognition models on your own audio.

Built for professors, researchers, and accessibility teams who need to pick an
ASR engine for their actual content — not whatever was on a public leaderboard
last year. Outputs markdown tables you can paste into a doc.

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
  ITM310 002 Week 16 Friday_default.mp4
  ITM310 002 Week 16 Friday_Captions_English (United States).txt
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

Output goes to stdout (markdown) and `./results/<timestamp>.md`.

### CPU-only run

The Whisper variants will use CUDA if available; otherwise CPU. Force CPU
explicitly with `--device cpu`. Expect large-v3 to take 5-10× the audio
duration on CPU — start it overnight.

## What's in the box

```
asr-bench/
├── README.md            ← this file
├── SPEC.md              ← full roadmap including WhisperX + Canary-Qwen
├── requirements.txt
├── asr_bench.py         ← the script
├── test-corpus/         ← sample (empty by default — drop your audio in)
└── results/             ← timestamped markdown outputs (gitignored)
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
