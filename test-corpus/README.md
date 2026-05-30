# test-corpus

Drop your audio + reference transcripts in here.

## Layouts the runner recognizes

### A — flat, name-matched pairs
```
test-corpus/
  week-3.mp4
  week-3.txt
```

### B — Panopto-export shape (auto-detected by `_default.mp4` + `_Captions...txt` pattern)
```
test-corpus/
  ITM310 002 Week 16 Friday_default.mp4
  ITM310 002 Week 16 Friday_Captions_English (United States).txt
```

### C — explicit pairing via manifest.json
```json
{
  "clips": [
    {"audio": "wk03.mp4", "reference": "wk03-corrected.txt"},
    {"audio": "wk04.mp4", "reference": "wk04-corrected.txt"}
  ]
}
```

## Reference transcript formats

The runner strips formatting before WER:

- **Plain text** — paragraphs of words. Best for hand-corrected gold.
- **SRT** — cue numbers, `HH:MM:SS,mmm --> HH:MM:SS,mmm`, then text. Panopto exports use this.
- **WebVTT** — same as SRT but with `WEBVTT` header and `.` decimal separator.

## A note on what to put in here

This folder is gitignored by default. The benchmark runs against YOUR audio +
YOUR reference. Bundling sample audio in the repo creates licensing headaches.
If you want public samples, see the v0.2 roadmap in `../SPEC.md` for using
LibriSpeech test-clean.
