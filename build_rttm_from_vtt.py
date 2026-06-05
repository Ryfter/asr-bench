#!/usr/bin/env python
"""One-off: convert the Zoom speaker-labeled transcript VTT into an RTTM
ground-truth sidecar for DER scoring. Zoom cues look like:

    00:07:58.100 --> 00:08:03.860
    Kevin Rank - BSU: Test one... there we go.

Each cue's "Name:" prefix is the ground-truth speaker for that time span.
Speaker labels are sanitized to single tokens (RTTM field 8 is whitespace-split).
"""
import re
import sys
from pathlib import Path

TS = re.compile(
    r"(\d\d):(\d\d):(\d\d)\.(\d{3})\s*-->\s*(\d\d):(\d\d):(\d\d)\.(\d{3})"
)


def hms(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_")


def convert(vtt_path: Path, uri: str) -> str:
    lines = vtt_path.read_text(encoding="utf-8").splitlines()
    rows = []
    speakers = set()
    i = 0
    while i < len(lines):
        m = TS.search(lines[i])
        if not m:
            i += 1
            continue
        start = hms(*m.group(1, 2, 3, 4))
        end = hms(*m.group(5, 6, 7, 8))
        # next non-empty line is the "Speaker: text" cue
        j = i + 1
        text = ""
        while j < len(lines) and lines[j].strip():
            text += (" " if text else "") + lines[j].strip()
            j += 1
        spk = "UNKNOWN"
        if ":" in text:
            cand, _ = text.split(":", 1)
            if len(cand) < 60:  # a name, not a sentence with a colon
                spk = sanitize(cand)
        speakers.add(spk)
        dur = max(0.0, end - start)
        if dur > 0:
            rows.append((start, dur, spk))
        i = j
    out = []
    for start, dur, spk in rows:
        out.append(
            f"SPEAKER {uri} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>"
        )
    sys.stderr.write(
        f"{len(rows)} turns, {len(speakers)} speakers: {sorted(speakers)}\n"
    )
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    vtt = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    uri = out_path.stem
    out_path.write_text(convert(vtt, uri), encoding="utf-8")
    sys.stderr.write(f"wrote {out_path}\n")
