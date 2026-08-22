#!/usr/bin/env python
"""Standalone Whisper Turbo MLX (wtm) runner for Apple Silicon.

Run inside .venv-wtm:

    python wtm_runner.py --audio lecture.wav --timestamps

Prints one JSON document on stdout:
  {"text": "...", "segments": [{"start": 0.0, "end": 1.2, "text": "..."}],
   "transcribe_sec": 12.3, "language": "en"}

Heavy import (whisper_turbo / MLX) stays inside run_wtm().
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Whisper Turbo MLX → JSON")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--timestamps", action="store_true",
                    help="Return segment timestamps (required for VTT output).")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--multilingual", action="store_true")
    return ap


def _segment_to_dict(seg: Any) -> Dict[str, Any]:
    if isinstance(seg, dict):
        start = float(seg.get("start", 0.0))
        end = seg.get("end")
        text = str(seg.get("text") or "").strip()
    else:
        start = float(seg[0])
        end = seg[1] if len(seg) > 1 else None
        text = str(seg[2] if len(seg) > 2 else "").strip()
    end_f = float(end) if end is not None else start + max(len(text.split()) * 0.35, 0.5)
    return {"start": start, "end": end_f, "text": text}


def run_wtm(
    audio: str,
    *,
    timestamps: bool = True,
    quick: bool = False,
    multilingual: bool = False,
) -> Dict[str, Any]:
    from whisper_turbo import transcribe

    t0 = time.time()
    txt, segs = transcribe(
        audio,
        timestamps=timestamps,
        quick=quick,
        multilingual=multilingual,
    )
    elapsed = time.time() - t0
    segments: List[Dict[str, Any]] = []
    if segs:
        for seg in segs:
            d = _segment_to_dict(seg)
            if d["text"]:
                segments.append(d)
    text = (txt or "").strip()
    if not text and segments:
        text = " ".join(s["text"] for s in segments).strip()
    return {
        "text": text,
        "segments": segments,
        "transcribe_sec": elapsed,
        "language": "en",
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        out = run_wtm(
            args.audio,
            timestamps=args.timestamps or True,
            quick=args.quick,
            multilingual=args.multilingual,
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
