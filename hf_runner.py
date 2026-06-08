#!/usr/bin/env python
"""Standalone HF-transformers ASR runner. Run under a torch-enabled 3.12 venv:

    python hf_runner.py --audio a.wav --model facebook/wav2vec2-large-960h --device cuda

Prints exactly ONE JSON document on stdout:
  {"text": "...", "segments":[{start,end,text}], "words":[{word,start,end}],
   "transcribe_sec": <float>, "vram_peak_bytes": <int|null>, "language": "en"}

wav2vec2 / Conformer are CTC models; the transformers ASR pipeline with
return_timestamps="word" + chunk_length_s handles long audio and yields per-word
timestamps, from which we build segments. Heavy imports (torch/transformers) are
INSIDE run_hf so the helpers import cleanly anywhere for testing."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import List, Optional

_SAMPLE_RATE = 16000


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="HF transformers ASR -> JSON")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--language", default="en")
    ap.add_argument("--chunk-len-secs", type=float, default=30.0,
                    help="pipeline chunk_length_s for long-form CTC inference.")
    ap.add_argument("--stride-len-secs", type=float, default=5.0,
                    help="pipeline stride_length_s overlap between chunks.")
    return ap


def _needs_decode(audio: str) -> bool:
    """True unless already a .wav. transformers/torchaudio cannot open mp4/m4a
    under torchaudio >=2.11 (torchaudio.io removed); decode non-wav up front."""
    return not audio.lower().endswith(".wav")


def _ffmpeg_to_wav(audio: str) -> str:
    """Decode to a temp 16 kHz mono WAV via ffmpeg CLI; caller deletes it.
    Mirrors nemo_runner._ffmpeg_to_wav."""
    fd, wav = tempfile.mkstemp(suffix=".wav", prefix="hf_")
    os.close(fd)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", audio, "-ac", "1", "-ar", str(_SAMPLE_RATE), "-vn", wav]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        os.path.exists(wav) and os.remove(wav)
        raise RuntimeError(
            "ffmpeg not found on PATH -- required to decode non-wav audio for the "
            "HF engine (torchaudio.io removed in torchaudio >=2.11). Install ffmpeg "
            "or pass a .wav file.")
    except subprocess.CalledProcessError as e:
        os.path.exists(wav) and os.remove(wav)
        raise RuntimeError(f"ffmpeg failed to decode {audio!r}: exit {e.returncode}")
    return wav


def _num(x) -> float:
    return float(x) if x is not None else 0.0


def _words_from_chunks(chunks) -> List[dict]:
    """Pipeline word chunks -> [{word,start,end}]. timestamp is a (start,end) tuple."""
    out: List[dict] = []
    for c in chunks or []:
        ts = c.get("timestamp") if isinstance(c, dict) else getattr(c, "timestamp", None)
        start, end = (ts or (None, None))[0], (ts or (None, None))[1]
        word = c.get("text") if isinstance(c, dict) else getattr(c, "text", "")
        out.append({"word": (word or "").strip(), "start": _num(start), "end": _num(end)})
    return out


def _segments_from_words(words, max_gap: float = 0.8, max_len: float = 12.0) -> List[dict]:
    """Group words into cue-sized segments: break on a > max_gap silence or when a
    segment would exceed max_len seconds. Keeps VTT cues readable."""
    segs: List[dict] = []
    cur: List[dict] = []
    for w in words or []:
        if cur:
            gap = w["start"] - cur[-1]["end"]
            span = w["end"] - cur[0]["start"]
            if gap > max_gap or span > max_len:
                segs.append(_flush(cur)); cur = []
        cur.append(w)
    if cur:
        segs.append(_flush(cur))
    return segs


def _flush(words: List[dict]) -> dict:
    return {"start": words[0]["start"], "end": words[-1]["end"],
            "text": " ".join(w["word"] for w in words if w["word"]).strip()}


def _peak_vram_bytes(torch_mod, device: str) -> Optional[int]:
    try:
        if device == "cuda" and torch_mod.cuda.is_available():
            return int(torch_mod.cuda.max_memory_allocated())
    except Exception:
        pass
    return None


def run_hf(audio: str, model: str, device: str, language: str = "en",
           chunk_len_secs: float = 30.0, stride_len_secs: float = 5.0) -> dict:
    import torch
    from transformers import pipeline
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tmp_wav = _ffmpeg_to_wav(audio) if _needs_decode(audio) else None
    if tmp_wav:
        audio = tmp_wav
    try:
        asr = pipeline("automatic-speech-recognition", model=model,
                       device=0 if (device == "cuda" and torch.cuda.is_available()) else -1,
                       chunk_length_s=chunk_len_secs, stride_length_s=stride_len_secs)
        t0 = time.time()
        out = asr(audio, return_timestamps="word")
        transcribe_sec = time.time() - t0
        words = _words_from_chunks(out.get("chunks"))
        segments = _segments_from_words(words)
        text = (out.get("text") or " ".join(w["word"] for w in words)).strip()
        return {"text": text, "segments": segments, "words": words,
                "transcribe_sec": transcribe_sec,
                "vram_peak_bytes": _peak_vram_bytes(torch, device),
                "language": language}
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.remove(tmp_wav)


def main() -> int:
    ns = build_arg_parser().parse_args()
    saved_stdout = sys.stdout
    sys.stdout.flush()
    saved_stdout_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    try:
        out = run_hf(audio=ns.audio, model=ns.model, device=ns.device,
                     language=ns.language, chunk_len_secs=ns.chunk_len_secs,
                     stride_len_secs=ns.stride_len_secs)
    finally:
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, 1)
        os.close(saved_stdout_fd)
        sys.stdout = saved_stdout
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
