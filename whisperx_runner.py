#!/usr/bin/env python
"""Standalone WhisperX runner. Run under a torch-enabled (3.12) venv:

    python whisperx_runner.py --audio a.wav --model large-v3-turbo --device cuda \
        --diarize [--rttm a.rttm]   # HF token via HF_TOKEN env or --hf-token

Prints one JSON document on stdout:
  {"segments":[{start,end,text,speaker?}], "words":[...], "speakers":[...],
   "der": <float|null>, "language": "en"}

All heavy imports (torch/whisperx/pyannote) are INSIDE functions so the
torch-free helpers (parse_rttm, compute_der_from_rttm, arg parsing) import
cleanly anywhere for testing.
"""
import argparse
import json
import os
import sys
from typing import List, Optional, Tuple


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="WhisperX transcribe+align+diarize → JSON")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--language", default="en")
    ap.add_argument("--diarize", action="store_true")
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument("--rttm", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    return ap


def parse_rttm(path: str) -> List[Tuple[float, float, str]]:
    """Parse NIST RTTM 'SPEAKER <uri> <chan> <start> <dur> <NA> <NA> <spk> ...'."""
    out: List[Tuple[float, float, str]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3]); dur = float(parts[4]); spk = parts[7]
            out.append((start, start + dur, spk))
    return out


def _segments_to_annotation(segments: List[Tuple[float, float, str]]):
    from pyannote.core import Annotation, Segment
    ann = Annotation()
    for start, end, spk in segments:
        if end > start:
            ann[Segment(start, end)] = spk
    return ann


def compute_der_from_rttm(hyp_segments: List[Tuple[float, float, str]], rttm_path: str) -> float:
    """DER of hyp_segments vs the RTTM reference, via pyannote.metrics (default
    collar/skip). pyannote.metrics has no torch dependency."""
    from pyannote.metrics.diarization import DiarizationErrorRate
    ref = _segments_to_annotation(parse_rttm(rttm_path))
    hyp = _segments_to_annotation(hyp_segments)
    return float(DiarizationErrorRate()(ref, hyp))


def run_whisperx(audio: str, model: str, device: str, language: str = "en",
                 diarize: bool = True, hf_token: Optional[str] = None,
                 min_speakers: Optional[int] = None, max_speakers: Optional[int] = None,
                 rttm: Optional[str] = None, batch_size: int = 16) -> dict:
    """Transcribe → align → (optional) diarize → (optional) DER. Returns a dict
    ready to JSON-serialize. Heavy imports are local."""
    import whisperx

    compute_type = "float16" if device == "cuda" else "int8"
    asr = whisperx.load_model(model, device, compute_type=compute_type, language=language)
    audio_arr = whisperx.load_audio(audio)
    result = asr.transcribe(audio_arr, batch_size=batch_size)
    lang = result.get("language", language)

    align_model, metadata = whisperx.load_align_model(language_code=lang, device=device)
    result = whisperx.align(result["segments"], align_model, metadata, audio_arr, device,
                            return_char_alignments=False)

    speakers: List[str] = []
    diarized = False
    if diarize and hf_token:
        try:
            try:
                from whisperx.diarize import DiarizationPipeline  # newer layout
            except Exception:
                from whisperx import DiarizationPipeline           # older layout
            dia = DiarizationPipeline(use_auth_token=hf_token, device=device)
            kw = {}
            if min_speakers is not None:
                kw["min_speakers"] = min_speakers
            if max_speakers is not None:
                kw["max_speakers"] = max_speakers
            diar_segments = dia(audio_arr, **kw)
            result = whisperx.assign_word_speakers(diar_segments, result)
            diarized = True
        except Exception as e:
            print(f"WARN: diarization failed ({e}); returning alignment-only", file=sys.stderr)

    segments = [{"start": float(s["start"]), "end": float(s["end"]),
                 "text": s.get("text", ""), "speaker": s.get("speaker")}
                for s in result.get("segments", [])]
    words = [{"word": w.get("word"), "start": w.get("start"), "end": w.get("end"),
              "score": w.get("score"), "speaker": w.get("speaker")}
             for w in result.get("word_segments", result.get("words", []))]
    if diarized:
        speakers = sorted({s["speaker"] for s in segments if s.get("speaker")})

    der = None
    if rttm:
        hyp = [(s["start"], s["end"], s["speaker"]) for s in segments if s.get("speaker")]
        if hyp:
            try:
                der = compute_der_from_rttm(hyp, rttm)
            except Exception as e:
                print(f"WARN: DER computation failed ({e})", file=sys.stderr)

    return {"segments": segments, "words": words, "speakers": speakers,
            "der": der, "language": lang}


def main() -> int:
    ns = build_arg_parser().parse_args()
    hf_token = ns.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    out = run_whisperx(
        audio=ns.audio, model=ns.model, device=ns.device, language=ns.language,
        diarize=ns.diarize, hf_token=hf_token, min_speakers=ns.min_speakers,
        max_speakers=ns.max_speakers, rttm=ns.rttm, batch_size=ns.batch_size,
    )
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
