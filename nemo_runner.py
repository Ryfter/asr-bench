#!/usr/bin/env python
"""Standalone NeMo runner. Run under a torch-enabled (3.12) venv:

    python nemo_runner.py --audio a.wav --model nvidia/parakeet-tdt-0.6b-v2 --device cuda

Prints exactly ONE JSON document on stdout:
  {"text": "...", "segments":[{start,end,text}], "words":[{word,start,end}],
   "transcribe_sec": <float>, "vram_peak_bytes": <int|null>, "language": "en"}

Parakeet (ASRModel.transcribe(timestamps=True)) emits segments+words. Canary-Qwen
(SALM.generate, 40s non-overlapping chunks) emits text only (no timestamps).

All heavy imports (torch/nemo) are INSIDE functions so the torch-free helpers
(arg parsing, JSON shaping) import cleanly anywhere for testing.
"""
import argparse
import json
import os
import sys
import time
from typing import List, Optional


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="NeMo transcribe -> JSON")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--language", default="en")
    ap.add_argument("--chunk-len-secs", type=float, default=40.0,
                    help="Canary-Qwen encoder window; non-overlapping chunk length.")
    ap.add_argument("--max-new-tokens", type=int, default=1024,
                    help="Per-chunk decode budget for SALM (example's 128 truncates).")
    return ap


def _looks_like_salm(model: str) -> bool:
    """Canary-Qwen is a Speech-LLM (SALM); everything else we ship is an ASRModel."""
    return "canary-qwen" in model.lower()


def _segments_to_json(segments) -> List[dict]:
    """Normalize NeMo segment timestamps (dicts with 'segment' text, or objects)
    into [{'start','end','text'}]."""
    out: List[dict] = []
    for s in segments or []:
        if isinstance(s, dict):
            start = s.get("start"); end = s.get("end")
            text = s.get("segment") or s.get("text") or ""
        else:
            start = getattr(s, "start", None); end = getattr(s, "end", None)
            text = getattr(s, "segment", None) or getattr(s, "text", "") or ""
        out.append({"start": float(start), "end": float(end), "text": str(text).strip()})
    return out


def _words_to_json(words) -> List[dict]:
    out: List[dict] = []
    for w in words or []:
        if isinstance(w, dict):
            word = w.get("word"); start = w.get("start"); end = w.get("end")
        else:
            word = getattr(w, "word", None); start = getattr(w, "start", None)
            end = getattr(w, "end", None)
        out.append({"word": word, "start": float(start), "end": float(end)})
    return out


def _peak_vram_bytes(torch_mod, device: str) -> Optional[int]:
    try:
        if device == "cuda" and torch_mod.cuda.is_available():
            return int(torch_mod.cuda.max_memory_allocated())
    except Exception:
        pass
    return None


def run_nemo(audio: str, model: str, device: str, language: str = "en",
             chunk_len_secs: float = 40.0, max_new_tokens: int = 1024) -> dict:
    """Transcribe with NeMo. Branches: SALM (Canary-Qwen, chunked generate, text
    only) vs ASRModel (Parakeet, timestamps=True). Heavy imports are local."""
    import torch
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if _looks_like_salm(model):
        from nemo.collections.speechlm2.models import SALM
        salm = SALM.from_pretrained(model)
        if device == "cuda" and torch.cuda.is_available():
            salm = salm.to("cuda").eval()
        from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
        seg = AudioSegment.from_file(audio, target_sr=16000)
        sr = 16000
        chunk = int(chunk_len_secs * sr)
        samples = seg.samples
        t0 = time.time()
        parts: List[str] = []
        for i in range(0, len(samples), chunk):
            window = samples[i:i + chunk]
            ans = salm.generate(
                prompts=[[{"role": "user",
                           "content": f"Transcribe the following: {salm.audio_locator_tag}"}]],
                audios=window[None, :], audio_lengths=[len(window)],
                max_new_tokens=max_new_tokens)
            parts.append(salm.tokenizer.ids_to_text(ans[0].cpu()).strip())
        transcribe_sec = time.time() - t0
        text = " ".join(p for p in parts if p).strip()
        return {"text": text, "segments": [], "words": [],
                "transcribe_sec": transcribe_sec,
                "vram_peak_bytes": _peak_vram_bytes(torch, device), "language": language}

    import nemo.collections.asr as nemo_asr
    asr = nemo_asr.models.ASRModel.from_pretrained(model_name=model)
    if device == "cuda" and torch.cuda.is_available():
        asr = asr.to("cuda")
    t0 = time.time()
    out = asr.transcribe([audio], timestamps=True)
    transcribe_sec = time.time() - t0
    hyp = out[0]
    ts = getattr(hyp, "timestamp", {}) or {}
    segments = _segments_to_json(ts.get("segment"))
    words = _words_to_json(ts.get("word"))
    text = getattr(hyp, "text", "") or " ".join(s["text"] for s in segments)
    return {"text": text.strip(), "segments": segments, "words": words,
            "transcribe_sec": transcribe_sec,
            "vram_peak_bytes": _peak_vram_bytes(torch, device), "language": language}


def main() -> int:
    ns = build_arg_parser().parse_args()
    # Contract: a SINGLE JSON document on stdout. NeMo/Lightning/HF emit progress
    # + logging there, so route everything to stderr during processing and
    # restore real stdout only for the final JSON (two levels, mirroring
    # whisperx_runner.py).
    saved_stdout = sys.stdout
    sys.stdout.flush()
    saved_stdout_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    try:
        out = run_nemo(
            audio=ns.audio, model=ns.model, device=ns.device, language=ns.language,
            chunk_len_secs=ns.chunk_len_secs, max_new_tokens=ns.max_new_tokens)
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
