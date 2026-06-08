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
import subprocess
import sys
import tempfile
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


_SAMPLE_RATE = 16000


def _needs_decode(audio: str) -> bool:
    """True unless the input is already a .wav. NeMo/lhotse reads .wav via
    soundfile, but cannot open compressed containers (mp4/m4a/...) under
    torchaudio >=2.11 -- torchaudio.io (the ffmpeg backend) was removed, so
    lhotse's Recording.from_file raises ModuleNotFoundError. We decode anything
    non-wav to a 16 kHz mono WAV up front."""
    return not audio.lower().endswith(".wav")


def _ffmpeg_to_wav(audio: str) -> str:
    """Decode `audio` to a temp 16 kHz mono WAV via the ffmpeg CLI and return its
    path (caller deletes it). Mirrors the WhisperX runner's self-contained decode
    so nemo_runner accepts the same mp4/m4a corpus files the rest of asr-bench
    does. Raises RuntimeError with a clear hint if ffmpeg is missing or fails."""
    fd, wav = tempfile.mkstemp(suffix=".wav", prefix="nemo_")
    os.close(fd)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", audio, "-ac", "1", "-ar", str(_SAMPLE_RATE), "-vn", wav]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        os.path.exists(wav) and os.remove(wav)
        raise RuntimeError(
            "ffmpeg not found on PATH -- required to decode non-wav audio for NeMo "
            "(torchaudio.io was removed in torchaudio >=2.11). Install ffmpeg or "
            "pass a .wav file.")
    except subprocess.CalledProcessError as e:
        os.path.exists(wav) and os.remove(wav)
        raise RuntimeError(f"ffmpeg failed to decode {audio!r}: exit {e.returncode}")
    return wav


def _looks_like_salm(model: str) -> bool:
    """Canary-Qwen is a Speech-LLM (SALM); everything else we ship is an ASRModel."""
    return "canary-qwen" in model.lower()


def _num(x) -> float:
    """Coerce a timestamp to float, treating a missing (None) value as 0.0 so a
    segment/word with absent timing keeps its text rather than crashing the run."""
    return float(x) if x is not None else 0.0


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
        out.append({"start": _num(start), "end": _num(end), "text": str(text).strip()})
    return out


def _words_to_json(words) -> List[dict]:
    out: List[dict] = []
    for w in words or []:
        if isinstance(w, dict):
            word = w.get("word"); start = w.get("start"); end = w.get("end")
        else:
            word = getattr(w, "word", None); start = getattr(w, "start", None)
            end = getattr(w, "end", None)
        out.append({"word": word, "start": _num(start), "end": _num(end)})
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
    # Windows load-order workaround: NeMo's import chain (sklearn -> pandas ->
    # pyarrow) segfaults (0xC0000005) if pyarrow's native libs load AFTER certain
    # other native deps pulled in earlier in the chain. Pre-importing pyarrow here
    # loads it cleanly first; NeMo's later `import pyarrow` becomes a no-op. Bare
    # `import pyarrow` is fine on its own -- only the in-chain ordering crashes.
    # Guarded so it's a harmless no-op where pyarrow isn't installed.
    try:
        import pyarrow  # noqa: F401
    except Exception:
        pass
    import torch
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # NeMo can't decode compressed containers (mp4/m4a) under torchaudio >=2.11;
    # normalize anything non-wav to a temp 16 kHz mono WAV and clean it up after.
    tmp_wav = _ffmpeg_to_wav(audio) if _needs_decode(audio) else None
    if tmp_wav:
        audio = tmp_wav
    try:
        if _looks_like_salm(model):
            from nemo.collections.speechlm2.models import SALM
            salm = SALM.from_pretrained(model)
            if device == "cuda" and torch.cuda.is_available():
                salm = salm.to("cuda").eval()
            from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
            seg = AudioSegment.from_file(audio, target_sr=_SAMPLE_RATE)
            chunk = int(chunk_len_secs * _SAMPLE_RATE)
            samples = seg.samples
            t0 = time.time()
            parts: List[str] = []
            for i in range(0, len(samples), chunk):
                window = samples[i:i + chunk]
                # SALM.generate wants torch tensors, NOT numpy: audios float32
                # (B, T), audio_lens int64 (B,). The kwarg is `audio_lens` (not
                # `audio_lengths` -- that name silently falls into
                # **generation_kwargs, leaving audio_lens=None and tripping
                # perception's input validation).
                audios = torch.as_tensor(window, dtype=torch.float32,
                                         device=salm.device)[None, :]
                audio_lens = torch.tensor([audios.shape[1]], dtype=torch.int64,
                                          device=salm.device)
                ans = salm.generate(
                    prompts=[[{"role": "user",
                               "content": f"Transcribe the following: {salm.audio_locator_tag}"}]],
                    audios=audios, audio_lens=audio_lens,
                    max_new_tokens=max_new_tokens)
                parts.append(salm.tokenizer.ids_to_text(ans[0].cpu()).strip())
            transcribe_sec = time.time() - t0
            text = " ".join(p for p in parts if p).strip()
            return {"text": text, "segments": [], "words": [],
                    "transcribe_sec": transcribe_sec,
                    "vram_peak_bytes": _peak_vram_bytes(torch, device),
                    "language": language}

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
                "vram_peak_bytes": _peak_vram_bytes(torch, device),
                "language": language}
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.remove(tmp_wav)


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
