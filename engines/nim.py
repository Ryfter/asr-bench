"""NimEngine — NVIDIA NIM / Riva gRPC ASR engine family.

The riva.client import is late (inside run()) so importing this module needs no
nvidia-riva-client and stays torch-free. Local self-hosted is the preferred
transport; a hosted endpoint is a flag-gated fallback (see build_nim_auth_kwargs)."""

from __future__ import annotations

import sys
import time
from typing import Dict, List, Optional, Tuple

from engines.base import (
    Engine, ModelResult, ClipResult, Pair, RunConfig,
    load_reference_text, detect_reference_origin, normalize_for_wer,
    compute_word_metrics, compute_hallucination_signals, write_whisper_vtt,
    _model_label, VramSampler, group_words_into_cues, decode_to_pcm16, _HAS_NVML,
)


def build_nim_auth_kwargs(url: str, api_key: Optional[str], ssl: bool) -> Dict:
    """Build kwargs for riva.client.Auth. An API key implies SSL + a Bearer
    authorization metadata header (the path a hosted endpoint needs)."""
    kw: Dict = {"uri": url, "use_ssl": bool(ssl)}
    if api_key:
        kw["use_ssl"] = True
        kw["metadata_args"] = [["authorization", f"Bearer {api_key}"]]
    return kw


def nim_response_to_hypothesis(response) -> str:
    """Concatenate the top alternative transcript across all results."""
    parts: List[str] = []
    for result in getattr(response, "results", []) or []:
        alts = getattr(result, "alternatives", None) or []
        if alts:
            parts.append(alts[0].transcript)
    return " ".join(p.strip() for p in parts if p).strip()


def nim_response_to_words(response) -> List[Tuple[float, float, str]]:
    """Flatten word-level timings (Riva reports ms) into (start_s, end_s, word)."""
    out: List[Tuple[float, float, str]] = []
    for result in getattr(response, "results", []) or []:
        alts = getattr(result, "alternatives", None) or []
        if not alts:
            continue
        for w in getattr(alts[0], "words", None) or []:
            out.append((float(w.start_time) / 1000.0, float(w.end_time) / 1000.0, w.word))
    return out


class NimEngine(Engine):
    name = "nim"

    def run(self, entry: Dict, pairs: List[Pair], cfg: RunConfig) -> ModelResult:
        riva_model = cfg.nim_model or entry.get("riva_model", "")
        print(
            f"\n[{entry['display']}] connecting to NIM at {cfg.nim_url} "
            f"(model={riva_model or 'server default'})...",
            flush=True,
        )

        def _fail(msg: str) -> ModelResult:
            print(f"  ERROR: {msg}", file=sys.stderr)
            # NIM has no local model file; fw_name/disk are N/A
            return ModelResult(
                model_id=entry["id"], display=entry["display"], fw_name="",  # fw_name: N/A for NIM
                params=entry.get("params", "—"), developer=entry.get("developer", "NVIDIA"),
                languages=entry.get("languages", "—"), notes=f"LOAD FAILED: {msg}",
                disk_bytes=None, load_sec=0.0, engine="nim", vram_is_total=True,
            )

        t0 = time.time()
        try:
            import riva.client  # late import; only needed for NIM runs
            auth = riva.client.Auth(**build_nim_auth_kwargs(cfg.nim_url, cfg.nim_api_key, cfg.nim_ssl))
            asr = riva.client.ASRService(auth)
        except ImportError:
            return _fail("nvidia-riva-client not installed (pip install nvidia-riva-client)")
        except Exception as e:
            return _fail(f"could not connect to NIM at {cfg.nim_url}: {e}")
        load_sec = time.time() - t0
        print(f"  connected in {load_sec:.1f}s", flush=True)

        # NIM has no local model file; fw_name/disk are N/A
        result = ModelResult(
            model_id=entry["id"], display=entry["display"], fw_name="",  # fw_name: N/A for NIM
            params=entry.get("params", "—"), developer=entry.get("developer", "NVIDIA"),
            languages=entry.get("languages", "—"), notes=entry.get("notes", ""),
            disk_bytes=None, load_sec=load_sec, engine="nim", vram_is_total=True,
        )

        for clip_idx, pair in enumerate(pairs, start=1):
            print(f"  [{clip_idx}/{len(pairs)}] transcribing {pair.audio.name}...", flush=True)
            ref_text = load_reference_text(pair.reference)
            ref_origin, ref_label = detect_reference_origin(pair.reference)

            try:
                pcm, n_samples = decode_to_pcm16(pair.audio)
            except Exception as e:
                print(f"    decode failed: {e}", file=sys.stderr)
                continue
            audio_sec = n_samples / 16000.0

            config = riva.client.RecognitionConfig(
                language_code=cfg.nim_language,
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True,
                max_alternatives=1,
            )
            # encoding / sample rate: LINEAR_PCM @ 16k mono
            config.sample_rate_hertz = 16000
            config.audio_channel_count = 1
            # AudioEncoding location varies across nvidia-riva-client versions:
            # newer exposes riva.client.AudioEncoding; older lives in the proto module.
            try:
                config.encoding = riva.client.AudioEncoding.LINEAR_PCM
            except AttributeError:
                from riva.client.proto.riva_audio_pb2 import AudioEncoding  # type: ignore
                config.encoding = AudioEncoding.LINEAR_PCM
            if riva_model:
                config.model = riva_model

            sampler = VramSampler().start() if _HAS_NVML else None
            print(f"    offline_recognize: {audio_sec:.1f}s audio, awaiting NIM response...", flush=True)
            t1 = time.time()
            try:
                response = asr.offline_recognize(pcm, config)
            except Exception as e:
                if sampler:
                    sampler.stop()
                print(f"    recognize failed: {e}", file=sys.stderr)
                continue
            transcribe_sec = time.time() - t1
            vram_peak = sampler.stop() if sampler else None

            hypothesis = nim_response_to_hypothesis(response)
            words = nim_response_to_words(response)
            cue_tuples = group_words_into_cues(words)
            vtt_path = write_whisper_vtt(pair.audio, _model_label(entry["id"]), cue_tuples)

            ref_norm = normalize_for_wer(ref_text)
            hyp_norm = normalize_for_wer(hypothesis)
            metrics = compute_word_metrics(ref_norm, hyp_norm)
            wer_val = metrics.wer

            rtfx = audio_sec / transcribe_sec if transcribe_sec > 0 else 0.0
            print(
                f"    {audio_sec:.1f}s audio in {transcribe_sec:.1f}s "
                f"(RTFx {rtfx:.2f}, WER {wer_val * 100:.1f}%)",
                flush=True,
            )

            rep_cov, comp_ratio = compute_hallucination_signals(hypothesis, hyp_norm)
            result.clips.append(
                ClipResult(
                    audio=pair.audio.name, audio_sec=audio_sec,
                    transcribe_sec=transcribe_sec, rtfx=rtfx,
                    vram_peak_bytes=vram_peak, hypothesis=hypothesis,
                    reference_normalized=ref_norm, hypothesis_normalized=hyp_norm,
                    wer=wer_val, mer=metrics.mer, wil=metrics.wil, cer=metrics.cer,
                    repeat_coverage=rep_cov, compression_ratio=comp_ratio,
                    hits=metrics.hits, substitutions=metrics.substitutions,
                    deletions=metrics.deletions, insertions=metrics.insertions,
                    cue_count=len(cue_tuples), vtt_path=str(vtt_path),
                    reference_origin=ref_origin, reference_label=ref_label,
                )
            )
        if not result.clips:
            result.notes = "ALL CLIPS FAILED — check NIM endpoint, audio decode, and stderr above"
        return result
