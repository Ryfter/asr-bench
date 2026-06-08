"""FasterWhisperEngine — the reference local-Whisper engine family.

Heavy imports (faster_whisper / ctranslate2) stay inside run() so importing this
module is torch-free. model_disk_bytes lives here because it is faster-whisper
specific (it sums the Systran/faster-whisper-* HF cache); asr_bench re-exports it
to keep the public surface unchanged."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from engines.base import (
    Engine, ModelResult, ClipResult, Pair, RunConfig,
    load_reference_text, detect_reference_origin, normalize_for_wer,
    compute_word_metrics, compute_hallucination_signals, write_whisper_vtt,
    _model_label, gpu_used_bytes, _HAS_NVML,
)


# ---- Model size on disk -----------------------------------------------------
def model_disk_bytes(fw_name: str) -> Optional[int]:
    """Sum the size of all files in the HF hub cache for this model.

    Returns None if not yet downloaded — that's fine, the script will fill the
    column after the first run.
    """
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache.exists():
        return None
    # faster-whisper models are mirrored under Systran/faster-whisper-<name>
    candidates = list(cache.glob(f"models--Systran--faster-whisper-{fw_name}"))
    candidates += list(cache.glob(f"models--openai--whisper-{fw_name}"))
    if not candidates:
        return None
    total = 0
    for d in candidates:
        for p in d.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    return total or None


class FasterWhisperEngine(Engine):
    name = "faster-whisper"

    def run(self, entry: Dict, pairs: List[Pair], cfg: RunConfig) -> ModelResult:
        model_id = entry["id"]
        assert entry.get("engine", "faster-whisper") == "faster-whisper", (
            f"FasterWhisperEngine requires a faster-whisper entry, got engine={entry.get('engine')!r}"
        )
        info = entry  # alias: the resolved entry carries the same keys the old run_model read from MODELS
        fw_name = info["fw_name"]
        device = cfg.device
        compute_type = cfg.compute_type
        batch_size = cfg.batch_size
        beam_size = cfg.beam_size
        vad_filter = cfg.vad_filter
        batched_note = f" batch_size={batch_size}" if batch_size > 1 else ""
        print(f"\n[{info['display']}] loading on device={device} compute_type={compute_type}{batched_note}...", flush=True)

        # Late import so the script can show --help without requiring the model dep
        from faster_whisper import WhisperModel

        t0 = time.time()
        try:
            model = WhisperModel(fw_name, device=device, compute_type=compute_type)
            # BatchedInferencePipeline is the path to high GPU utilization. Sequential
            # decoding tops out around 50% on big GPUs; batching pushes it past 80%.
            if batch_size > 1:
                from faster_whisper import BatchedInferencePipeline  # type: ignore
                transcribe_target = BatchedInferencePipeline(model=model)
            else:
                transcribe_target = model
        except Exception as e:
            print(f"  ERROR loading {fw_name}: {e}", file=sys.stderr)
            # Return a model result with a zero-clip note so the table shows the failure
            return ModelResult(
                model_id=model_id, display=info["display"], fw_name=fw_name,
                params=info["params"], developer=info["developer"],
                languages=info["languages"], notes=f"LOAD FAILED: {e}",
                disk_bytes=model_disk_bytes(fw_name), load_sec=0.0,
                engine="faster-whisper", vram_is_total=False,
            )
        load_sec = time.time() - t0
        print(f"  loaded in {load_sec:.1f}s", flush=True)

        result = ModelResult(
            model_id=model_id, display=info["display"], fw_name=fw_name,
            params=info["params"], developer=info["developer"],
            languages=info["languages"], notes=info["notes"],
            disk_bytes=model_disk_bytes(fw_name), load_sec=load_sec,
            engine="faster-whisper", vram_is_total=False,
        )

        for clip_idx, pair in enumerate(pairs, start=1):
            print(f"  [{clip_idx}/{len(pairs)}] transcribing {pair.audio.name}...", flush=True)
            ref_text = load_reference_text(pair.reference)
            ref_origin, ref_label = detect_reference_origin(pair.reference)

            # Track peak VRAM during this clip's transcription
            vram_baseline = gpu_used_bytes()
            vram_peak = vram_baseline

            t0 = time.time()
            transcribe_kwargs = dict(language="en", beam_size=beam_size, vad_filter=vad_filter)
            if batch_size > 1:
                transcribe_kwargs["batch_size"] = batch_size
            segments, audio_info = transcribe_target.transcribe(
                str(pair.audio),
                **transcribe_kwargs,
            )
            text_parts: List[str] = []
            cue_tuples: List[Tuple[float, float, str]] = []
            duration_sec = float(audio_info.duration) or 1.0
            last_pct_printed = -10.0  # so first segment can trigger 0% line; tunable
            for seg in segments:
                text_parts.append(seg.text)
                cue_tuples.append((float(seg.start), float(seg.end), seg.text))
                cur = gpu_used_bytes()
                if cur > vram_peak:
                    vram_peak = cur
                # Streaming progress: print every 10% of audio crossed so the user can
                # see the run is alive (transcription is otherwise silent for minutes).
                pct = (float(seg.end) / duration_sec) * 100.0
                if pct - last_pct_printed >= 10.0:
                    elapsed = time.time() - t0
                    eta = (duration_sec - float(seg.end)) / max(float(seg.end), 1.0) * elapsed
                    print(
                        f"    {pct:5.1f}%  audio {int(seg.end):>5d}s/{int(duration_sec):>5d}s  "
                        f"elapsed {elapsed:5.1f}s  eta {eta:5.1f}s",
                        flush=True,
                    )
                    last_pct_printed = pct
            transcribe_sec = time.time() - t0
            hypothesis = " ".join(text_parts).strip()

            # Write the per-model VTT next to the source audio so it stands alongside
            # Panopto's own caption file.
            vtt_path = write_whisper_vtt(pair.audio, _model_label(model_id), cue_tuples)

            ref_norm = normalize_for_wer(ref_text)
            hyp_norm = normalize_for_wer(hypothesis)
            metrics = compute_word_metrics(ref_norm, hyp_norm)
            wer_val = metrics.wer

            audio_sec = float(audio_info.duration)
            rtfx = audio_sec / transcribe_sec if transcribe_sec > 0 else 0.0
            vram_used = (vram_peak - vram_baseline) if _HAS_NVML and device == "cuda" else None

            print(
                f"    {audio_sec:.1f}s audio in {transcribe_sec:.1f}s "
                f"(RTFx {rtfx:.2f}, WER {wer_val * 100:.1f}%)",
                flush=True,
            )

            rep_cov, comp_ratio = compute_hallucination_signals(hypothesis, hyp_norm)
            result.clips.append(
                ClipResult(
                    audio=pair.audio.name,
                    audio_sec=audio_sec,
                    transcribe_sec=transcribe_sec,
                    rtfx=rtfx,
                    vram_peak_bytes=vram_used,
                    hypothesis=hypothesis,
                    reference_normalized=ref_norm,
                    hypothesis_normalized=hyp_norm,
                    wer=wer_val,
                    mer=metrics.mer,
                    wil=metrics.wil,
                    cer=metrics.cer,
                    repeat_coverage=rep_cov, compression_ratio=comp_ratio,
                    hits=metrics.hits,
                    substitutions=metrics.substitutions,
                    deletions=metrics.deletions,
                    insertions=metrics.insertions,
                    cue_count=len(cue_tuples),
                    vtt_path=str(vtt_path),
                    reference_origin=ref_origin,
                    reference_label=ref_label,
                )
            )

            # Refresh disk-size measurement now that the model has fully downloaded
            if result.disk_bytes is None:
                result.disk_bytes = model_disk_bytes(fw_name)

        # Drop the model reference so Python can release memory between runs
        del model
        return result
