#!/usr/bin/env python3
"""
asr-bench v0.1 — benchmark local Whisper variants on your own audio.

Usage:
  python asr_bench.py --corpus ./test-corpus
  python asr_bench.py --corpus ./test-corpus --models small,medium
  python asr_bench.py --corpus ./test-corpus --device cpu

See README.md for corpus layout. See SPEC.md for the v0.2/v0.3 roadmap.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---- Optional VRAM tracking via NVIDIA NVML ---------------------------------
try:
    import pynvml  # provided by nvidia-ml-py3
    pynvml.nvmlInit()
    _HAS_NVML = True
    _NVML_DEVICE_COUNT = pynvml.nvmlDeviceGetCount()
except Exception:
    _HAS_NVML = False
    _NVML_DEVICE_COUNT = 0


# ---- Model registry ---------------------------------------------------------
MODELS: Dict[str, Dict] = {
    "small": {
        "display": "Whisper Small",
        "params": "244M",
        "developer": "OpenAI",
        "languages": "99 (multilingual)",
        "fw_name": "small",
        "notes": "Real-time on CPU. Decent for clear single speaker.",
    },
    "medium": {
        "display": "Whisper Medium",
        "params": "769M",
        "developer": "OpenAI",
        "languages": "99 (multilingual)",
        "fw_name": "medium",
        "notes": "Production sweet spot. ~2-3x realtime on CPU.",
    },
    "large-v3": {
        "display": "Whisper Large V3",
        "params": "1550M",
        "developer": "OpenAI",
        "languages": "99 (multilingual)",
        "fw_name": "large-v3",
        "notes": "State-of-art OpenAI accuracy. CPU is slow; GPU recommended.",
    },
    "large-v3-turbo": {
        "display": "Whisper Large V3 Turbo",
        "params": "809M",
        "developer": "OpenAI",
        "languages": "99 (multilingual)",
        "fw_name": "large-v3-turbo",
        "notes": "Distilled large-v3. Accuracy close to large at medium-class speed.",
    },
}


# ---- Reference / hypothesis text loading ------------------------------------
_TS_RE = re.compile(
    r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}[,.]\d{3}"
)
_CUE_NUM_RE = re.compile(r"^\d+$")
_BRACKETED_RE = re.compile(r"^\[.*\]$")


def load_reference_text(path: Path) -> str:
    """Strip SRT/VTT/Panopto formatting and return one flat string of words."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    out: List[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.upper() == "WEBVTT":
            continue
        if _CUE_NUM_RE.match(s):
            continue
        if _TS_RE.search(s):
            continue
        # Strip Panopto's "[Auto-generated transcript. Edits may have been applied for clarity.]" header
        if _BRACKETED_RE.match(s):
            continue
        out.append(s)
    return " ".join(out)


def normalize_for_wer(text: str) -> str:
    """Lowercase, strip punctuation except apostrophes, collapse whitespace.

    Keeps "don't" intact rather than splitting into "don" + "t". Most WER
    implementations do this; we do it explicitly for reproducibility.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---- Pair discovery ---------------------------------------------------------
AUDIO_EXTS = {".mp4", ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}


@dataclass
class Pair:
    audio: Path
    reference: Path

    @property
    def stem(self) -> str:
        return self.audio.stem.replace("_default", "")


def discover_pairs(corpus: Path) -> List[Pair]:
    """Find (audio, reference) pairs in three supported layouts."""
    manifest = corpus / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return [Pair(corpus / c["audio"], corpus / c["reference"]) for c in data["clips"]]

    pairs: List[Pair] = []
    for audio in sorted(corpus.iterdir()):
        if not audio.is_file() or audio.suffix.lower() not in AUDIO_EXTS:
            continue
        # Layout A: sibling with same stem + .txt / .srt / .vtt
        for ext in (".txt", ".srt", ".vtt"):
            sibling = audio.with_suffix(ext)
            if sibling.exists():
                pairs.append(Pair(audio, sibling))
                break
        else:
            # Layout B: Panopto export shape — "<base>_default.mp4" + "<base>_Captions*.txt"
            base = audio.stem
            if base.endswith("_default"):
                base = base[: -len("_default")]
                candidates = sorted(corpus.glob(f"{base}*Captions*.txt"))
                # Prefer the SRT-shaped one (cue 1 at top of file)
                preferred: Optional[Path] = None
                for c in candidates:
                    head = c.read_text(encoding="utf-8", errors="replace")[:300]
                    if re.match(r"\s*\d+\s*\n\s*\d{2}:\d{2}:\d{2}[,.]\d{3}", head):
                        preferred = c
                        break
                if preferred is None and candidates:
                    preferred = candidates[0]
                if preferred is not None:
                    pairs.append(Pair(audio, preferred))
    return pairs


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


def fmt_bytes(n: Optional[int]) -> str:
    if n is None:
        return "?"
    units = ["B", "KB", "MB", "GB"]
    f = float(n)
    for u in units:
        if f < 1024:
            return f"{f:.1f}{u}"
        f /= 1024
    return f"{f:.1f}TB"


# ---- VRAM tracking ----------------------------------------------------------
def gpu_used_bytes() -> int:
    if not _HAS_NVML or _NVML_DEVICE_COUNT == 0:
        return 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    return pynvml.nvmlDeviceGetMemoryInfo(handle).used


# ---- Per-model run ----------------------------------------------------------
@dataclass
class ClipResult:
    audio: str
    audio_sec: float
    transcribe_sec: float
    rtfx: float
    vram_peak_bytes: Optional[int]
    hypothesis: str
    reference_normalized: str
    hypothesis_normalized: str
    wer: float


@dataclass
class ModelResult:
    model_id: str
    display: str
    fw_name: str
    params: str
    developer: str
    languages: str
    notes: str
    disk_bytes: Optional[int]
    load_sec: float
    clips: List[ClipResult] = field(default_factory=list)

    @property
    def avg_wer(self) -> float:
        if not self.clips:
            return 0.0
        return sum(c.wer for c in self.clips) / len(self.clips)

    @property
    def total_audio_sec(self) -> float:
        return sum(c.audio_sec for c in self.clips)

    @property
    def total_transcribe_sec(self) -> float:
        return sum(c.transcribe_sec for c in self.clips)

    @property
    def aggregate_rtfx(self) -> float:
        if self.total_transcribe_sec == 0:
            return 0.0
        return self.total_audio_sec / self.total_transcribe_sec

    @property
    def peak_vram_bytes(self) -> Optional[int]:
        peaks = [c.vram_peak_bytes for c in self.clips if c.vram_peak_bytes is not None]
        return max(peaks) if peaks else None


def run_model(model_id: str, pairs: List[Pair], device: str, compute_type: str) -> ModelResult:
    """Load model once, transcribe each clip, compute WER per clip."""
    info = MODELS[model_id]
    fw_name = info["fw_name"]
    print(f"\n[{info['display']}] loading on device={device} compute_type={compute_type}...", flush=True)

    # Late import so the script can show --help without requiring the model dep
    from faster_whisper import WhisperModel
    from jiwer import wer as jiwer_wer

    t0 = time.time()
    try:
        model = WhisperModel(fw_name, device=device, compute_type=compute_type)
    except Exception as e:
        print(f"  ERROR loading {fw_name}: {e}", file=sys.stderr)
        # Return a model result with a zero-clip note so the table shows the failure
        return ModelResult(
            model_id=model_id, display=info["display"], fw_name=fw_name,
            params=info["params"], developer=info["developer"],
            languages=info["languages"], notes=f"LOAD FAILED: {e}",
            disk_bytes=model_disk_bytes(fw_name), load_sec=0.0,
        )
    load_sec = time.time() - t0
    print(f"  loaded in {load_sec:.1f}s", flush=True)

    result = ModelResult(
        model_id=model_id, display=info["display"], fw_name=fw_name,
        params=info["params"], developer=info["developer"],
        languages=info["languages"], notes=info["notes"],
        disk_bytes=model_disk_bytes(fw_name), load_sec=load_sec,
    )

    for pair in pairs:
        print(f"  transcribing {pair.audio.name}...", flush=True)
        ref_text = load_reference_text(pair.reference)

        # Track peak VRAM during this clip's transcription
        vram_baseline = gpu_used_bytes()
        vram_peak = vram_baseline

        t0 = time.time()
        segments, audio_info = model.transcribe(
            str(pair.audio),
            language="en",
            beam_size=5,
        )
        text_parts: List[str] = []
        for seg in segments:
            text_parts.append(seg.text)
            cur = gpu_used_bytes()
            if cur > vram_peak:
                vram_peak = cur
        transcribe_sec = time.time() - t0
        hypothesis = " ".join(text_parts).strip()

        ref_norm = normalize_for_wer(ref_text)
        hyp_norm = normalize_for_wer(hypothesis)
        try:
            wer_val = jiwer_wer(ref_norm, hyp_norm)
        except Exception:
            wer_val = float("nan")

        audio_sec = float(audio_info.duration)
        rtfx = audio_sec / transcribe_sec if transcribe_sec > 0 else 0.0
        vram_used = (vram_peak - vram_baseline) if _HAS_NVML and device == "cuda" else None

        print(
            f"    {audio_sec:.1f}s audio in {transcribe_sec:.1f}s "
            f"(RTFx {rtfx:.2f}, WER {wer_val * 100:.1f}%)",
            flush=True,
        )

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
            )
        )

        # Refresh disk-size measurement now that the model has fully downloaded
        if result.disk_bytes is None:
            result.disk_bytes = model_disk_bytes(fw_name)

    # Drop the model reference so Python can release memory between runs
    del model
    return result


# ---- Output -----------------------------------------------------------------
def render_markdown(
    results: List[ModelResult],
    corpus_path: Path,
    args: argparse.Namespace,
    gold_label: str,
) -> str:
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines: List[str] = []
    lines.append("# ASR Benchmark Results")
    lines.append("")
    lines.append(f"- **Date:** {now}")
    lines.append(f"- **Corpus:** `{corpus_path}`")
    lines.append(f"- **Reference quality:** {gold_label}")
    lines.append(f"- **Device:** {args.device}")
    lines.append(f"- **Compute type:** {args.compute_type}")
    lines.append(f"- **Clips:** {len(results[0].clips) if results else 0}")
    if results:
        total_audio_min = results[0].total_audio_sec / 60.0
        lines.append(f"- **Total audio:** {total_audio_min:.1f} min")
    lines.append(f"- **VRAM tracking:** {'on (NVML)' if _HAS_NVML else 'off (nvidia-ml-py3 not installed or no NVIDIA GPU)'}")
    lines.append("")

    # Aggregate table
    lines.append("## Aggregate")
    lines.append("")
    lines.append(
        "| Model | Params | Disk | WER% | RTFx | Wall clock | Peak VRAM | Load | Notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        wall_clock = f"{r.total_transcribe_sec:.1f}s"
        wer_pct = f"{r.avg_wer * 100:.1f}" if r.clips else "—"
        rtfx = f"{r.aggregate_rtfx:.2f}x" if r.clips else "—"
        vram = fmt_bytes(r.peak_vram_bytes) if r.peak_vram_bytes else ("n/a" if not _HAS_NVML else "0")
        disk = fmt_bytes(r.disk_bytes)
        load = f"{r.load_sec:.1f}s" if r.load_sec else "—"
        lines.append(
            f"| {r.display} | {r.params} | {disk} | {wer_pct} | {rtfx} | {wall_clock} | {vram} | {load} | {r.notes} |"
        )
    lines.append("")

    # Per-clip detail
    if results and results[0].clips:
        lines.append("## Per-clip detail")
        lines.append("")
        clip_count = len(results[0].clips)
        for i in range(clip_count):
            sample = results[0].clips[i]
            audio_min = sample.audio_sec / 60.0
            lines.append(f"### {sample.audio} — {audio_min:.1f} min")
            lines.append("")
            lines.append("| Model | WER% | RTFx | Transcribe time | VRAM peak |")
            lines.append("|---|---|---|---|---|")
            for r in results:
                if i < len(r.clips):
                    c = r.clips[i]
                    wer_pct = f"{c.wer * 100:.1f}"
                    vram = fmt_bytes(c.vram_peak_bytes) if c.vram_peak_bytes else ("n/a" if not _HAS_NVML else "0")
                    lines.append(
                        f"| {r.display} | {wer_pct} | {c.rtfx:.2f}x | {c.transcribe_sec:.1f}s | {vram} |"
                    )
            lines.append("")

    # Footnotes
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- Command: `python asr_bench.py --corpus '{corpus_path}' --models {','.join(args.models)} --device {args.device} --compute-type {args.compute_type}`")
    lines.append(f"- Reference normalization: lowercase, strip punctuation (keep apostrophes), collapse whitespace.")
    lines.append(f"- WER computed via [jiwer](https://github.com/jitsi/jiwer).")
    lines.append("")
    return "\n".join(lines)


# ---- CLI --------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark local Whisper variants on your own audio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--corpus", required=True, help="Path to a folder of (audio, reference) pairs.")
    ap.add_argument(
        "--models",
        default="small,medium,large-v3,large-v3-turbo",
        help="Comma-separated model IDs. Choices: " + ", ".join(MODELS.keys()),
    )
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device. 'auto' picks CUDA if available, else CPU.",
    )
    ap.add_argument(
        "--compute-type",
        default="auto",
        help="ctranslate2 compute type: 'auto', 'int8', 'int8_float16', 'float16', 'float32'.",
    )
    ap.add_argument(
        "--gold",
        action="store_true",
        help="Label the WER as gold-standard (hand-corrected reference). Without this, output marks WER as proxy.",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Where to save the markdown report. Default: ./results/<timestamp>.md",
    )
    args = ap.parse_args()

    corpus = Path(args.corpus).resolve()
    if not corpus.is_dir():
        print(f"ERROR: --corpus {corpus} is not a directory", file=sys.stderr)
        return 2

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in requested if m not in MODELS]
    if unknown:
        print(f"ERROR: unknown models: {', '.join(unknown)}", file=sys.stderr)
        print(f"Available: {', '.join(MODELS.keys())}", file=sys.stderr)
        return 2

    pairs = discover_pairs(corpus)
    if not pairs:
        print(f"ERROR: no (audio, reference) pairs discovered in {corpus}", file=sys.stderr)
        print("See test-corpus/README.md for supported layouts.", file=sys.stderr)
        return 2

    print(f"Discovered {len(pairs)} clip(s) under {corpus}:")
    for p in pairs:
        print(f"  - {p.audio.name} ({p.audio.stat().st_size / 1e6:.1f}MB) ← ref {p.reference.name}")

    # Device resolution
    device = args.device
    if device == "auto":
        device = "cuda" if (_HAS_NVML and _NVML_DEVICE_COUNT > 0) else "cpu"
    args.device = device

    if args.compute_type == "auto":
        args.compute_type = "float16" if device == "cuda" else "int8"
    args.models = requested

    gold_label = "**gold (hand-corrected)**" if args.gold else "**proxy** (auto-generated reference — WER is relative divergence, not absolute accuracy)"
    print(f"Reference quality: {gold_label}")

    results: List[ModelResult] = []
    for model_id in requested:
        results.append(run_model(model_id, pairs, device, args.compute_type))

    md = render_markdown(results, corpus, args, gold_label)
    print()
    print(md)

    # Save
    output_path = Path(args.output) if args.output else None
    if output_path is None:
        results_dir = Path(__file__).resolve().parent / "results"
        results_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = results_dir / f"{ts}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    print(f"\nSaved report to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
