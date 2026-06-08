#!/usr/bin/env python3
"""
asr-bench — benchmark local Whisper variants on your own audio.

Usage:
  python asr_bench.py --corpus ./test-corpus
  python asr_bench.py --corpus ./test-corpus --models small,medium
  python asr_bench.py --corpus ./test-corpus --device cpu

Installed (pip install .): the `asr-bench` console command is equivalent.

See README.md for corpus layout. See SPEC.md for the v0.2/v0.3 roadmap.
"""
from __future__ import annotations

__version__ = "0.3.5"

import argparse
import copy
import json
import math
import os
import re
import shutil
import statistics
import sys
import subprocess
import threading
import time
import importlib.util
import urllib.request

# Windows console defaults to cp1252 and chokes on most non-ASCII glyphs.
# Force UTF-8 on stdout/stderr so the progress + report render correctly.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


def _add_nvidia_dll_directories() -> None:
    """On Windows, make the nvidia-cublas-cu12 + nvidia-cudnn-cu12 wheel DLL
    dirs visible to ctranslate2's C++ loader. We use TWO mechanisms because the
    Python `os.add_dll_directory()` flag doesn't always reach `LoadLibrary`
    calls from native code:

    1. Prepend the dirs to PATH — universal, works for any LoadLibrary call.
    2. Call os.add_dll_directory() — belt-and-suspenders for pure-Python loads.

    No-op on non-Windows or when the wheels aren't installed."""
    try:
        import sysconfig
        import site
        candidates: List[str] = []
        for key in ("purelib", "platlib"):
            sp = sysconfig.get_paths().get(key)
            if sp:
                candidates.append(sp)
        user_sp = site.getusersitepackages() if hasattr(site, "getusersitepackages") else None
        if isinstance(user_sp, str):
            candidates.append(user_sp)
        elif isinstance(user_sp, list):
            candidates.extend(user_sp)
        # Look for nvidia/*/bin under each site-packages root.
        nvidia_bins: List[str] = []
        seen: set = set()
        for sp in candidates:
            for sub in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
                p = Path(sp) / "nvidia" / sub / "bin"
                if p.is_dir():
                    sp_str = str(p)
                    if sp_str not in seen:
                        nvidia_bins.append(sp_str)
                        seen.add(sp_str)
        if not nvidia_bins:
            return
        # Universal: prepend to PATH so the OS loader finds the DLLs.
        prepend = os.pathsep.join(nvidia_bins)
        os.environ["PATH"] = prepend + os.pathsep + os.environ.get("PATH", "")
        # Belt-and-suspenders: also register with Python's loader.
        if hasattr(os, "add_dll_directory"):
            for p in nvidia_bins:
                try:
                    os.add_dll_directory(p)
                except (FileNotFoundError, OSError):
                    pass
    except Exception:
        pass


# Path is needed inside the helper above; safe to import early.
from pathlib import Path  # noqa: E402

_add_nvidia_dll_directories()
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
# Path is imported earlier above for _add_nvidia_dll_directories — already in scope.

# ---- Re-exported shared support layer (lives in engines/base.py) ------------
# The Engine ABC, result/config dataclasses, metrics, VRAM sampling, audio
# decode, reference/caption parsing, and the VTT/words writers now live in the
# torch-free engines/base.py. They are re-exported here so the public surface
# (import asr_bench / python asr_bench.py) and the entire test suite keep working
# unchanged.
from engines.base import (  # noqa: E402,F401
    _HAS_NVML, _NVML_DEVICE_COUNT,
    HALLUCINATION_NGRAM, HALLUCINATION_MIN_WORDS, HALLUCINATION_MIN_CHARS,
    HALLUCINATION_REPEAT_COVERAGE, HALLUCINATION_COMPRESSION_RATIO,
    HALLUCINATION_INSERTION_RATE,
    gpu_used_bytes, gpu_total_and_free_bytes, gpu_name,
    detect_reference_origin, load_reference_text, Cue, parse_caption_cues,
    normalize_for_wer, WordMetrics, _repeat_coverage, _compression_ratio,
    compute_hallucination_signals, compute_word_metrics,
    Pair, _fused_base, _fmt_vtt_time, write_whisper_vtt, write_whisperx_vtt,
    write_words_sidecar, find_rttm, _model_label,
    ClipResult, ModelResult, RunConfig, Engine,
    VramSampler, group_words_into_cues, decode_to_pcm16, _audio_duration_sec,
)

# ---- Re-exported engine families (live in engines/<name>.py) ----------------
# Each engine class now lives in its own module under engines/; the ENGINES
# registry is assembled in engines/__init__.py. Re-exported here so the public
# surface (asr_bench.FasterWhisperEngine, etc.) and the test suite are unchanged.
from engines.faster_whisper import FasterWhisperEngine, model_disk_bytes  # noqa: E402,F401


# ---- Model registry ---------------------------------------------------------
MODELS: Dict[str, Dict] = {
    "small": {
        "engine": "faster-whisper",
        "display": "Whisper Small",
        "params": "244M",
        "developer": "OpenAI",
        "languages": "99 (multilingual)",
        "fw_name": "small",
        "notes": "Real-time on CPU. Decent for clear single speaker.",
    },
    "medium": {
        "engine": "faster-whisper",
        "display": "Whisper Medium",
        "params": "769M",
        "developer": "OpenAI",
        "languages": "99 (multilingual)",
        "fw_name": "medium",
        "notes": "Production sweet spot. ~2-3x realtime on CPU.",
    },
    "large-v3": {
        "engine": "faster-whisper",
        "display": "Whisper Large V3",
        "params": "1550M",
        "developer": "OpenAI",
        "languages": "99 (multilingual)",
        "fw_name": "large-v3",
        "notes": "State-of-art OpenAI accuracy. CPU is slow; GPU recommended.",
    },
    "large-v3-turbo": {
        "engine": "faster-whisper",
        "display": "Whisper Large V3 Turbo",
        "params": "809M",
        "developer": "OpenAI",
        "languages": "99 (multilingual)",
        "fw_name": "large-v3-turbo",
        "notes": "Distilled large-v3. Accuracy close to large at medium-class speed.",
    },
    "canary-nim": {
        "engine": "nim",
        "display": "Canary (NIM)",
        "params": "—",
        "developer": "NVIDIA",
        "languages": "en (+multi)",
        "riva_model": "",  # "" => let the NIM server pick its default model
        "notes": "NVIDIA NIM ASR via Riva gRPC. Endpoint set by --nim-url.",
    },
    "parakeet-tdt-0.6b-v2": {
        "engine": "nemo",
        "display": "Parakeet TDT 0.6B v2 (NeMo)",
        "params": "600M",
        "developer": "NVIDIA",
        "languages": "en",
        "nemo_model": "nvidia/parakeet-tdt-0.6b-v2",
        "notes": "NeMo FastConformer-TDT. Native word/segment timestamps -> VTT.",
    },
    "canary-qwen-2.5b": {
        "engine": "nemo",
        "display": "Canary-Qwen 2.5B (NeMo)",
        "params": "2.5B",
        "developer": "NVIDIA",
        "languages": "en",
        "nemo_model": "nvidia/canary-qwen-2.5b",
        "notes": "NeMo SALM (FastConformer enc + Qwen3 dec). WER-only - no native timestamps.",
    },
}

_NIM_ADHOC_RE = re.compile(r"^nim:(.+)$")
_NEMO_ADHOC_RE = re.compile(r"^nemo:(.*)$")
_WHISPERX_RE = re.compile(r"^(.+)\+whisperx$")
_WHISPERX_SIZES = {"small", "medium", "large-v3", "large-v3-turbo"}


def resolve_model_entry(model_id: str) -> Dict:
    """Resolve a --models token to a full engine entry.

    Returns a dict that always carries: id, engine, display, developer, params,
    languages, notes. NIM entries also carry riva_model; whisper entries carry
    fw_name. WhisperX entries also carry fw_name. Raises ValueError for unknown ids.
    """
    if model_id in MODELS:
        entry = dict(MODELS[model_id])
        entry.setdefault("engine", "faster-whisper")
        entry["id"] = model_id
        return entry
    wx = _WHISPERX_RE.match(model_id)
    if wx:
        size = wx.group(1).strip()
        if size not in _WHISPERX_SIZES:
            raise ValueError(
                f"unknown WhisperX base size '{size}' in '{model_id}' "
                f"(choices: {', '.join(sorted(_WHISPERX_SIZES))})"
            )
        base = MODELS[size]
        return {
            "id": model_id,
            "engine": "whisperx",
            "display": f"{base['display']} + WhisperX",
            "developer": base["developer"],
            "params": base["params"],
            "languages": base["languages"],
            "fw_name": size,
            "notes": "WhisperX: wav2vec2 word alignment + pyannote diarization.",
        }
    m = _NIM_ADHOC_RE.match(model_id)
    if m:
        name = m.group(1).strip()
        if not name:
            raise ValueError(f"empty NIM model name in '{model_id}'")
        return {
            "id": model_id,
            "engine": "nim",
            "display": f"NIM ({name})",
            "developer": "NVIDIA",
            "params": "—",
            "languages": "—",
            "riva_model": name,
            "notes": f"Ad-hoc NIM model '{name}' via Riva gRPC.",
        }
    nm = _NEMO_ADHOC_RE.match(model_id)
    if nm:
        name = nm.group(1).strip()
        if not name:
            raise ValueError(f"empty NeMo model name in '{model_id}'")
        return {
            "id": model_id,
            "engine": "nemo",
            "display": f"NeMo ({name})",
            "developer": "NVIDIA",
            "params": "—",
            "languages": "—",
            "nemo_model": name,
            "notes": f"Ad-hoc NeMo model '{name}'.",
        }
    raise ValueError(f"unknown model id: {model_id}")


# ---- Pair discovery ---------------------------------------------------------
AUDIO_EXTS = {".mp4", ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}


def discover_pairs(corpus: Path) -> List[Pair]:
    """Find (audio, reference) pairs in three supported layouts.

    Deduplication rule: if both a .mp4 and a same-stem .mp3 exist (common when
    an extraction step left an mp3 leftover), prefer the .mp4 and skip the mp3.
    The user benchmarks the source-of-truth video; ad-hoc extractions are noise.
    """
    manifest = corpus / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return [Pair(corpus / c["audio"], corpus / c["reference"]) for c in data["clips"]]

    # Pre-compute which stems have a primary video (.mp4/.webm) so we can skip
    # secondary audio files that just shadow them.
    PRIMARY_EXTS = {".mp4", ".webm", ".m4a"}
    SECONDARY_EXTS = {".mp3", ".wav", ".flac", ".ogg"}
    primary_stems = {
        p.stem.lower()
        for p in corpus.iterdir()
        if p.is_file() and p.suffix.lower() in PRIMARY_EXTS
    }

    pairs: List[Pair] = []
    for audio in sorted(corpus.iterdir()):
        if not audio.is_file() or audio.suffix.lower() not in AUDIO_EXTS:
            continue
        if audio.suffix.lower() in SECONDARY_EXTS and audio.stem.lower() in primary_stems:
            continue  # mp4 sibling already covers this stem
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


# ---- GPU probe + batch-size recommendation ----------------------------------
# Rough VRAM cost per model at compute_type=float16, including a base + per-batch-item slope.
# Numbers come from observed peaks; conservative so the recommendation doesn't OOM.
_MODEL_VRAM_COST: Dict[str, Tuple[int, int]] = {
    # model_id -> (base_bytes, per_batch_item_bytes). faster-whisper only — it is
    # the only engine that uses --batch-size. nim/whisperx/nemo are intentionally
    # absent: they manage their own batching (in-container or in the subprocess
    # runner), so recommend_batch_size correctly ignores them. NeMo peak VRAM is
    # measured live by the runner (torch.cuda.max_memory_allocated), not sized here.
    "small":          (int(0.8 * 1024**3), int(0.18 * 1024**3)),
    "medium":         (int(1.8 * 1024**3), int(0.45 * 1024**3)),
    "large-v3":       (int(4.0 * 1024**3), int(0.90 * 1024**3)),
    "large-v3-turbo": (int(2.0 * 1024**3), int(0.45 * 1024**3)),
}


def recommend_batch_size(model_ids: List[str], headroom_bytes: int = int(2 * 1024**3)) -> Tuple[int, str]:
    """Suggest a batch size that fits the largest queued model in available VRAM.

    Returns (batch_size, reason). Falls back to 1 with explanation when we can't
    probe or when free VRAM is too small to safely batch.
    """
    _, free = gpu_total_and_free_bytes()
    if free is None:
        return (1, "no NVIDIA GPU detected — staying sequential")
    # The model that constrains us is the one with the highest per-item slope.
    worst = max(
        (mid for mid in model_ids if mid in _MODEL_VRAM_COST),
        key=lambda mid: _MODEL_VRAM_COST[mid][1],
        default=None,
    )
    if worst is None:
        return (1, "no batch-cost data for selected models — staying sequential")
    base, per_item = _MODEL_VRAM_COST[worst]
    usable = free - headroom_bytes - base
    if usable <= 0:
        return (1, f"only {fmt_bytes(free)} free — keeping batch=1 to avoid OOM (constraining model: {worst})")
    candidate = max(1, usable // per_item)
    # Cap at 32 — diminishing returns past that for Whisper-sized models, and
    # extremely large batches hurt latency without helping throughput.
    candidate = min(candidate, 32)
    return (
        candidate,
        f"{fmt_bytes(free)} free, constraining model {worst} "
        f"({fmt_bytes(base)} base + {fmt_bytes(per_item)}/batch item, {fmt_bytes(headroom_bytes)} safety headroom)",
    )


# ---- Helpers ----------------------------------------------------------------
def _fmt_pct(value: float) -> str:
    """Format a 0-1 metric as a 1-decimal percentage, or '—' if NaN/None."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value * 100:.1f}"


def _vram_cell(value: Optional[int], is_total: bool) -> str:
    """Render a VRAM cell; mark NIM 'total used' values with a trailing '*'."""
    if value is None:
        return "n/a"
    return fmt_bytes(value) + ("*" if is_total else "")


def _disk_cell(result: "ModelResult") -> str:
    return "n/a" if result.disk_bytes is None else fmt_bytes(result.disk_bytes)


def _md_escape(text: object) -> str:
    """Escape characters that would break a Markdown table cell.

    A literal `|` in a filename or model name would otherwise split the row into
    extra columns; newlines would terminate it. Escape the pipe and flatten any
    newlines so free text (clip names, model display, notes) is row-safe."""
    return str(text).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


# ---- NIM helpers ------------------------------------------------------------
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


ENGINES: Dict[str, type] = {
    "faster-whisper": FasterWhisperEngine,
    "nim": NimEngine,
}


# ---- WhisperX ---------------------------------------------------------------
@dataclass
class WhisperXResult:
    """Parsed output of a WhisperX run (transcribe + align + optional diarize)."""
    segments: List[Dict]                 # [{start, end, text, speaker?}]
    words: List[Dict] = field(default_factory=list)
    speakers: List[str] = field(default_factory=list)
    der: Optional[float] = None
    language: str = ""

    @classmethod
    def from_dict(cls, d: Dict) -> "WhisperXResult":
        return cls(
            segments=list(d.get("segments") or []),
            words=list(d.get("words") or []),
            speakers=list(d.get("speakers") or []),
            der=d.get("der"),
            language=d.get("language") or "",
        )

    def text(self) -> str:
        return " ".join(s.get("text", "").strip() for s in self.segments).strip()

    def speaker_segments(self) -> List[Tuple[float, float, str]]:
        out: List[Tuple[float, float, str]] = []
        for s in self.segments:
            if s.get("speaker"):
                out.append((float(s["start"]), float(s["end"]), s["speaker"]))
        return out


# ---- NeMo result ------------------------------------------------------------
@dataclass
class NeMoResult:
    """Parsed output of a NeMo run. Parakeet yields segments+words (native
    timestamps); Canary-Qwen yields full_text only (no timestamps). Duck-typed to
    reuse write_whisperx_vtt / write_words_sidecar (they read .segments / .words)."""
    segments: List[Dict] = field(default_factory=list)   # [{start, end, text}]
    words: List[Dict] = field(default_factory=list)       # [{word, start, end}]
    full_text: str = ""
    transcribe_sec: Optional[float] = None
    vram_peak_bytes: Optional[int] = None
    language: str = ""

    @classmethod
    def from_dict(cls, d: Dict) -> "NeMoResult":
        return cls(
            segments=list(d.get("segments") or []),
            words=list(d.get("words") or []),
            full_text=d.get("text") or "",
            transcribe_sec=d.get("transcribe_sec"),
            vram_peak_bytes=d.get("vram_peak_bytes"),
            language=d.get("language") or "",
        )

    def text(self) -> str:
        if self.segments:
            return " ".join(s.get("text", "").strip() for s in self.segments).strip()
        return self.full_text.strip()

    def has_timestamps(self) -> bool:
        return bool(self.segments)

    def speaker_segments(self) -> List[Tuple[float, float, str]]:
        return []   # v0.4 NeMo is transcription-only


# ---- WhisperX adapter -------------------------------------------------------
_RUNNER_PATH = str(Path(__file__).resolve().parent / "whisperx_runner.py")


class WhisperXAdapter(ABC):
    """Turns an audio file into a WhisperXResult. Two real impls (in-process /
    subprocess to a 3.12 venv) + a Fake for tests."""
    name: str = ""

    @abstractmethod
    def transcribe(self, audio_path: str, model: str, cfg: "RunConfig",
                   rttm: Optional[str]) -> "WhisperXResult":
        ...


class FakeWhisperXAdapter(WhisperXAdapter):
    name = "fake"

    def __init__(self, result: "WhisperXResult"):
        self._result = result

    def transcribe(self, audio_path, model, cfg, rttm):
        return self._result


def _runner_args(audio_path: str, model: str, cfg: "RunConfig", rttm: Optional[str]) -> List[str]:
    args = [_RUNNER_PATH, "--audio", audio_path, "--model", model,
            "--device", cfg.device, "--language", "en"]
    if cfg.diarize:
        args.append("--diarize")
        if cfg.min_speakers is not None:
            args += ["--min-speakers", str(cfg.min_speakers)]
        if cfg.max_speakers is not None:
            args += ["--max-speakers", str(cfg.max_speakers)]
    if rttm:
        args += ["--rttm", rttm]
    return args


class InProcessWhisperX(WhisperXAdapter):
    """Calls whisperx_runner.run_whisperx directly (torch importable here)."""
    name = "in-process"

    def transcribe(self, audio_path, model, cfg, rttm):
        import whisperx_runner
        d = whisperx_runner.run_whisperx(
            audio=audio_path, model=model, device=cfg.device, language="en",
            diarize=cfg.diarize, hf_token=cfg.hf_token,
            min_speakers=cfg.min_speakers, max_speakers=cfg.max_speakers, rttm=rttm,
        )
        return WhisperXResult.from_dict(d)


class SubprocessWhisperX(WhisperXAdapter):
    """Runs whisperx_runner.py under a configured 3.12 venv python; parses JSON."""
    name = "subprocess"

    def __init__(self, python: str, timeout: float = 3600.0):
        self.python = python
        self.timeout = timeout

    def transcribe(self, audio_path, model, cfg, rttm):
        exe = shutil.which(self.python) or self.python
        cmd = [exe, *_runner_args(audio_path, model, cfg, rttm)]
        env = dict(os.environ)
        if cfg.hf_token:
            env["HF_TOKEN"] = cfg.hf_token
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=self.timeout, check=False, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"whisperx_runner failed ({proc.returncode}): {proc.stderr[:500]}")
        return WhisperXResult.from_dict(json.loads(proc.stdout))


def _default_whisperx_python() -> Optional[str]:
    """Look for a conventional sibling venv (./.venv-whisperx)."""
    root = Path(__file__).resolve().parent
    for rel in (".venv-whisperx/Scripts/python.exe", ".venv-whisperx/bin/python"):
        cand = root / rel
        if cand.is_file():
            return str(cand)
    return None


def make_whisperx_adapter(cfg: "RunConfig") -> WhisperXAdapter:
    """Auto-select: in-process if torch importable; else subprocess to a venv
    python (cfg.whisperx_python or a default ./.venv-whisperx); else error."""
    if importlib.util.find_spec("torch") is not None:
        return InProcessWhisperX()
    venv_py = cfg.whisperx_python or _default_whisperx_python()
    if venv_py:
        return SubprocessWhisperX(venv_py)
    raise RuntimeError(
        "WhisperX needs torch (not importable here) or a 3.12 venv. Create one "
        "(py -3.12 -m venv .venv-whisperx && .venv-whisperx/Scripts/pip install whisperx) "
        "and pass --whisperx-python, or run asr-bench under a torch-enabled interpreter."
    )


# ---- NeMo adapter -----------------------------------------------------------
_NEMO_RUNNER_PATH = str(Path(__file__).resolve().parent / "nemo_runner.py")


class NeMoAdapter(ABC):
    """Turns an audio file into a NeMoResult. Subprocess (to a 3.12 .venv-nemo)
    is the only real impl — torch has no 3.14 wheels — plus a Fake for tests."""
    name: str = ""

    @abstractmethod
    def transcribe(self, audio_path: str, model: str, cfg: "RunConfig") -> "NeMoResult":
        ...


class FakeNeMoAdapter(NeMoAdapter):
    name = "fake"

    def __init__(self, result: "NeMoResult"):
        self._result = result

    def transcribe(self, audio_path, model, cfg):
        return self._result


def _nemo_runner_args(audio_path: str, model: str, cfg: "RunConfig") -> List[str]:
    return [_NEMO_RUNNER_PATH, "--audio", audio_path, "--model", model,
            "--device", cfg.device, "--language", "en"]


class SubprocessNeMo(NeMoAdapter):
    """Runs nemo_runner.py under a configured 3.12 venv python; parses JSON."""
    name = "subprocess"

    def __init__(self, python: str, timeout: float = 7200.0):  # NeMo first-run downloads can exceed WhisperX's 1h
        self.python = python
        self.timeout = timeout

    def transcribe(self, audio_path, model, cfg):
        exe = shutil.which(self.python) or self.python
        cmd = [exe, *_nemo_runner_args(audio_path, model, cfg)]
        env = dict(os.environ)
        # torch 2.6+ defaults weights_only=True; some NeMo checkpoints need this off.
        env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=self.timeout, check=False, env=env)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"nemo_runner timed out after {self.timeout}s — increase "
                "SubprocessNeMo(timeout=...) or use faster storage for the model cache")
        if proc.returncode != 0:
            raise RuntimeError(f"nemo_runner failed ({proc.returncode}): {proc.stderr[:500]}")
        return NeMoResult.from_dict(json.loads(proc.stdout))


def _default_nemo_python() -> Optional[str]:
    """Look for a conventional sibling venv (./.venv-nemo)."""
    root = Path(__file__).resolve().parent
    for rel in (".venv-nemo/Scripts/python.exe", ".venv-nemo/bin/python"):
        cand = root / rel
        if cand.is_file():
            return str(cand)
    return None


def make_nemo_adapter(cfg: "RunConfig") -> NeMoAdapter:
    """Subprocess to a 3.12 venv python (cfg.nemo_python or ./.venv-nemo). NeMo is
    never run in-process: core is Python 3.14 (no torch wheels) and a fresh
    subprocess gives clean per-model VRAM teardown."""
    venv_py = cfg.nemo_python or _default_nemo_python()
    if venv_py:
        return SubprocessNeMo(venv_py)
    raise RuntimeError(
        "NeMo needs a Python 3.12 venv with torch (cu128) + nemo_toolkit. "
        "Run setup_nemo_venv.ps1 (creates ./.venv-nemo) or pass --nemo-python."
    )


class WhisperXEngine(Engine):
    name = "whisperx"

    def run(self, entry: Dict, pairs: List[Pair], cfg: RunConfig) -> ModelResult:
        adapter = make_whisperx_adapter(cfg)
        print(f"\n[{entry['display']}] using WhisperX adapter: {adapter.name}", flush=True)
        result_model = ModelResult(
            model_id=entry["id"], display=entry["display"], fw_name=entry.get("fw_name", ""),
            params=entry["params"], developer=entry["developer"], languages=entry["languages"],
            notes=entry["notes"], disk_bytes=None, load_sec=0.0,
            engine="whisperx", vram_is_total=False,
        )
        for clip_idx, pair in enumerate(pairs, start=1):
            print(f"  [{clip_idx}/{len(pairs)}] transcribing {pair.audio.name}...", flush=True)
            ref_text = load_reference_text(pair.reference)
            ref_origin, ref_label = detect_reference_origin(pair.reference)
            rttm = find_rttm(pair.audio)
            t0 = time.time()
            try:
                wx = adapter.transcribe(str(pair.audio), entry["fw_name"], cfg,
                                        str(rttm) if rttm else None)
            except Exception as e:
                print(f"  ERROR whisperx on {pair.audio.name}: {e}", file=sys.stderr)
                continue
            transcribe_sec = time.time() - t0

            hypothesis = wx.text()
            ref_norm = normalize_for_wer(ref_text)
            hyp_norm = normalize_for_wer(hypothesis)
            metrics = compute_word_metrics(ref_norm, hyp_norm)

            label = _model_label(entry["id"])
            vtt_path = write_whisperx_vtt(pair.audio, label, wx)
            write_words_sidecar(pair.audio, label, wx)

            spk_segs = wx.speaker_segments()
            audio_sec = _audio_duration_sec(str(pair.audio)) or (
                wx.segments[-1]["end"] if wx.segments else 0.0)
            rtfx = audio_sec / transcribe_sec if transcribe_sec > 0 else 0.0
            der_val = wx.der if wx.der is not None else float("nan")

            print(f"    {audio_sec:.1f}s in {transcribe_sec:.1f}s "
                  f"(RTFx {rtfx:.2f}, WER {metrics.wer*100:.1f}%, "
                  f"{len(wx.speakers)} speaker(s))", flush=True)

            rep_cov, comp_ratio = compute_hallucination_signals(hypothesis, hyp_norm)
            result_model.clips.append(ClipResult(
                audio=pair.audio.name, audio_sec=audio_sec, transcribe_sec=transcribe_sec,
                rtfx=rtfx, vram_peak_bytes=None, hypothesis=hypothesis,
                reference_normalized=ref_norm, hypothesis_normalized=hyp_norm,
                wer=metrics.wer, mer=metrics.mer, wil=metrics.wil, cer=metrics.cer,
                repeat_coverage=rep_cov, compression_ratio=comp_ratio,
                hits=metrics.hits,
                substitutions=metrics.substitutions, deletions=metrics.deletions,
                insertions=metrics.insertions, cue_count=len(wx.segments),
                vtt_path=str(vtt_path), reference_origin=ref_origin, reference_label=ref_label,
                speaker_segments=spk_segs, num_speakers=len(wx.speakers), der=der_val,
            ))
        if not result_model.clips:
            result_model.notes = "ALL CLIPS FAILED — check WhisperX setup/venv/token and stderr above"
        return result_model


ENGINES["whisperx"] = WhisperXEngine


class NeMoEngine(Engine):
    name = "nemo"

    def run(self, entry: Dict, pairs: List[Pair], cfg: RunConfig) -> ModelResult:
        adapter = make_nemo_adapter(cfg)
        print(f"\n[{entry['display']}] using NeMo adapter: {adapter.name}", flush=True)
        result_model = ModelResult(
            model_id=entry["id"], display=entry["display"], fw_name=entry.get("fw_name", ""),
            params=entry["params"], developer=entry["developer"], languages=entry["languages"],
            notes=entry["notes"], disk_bytes=None, load_sec=0.0,
            engine="nemo", vram_is_total=False,
        )
        for clip_idx, pair in enumerate(pairs, start=1):
            print(f"  [{clip_idx}/{len(pairs)}] transcribing {pair.audio.name}...", flush=True)
            ref_text = load_reference_text(pair.reference)
            ref_origin, ref_label = detect_reference_origin(pair.reference)
            t0 = time.time()
            try:
                nm = adapter.transcribe(str(pair.audio), entry["nemo_model"], cfg)
            except Exception as e:
                print(f"  ERROR nemo on {pair.audio.name}: {e}", file=sys.stderr)
                continue
            wall = time.time() - t0
            transcribe_sec = nm.transcribe_sec if nm.transcribe_sec is not None else wall

            hypothesis = nm.text()
            ref_norm = normalize_for_wer(ref_text)
            hyp_norm = normalize_for_wer(hypothesis)
            metrics = compute_word_metrics(ref_norm, hyp_norm)

            label = _model_label(entry["id"])
            vtt_path = None
            cue_count = 0
            if nm.has_timestamps():
                vtt_path = str(write_whisperx_vtt(pair.audio, label, nm))
                write_words_sidecar(pair.audio, label, nm)
                cue_count = len(nm.segments)

            audio_sec = _audio_duration_sec(str(pair.audio)) or (
                nm.segments[-1]["end"] if nm.segments else 0.0)
            rtfx = audio_sec / transcribe_sec if transcribe_sec > 0 else 0.0

            print(f"    {audio_sec:.1f}s in {transcribe_sec:.1f}s "
                  f"(RTFx {rtfx:.2f}, WER {metrics.wer*100:.1f}%, "
                  f"{'timestamps' if nm.has_timestamps() else 'text-only'})", flush=True)

            rep_cov, comp_ratio = compute_hallucination_signals(hypothesis, hyp_norm)
            result_model.clips.append(ClipResult(
                audio=pair.audio.name, audio_sec=audio_sec, transcribe_sec=transcribe_sec,
                rtfx=rtfx, vram_peak_bytes=nm.vram_peak_bytes, hypothesis=hypothesis,
                reference_normalized=ref_norm, hypothesis_normalized=hyp_norm,
                wer=metrics.wer, mer=metrics.mer, wil=metrics.wil, cer=metrics.cer,
                repeat_coverage=rep_cov, compression_ratio=comp_ratio,
                hits=metrics.hits, substitutions=metrics.substitutions,
                deletions=metrics.deletions, insertions=metrics.insertions,
                cue_count=cue_count, vtt_path=vtt_path,
                reference_origin=ref_origin, reference_label=ref_label,
            ))
        if not result_model.clips:
            result_model.notes = "ALL CLIPS FAILED — check NeMo setup/.venv-nemo and stderr above"
        return result_model


ENGINES["nemo"] = NeMoEngine


# ---- Fusion -----------------------------------------------------------------
_CONTEXT_GLOSSARY_HEADER_RE = re.compile(r"^#+\s*glossary\b", re.IGNORECASE | re.MULTILINE)


def init_context_template() -> str:
    """A guided context.md the user fills in, then passes via --context."""
    return """# Fusion context

Fill in what the fusion LLM should know about this corpus. Everything here is
fed to the model for every clip. Delete sections you don't need.

## Topic / course
<!-- e.g. "Undergraduate intro statistics; lectures cover hypothesis testing." -->

## Schedule & recurring times
<!-- e.g. "I teach 9-11am; there are no evening sessions, so 'final at 9pm' is wrong." -->

## Names (people, places) — canonical spelling
<!-- e.g. "Dr. Nguyen; the dataset is called CIFAR-10." -->

## Jargon & acronyms
<!-- e.g. "Spell 'AI' (not 'I'); 'p-value' (not 'p value')." -->

## Known mishearings to watch for
<!-- e.g. "'their' vs 'there'; 'affect' vs 'effect'." -->

## Style preferences
<!-- e.g. captions: keep verbatim; KB: full sentences, normalize numbers. -->

## Glossary
<!-- One correction per line, e.g.:
AI not I
CIFAR-10 not cipher ten
-->
"""


def load_context(context_path: Optional[str], glossary_path: Optional[str]) -> Tuple[str, str]:
    """Return (context_text, glossary_text).

    The glossary is the '## Glossary' section of the context file, unless a
    separate --glossary file is given (which overrides it). Missing files -> "".
    """
    context_text = ""
    glossary_text = ""
    if context_path:
        raw = Path(context_path).read_text(encoding="utf-8", errors="replace")
        m = _CONTEXT_GLOSSARY_HEADER_RE.search(raw)
        if m:
            context_text = raw[: m.start()].strip()
            glossary_text = raw[m.end():].strip()
        else:
            context_text = raw.strip()
    if glossary_path:
        glossary_text = Path(glossary_path).read_text(encoding="utf-8", errors="replace").strip()
    return context_text, glossary_text


def build_windows(duration: float, window: float, overlap: float) -> List[Tuple[float, float]]:
    """Tile [0, duration] into (start, end) spans of length `window`, stepping by
    stride = window - overlap. The overlap is carried into prompts as context; the
    final window is clamped to `duration`. Returns a single full-span window when
    the clip is shorter than one window.
    """
    if duration <= window or window <= 0:
        return [(0.0, duration)]
    stride = max(window - overlap, 1.0)
    spans: List[Tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(start + window, duration)
        spans.append((round(start, 3), round(end, 3)))
        if end >= duration:
            break
        start += stride
    return spans


def collect_window_text(cues: List[Cue], start: float, end: float) -> str:
    """Concatenate the text of all cues that overlap [start, end)."""
    parts = [c.text for c in cues if c.end > start and c.start < end]
    return " ".join(parts).strip()


def collect_window_text_midpoint(cues: List[Cue], start: float, end: float) -> str:
    """Text of cues whose MIDPOINT falls in [start, end). Assigns each cue to
    exactly one window (no shared boundary cues) — used for non-overlapping
    verbatim tiling so captions don't duplicate across cues."""
    parts = [c.text for c in cues if start <= (c.start + c.end) / 2.0 < end]
    return " ".join(parts).strip()


@dataclass
class WindowPayload:
    start: float
    end: float
    sources: Dict[str, str]          # source label -> text in this window
    prev_fused: str = ""             # previous window's fused output (carryover)


_VERBATIM_INSTRUCTIONS = (
    "You are reconciling several speech-to-text transcripts of the SAME audio span "
    "into the single most accurate VERBATIM transcript of what was actually said.\n"
    "Rules:\n"
    "- Restore the actually-spoken words. When sources disagree (e.g. 'AI' vs 'I'), "
    "choose the reading that fits the context and glossary.\n"
    "- Do NOT rephrase, summarize, or clean up grammar. Preserve the speaker's wording "
    "and disfluencies.\n"
    "- Output ONLY the corrected transcript text for this span. No commentary, no labels."
)

_KB_INSTRUCTIONS = (
    "You are merging several speech-to-text transcripts of the SAME audio span into "
    "one clean, readable passage for a searchable knowledge base.\n"
    "Rules:\n"
    "- Rewrite for clarity and correct grammar. Normalize times, numbers and dates "
    "(e.g. '9 to 11' -> '9:00-11:00 am') using the context.\n"
    "- Fix mishearings and proper nouns using the glossary and context. Prefer meaning "
    "over literal wording, but never invent facts.\n"
    "- Output ONLY the cleaned passage text for this span. No commentary, no labels."
)


def build_fusion_prompt(payload: "WindowPayload", profile: str, context: str, glossary: str) -> str:
    instructions = _KB_INSTRUCTIONS if profile == "kb" else _VERBATIM_INSTRUCTIONS
    parts: List[str] = [instructions, ""]
    if context.strip():
        parts += ["## Context", context.strip(), ""]
    if glossary.strip():
        parts += ["## Glossary (canonical spellings / corrections)", glossary.strip(), ""]
    if payload.prev_fused.strip():
        parts += ["## Preceding text (already finalized — for continuity only, do not repeat)",
                  payload.prev_fused.strip(), ""]
    parts.append(f"## Transcripts for span {payload.start:.1f}s-{payload.end:.1f}s")
    for label, text in payload.sources.items():
        parts.append(f"### {label}")
        parts.append(text.strip() or "(empty)")
    parts.append("")
    parts.append("## Output")
    return "\n".join(parts)


# ---- Fusion orchestrator ----------------------------------------------------
@dataclass
class FusionResult:
    verbatim_cues: List[Cue] = field(default_factory=list)
    kb_chunks: List[Dict] = field(default_factory=list)   # {start, end, text}
    flags: List[str] = field(default_factory=list)        # drift warnings


def write_fused_vtt(audio_path: Path, cues: List[Cue]) -> Path:
    """Write a WebVTT from fused cues, named <base>_Captions_Fused.vtt."""
    out = audio_path.parent / f"{_fused_base(audio_path)}_Captions_Fused.vtt"
    lines: List[str] = ["WEBVTT", ""]
    cue_num = 0
    for c in cues:
        text = c.text.strip()
        if not text:
            continue
        cue_num += 1
        lines.append(str(cue_num))
        lines.append(f"{_fmt_vtt_time(c.start)} --> {_fmt_vtt_time(c.end)}")
        lines.append(text)
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_kb_jsonl(audio_path: Path, chunks: List[Dict]) -> Path:
    """Write overlapping KB chunks as newline-delimited JSON, named <base>_KB_Fused.jsonl."""
    out = audio_path.parent / f"{_fused_base(audio_path)}_KB_Fused.jsonl"
    lines = [json.dumps(c, ensure_ascii=False) for c in chunks]
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


def write_kb_md(audio_path: Path, chunks: List[Dict]) -> Path:
    """Write overlapping KB chunks as a readable Markdown file, named <base>_KB_Fused.md."""
    out = audio_path.parent / f"{_fused_base(audio_path)}_KB_Fused.md"
    lines: List[str] = [f"# Knowledge base — {_fused_base(audio_path)}", ""]
    for c in chunks:
        lines.append(f"## {_fmt_vtt_time(c['start'])} – {_fmt_vtt_time(c['end'])}")
        lines.append("")
        lines.append(c["text"].strip())
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def fuse_clip(
    duration: float,
    base_label: str,
    sources: Dict[str, List[Cue]],
    profiles: List[str],
    backend: "LLMBackend",
    context: str,
    glossary: str,
    window: float,
    overlap: float,
    drift_threshold: float,
) -> FusionResult:
    """Window the timeline, fuse each window per requested profile, assemble.

    - verbatim: non-overlapping window tiling (overlap=0) with midpoint cue
      assignment — each source cue belongs to exactly one window (no shared
      boundary cues), so caption text never duplicates across adjacent cues.
      Verbatim cue span equals the non-overlapping window itself.
    - kb: overlapping windows (overlap as supplied) with full boundary-cue
      inclusion via collect_window_text — preserves RAG context continuity.
    - drift guard: per window, WER(fused vs base text); flagged if > threshold.
    """
    res = FusionResult()
    for profile in profiles:
        if profile == "verbatim":
            wins = build_windows(duration, window, 0.0)        # non-overlapping tiles
            collect = collect_window_text_midpoint
        else:
            wins = build_windows(duration, window, overlap)    # overlapping (RAG)
            collect = collect_window_text
        prev = ""
        for (w_start, w_end) in wins:
            payload_sources = {label: collect(cues, w_start, w_end) for label, cues in sources.items()}
            base_text = payload_sources.get(base_label, "")
            payload = WindowPayload(w_start, w_end, payload_sources, prev_fused=prev)
            prompt = build_fusion_prompt(payload, profile, context, glossary)
            try:
                fused = backend.generate(prompt).strip()
            except Exception as e:
                fused = ""
                res.flags.append(f"[{w_start:.0f}-{w_end:.0f}s {profile}] backend error: {e}")
            prev = fused
            # Drift guard: high WER between base source and fused output signals
            # the LLM may have hallucinated or radically paraphrased.
            if base_text and fused:
                drift = compute_word_metrics(
                    normalize_for_wer(base_text), normalize_for_wer(fused)
                ).wer
                if not math.isnan(drift) and drift > drift_threshold:
                    res.flags.append(
                        f"[{w_start:.0f}-{w_end:.0f}s {profile}] drift WER {drift*100:.0f}% vs base — review"
                    )
            if not fused.strip():
                continue
            if profile == "verbatim":
                res.verbatim_cues.append(Cue(w_start, w_end, fused.strip()))
            else:
                res.kb_chunks.append({"start": w_start, "end": w_end, "text": fused.strip()})
    return res


# ---- Fusion re-scoring -------------------------------------------------------

def rescore_against_reference(
    results: List["ModelResult"],
    reference_cues_by_clip: Dict[str, List[Cue]],
) -> List["ModelResult"]:
    """Return deep copies of `results` with each clip's metrics recomputed against
    the fused verbatim reference (keyed by clip audio filename).

    Models are scored on their stored `hypothesis`. Clips with no matching
    reference are left unscored (NaN). The originals are not mutated.
    """
    out: List[ModelResult] = []
    for r in results:
        r2 = copy.deepcopy(r)
        for c in r2.clips:
            ref_cues = reference_cues_by_clip.get(c.audio)
            if not ref_cues:
                c.wer = c.mer = c.wil = float("nan")
                continue
            ref_text = normalize_for_wer(" ".join(cu.text for cu in ref_cues))
            hyp_text = normalize_for_wer(c.hypothesis)
            m = compute_word_metrics(ref_text, hyp_text)
            c.wer, c.mer, c.wil = m.wer, m.mer, m.wil
            c.hits, c.substitutions, c.deletions, c.insertions = (
                m.hits, m.substitutions, m.deletions, m.insertions,
            )
        out.append(r2)
    return out


def render_fused_rescore_table(results: List["ModelResult"]) -> str:
    lines: List[str] = []
    lines.append("## Scores vs fused verbatim reference")
    lines.append("")
    lines.append(
        "> **Reference = fused verbatim consensus (agreement-biased).** This reference "
        "was built from the models below, so scores favor models that agreed with the "
        "majority. Treat these as *relative*, not absolute accuracy."
    )
    lines.append("")
    lines.append("| Model | WER% | MER% | WIL% |")
    lines.append("|---|---|---|---|")
    for r in results:
        wer = _fmt_pct(r.avg_wer) if r.clips else "—"
        mer = _fmt_pct(r.avg_mer) if r.clips else "—"
        wil = _fmt_pct(r.avg_wil) if r.clips else "—"
        lines.append(f"| {r.display} | {wer} | {mer} | {wil} |")
    lines.append("")
    return "\n".join(lines)


def run_fusion_stage(
    results: List["ModelResult"],
    pairs: List["Pair"],
    backend: "LLMBackend",
    profiles: List[str],
    base_label: str,
    context: str,
    glossary: str,
    window: float,
    overlap: float,
    drift_threshold: float,
    rescore: bool,
) -> Tuple[str, Optional[List["ModelResult"]]]:
    """Fuse every clip and write outputs. Returns (markdown_section, rescored_or_None).

    Sources per clip = each model's written VTT (parsed back to timed cues) +
    the Panopto/reference caption file (if it parses as timed cues).
    """
    lines: List[str] = ["## Fusion", ""]
    lines.append(f"- Backend: `{backend.name}`  Profiles: `{', '.join(profiles)}`  "
                 f"Window: {window:.0f}s / overlap {overlap:.0f}s  Base: `{base_label}`")
    lines.append("")
    lines.append(
        "> **Accessibility note:** only the *verbatim* output targets ADA/WCAG caption "
        "fidelity. The *kb* output is rephrased and is **not** compliant captions."
    )
    lines.append("")

    verbatim_ref_by_clip: Dict[str, List[Cue]] = {}
    pair_by_audio = {p.audio.name: p for p in pairs}

    for clip_idx in range(len(results[0].clips) if results else 0):
        audio_name = results[0].clips[clip_idx].audio
        pair = pair_by_audio.get(audio_name)
        if pair is None:
            continue
        audio_path = pair.audio

        sources: Dict[str, List[Cue]] = {}
        duration = results[0].clips[clip_idx].audio_sec or 1.0
        for r in results:
            if clip_idx < len(r.clips) and r.clips[clip_idx].vtt_path:
                vp = Path(r.clips[clip_idx].vtt_path)
                if vp.is_file():
                    sources[r.model_id] = parse_caption_cues(vp)
        try:
            ref_cues = parse_caption_cues(pair.reference)
            if ref_cues:
                sources["Panopto"] = ref_cues
        except Exception:
            pass

        if not sources:
            lines.append(f"- {audio_name}: no parseable sources — skipped")
            continue

        res = fuse_clip(
            duration=duration, base_label=base_label, sources=sources,
            profiles=profiles, backend=backend, context=context, glossary=glossary,
            window=window, overlap=overlap, drift_threshold=drift_threshold,
        )
        written: List[str] = []
        if "verbatim" in profiles and res.verbatim_cues:
            vtt_out = write_fused_vtt(audio_path, res.verbatim_cues)
            verbatim_ref_by_clip[audio_name] = res.verbatim_cues
            written.append(vtt_out.name)
        if "kb" in profiles and res.kb_chunks:
            written.append(write_kb_jsonl(audio_path, res.kb_chunks).name)
            written.append(write_kb_md(audio_path, res.kb_chunks).name)
        lines.append(f"- **{audio_name}** → {', '.join(f'`{w}`' for w in written) or '(nothing written)'}")
        for flag in res.flags:
            lines.append(f"  - ⚠️ {flag}")
    lines.append("")

    rescored = None
    if rescore and verbatim_ref_by_clip:
        rescored = rescore_against_reference(results, verbatim_ref_by_clip)
    return "\n".join(lines), rescored


# ---- LLM backends -----------------------------------------------------------
class LLMBackend(ABC):
    """Minimal contract: turn a prompt into text. Fusion builds the prompt; the
    backend only generates. Keeps profile/prompt logic in one place (DRY)."""
    name: str = ""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...


class FakeLLMBackend(LLMBackend):
    """Deterministic, dependency-free backend for tests."""
    name = "fake"

    def __init__(self, fn=None):
        self._fn = fn or (lambda prompt: prompt)

    def generate(self, prompt: str) -> str:
        return self._fn(prompt)


class OllamaBackend(LLMBackend):
    """Local Ollama HTTP backend (default). Offline, free, no API key."""
    name = "ollama"

    def __init__(self, model: str = "qwen2.5", host: str = "http://localhost:11434", timeout: float = 300.0):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("response") or "").strip()


class CliBackend(LLMBackend):
    """Shell out to an authenticated frontier CLI (e.g. `claude -p`, `gemini -p {prompt}`).

    If the command contains a ``{prompt}`` token it is substituted with the
    prompt as an argument (for CLIs like ``gemini -p`` that take the prompt as
    an arg); otherwise the prompt is piped on stdin (for CLIs like ``claude
    -p``).  Arg substitution is subject to OS arg-length limits; prefer stdin
    for very large prompts.

    The executable (cmd[0]) is resolved via ``shutil.which`` before invoking
    subprocess so that npm-installed CLI shims (e.g. ``gemini.CMD``,
    ``codex.CMD`` on Windows) are found without requiring ``shell=True``.
    Falls back to the bare name when ``which`` returns None, so a missing CLI
    still raises a clear ``FileNotFoundError``.

    Uses the operator's existing subscription — no API key is stored in
    asr-bench.
    """
    name = "cli"

    def __init__(self, command: List[str], timeout: float = 300.0):
        self.command = command
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        if any("{prompt}" in part for part in self.command):
            cmd = [part.replace("{prompt}", prompt) for part in self.command]
            stdin = None
        else:
            cmd = list(self.command)
            stdin = prompt
        exe = shutil.which(cmd[0]) or cmd[0]
        cmd = [exe, *cmd[1:]]
        proc = subprocess.run(
            cmd, input=stdin, capture_output=True, text=True,
            timeout=self.timeout, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"LLM CLI {cmd} exited {proc.returncode}: {proc.stderr[:500]}")
        return (proc.stdout or "").strip()


def make_llm_backend(spec: str) -> LLMBackend:
    """Parse a --llm spec into a backend.

    'fake'                 -> FakeLLMBackend (echo)
    'ollama:<model>'       -> OllamaBackend  (default model qwen2.5 if omitted)
    'cli:<command words>'  -> CliBackend     (command split on whitespace)
    """
    spec = (spec or "").strip()
    if spec == "fake":
        return FakeLLMBackend()
    kind, _, rest = spec.partition(":")
    kind = kind.strip().lower()
    rest = rest.strip()
    if kind == "ollama":
        return OllamaBackend(model=rest or "qwen2.5")
    if kind == "cli":
        if not rest:
            raise ValueError("cli backend needs a command, e.g. --llm cli:claude")
        return CliBackend(rest.split())
    raise ValueError(f"unknown --llm backend '{spec}' (use fake, ollama:<model>, or cli:<command>)")


# ---- Results JSON sidecar ---------------------------------------------------
def _json_sanitize(obj):
    """Make a value strictly-JSON-safe: NaN/Inf floats -> None, tuples -> lists,
    recursing through dicts and lists. (json allows NaN by default but emits the
    literal token `NaN`, which is invalid JSON for strict parsers.)"""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return obj


def _reproducibility_command(args: argparse.Namespace, corpus_path: Path, results: List["ModelResult"]) -> str:
    """The `python asr_bench.py ...` command that reproduces this run (no backticks).
    Shared by the markdown reproducibility footnote and the JSON sidecar."""
    batch_flag = f" --batch-size {args.batch_size}" if args.batch_size > 1 else ""
    beam_flag = f" --beam-size {args.beam_size}" if args.beam_size != 5 else ""
    vad_flag = "" if args.vad_filter else " --no-vad-filter"
    nim_flag = ""
    if any(r.engine == "nim" for r in results):
        nim_flag = f" --nim-url {args.nim_url} --nim-language {args.nim_language}"
        if args.nim_model:
            nim_flag += f" --nim-model {args.nim_model}"
    return (f"python asr_bench.py --corpus '{corpus_path}' --models {','.join(args.models)} "
            f"--device {args.device} --compute-type {args.compute_type}"
            f"{batch_flag}{beam_flag}{vad_flag}{nim_flag}")


def _fusion_output_paths(pairs: List["Pair"], profiles: List[str]) -> List[str]:
    """The fusion output files this run is expected to have written, per profile
    (verbatim -> <base>_Captions_Fused.vtt; kb -> <base>_KB_Fused.jsonl + .md).
    Reconstructed from the same `_fused_base` convention the writers use."""
    out: List[str] = []
    for p in pairs:
        base = p.audio.parent / _fused_base(p.audio)
        if "verbatim" in profiles:
            out.append(f"{base}_Captions_Fused.vtt")
        if "kb" in profiles:
            out.append(f"{base}_KB_Fused.jsonl")
            out.append(f"{base}_KB_Fused.md")
    return out


def _clip_to_dict(c: "ClipResult") -> Dict:
    return {
        "audio": c.audio, "audio_sec": c.audio_sec, "transcribe_sec": c.transcribe_sec,
        "rtfx": c.rtfx, "vram_peak_bytes": c.vram_peak_bytes,
        "wer": c.wer, "mer": c.mer, "wil": c.wil, "cer": c.cer,
        "hits": c.hits, "substitutions": c.substitutions,
        "deletions": c.deletions, "insertions": c.insertions,
        "cue_count": c.cue_count, "num_speakers": c.num_speakers, "der": c.der,
        "speaker_segments": [{"start": s, "end": e, "speaker": spk}
                             for (s, e, spk) in c.speaker_segments],
        "vtt_path": c.vtt_path,
        "reference_origin": c.reference_origin, "reference_label": c.reference_label,
        "hypothesis": c.hypothesis,
        "reference_normalized": c.reference_normalized,
        "hypothesis_normalized": c.hypothesis_normalized,
        "repeat_coverage": c.repeat_coverage,
        "compression_ratio": c.compression_ratio,
        "hallucination_suspect": c.is_hallucination_suspect,
    }


def _model_to_dict(m: "ModelResult") -> Dict:
    return {
        "model_id": m.model_id, "display": m.display, "engine": m.engine,
        "fw_name": m.fw_name, "params": m.params, "developer": m.developer,
        "languages": m.languages, "disk_bytes": m.disk_bytes, "load_sec": m.load_sec,
        "vram_is_total": m.vram_is_total, "notes": m.notes,
        "aggregates": {
            "avg_wer": m.avg_wer, "avg_mer": m.avg_mer, "avg_wil": m.avg_wil,
            "avg_cer": m.avg_cer,
            "total_audio_sec": m.total_audio_sec,
            "total_transcribe_sec": m.total_transcribe_sec,
            "aggregate_rtfx": m.aggregate_rtfx,
            "median_rtfx": m.median_rtfx,
            "median_sec_per_audio_min": m.median_sec_per_audio_min,
            "peak_vram_bytes": m.peak_vram_bytes,
            "hallucination_rate": m.hallucination_rate,
        },
        "clips": [_clip_to_dict(c) for c in m.clips],
    }


def _config_to_dict(cfg: "RunConfig") -> Dict:
    """RunConfig as a dict, OMITTING secrets (hf_token, nim_api_key)."""
    return {
        "device": cfg.device, "compute_type": cfg.compute_type,
        "batch_size": cfg.batch_size, "beam_size": cfg.beam_size,
        "vad_filter": cfg.vad_filter,
        "nim_url": cfg.nim_url, "nim_model": cfg.nim_model,
        "nim_language": cfg.nim_language, "nim_ssl": cfg.nim_ssl,
        "whisperx_python": cfg.whisperx_python, "nemo_python": cfg.nemo_python,
        "diarize": cfg.diarize,
        "min_speakers": cfg.min_speakers, "max_speakers": cfg.max_speakers,
    }


def build_results_document(results: List["ModelResult"], *, corpus: Path,
                           cfg: "RunConfig", args: argparse.Namespace, gold_label: str,
                           pairs: List["Pair"], report_path: Path,
                           generated_at: str) -> Dict:
    """Build the JSON sidecar document (a plain, strictly-JSON-safe dict) from a
    completed run. Secrets are omitted; NaN/Inf become null."""
    ref_quality = ("gold" if gold_label.replace("*", "").strip().lower().startswith("gold")
                   else "proxy")
    first = results[0] if results else None
    fusion: Dict = {"ran": bool(getattr(args, "fuse", False))}
    if fusion["ran"]:
        fusion["profiles"] = (["verbatim", "kb"] if args.profile == "both"
                              else [args.profile])
        fusion["outputs"] = _fusion_output_paths(pairs, fusion["profiles"])
    doc: Dict = {
        "schema_version": 1,
        "generated_at": generated_at,
        "report_markdown": str(report_path),
        "command": _reproducibility_command(args, corpus, results),
        "run": {
            "corpus": str(corpus),
            "device": cfg.device,
            "compute_type": cfg.compute_type,
            "reference_quality": ref_quality,
            "reference_quality_label": gold_label,
            "clips_count": len(first.clips) if first else 0,
            "total_audio_sec": first.total_audio_sec if first else 0.0,
            "vram_tracking": any(c.vram_peak_bytes is not None
                                 for r in results for c in r.clips),
            "config": _config_to_dict(cfg),
        },
        "models": [_model_to_dict(m) for m in results],
        "fusion": fusion,
    }
    return _json_sanitize(doc)


def write_results_json(document: Dict, json_path: Path) -> Path:
    """Serialize the results document to json_path. allow_nan=False is a guard:
    any NaN/Inf that escaped _json_sanitize raises ValueError rather than writing
    invalid JSON."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return json_path


# ---- Output -----------------------------------------------------------------
def render_markdown(
    results: List[ModelResult],
    corpus_path: Path,
    args: argparse.Namespace,
    gold_label: str,
) -> str:
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    # Collect distinct reference-origin labels across all clips so we can surface
    # auto-detected proxies prominently if any are present.
    ref_origins: Dict[str, int] = {}
    if results:
        for c in results[0].clips:
            key = c.reference_label or "(unlabeled)"
            ref_origins[key] = ref_origins.get(key, 0) + 1

    lines: List[str] = []
    lines.append("# ASR Benchmark Results")
    lines.append("")
    lines.append(f"- **Date:** {now}")
    lines.append(f"- **Corpus:** `{corpus_path}`")
    lines.append(f"- **Reference quality (declared):** {gold_label}")
    if ref_origins:
        for label, count in ref_origins.items():
            lines.append(f"- **Reference origin (detected):** {label} — {count} clip(s)")
    lines.append(f"- **Device:** {args.device}")
    lines.append(f"- **Compute type:** {args.compute_type}")
    lines.append(f"- **Clips:** {len(results[0].clips) if results else 0}")
    if results:
        total_audio_min = results[0].total_audio_sec / 60.0
        lines.append(f"- **Total audio:** {total_audio_min:.1f} min")
    lines.append(f"- **VRAM tracking:** {'on (NVML)' if _HAS_NVML else 'off — install nvidia-ml-py'}")
    lines.append("")

    # ---- Headline: one row per model, the key numbers ----
    any_diar = any(
        (not math.isnan(c.der)) or c.num_speakers > 0
        for r in results for c in r.clips
    )

    lines.append("## Headline")
    lines.append("")
    diar_hdr = " DER% | Speakers |" if any_diar else ""
    diar_sep = "---|---|" if any_diar else ""
    lines.append("| Model | Params | Disk | Overall WER% | MER% | WIL% | CER% | RTFx | RTFx (med) | s/aud-min (med) | Total time | Peak VRAM |" + diar_hdr + " Notes |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|" + diar_sep + "---|")
    for r in results:
        wall_clock = f"{r.total_transcribe_sec:.1f}s"
        wer_pct = _fmt_pct(r.avg_wer) if r.clips else "—"
        mer_pct = _fmt_pct(r.avg_mer) if r.clips else "—"
        wil_pct = _fmt_pct(r.avg_wil) if r.clips else "—"
        cer_pct = _fmt_pct(r.avg_cer) if r.clips else "—"
        rtfx = f"{r.aggregate_rtfx:.2f}x" if r.clips else "—"
        rtfx_med = f"{r.median_rtfx:.2f}x" if r.clips else "—"
        spm = f"{r.median_sec_per_audio_min:.2f}s" if r.clips else "—"
        vram = _vram_cell(r.peak_vram_bytes, r.vram_is_total)
        disk = _disk_cell(r)
        diar_cells = ""
        if any_diar:
            der_vals = [c.der for c in r.clips if not math.isnan(c.der)]
            der_avg = _fmt_pct(sum(der_vals) / len(der_vals)) if der_vals else "—"
            spk = max((c.num_speakers for c in r.clips), default=0) or "—"
            diar_cells = f" {der_avg} | {spk} |"
        lines.append(
            f"| {_md_escape(r.display)} | {r.params} | {disk} | {wer_pct} | {mer_pct} | {wil_pct} | {cer_pct} | {rtfx} | {rtfx_med} | {spm} | {wall_clock} | {vram} |{diar_cells} {_md_escape(r.notes)} |"
        )
    lines.append("")

    if any_diar:
        lines.append("> **Diarization:** speaker labels are pyannote hypotheses. **DER%** is "
                     "shown only for clips with an `<base>.rttm` ground-truth sidecar (pyannote.metrics "
                     "defaults). Speakers = detected speaker count.")
        lines.append("")

    # ---- Engines note: explain metric-fidelity differences when NIM is present ----
    if any(r.engine == "nim" for r in results):
        lines.append("## Engines in this run")
        lines.append("")
        lines.append(
            "This run mixes engine families. The in-process **faster-whisper** engine "
            "returns the fuller, more directly-comparable set of readings: a per-clip "
            "**VRAM allocation delta**, the **on-disk** model size, and weights **load** time."
        )
        lines.append("")
        lines.append(
            "The **NIM** engine runs behind a gRPC service, so it exposes less. "
            "Its VRAM figures are marked `*` = **total GPU memory** in use during the clip "
            "(the model is pre-resident in the NIM container), **not** the per-clip allocation "
            "delta that Whisper rows report — the two are not directly comparable. Disk size "
            "is shown as `n/a` (it is a container image, not an HF cache dir). NIM is still "
            "fully benchmarkable for **WER**, **RTFx**, and **wall clock** — run it and see how it does."
        )
        lines.append("")

    # ---- Per-clip view: each clip first, with one row per model ----
    if results and results[0].clips:
        lines.append("## Per-clip view")
        lines.append("")
        lines.append("Each clip's table shows one row per model. Compare how the engines stack up on the *same* audio.")
        lines.append("")
        clip_count = len(results[0].clips)
        for i in range(clip_count):
            sample = results[0].clips[i]
            audio_min = sample.audio_sec / 60.0
            lines.append(f"### {sample.audio} — {audio_min:.1f} min")
            lines.append("")
            lines.append("| Model | WER% | MER% | WIL% | CER% | S | D | I | RTFx | Transcribe time | VRAM peak |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for r in results:
                if i < len(r.clips):
                    c = r.clips[i]
                    wer_pct = _fmt_pct(c.wer)
                    mer_pct = _fmt_pct(c.mer)
                    wil_pct = _fmt_pct(c.wil)
                    cer_pct = _fmt_pct(c.cer)
                    vram = _vram_cell(c.vram_peak_bytes, r.vram_is_total)
                    lines.append(
                        f"| {_md_escape(r.display)} | {wer_pct} | {mer_pct} | {wil_pct} | {cer_pct} | {c.substitutions} | {c.deletions} | {c.insertions} | {c.rtfx:.2f}x | {c.transcribe_sec:.1f}s | {vram} |"
                    )
            lines.append("")

    # ---- Per-model breakdown: each model gets a table showing per-clip rows + overall ----
    lines.append("## Per-model breakdown")
    lines.append("")
    lines.append("Each model's table lists every clip plus an **OVERALL** row aggregating that model's run.")
    lines.append("")
    for r in results:
        lines.append(f"### {r.display}")
        lines.append("")
        lines.append("| Clip | Audio | WER% | MER% | WIL% | CER% | RTFx | Transcribe time | VRAM peak |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for c in r.clips:
            wer_pct = _fmt_pct(c.wer)
            mer_pct = _fmt_pct(c.mer)
            wil_pct = _fmt_pct(c.wil)
            cer_pct = _fmt_pct(c.cer)
            vram = _vram_cell(c.vram_peak_bytes, r.vram_is_total)
            audio_label = f"{c.audio_sec / 60:.1f} min"
            lines.append(
                f"| {_md_escape(c.audio)} | {audio_label} | {wer_pct} | {mer_pct} | {wil_pct} | {cer_pct} | {c.rtfx:.2f}x | {c.transcribe_sec:.1f}s | {vram} |"
            )
        overall_audio = f"{r.total_audio_sec / 60:.1f} min"
        overall_wer = _fmt_pct(r.avg_wer) if r.clips else "—"
        overall_mer = _fmt_pct(r.avg_mer) if r.clips else "—"
        overall_wil = _fmt_pct(r.avg_wil) if r.clips else "—"
        overall_cer = _fmt_pct(r.avg_cer) if r.clips else "—"
        overall_rtfx = f"{r.aggregate_rtfx:.2f}x" if r.clips else "—"
        overall_vram = _vram_cell(r.peak_vram_bytes, r.vram_is_total)
        lines.append(
            f"| **OVERALL** | **{overall_audio}** | **{overall_wer}** | **{overall_mer}** | **{overall_wil}** | **{overall_cer}** | **{overall_rtfx}** | **{r.total_transcribe_sec:.1f}s** | **{overall_vram}** |"
        )
        lines.append("")

    # ---- Cue-density anomaly detection ----
    # Flag (model, clip) pairs whose cue count is >= 1.5x the median across the OTHER models
    # on the same clip. Catches Whisper-Large-style 1-second-cue decoder lockups automatically.
    if results and len(results) >= 2 and results[0].clips:
        clip_count = len(results[0].clips)
        anomalies: List[Tuple[str, str, int, float]] = []  # (model_display, clip_name, cues, ratio)
        for i in range(clip_count):
            counts_by_model: List[Tuple[ModelResult, int]] = [
                (r, r.clips[i].cue_count) for r in results if i < len(r.clips)
            ]
            for r_target, n_target in counts_by_model:
                others = [n for r_o, n in counts_by_model if r_o is not r_target and n > 0]
                if len(others) < 1:
                    continue
                others_sorted = sorted(others)
                mid = len(others_sorted) // 2
                if len(others_sorted) % 2 == 1:
                    median_others = float(others_sorted[mid])
                else:
                    median_others = (others_sorted[mid - 1] + others_sorted[mid]) / 2.0
                if median_others <= 0:
                    continue
                ratio = n_target / median_others
                if ratio >= 1.5:
                    anomalies.append((r_target.display, r_target.clips[i].audio, n_target, ratio))
        if anomalies:
            lines.append("## ⚠️ Cue-density anomalies")
            lines.append("")
            lines.append(
                "These (model, clip) pairs produced **1.5×+ more cues** than the median of "
                "the other models on the same clip. Common cause: the Whisper decoder enters "
                "a 1-second-per-cue lockup (a known faster-whisper failure mode) — the content "
                "is usually still transcribed but the WER is inflated by alignment churn. "
                "Already mitigated by `vad_filter=True` (the default); if you re-run with "
                "`--no-vad-filter` you'll likely see these reappear."
            )
            lines.append("")
            lines.append("| Model | Clip | Cue count | × median of others |")
            lines.append("|---|---|---|---|")
            for model_display, clip_name, n, ratio in anomalies:
                lines.append(f"| {model_display} | {clip_name} | {n} | {ratio:.2f}× |")
            lines.append("")

    # ---- Hallucination signals (reference-free, per clip) ----
    flagged: List[Tuple[str, ClipResult]] = [
        (r.display, c) for r in results for c in r.clips if c.is_hallucination_suspect
    ]
    if flagged:
        lines.append("## ⚠️ Hallucination signals")
        lines.append("")
        lines.append(
            "Reference-free heuristics that flag a clip for **manual inspection** "
            "(not a definitive error): **repeat coverage** = fraction of the "
            "transcript made of repeated 4-grams; **compression** = gzip ratio of "
            "the text (Whisper's own internal hallucination signal — normal prose "
            "is ~1.5–2.2, looped output is higher). A clip is flagged when repeat "
            f"coverage > {HALLUCINATION_REPEAT_COVERAGE:.0%} or compression > "
            f"{HALLUCINATION_COMPRESSION_RATIO:.1f}."
        )
        lines.append("")
        for r in results:
            n_flag = sum(1 for c in r.clips if c.is_hallucination_suspect)
            if n_flag:
                lines.append(f"- **{r.display}:** {n_flag}/{len(r.clips)} clips flagged")
        lines.append("")
        lines.append("| Model | Clip | Repeat cov % | Compression | Insertion rate | Note |")
        lines.append("|---|---|---|---|---|---|")
        for model_display, c in flagged:
            ref_words = c.hits + c.substitutions + c.deletions
            ins_rate = (c.insertions / ref_words) if ref_words > 0 else None
            ins_cell = f"{ins_rate * 100:.0f}%" if ins_rate is not None else "—"
            reasons: List[str] = []
            if c.repeat_coverage > HALLUCINATION_REPEAT_COVERAGE:
                reasons.append("repetition")
            if c.compression_ratio > HALLUCINATION_COMPRESSION_RATIO:
                reasons.append("high compression")
            if ins_rate is not None and ins_rate > HALLUCINATION_INSERTION_RATE:
                reasons.append("insertion burst")
            note = ", ".join(reasons)
            lines.append(
                f"| {model_display} | {c.audio} | {c.repeat_coverage * 100:.1f} | "
                f"{c.compression_ratio:.2f} | {ins_cell} | {note} |"
            )
        lines.append("")

    # ---- Generated VTT files ----
    if results and any(c.vtt_path for c in results[0].clips):
        lines.append("## Generated VTT outputs")
        lines.append("")
        lines.append("Each model writes a WebVTT caption file next to the source audio:")
        lines.append("")
        for r in results:
            written: List[str] = []
            for c in r.clips:
                if c.vtt_path:
                    written.append(Path(c.vtt_path).name)
            if written:
                lines.append(f"- **{r.display}**:")
                for name in written:
                    lines.append(f"  - `{name}`")
        lines.append("")

    # ---- Optional alignment detail ----
    if getattr(args, "show_alignment", False) and results and results[0].clips:
        from jiwer import process_words, visualize_alignment
        blocks: List[str] = []
        for r in results:
            for c in r.clips:
                if not c.reference_normalized or not c.hypothesis_normalized:
                    continue
                name = Path(c.audio).name
                try:
                    viz = visualize_alignment(
                        process_words(c.reference_normalized, c.hypothesis_normalized),
                        show_measures=False,
                    )
                except Exception:
                    blocks.append(f"<!-- alignment unavailable for {name} -->")
                    continue
                blocks.append(f"### {r.display} — {name}")
                blocks.append("")
                blocks.append("```")
                blocks.append(viz.rstrip())
                blocks.append("```")
                blocks.append("")
        if blocks:
            lines.append("## Alignment detail")
            lines.append("")
            lines.append("Word-level reference→hypothesis alignment (S=substitution, D=deletion, I=insertion).")
            lines.append("")
            lines.extend(blocks)

    # ---- Reproducibility footnote ----
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- Command: `{_reproducibility_command(args, corpus_path, results)}`")
    lines.append(f"- VAD filter: {'on (Silero VAD pre-segments audio — prevents the Whisper-Large 1-second-cue decoder lock)' if args.vad_filter else 'off (--no-vad-filter)'}")
    lines.append(f"- Reference normalization: lowercase, strip punctuation (keep apostrophes), collapse whitespace.")
    lines.append(f"- WER computed via [jiwer](https://github.com/jitsi/jiwer).")
    lines.append("- **MER** (match error rate) and **WIL** (word information lost) are the bounded-[0,1] measures from Morris, Maier & Green (2004); both derive from the same H/S/D/I alignment as WER. S/D/I in the per-clip table are raw substitution/deletion/insertion counts.")
    if any(c.reference_origin in {"panopto-asr", "asr-generic"} for r in results for c in r.clips):
        lines.append(
            "- **Reference origin warning:** at least one clip's reference was auto-detected as ASR-generated "
            "(Panopto captions or similar). WER numbers should be read as *relative divergence* between engines, "
            "not absolute accuracy."
        )
    lines.append("")
    return "\n".join(lines)


# ---- CLI --------------------------------------------------------------------
# ---- prepare-gold subcommand ------------------------------------------------
# Files asr-bench itself emits — never treat these as source captions to convert,
# or a model's own output would become its own "reference" (circular).
_GENERATED_CAPTION_RE = re.compile(r"_Captions_.*\.vtt$", re.IGNORECASE)
_CAPTION_SOURCE_EXTS = {".vtt", ".srt"}
_PROXY_TXT_HEADER = (
    "[Auto-generated transcript. Converted by asr-bench prepare-gold "
    "— proxy reference, not verified gold.]"
)


def _is_generated_caption(name: str) -> bool:
    return bool(_GENERATED_CAPTION_RE.search(name))


def find_caption_sources(paths: List[Path]) -> List[Path]:
    """Expand file/dir args into a sorted list of convertible .vtt/.srt sources,
    excluding asr-bench's own generated `_Captions_*.vtt` outputs."""
    out: List[Path] = []
    for p in paths:
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if (f.is_file() and f.suffix.lower() in _CAPTION_SOURCE_EXTS
                        and not _is_generated_caption(f.name)):
                    out.append(f)
        elif (p.is_file() and p.suffix.lower() in _CAPTION_SOURCE_EXTS
              and not _is_generated_caption(p.name)):
            out.append(p)
    return out


def prepare_gold_main(argv: List[str]) -> int:
    """Convert VTT/SRT caption files into the plain `.txt` reference files
    asr-bench scores against (timing stripped, cues joined). Proxy/ASR-generated
    sources keep an auto-gen marker so they stay detectable as non-gold."""
    ap = argparse.ArgumentParser(
        prog="asr_bench.py prepare-gold",
        description="Convert VTT/SRT captions into plain-text .txt references. "
                    "Proxy/ASR-generated sources stay labeled proxy.",
    )
    ap.add_argument("paths", nargs="*",
                    help="Caption files or directories. Defaults to --corpus.")
    ap.add_argument("--corpus", default="test-corpus",
                    help="Directory of captions, used when no paths are given.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace existing .txt references.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would happen; write nothing.")
    ns = ap.parse_args(argv)

    inputs = [Path(p) for p in ns.paths] or [Path(ns.corpus)]
    sources = find_caption_sources(inputs)
    if not sources:
        print("No .vtt/.srt caption files found to convert "
              "(asr-bench's own _Captions_*.vtt outputs are skipped).",
              file=sys.stderr)
        return 1

    converted = skipped = proxy_count = 0
    for src in sources:
        dst = src.with_suffix(".txt")
        origin, label = detect_reference_origin(src)
        is_proxy = origin != "unknown"
        if is_proxy:
            proxy_count += 1
        tag = f"PROXY ({label})" if is_proxy else "gold-eligible (proofread to confirm)"
        if dst.exists() and not ns.overwrite:
            print(f"skip   {src.name} -> {dst.name} (exists; --overwrite to replace) [{tag}]")
            skipped += 1
            continue
        text = load_reference_text(src)
        word_n = len(text.split())
        if ns.dry_run:
            print(f"dry    {src.name} -> {dst.name} [{tag}] ({word_n} words)")
            converted += 1
            continue
        # Preserve the proxy signal: the bracketed header is stripped at scoring
        # time by load_reference_text, but detect_reference_origin still sees it.
        body = (f"{_PROXY_TXT_HEADER}\n{text}" if is_proxy else text)
        dst.write_text(body + "\n", encoding="utf-8")
        print(f"write  {src.name} -> {dst.name} [{tag}] ({word_n} words)")
        converted += 1

    verb = "would convert" if ns.dry_run else "converted"
    print(f"\n{verb} {converted} file(s), skipped {skipped}.")
    if proxy_count:
        print(f"Note: {proxy_count} PROXY source(s) produced proxy .txt references "
              "(auto-gen header preserved). Proofread and delete that header line "
              "to promote a file to gold.")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "compare":
        from asr_compare import compare_main
        return compare_main(argv[1:])
    if argv and argv[0] == "prepare-gold":
        return prepare_gold_main(argv[1:])
    ap = argparse.ArgumentParser(
        description="Benchmark local Whisper variants on your own audio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--version", action="version", version=f"asr-bench {__version__}")
    ap.add_argument(
        "--corpus",
        default=str(Path(__file__).resolve().parent / "test-corpus"),
        help="Path to a folder of (audio, reference) pairs. Defaults to ./test-corpus next to the script.",
    )
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
        "--batch-size",
        default="auto",
        help="Batch size for transcription. 'auto' probes free VRAM and recommends a fit "
             "(uses BatchedInferencePipeline when > 1). '1' = sequential. Otherwise integer.",
    )
    ap.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam search width. 1 = greedy decoding (fastest, slightly lower accuracy).",
    )
    ap.add_argument(
        "--vad-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pre-segment audio with Silero VAD before transcription. Prevents the "
             "Whisper-Large 1-second-cue decoder lock and generally improves WER. "
             "Use --no-vad-filter to disable.",
    )
    ap.add_argument(
        "--nim-url", default="localhost:50051",
        help="NIM/Riva gRPC endpoint for nim-engine models (self-hosted or hosted).",
    )
    ap.add_argument(
        "--nim-model", default="",
        help="Override the Riva model name for the canary-nim entry. '' = server default.",
    )
    ap.add_argument(
        "--nim-language", default="en-US",
        help="Riva language code for NIM models (note: Whisper uses 'en').",
    )
    ap.add_argument(
        "--nim-api-key", default=None,
        help="Bearer token for a secured NIM endpoint. Presence auto-enables SSL.",
    )
    ap.add_argument(
        "--nim-ssl", action="store_true",
        help="Force SSL for the NIM endpoint without an API key (e.g. self-signed TLS).",
    )
    ap.add_argument("--whisperx-python", default=None,
                    help="Path to a 3.12 venv python with whisperx+torch+pyannote "
                         "(for the subprocess adapter; auto-detects ./.venv-whisperx if omitted).")
    ap.add_argument("--nemo-python", default=None,
                    help="Path to a 3.12 venv python with torch (cu128) + nemo_toolkit "
                         "(subprocess adapter; auto-detects ./.venv-nemo if omitted).")
    ap.add_argument("--diarize", action=argparse.BooleanOptionalAction, default=True,
                    help="Run pyannote speaker diarization for whisperx models (default on). "
                         "Without an HF token it warns and falls back to alignment-only.")
    ap.add_argument("--hf-token", default=None,
                    help="HuggingFace token for pyannote diarization (else HF_TOKEN/HUGGINGFACE_TOKEN env).")
    ap.add_argument("--min-speakers", type=int, default=None, help="pyannote min speakers hint.")
    ap.add_argument("--max-speakers", type=int, default=None, help="pyannote max speakers hint.")
    ap.add_argument(
        "--gold",
        action="store_true",
        help="Label the WER as gold-standard (hand-corrected reference). Without this, output marks WER as proxy.",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Where to save the markdown report. Default: ./report/<timestamp>.md",
    )
    ap.add_argument("--json", action=argparse.BooleanOptionalAction, default=True,
                    help="Write a machine-readable results JSON sidecar next to the report "
                         "(default on). results/<timestamp>.json, or <output>.json with --output.")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N clip pairs. Handy for smoke runs over a big corpus.",
    )
    ap.add_argument(
        "--include",
        default=None,
        help="Regex; only include clips whose audio filename matches.",
    )
    ap.add_argument(
        "--show-alignment",
        action="store_true",
        help="Append a per-clip word-level alignment diff (jiwer) to the report. Verbose — one fenced block per (model, clip) pair; can add many hundreds of lines.",
    )
    ap.add_argument("--fuse", action="store_true",
                    help="After benchmarking, fuse all models + reference into a best transcript.")
    ap.add_argument("--profile", default="both", choices=["verbatim", "kb", "both"],
                    help="Fusion profile(s). verbatim=captions/reference, kb=RAG knowledge base.")
    ap.add_argument("--fuse-base", default="large-v3-turbo",
                    help="Model whose cue timing anchors the fusion windows.")
    ap.add_argument("--llm", default="ollama:qwen2.5",
                    help="Fusion LLM backend: fake | ollama:<model> | cli:<command>. "
                         "cli pipes the prompt on stdin (e.g. 'cli:claude -p'); a {prompt} "
                         "token is substituted as an arg instead (e.g. 'cli:gemini -p {prompt}'). "
                         "Agentic CLIs are slow per call — prefer ollama for bulk runs.")
    ap.add_argument("--context", default=None, help="Path to a fusion context file (see --init-context).")
    ap.add_argument("--glossary", default=None, help="Optional separate glossary file (overrides in-context glossary).")
    ap.add_argument("--window", type=float, default=25.0, help="Fusion window length in seconds.")
    ap.add_argument("--overlap", type=float, default=5.0, help="Fusion window overlap in seconds (context carryover).")
    ap.add_argument("--drift-threshold", type=float, default=1.0,
                    help="Flag a fused window whose WER vs the base model exceeds this (1.0 = 100%%).")
    ap.add_argument("--rescore-against-fused", action="store_true",
                    help="Re-score every model against the verbatim fused reference (agreement-biased; labeled as such).")
    ap.add_argument("--init-context", nargs="?", const="context.md", default=None,
                    metavar="PATH", help="Write a context.md template to PATH (default context.md) and exit.")
    args = ap.parse_args()

    if args.init_context is not None:
        dest = Path(args.init_context)
        if dest.exists():
            print(f"ERROR: {dest} already exists — refusing to overwrite", file=sys.stderr)
            return 2
        dest.write_text(init_context_template(), encoding="utf-8")
        print(f"Wrote fusion context template to {dest}. Edit it, then pass --context {dest} --fuse.")
        return 0

    corpus = Path(args.corpus).resolve()
    if not corpus.is_dir():
        print(f"ERROR: --corpus {corpus} is not a directory", file=sys.stderr)
        return 2

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = []
    for m in requested:
        try:
            resolve_model_entry(m)
        except ValueError:
            unknown.append(m)
    if unknown:
        print(f"ERROR: unknown models: {', '.join(unknown)}", file=sys.stderr)
        print(f"Available: {', '.join(MODELS.keys())} (or ad-hoc 'nim:<riva-model-name>')", file=sys.stderr)
        return 2

    # Pre-flight fusion setup (fail fast before the expensive benchmark run)
    fusion_backend = None
    fusion_context = ""
    fusion_glossary = ""
    if args.fuse:
        try:
            fusion_backend = make_llm_backend(args.llm)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        for label, pth in (("--context", args.context), ("--glossary", args.glossary)):
            if pth and not Path(pth).is_file():
                print(f"ERROR: {label} file not found: {pth}", file=sys.stderr)
                return 2
        fusion_context, fusion_glossary = load_context(args.context, args.glossary)
        if args.fuse_base not in requested:
            print(
                f"WARNING: --fuse-base '{args.fuse_base}' is not in --models ({', '.join(requested)}); "
                f"its cues won't exist, so windows will have no timing anchor and drift checks won't fire.",
                file=sys.stderr,
            )

    # Pre-flight: warn early if diarization is requested for whisperx models without a token.
    wants_whisperx = any(resolve_model_entry(m)["engine"] == "whisperx" for m in requested)
    if wants_whisperx and args.diarize and not (
            args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")):
        print("WARNING: --diarize is on but no HF token found (--hf-token / HF_TOKEN). "
              "WhisperX will run alignment-only. Get a free token and accept the gated "
              "pyannote/speaker-diarization-3.1 model to enable diarization.", file=sys.stderr)

    # Pre-flight: NeMo models need a .venv-nemo (or --nemo-python). Skip just those
    # models (not the whole run) when absent, so Whisper/NIM models still benchmark.
    nemo_requested = [m for m in requested if resolve_model_entry(m)["engine"] == "nemo"]
    if nemo_requested and not (args.nemo_python or _default_nemo_python()):
        print(f"WARNING: NeMo model(s) {', '.join(nemo_requested)} requested but no "
              f".venv-nemo found and --nemo-python not given. Skipping them. "
              f"Run setup_nemo_venv.ps1 to enable NeMo.", file=sys.stderr)
        requested = [m for m in requested if m not in nemo_requested]
        if not requested:
            print("ERROR: no runnable models left after skipping NeMo "
                  "(no .venv-nemo). See setup_nemo_venv.ps1.", file=sys.stderr)
            return 2

    pairs = discover_pairs(corpus)
    if args.include:
        include_re = re.compile(args.include, re.IGNORECASE)
        pairs = [p for p in pairs if include_re.search(p.audio.name)]
    if args.limit is not None:
        pairs = pairs[: args.limit]

    if not pairs:
        print(f"ERROR: no (audio, reference) pairs discovered in {corpus}", file=sys.stderr)
        print("See test-corpus/README.md for supported layouts.", file=sys.stderr)
        return 2

    print(f"Discovered {len(pairs)} clip(s) under {corpus}:")
    for p in pairs:
        print(f"  - {p.audio.name} ({p.audio.stat().st_size / 1e6:.1f}MB) <- ref {p.reference.name}")

    # Device resolution
    device = args.device
    if device == "auto":
        device = "cuda" if (_HAS_NVML and _NVML_DEVICE_COUNT > 0) else "cpu"
    args.device = device

    if args.compute_type == "auto":
        args.compute_type = "float16" if device == "cuda" else "int8"
    args.models = requested

    # Resolve --batch-size (auto → recommend based on free VRAM + queued models)
    if str(args.batch_size).lower() == "auto":
        if device != "cuda":
            args.batch_size = 1
            print("Batch size: 1 (CPU device — batching not used)")
        else:
            bsz, why = recommend_batch_size(requested)
            args.batch_size = bsz
            gname = gpu_name() or "GPU"
            print(f"Batch size: {bsz} (auto)  GPU={gname}  reason: {why}")
    else:
        try:
            args.batch_size = int(args.batch_size)
        except ValueError:
            print(f"ERROR: --batch-size must be 'auto' or an integer, got {args.batch_size}", file=sys.stderr)
            return 2

    # Auto-detect ASR-shaped references; surface even if the user passed --gold.
    auto_origins = {detect_reference_origin(p.reference)[0] for p in pairs}
    asr_detected = any(o in {"panopto-asr", "asr-generic"} for o in auto_origins)
    if args.gold and asr_detected:
        gold_label = "**gold (claimed via --gold)** ⚠️ but auto-detection flagged at least one reference as ASR-generated — verify before trusting WER as absolute"
    elif args.gold:
        gold_label = "**gold (hand-corrected, declared via --gold)**"
    elif asr_detected:
        gold_label = "**proxy** (auto-detected ASR-generated reference — WER is relative divergence, not absolute accuracy)"
    else:
        gold_label = "**proxy** (default: pass --gold if your reference is hand-corrected)"
    print(f"Reference quality: {gold_label}")

    cfg = RunConfig(
        device=device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
        nim_url=args.nim_url,
        nim_model=args.nim_model,
        nim_language=args.nim_language,
        nim_api_key=args.nim_api_key,
        nim_ssl=args.nim_ssl,
        whisperx_python=args.whisperx_python,
        nemo_python=args.nemo_python,
        diarize=args.diarize,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    results: List[ModelResult] = []
    for model_id in requested:
        entry = resolve_model_entry(model_id)
        engine_cls = ENGINES.get(entry["engine"])
        if engine_cls is None:
            print(f"ERROR: no engine registered for '{entry['engine']}' (model {model_id})", file=sys.stderr)
            return 2
        results.append(engine_cls().run(entry, pairs, cfg))

    md = render_markdown(results, corpus, args, gold_label)

    if args.fuse:
        profiles = ["verbatim", "kb"] if args.profile == "both" else [args.profile]
        fusion_md, rescored = run_fusion_stage(
            results=results, pairs=pairs, backend=fusion_backend, profiles=profiles,
            base_label=args.fuse_base, context=fusion_context, glossary=fusion_glossary,
            window=args.window, overlap=args.overlap, drift_threshold=args.drift_threshold,
            rescore=args.rescore_against_fused,
        )
        md = md + "\n" + fusion_md
        if rescored is not None:
            md = md + "\n" + render_fused_rescore_table(rescored)

    print()
    print(md)

    # Save markdown report
    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    output_path = Path(args.output) if args.output else None
    if output_path is None:
        report_dir = Path(__file__).resolve().parent / "report"
        report_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = report_dir / f"{ts}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    print(f"\nSaved report to {output_path}")

    # Save JSON results sidecar (default on; --no-json opts out)
    if args.json:
        if args.output:
            json_path = output_path.with_suffix(".json")
        else:
            results_dir = Path(__file__).resolve().parent / "results"
            results_dir.mkdir(exist_ok=True)
            json_path = results_dir / f"{output_path.stem}.json"
        document = build_results_document(
            results, corpus=corpus, cfg=cfg, args=args, gold_label=gold_label,
            pairs=pairs, report_path=output_path, generated_at=generated_at)
        write_results_json(document, json_path)
        print(f"Saved results JSON to {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
