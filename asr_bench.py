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
import math
import os
import re
import sys
import subprocess
import threading
import time
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
}

_NIM_ADHOC_RE = re.compile(r"^nim:(.+)$")


def resolve_model_entry(model_id: str) -> Dict:
    """Resolve a --models token to a full engine entry.

    Returns a dict that always carries: id, engine, display, developer, params,
    languages, notes. NIM entries also carry riva_model; whisper entries carry
    fw_name. Raises ValueError for unknown ids.
    """
    if model_id in MODELS:
        entry = dict(MODELS[model_id])
        entry.setdefault("engine", "faster-whisper")
        entry["id"] = model_id
        return entry
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
    raise ValueError(f"unknown model id: {model_id}")


# ---- Reference origin detection ---------------------------------------------
_PANOPTO_FILENAME_RE = re.compile(r"_Captions_[A-Za-z]+(?:\s*\([^)]+\))?(?:\s*\(\d+\))?\.txt$")
_ASR_HEADER_RE = re.compile(r"\[Auto-generated transcript", re.IGNORECASE)


def detect_reference_origin(path: Path) -> Tuple[str, str]:
    """Return (origin, label) for a reference file.

    origin: 'panopto-asr' | 'asr-generic' | 'unknown'
    label: short human-readable string for the report
    """
    name = path.name
    if _PANOPTO_FILENAME_RE.search(name):
        return ("panopto-asr", "Panopto auto-generated captions")
    head = ""
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:2000]
    except Exception:
        pass
    if _ASR_HEADER_RE.search(head):
        return ("asr-generic", "ASR-generated captions (auto-detected from header)")
    return ("unknown", "user-provided reference (gold unless --proxy-anyway)")


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


# ---- Caption cue parsing ----------------------------------------------------
@dataclass
class Cue:
    start: float
    end: float
    text: str


_VTT_TS_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
)


def _ts_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_caption_cues(path: Path) -> List[Cue]:
    """Parse a VTT or SRT file into timed cues.

    Tolerant of both '.' (VTT) and ',' (SRT) millisecond separators. Drops the
    WEBVTT header, numeric cue indices, and any fully bracketed line (e.g.
    ``[Applause]``, ``[Music]``, Panopto's ``[Auto-generated transcript...]``).
    Multi-line cue text is joined with spaces.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    cues: List[Cue] = []
    start = end = None
    buf: List[str] = []

    def flush() -> None:
        nonlocal start, end, buf
        if start is not None and buf:
            text = " ".join(buf).strip()
            if text:
                cues.append(Cue(start, end, text))
        start = end = None
        buf = []

    for line in raw.splitlines():
        s = line.strip()
        m = _VTT_TS_RE.search(s)
        if m:
            flush()
            start = _ts_to_seconds(m.group(1), m.group(2), m.group(3), m.group(4))
            end = _ts_to_seconds(m.group(5), m.group(6), m.group(7), m.group(8))
            continue
        if not s:
            flush()
            continue
        # Also drops a cue body that is a bare integer or fully bracketed — harmless for ASR content.
        if s.upper() == "WEBVTT" or _CUE_NUM_RE.match(s) or _BRACKETED_RE.match(s):
            continue
        if start is not None:
            buf.append(s)
    flush()
    return cues


def normalize_for_wer(text: str) -> str:
    """Lowercase, strip punctuation except apostrophes, collapse whitespace.

    Keeps "don't" intact rather than splitting into "don" + "t". Most WER
    implementations do this; we do it explicitly for reproducibility.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---- Word metrics -----------------------------------------------------------
@dataclass
class WordMetrics:
    """All word-level scores from a single jiwer alignment.

    WER  = (S+D+I)/N1                      (edit cost; can exceed 1.0)
    MER  = (S+D+I)/(H+S+D+I)               (Morris et al.; bounded [0,1])
    WIL  = 1 - H*H/(N1*N2)                 (Morris et al.; bounded [0,1])
    where N1 = ref words = H+S+D, N2 = hyp words = H+S+I.
    """
    wer: float
    mer: float
    wil: float
    hits: int
    substitutions: int
    deletions: int
    insertions: int


def compute_word_metrics(reference: str, hypothesis: str) -> WordMetrics:
    """One jiwer.process_words call -> WER, MER, WIL, and H/S/D/I counts.

    Inputs should already be normalized (see normalize_for_wer). Returns NaN
    metrics (not an exception) when alignment is impossible (e.g. empty ref).
    """
    nan = float("nan")
    if not reference.strip():
        return WordMetrics(nan, nan, nan, 0, 0, 0, 0)
    from jiwer import process_words
    try:
        out = process_words(reference, hypothesis)
        return WordMetrics(
            wer=float(out.wer),
            mer=float(out.mer),
            wil=float(out.wil),
            hits=int(out.hits),
            substitutions=int(out.substitutions),
            deletions=int(out.deletions),
            insertions=int(out.insertions),
        )
    except Exception:
        return WordMetrics(nan, nan, nan, 0, 0, 0, 0)


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


# ---- GPU probe + batch-size recommendation ----------------------------------
def gpu_total_and_free_bytes() -> Tuple[Optional[int], Optional[int]]:
    """Return (total, free) VRAM bytes for GPU 0, or (None, None) if NVML is off."""
    if not _HAS_NVML or _NVML_DEVICE_COUNT == 0:
        return (None, None)
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    return (info.total, info.free)


def gpu_name() -> Optional[str]:
    if not _HAS_NVML or _NVML_DEVICE_COUNT == 0:
        return None
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(h)
    return name if isinstance(name, str) else name.decode("utf-8", errors="replace")


# Rough VRAM cost per model at compute_type=float16, including a base + per-batch-item slope.
# Numbers come from observed peaks; conservative so the recommendation doesn't OOM.
_MODEL_VRAM_COST: Dict[str, Tuple[int, int]] = {
    # model_id -> (base_bytes, per_batch_item_bytes)
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


# ---- VTT output -------------------------------------------------------------
def _fmt_vtt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - (h * 3600) - (m * 60)
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def write_whisper_vtt(audio_path: Path, model_label: str, segments: List[Tuple[float, float, str]]) -> Path:
    """Write a WebVTT file next to the audio, named to mirror Panopto's pattern.

    Audio:  <base>_default.mp4  (or any audio extension)
    Output: <base>_Captions_<Model>.vtt
    If the stem doesn't end with `_default`, the bare stem is used.
    """
    stem = audio_path.stem
    base = stem[: -len("_default")] if stem.endswith("_default") else stem
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model_label).strip("-")
    out = audio_path.parent / f"{base}_Captions_{safe_model}.vtt"
    lines: List[str] = ["WEBVTT", ""]
    for i, (start, end, text) in enumerate(segments, start=1):
        text = text.strip()
        if not text:
            continue
        lines.append(str(i))
        lines.append(f"{_fmt_vtt_time(start)} --> {_fmt_vtt_time(end)}")
        lines.append(text)
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ---- VRAM tracking ----------------------------------------------------------
def gpu_used_bytes() -> int:
    if not _HAS_NVML or _NVML_DEVICE_COUNT == 0:
        return 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    return pynvml.nvmlDeviceGetMemoryInfo(handle).used


# ---- Helpers ----------------------------------------------------------------
def _model_label(model_id: str) -> str:
    """small -> Small, large-v3 -> LargeV3, large-v3-turbo -> LargeV3Turbo.

    Used for filename suffixes (`_Captions_<Label>.vtt`) and short report cells.
    """
    return "".join(p.capitalize() for p in model_id.split("-"))


def _fmt_pct(value: float) -> str:
    """Format a 0-1 metric as a 1-decimal percentage, or '—' if NaN/None."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value * 100:.1f}"


def _vram_cell(value: Optional[int], is_total: bool) -> str:
    """Render a VRAM cell; mark NIM 'total used' values with a trailing '*'."""
    if value is None:
        return "n/a" if not _HAS_NVML else "0"
    return fmt_bytes(value) + ("*" if is_total else "")


def _disk_cell(result: "ModelResult") -> str:
    return "n/a" if result.engine == "nim" else fmt_bytes(result.disk_bytes)


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
    mer: float = float("nan")
    wil: float = float("nan")
    hits: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    cue_count: int = 0
    vtt_path: Optional[str] = None
    reference_origin: str = "unknown"
    reference_label: str = ""


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
    engine: str = "faster-whisper"
    vram_is_total: bool = False
    clips: List[ClipResult] = field(default_factory=list)

    @property
    def avg_wer(self) -> float:
        if not self.clips:
            return 0.0
        return sum(c.wer for c in self.clips) / len(self.clips)

    @property
    def avg_mer(self) -> float:
        if not self.clips:
            return 0.0
        return sum(c.mer for c in self.clips) / len(self.clips)

    @property
    def avg_wil(self) -> float:
        if not self.clips:
            return 0.0
        return sum(c.wil for c in self.clips) / len(self.clips)

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


@dataclass
class RunConfig:
    """Settings passed to an engine's run(). Engine-specific fields are ignored
    by the engine they don't apply to."""
    # shared
    device: str
    compute_type: str
    # faster-whisper only
    batch_size: int = 1
    beam_size: int = 5
    vad_filter: bool = True
    # nim only
    nim_url: str = "localhost:50051"
    nim_model: str = ""
    nim_language: str = "en-US"
    nim_api_key: Optional[str] = None
    nim_ssl: bool = False


class Engine(ABC):
    """Contract every ASR engine family implements. Returns a ModelResult so the
    report renderer is engine-agnostic."""
    name: str = ""

    @abstractmethod
    def run(self, entry: Dict, pairs: List[Pair], cfg: RunConfig) -> "ModelResult":
        ...


# ---- NIM helpers ------------------------------------------------------------
class VramSampler:
    """Background poller that records peak total GPU memory used during a call.

    Used for the NIM path, where a single blocking offline_recognize RPC offers
    no per-segment loop to sample in. Reports TOTAL used (model is pre-resident
    in the container), not a per-clip delta — callers must mark it as such.
    """
    def __init__(self, read_fn=gpu_used_bytes, interval: float = 0.1):
        self._read_fn = read_fn
        self._interval = interval
        self.peak: int = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _record(self, value: int) -> None:
        if value > self.peak:
            self.peak = value

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._record(self._read_fn())
            self._stop.wait(self._interval)

    def start(self) -> "VramSampler":
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self.peak


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


def group_words_into_cues(
    words: List[Tuple[float, float, str]],
    max_words: int = 12,
    max_span: float = 6.0,
) -> List[Tuple[float, float, str]]:
    """Group word timings into VTT-style cues. Close a cue on sentence-final
    punctuation, or when it reaches max_words, or spans >= max_span seconds."""
    cues: List[Tuple[float, float, str]] = []
    buf: List[str] = []
    start: Optional[float] = None
    end: float = 0.0
    for (ws, we, text) in words:
        if start is None:
            start = ws
        buf.append(text)
        end = we
        ends_sentence = text.rstrip().endswith((".", "?", "!"))
        if ends_sentence or len(buf) >= max_words or (end - start) >= max_span:
            cues.append((start, end, " ".join(buf)))
            buf, start = [], None
    if buf and start is not None:
        cues.append((start, end, " ".join(buf)))
    return cues


def decode_to_pcm16(path: Path, target_rate: int = 16000) -> Tuple[bytes, int]:
    """Decode any audio/video file to 16kHz mono s16le PCM bytes.

    Primary: pyav (already installed via faster-whisper). Fallback: an ffmpeg
    subprocess. Returns (pcm_bytes, n_samples). Raises RuntimeError if neither
    path works.
    """
    # --- Primary: pyav ---
    pyav_error = None
    try:
        import av  # type: ignore
        from av.audio.resampler import AudioResampler  # type: ignore

        container = av.open(str(path))
        resampler = AudioResampler(format="s16", layout="mono", rate=target_rate)
        chunks: List[bytes] = []
        for frame in container.decode(audio=0):
            for rframe in resampler.resample(frame):
                chunks.append(rframe.to_ndarray().astype("<i2").tobytes())
        # Flush the resampler.
        for rframe in resampler.resample(None):
            chunks.append(rframe.to_ndarray().astype("<i2").tobytes())
        container.close()
        pcm = b"".join(chunks)
        if pcm:
            return pcm, len(pcm) // 2
    except Exception as e:
        pyav_error = e  # fall through to ffmpeg

    # --- Fallback: ffmpeg subprocess ---
    import subprocess
    try:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-i", str(path),
             "-ar", str(target_rate), "-ac", "1", "-f", "s16le", "-"],
            capture_output=True, check=True,
        )
        pcm = proc.stdout
        return pcm, len(pcm) // 2
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            f"could not decode {path}: pyav error={pyav_error!r}; ffmpeg error={e}"
        )


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

            result.clips.append(
                ClipResult(
                    audio=pair.audio.name, audio_sec=audio_sec,
                    transcribe_sec=transcribe_sec, rtfx=rtfx,
                    vram_peak_bytes=vram_peak, hypothesis=hypothesis,
                    reference_normalized=ref_norm, hypothesis_normalized=hyp_norm,
                    wer=wer_val, mer=metrics.mer, wil=metrics.wil,
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


# ---- Fusion -----------------------------------------------------------------
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
    """Shell out to an authenticated frontier CLI (e.g. `claude -p`, `gemini`).

    The prompt is passed on stdin to avoid arg-length limits. Uses the operator's
    existing subscription — no API key is stored in asr-bench.
    """
    name = "cli"

    def __init__(self, command: List[str], timeout: float = 300.0):
        self.command = command
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        proc = subprocess.run(
            self.command, input=prompt, capture_output=True, text=True,
            timeout=self.timeout, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"LLM CLI {self.command} exited {proc.returncode}: {proc.stderr[:500]}")
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
    lines.append("## Headline")
    lines.append("")
    lines.append("| Model | Params | Disk | Overall WER% | MER% | WIL% | RTFx | Total time | Peak VRAM | Notes |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        wall_clock = f"{r.total_transcribe_sec:.1f}s"
        wer_pct = _fmt_pct(r.avg_wer) if r.clips else "—"
        mer_pct = _fmt_pct(r.avg_mer) if r.clips else "—"
        wil_pct = _fmt_pct(r.avg_wil) if r.clips else "—"
        rtfx = f"{r.aggregate_rtfx:.2f}x" if r.clips else "—"
        vram = _vram_cell(r.peak_vram_bytes, r.vram_is_total)
        disk = _disk_cell(r)
        lines.append(
            f"| {r.display} | {r.params} | {disk} | {wer_pct} | {mer_pct} | {wil_pct} | {rtfx} | {wall_clock} | {vram} | {r.notes} |"
        )
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
            lines.append("| Model | WER% | MER% | WIL% | S | D | I | RTFx | Transcribe time | VRAM peak |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for r in results:
                if i < len(r.clips):
                    c = r.clips[i]
                    wer_pct = _fmt_pct(c.wer)
                    mer_pct = _fmt_pct(c.mer)
                    wil_pct = _fmt_pct(c.wil)
                    vram = _vram_cell(c.vram_peak_bytes, r.vram_is_total)
                    lines.append(
                        f"| {r.display} | {wer_pct} | {mer_pct} | {wil_pct} | {c.substitutions} | {c.deletions} | {c.insertions} | {c.rtfx:.2f}x | {c.transcribe_sec:.1f}s | {vram} |"
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
        lines.append("| Clip | Audio | WER% | MER% | WIL% | RTFx | Transcribe time | VRAM peak |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for c in r.clips:
            wer_pct = _fmt_pct(c.wer)
            mer_pct = _fmt_pct(c.mer)
            wil_pct = _fmt_pct(c.wil)
            vram = _vram_cell(c.vram_peak_bytes, r.vram_is_total)
            audio_label = f"{c.audio_sec / 60:.1f} min"
            lines.append(
                f"| {c.audio} | {audio_label} | {wer_pct} | {mer_pct} | {wil_pct} | {c.rtfx:.2f}x | {c.transcribe_sec:.1f}s | {vram} |"
            )
        overall_audio = f"{r.total_audio_sec / 60:.1f} min"
        overall_wer = _fmt_pct(r.avg_wer) if r.clips else "—"
        overall_mer = _fmt_pct(r.avg_mer) if r.clips else "—"
        overall_wil = _fmt_pct(r.avg_wil) if r.clips else "—"
        overall_rtfx = f"{r.aggregate_rtfx:.2f}x" if r.clips else "—"
        overall_vram = _vram_cell(r.peak_vram_bytes, r.vram_is_total)
        lines.append(
            f"| **OVERALL** | **{overall_audio}** | **{overall_wer}** | **{overall_mer}** | **{overall_wil}** | **{overall_rtfx}** | **{r.total_transcribe_sec:.1f}s** | **{overall_vram}** |"
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
    batch_flag = f" --batch-size {args.batch_size}" if args.batch_size > 1 else ""
    beam_flag = f" --beam-size {args.beam_size}" if args.beam_size != 5 else ""
    vad_flag = "" if args.vad_filter else " --no-vad-filter"
    nim_flag = ""
    if any(r.engine == "nim" for r in results):
        nim_flag = f" --nim-url {args.nim_url} --nim-language {args.nim_language}"
        if args.nim_model:
            nim_flag += f" --nim-model {args.nim_model}"
    lines.append(f"- Command: `python asr_bench.py --corpus '{corpus_path}' --models {','.join(args.models)} --device {args.device} --compute-type {args.compute_type}{batch_flag}{beam_flag}{vad_flag}{nim_flag}`")
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
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark local Whisper variants on your own audio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
    args = ap.parse_args()

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
    print()
    print(md)

    # Save
    output_path = Path(args.output) if args.output else None
    if output_path is None:
        report_dir = Path(__file__).resolve().parent / "report"
        report_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = report_dir / f"{ts}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    print(f"\nSaved report to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
