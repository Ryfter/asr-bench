"""Shared, torch-free support layer for every Engine family: the Engine ABC, the
result/config dataclasses, metrics, VRAM sampling, audio decode, and the VTT/words
writers. Engine modules import from here; asr_bench re-exports these names so the
public surface (import asr_bench / python asr_bench.py) is unchanged.

Torch-free at module scope by contract -- any heavy import belongs inside an
engine's run()/adapter, not here."""

from __future__ import annotations

import json
import re
import statistics
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    """All alignment scores from a single jiwer alignment (word-level + CER).

    WER  = (S+D+I)/N1                      (edit cost; can exceed 1.0)
    MER  = (S+D+I)/(H+S+D+I)               (Morris et al.; bounded [0,1])
    WIL  = 1 - H*H/(N1*N2)                 (Morris et al.; bounded [0,1])
    CER  = char edit distance / len(reference)  (character-level WER; bounded [0,inf])
    where N1 = ref words = H+S+D, N2 = hyp words = H+S+I.
    """
    wer: float
    mer: float
    wil: float
    cer: float
    hits: int
    substitutions: int
    deletions: int
    insertions: int


# ---- Hallucination signals (reference-free) ---------------------------------
HALLUCINATION_NGRAM = 4
HALLUCINATION_MIN_WORDS = 8          # below this, repeat_coverage is unreliable -> 0.0
HALLUCINATION_MIN_CHARS = 200        # below this, compression_ratio is meaningless -> 1.0
HALLUCINATION_REPEAT_COVERAGE = 0.30  # flag threshold
HALLUCINATION_COMPRESSION_RATIO = 2.4  # flag threshold (Whisper's own default)
HALLUCINATION_INSERTION_RATE = 0.5   # report annotation threshold (reference-based)


def _repeat_coverage(normalized_hypothesis: str, n: int = HALLUCINATION_NGRAM) -> float:
    """Fraction of word positions covered by an n-gram that occurs >= 2 times.
    0.0 when there are fewer than HALLUCINATION_MIN_WORDS words."""
    words = normalized_hypothesis.split()
    if len(words) < HALLUCINATION_MIN_WORDS:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    counts: Dict[tuple, int] = {}
    for g in ngrams:
        counts[g] = counts.get(g, 0) + 1
    covered = [False] * len(words)
    for i, g in enumerate(ngrams):
        if counts[g] >= 2:
            for j in range(i, i + n):
                covered[j] = True
    return sum(covered) / len(words)


def _compression_ratio(text: str) -> float:
    """len(utf8 bytes) / len(gzip(bytes)). ~1.5-2.2 normal prose, >2.4 repetitive.
    Returns 1.0 for text shorter than HALLUCINATION_MIN_CHARS (gzip overhead makes
    tiny inputs meaningless)."""
    raw = text.encode("utf-8")
    if len(raw) < HALLUCINATION_MIN_CHARS:
        return 1.0
    import gzip
    compressed = gzip.compress(raw)
    return len(raw) / len(compressed) if compressed else 1.0


def compute_hallucination_signals(hypothesis: str, hypothesis_normalized: str) -> Tuple[float, float]:
    """(repeat_coverage, compression_ratio) for a clip. Reference-free."""
    return _repeat_coverage(hypothesis_normalized), _compression_ratio(hypothesis)


def compute_word_metrics(reference: str, hypothesis: str) -> WordMetrics:
    """One jiwer.process_words call -> WER, MER, WIL, and H/S/D/I counts.

    Inputs should already be normalized (see normalize_for_wer). Returns NaN
    metrics (not an exception) when alignment is impossible (e.g. empty ref).
    """
    nan = float("nan")
    if not reference.strip():
        return WordMetrics(nan, nan, nan, nan, 0, 0, 0, 0)
    from jiwer import process_words, cer as jiwer_cer
    try:
        out = process_words(reference, hypothesis)
        return WordMetrics(
            wer=float(out.wer),
            mer=float(out.mer),
            wil=float(out.wil),
            cer=float(jiwer_cer(reference, hypothesis)),
            hits=int(out.hits),
            substitutions=int(out.substitutions),
            deletions=int(out.deletions),
            insertions=int(out.insertions),
        )
    except Exception:
        return WordMetrics(nan, nan, nan, nan, 0, 0, 0, 0)


# ---- Corpus pair type -------------------------------------------------------
@dataclass
class Pair:
    audio: Path
    reference: Path

    @property
    def stem(self) -> str:
        return self.audio.stem.replace("_default", "")


# ---- GPU probe --------------------------------------------------------------
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


def gpu_used_bytes() -> int:
    if not _HAS_NVML or _NVML_DEVICE_COUNT == 0:
        return 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    return pynvml.nvmlDeviceGetMemoryInfo(handle).used


# ---- VTT output -------------------------------------------------------------
def _fused_base(audio_path: Path) -> str:
    stem = audio_path.stem
    return stem[: -len("_default")] if stem.endswith("_default") else stem


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


def write_whisperx_vtt(audio_path: Path, model_label: str, result: "WhisperXResult") -> Path:
    """WebVTT with speaker-prefixed cues (e.g. 'SPEAKER_00: text'). Named like the
    other engines' VTTs: <base>_Captions_<Model>.vtt."""
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model_label).strip("-")
    out = audio_path.parent / f"{_fused_base(audio_path)}_Captions_{safe_model}.vtt"
    lines: List[str] = ["WEBVTT", ""]
    n = 0
    for s in result.segments:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        spk = s.get("speaker")
        if spk:
            text = f"{spk}: {text}"
        n += 1
        lines.append(str(n))
        lines.append(f"{_fmt_vtt_time(float(s['start']))} --> {_fmt_vtt_time(float(s['end']))}")
        lines.append(text)
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_words_sidecar(audio_path: Path, model_label: str, result: "WhisperXResult") -> Path:
    """Write word-level timestamps (and speaker, if present) to a JSON sidecar."""
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model_label).strip("-")
    out = audio_path.parent / f"{_fused_base(audio_path)}_Words_{safe_model}.json"
    out.write_text(json.dumps(result.words, ensure_ascii=False, indent=0), encoding="utf-8")
    return out


def find_rttm(audio_path: Path) -> Optional[Path]:
    """Return the <base>.rttm ground-truth sidecar next to the audio, or None.
    Matches the same base as the VTT writers (strips a trailing _default)."""
    cand = audio_path.parent / f"{_fused_base(audio_path)}.rttm"
    return cand if cand.is_file() else None


# ---- Helpers ----------------------------------------------------------------
def _model_label(model_id: str) -> str:
    """small -> Small, large-v3 -> LargeV3, large-v3-turbo -> LargeV3Turbo.

    Used for filename suffixes (`_Captions_<Label>.vtt`) and short report cells.
    """
    return "".join(p.capitalize() for p in model_id.split("-"))


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
    cer: float = float("nan")
    repeat_coverage: float = 0.0
    compression_ratio: float = 1.0
    hits: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    speaker_segments: List[Tuple[float, float, str]] = field(default_factory=list)
    num_speakers: int = 0
    der: float = float("nan")
    cue_count: int = 0
    vtt_path: Optional[str] = None
    reference_origin: str = "unknown"
    reference_label: str = ""

    @property
    def is_hallucination_suspect(self) -> bool:
        return (self.repeat_coverage > HALLUCINATION_REPEAT_COVERAGE
                or self.compression_ratio > HALLUCINATION_COMPRESSION_RATIO)


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
    def avg_cer(self) -> float:
        if not self.clips:
            return 0.0
        return sum(c.cer for c in self.clips) / len(self.clips)

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
    def median_rtfx(self) -> float:
        """Median per-clip RTFx — robust to a single decoder-lockup outlier that
        would drag down the totals-based aggregate_rtfx."""
        if not self.clips:
            return 0.0
        return statistics.median(c.rtfx for c in self.clips)

    @property
    def median_sec_per_audio_min(self) -> float:
        """Median per-clip compute-seconds per minute of audio (lower = faster)."""
        vals = [c.transcribe_sec * 60.0 / c.audio_sec
                for c in self.clips if c.audio_sec > 0]
        return statistics.median(vals) if vals else 0.0

    @property
    def peak_vram_bytes(self) -> Optional[int]:
        peaks = [c.vram_peak_bytes for c in self.clips if c.vram_peak_bytes is not None]
        return max(peaks) if peaks else None

    @property
    def hallucination_rate(self) -> float:
        """Fraction of this model's clips flagged as hallucination-suspect."""
        if not self.clips:
            return 0.0
        return sum(1 for c in self.clips if c.is_hallucination_suspect) / len(self.clips)


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
    # whisperx only
    whisperx_python: Optional[str] = None
    diarize: bool = True
    hf_token: Optional[str] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    # nemo only
    nemo_python: Optional[str] = None
    # hf only
    hf_python: Optional[str] = None


class Engine(ABC):
    """Contract every ASR engine family implements. Returns a ModelResult so the
    report renderer is engine-agnostic."""
    name: str = ""

    @abstractmethod
    def run(self, entry: Dict, pairs: List[Pair], cfg: RunConfig) -> "ModelResult":
        ...


# ---- VRAM sampling ----------------------------------------------------------
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


def _audio_duration_sec(audio_path: str) -> float:
    """Best-effort audio duration in seconds via pyav (a faster-whisper dep).
    Returns 0.0 on failure; callers fall back to the last segment end."""
    try:
        import av
        with av.open(audio_path) as container:
            if container.duration:
                return float(container.duration) / 1_000_000.0
    except Exception:
        pass
    return 0.0
