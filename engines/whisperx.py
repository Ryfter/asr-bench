"""WhisperXEngine — Whisper + WhisperX word alignment + pyannote diarization.

Two adapters auto-selected at runtime: in-process (torch importable in this
interpreter) or subprocess to a Python <=3.13 venv (.venv-whisperx). Heavy imports
(torch/whisperx/pyannote) live inside the adapters / whisperx_runner, never here,
so importing this module is torch-free."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from engines.base import (
    Engine, ModelResult, ClipResult, Pair, RunConfig,
    load_reference_text, detect_reference_origin, normalize_for_wer,
    compute_word_metrics, compute_hallucination_signals,
    write_whisperx_vtt, write_words_sidecar, find_rttm, _model_label,
    _audio_duration_sec,
)


# ---- WhisperX result --------------------------------------------------------
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


# ---- WhisperX adapter -------------------------------------------------------
# The runner script lives at the repo root, one directory above engines/.
_RUNNER_PATH = str(Path(__file__).resolve().parent.parent / "whisperx_runner.py")


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
    root = Path(__file__).resolve().parent.parent
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
