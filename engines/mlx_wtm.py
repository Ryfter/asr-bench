"""WtmEngine — Whisper Turbo MLX (Apple Silicon) via whisper-turbo-mlx / wtm.

Runs as a subprocess into .venv-wtm (MLX has no Windows/CUDA path). Heavy imports
live in wtm_runner.py so this module stays torch-free."""

from __future__ import annotations

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
    write_whisper_vtt, _model_label, _audio_duration_sec,
)


@dataclass
class WtmResult:
    segments: List[Dict] = field(default_factory=list)
    full_text: str = ""
    transcribe_sec: Optional[float] = None
    language: str = ""

    @classmethod
    def from_dict(cls, d: Dict) -> "WtmResult":
        return cls(
            segments=list(d.get("segments") or []),
            full_text=d.get("text") or "",
            transcribe_sec=d.get("transcribe_sec"),
            language=d.get("language") or "",
        )

    def text(self) -> str:
        if self.segments:
            return " ".join(s.get("text", "").strip() for s in self.segments).strip()
        return self.full_text.strip()

    def cue_tuples(self) -> List[Tuple[float, float, str]]:
        out: List[Tuple[float, float, str]] = []
        for s in self.segments:
            t = s.get("text", "").strip()
            if t:
                out.append((float(s["start"]), float(s["end"]), t))
        return out


_WTM_RUNNER_PATH = str(Path(__file__).resolve().parent.parent / "wtm_runner.py")


class WtmAdapter(ABC):
    name: str = ""

    @abstractmethod
    def transcribe(self, audio_path: str, cfg: "RunConfig") -> "WtmResult":
        ...


class FakeWtmAdapter(WtmAdapter):
    name = "fake"

    def __init__(self, result: "WtmResult"):
        self._result = result

    def transcribe(self, audio_path, cfg):
        return self._result


def _wtm_runner_args(audio_path: str, cfg: "RunConfig") -> List[str]:
    args = [_WTM_RUNNER_PATH, "--audio", audio_path, "--timestamps"]
    if cfg.wtm_quick:
        args.append("--quick")
    if cfg.wtm_multilingual:
        args.append("--multilingual")
    return args


class SubprocessWtm(WtmAdapter):
    name = "subprocess"

    def __init__(self, python: str, timeout: float = 7200.0):
        self.python = python
        self.timeout = timeout

    def transcribe(self, audio_path, cfg):
        exe = shutil.which(self.python) or self.python
        cmd = [exe, *_wtm_runner_args(audio_path, cfg)]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.timeout, check=False, env=dict(os.environ),
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"wtm_runner timed out after {self.timeout}s")
        if proc.returncode != 0:
            raise RuntimeError(f"wtm_runner failed ({proc.returncode}): {proc.stderr[:500]}")
        return WtmResult.from_dict(json.loads(proc.stdout))


def _default_wtm_python() -> Optional[str]:
    root = Path(__file__).resolve().parent.parent
    for rel in (".venv-wtm/bin/python", ".venv-wtm/Scripts/python.exe"):
        cand = root / rel
        if cand.is_file():
            return str(cand)
    return None


def make_wtm_adapter(cfg: "RunConfig") -> WtmAdapter:
    venv_py = cfg.wtm_python or _default_wtm_python()
    if venv_py:
        return SubprocessWtm(venv_py)
    raise RuntimeError(
        "MLX Whisper Turbo needs .venv-wtm with whisper-turbo-mlx. "
        "Run setup_wtm_venv.sh (macOS Apple Silicon) or pass --wtm-python."
    )


class WtmEngine(Engine):
    name = "mlx-wtm"

    def run(self, entry: Dict, pairs: List[Pair], cfg: RunConfig) -> ModelResult:
        adapter = make_wtm_adapter(cfg)
        print(f"\n[{entry['display']}] using MLX WTM adapter: {adapter.name}", flush=True)
        result_model = ModelResult(
            model_id=entry["id"], display=entry["display"], fw_name=entry.get("fw_name", ""),
            params=entry["params"], developer=entry["developer"], languages=entry["languages"],
            notes=entry["notes"], disk_bytes=None, load_sec=0.0,
            engine="mlx-wtm", vram_is_total=False,
        )
        for clip_idx, pair in enumerate(pairs, start=1):
            print(f"  [{clip_idx}/{len(pairs)}] transcribing {pair.audio.name}...", flush=True)
            ref_text = load_reference_text(pair.reference)
            ref_origin, ref_label = detect_reference_origin(pair.reference)
            t0 = time.time()
            try:
                wtm = adapter.transcribe(str(pair.audio), cfg)
            except Exception as e:
                print(f"  ERROR wtm on {pair.audio.name}: {e}", file=sys.stderr)
                continue
            wall = time.time() - t0
            transcribe_sec = wtm.transcribe_sec if wtm.transcribe_sec is not None else wall

            hypothesis = wtm.text()
            cue_tuples = wtm.cue_tuples()
            label = _model_label(entry["id"])
            vtt_path = write_whisper_vtt(pair.audio, label, cue_tuples) if cue_tuples else None

            ref_norm = normalize_for_wer(ref_text)
            hyp_norm = normalize_for_wer(hypothesis)
            metrics = compute_word_metrics(ref_norm, hyp_norm)

            try:
                audio_sec = _audio_duration_sec(pair.audio)
            except Exception:
                audio_sec = (
                    float(cue_tuples[-1][1]) if cue_tuples else transcribe_sec or 1.0
                )
            rtfx = audio_sec / transcribe_sec if transcribe_sec > 0 else 0.0

            print(
                f"    {audio_sec:.1f}s audio in {transcribe_sec:.1f}s "
                f"(RTFx {rtfx:.2f}, WER {metrics.wer * 100:.1f}%)",
                flush=True,
            )

            rep_cov, comp_ratio = compute_hallucination_signals(hypothesis, hyp_norm)
            result_model.clips.append(
                ClipResult(
                    audio=pair.audio.name,
                    audio_sec=audio_sec,
                    transcribe_sec=transcribe_sec,
                    rtfx=rtfx,
                    vram_peak_bytes=None,
                    hypothesis=hypothesis,
                    reference_normalized=ref_norm,
                    hypothesis_normalized=hyp_norm,
                    wer=metrics.wer,
                    mer=metrics.mer,
                    wil=metrics.wil,
                    cer=metrics.cer,
                    repeat_coverage=rep_cov, compression_ratio=comp_ratio,
                    hits=metrics.hits,
                    substitutions=metrics.substitutions,
                    deletions=metrics.deletions,
                    insertions=metrics.insertions,
                    cue_count=len(cue_tuples),
                    vtt_path=str(vtt_path) if vtt_path else None,
                    reference_origin=ref_origin,
                    reference_label=ref_label,
                )
            )
        return result_model
