"""NeMoEngine — NVIDIA NeMo (Parakeet TDT + Canary-Qwen) engine family.

Runs only as a subprocess into a Python <=3.13 .venv-nemo (torch has no 3.14
wheels) via nemo_runner.py. Heavy imports live in that runner, so importing this
module is torch-free. Parakeet emits native timestamps (full VTT + words sidecar);
Canary-Qwen is WER-only (no timestamps)."""

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
    write_whisperx_vtt, write_words_sidecar, _model_label, _audio_duration_sec,
)


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


# ---- NeMo adapter -----------------------------------------------------------
# The runner script lives at the repo root, one directory above engines/.
_NEMO_RUNNER_PATH = str(Path(__file__).resolve().parent.parent / "nemo_runner.py")


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
    root = Path(__file__).resolve().parent.parent
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
