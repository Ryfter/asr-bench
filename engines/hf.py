"""HuggingFace-transformers engine (wav2vec2 / Conformer CTC). Subprocess into a
3.12 .venv-hf (torch has no 3.14 wheels) via hf_runner.py — pure-JSON stdout.
Mirrors engines/nemo.py. CTC word timestamps -> full VTT + _Words_*.json."""
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

_HF_RUNNER_PATH = str(Path(__file__).resolve().parent.parent / "hf_runner.py")


@dataclass
class HFResult:
    """Parsed hf_runner output. Duck-typed (.segments/.words) to reuse
    write_whisperx_vtt / write_words_sidecar, like NeMoResult."""
    segments: List[Dict] = field(default_factory=list)
    words: List[Dict] = field(default_factory=list)
    full_text: str = ""
    transcribe_sec: Optional[float] = None
    vram_peak_bytes: Optional[int] = None
    language: str = ""

    @classmethod
    def from_dict(cls, d: Dict) -> "HFResult":
        return cls(segments=list(d.get("segments") or []),
                   words=list(d.get("words") or []),
                   full_text=d.get("text") or "",
                   transcribe_sec=d.get("transcribe_sec"),
                   vram_peak_bytes=d.get("vram_peak_bytes"),
                   language=d.get("language") or "")

    def text(self) -> str:
        if self.segments:
            return " ".join(s.get("text", "").strip() for s in self.segments).strip()
        return self.full_text.strip()

    def has_timestamps(self) -> bool:
        return bool(self.segments)

    def speaker_segments(self) -> List[Tuple[float, float, str]]:
        return []   # HF engine is transcription-only


class HFAdapter(ABC):
    name: str = ""
    @abstractmethod
    def transcribe(self, audio_path: str, model: str, cfg: RunConfig) -> HFResult: ...


class FakeHFAdapter(HFAdapter):
    name = "fake"
    def __init__(self, result: HFResult): self._result = result
    def transcribe(self, audio_path, model, cfg): return self._result


def _hf_runner_args(audio_path: str, model: str, cfg: RunConfig) -> List[str]:
    return [_HF_RUNNER_PATH, "--audio", audio_path, "--model", model,
            "--device", cfg.device, "--language", "en"]


class SubprocessHF(HFAdapter):
    name = "subprocess"
    def __init__(self, python: str, timeout: float = 7200.0):
        self.python = python
        self.timeout = timeout
    def transcribe(self, audio_path, model, cfg):
        exe = shutil.which(self.python) or self.python
        cmd = [exe, *_hf_runner_args(audio_path, model, cfg)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=self.timeout, check=False, env=dict(os.environ))
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"hf_runner timed out after {self.timeout}s")
        if proc.returncode != 0:
            raise RuntimeError(f"hf_runner failed ({proc.returncode}): {proc.stderr[:500]}")
        return HFResult.from_dict(json.loads(proc.stdout))


def _default_hf_python() -> Optional[str]:
    root = Path(__file__).resolve().parent.parent
    for rel in (".venv-hf/Scripts/python.exe", ".venv-hf/bin/python"):
        cand = root / rel
        if cand.is_file():
            return str(cand)
    return None


def make_hf_adapter(cfg: RunConfig) -> HFAdapter:
    venv_py = cfg.hf_python or _default_hf_python()
    if venv_py:
        return SubprocessHF(venv_py)
    raise RuntimeError(
        "HF engine needs a Python 3.12 venv with torch (cu128) + transformers. "
        "Run setup_hf_venv.ps1 (creates ./.venv-hf) or pass --hf-python.")


class HFTransformersEngine(Engine):
    name = "hf"

    def run(self, entry: Dict, pairs: List[Pair], cfg: RunConfig) -> ModelResult:
        adapter = make_hf_adapter(cfg)
        print(f"\n[{entry['display']}] using HF adapter: {adapter.name}", flush=True)
        result_model = ModelResult(
            model_id=entry["id"], display=entry["display"], fw_name=entry.get("fw_name", ""),
            params=entry["params"], developer=entry["developer"], languages=entry["languages"],
            notes=entry["notes"], disk_bytes=None, load_sec=0.0,
            engine="hf", vram_is_total=False)
        for clip_idx, pair in enumerate(pairs, start=1):
            print(f"  [{clip_idx}/{len(pairs)}] transcribing {pair.audio.name}...", flush=True)
            ref_text = load_reference_text(pair.reference)
            ref_origin, ref_label = detect_reference_origin(pair.reference)
            t0 = time.time()
            try:
                hr = adapter.transcribe(str(pair.audio), entry["hf_model"], cfg)
            except Exception as e:
                print(f"  ERROR hf on {pair.audio.name}: {e}", file=sys.stderr)
                continue
            wall = time.time() - t0
            transcribe_sec = hr.transcribe_sec if hr.transcribe_sec is not None else wall
            hypothesis = hr.text()
            ref_norm = normalize_for_wer(ref_text)
            hyp_norm = normalize_for_wer(hypothesis)
            metrics = compute_word_metrics(ref_norm, hyp_norm)
            label = _model_label(entry["id"])
            vtt_path = None
            cue_count = 0
            if hr.has_timestamps():
                vtt_path = str(write_whisperx_vtt(pair.audio, label, hr))
                write_words_sidecar(pair.audio, label, hr)
                cue_count = len(hr.segments)
            audio_sec = _audio_duration_sec(str(pair.audio)) or (
                hr.segments[-1]["end"] if hr.segments else 0.0)
            rtfx = audio_sec / transcribe_sec if transcribe_sec > 0 else 0.0
            print(f"    {audio_sec:.1f}s in {transcribe_sec:.1f}s (RTFx {rtfx:.2f}, "
                  f"WER {metrics.wer*100:.1f}%)", flush=True)
            rep_cov, comp_ratio = compute_hallucination_signals(hypothesis, hyp_norm)
            result_model.clips.append(ClipResult(
                audio=pair.audio.name, audio_sec=audio_sec, transcribe_sec=transcribe_sec,
                rtfx=rtfx, vram_peak_bytes=hr.vram_peak_bytes, hypothesis=hypothesis,
                reference_normalized=ref_norm, hypothesis_normalized=hyp_norm,
                wer=metrics.wer, mer=metrics.mer, wil=metrics.wil, cer=metrics.cer,
                repeat_coverage=rep_cov, compression_ratio=comp_ratio,
                hits=metrics.hits, substitutions=metrics.substitutions,
                deletions=metrics.deletions, insertions=metrics.insertions,
                cue_count=cue_count, vtt_path=vtt_path,
                reference_origin=ref_origin, reference_label=ref_label))
        if not result_model.clips:
            result_model.notes = "ALL CLIPS FAILED — check HF setup/.venv-hf and stderr above"
        return result_model
