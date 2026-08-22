"""Multi-model ASR consensus — deterministic voting across engine outputs.

Complements the LLM-based ``--fuse`` stage: no cloud/local LLM required.
Builds a transcript from agreement across several model VTTs; flags windows
where models disagree. Panopto/proxy references stay scoring baselines only
unless ``include_reference=True``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from engines.base import Cue, normalize_for_wer, _fused_base, _fmt_vtt_time


def build_windows(duration: float, window: float, overlap: float) -> List[Tuple[float, float]]:
    if window <= 0:
        return [(0.0, max(duration, 0.0))]
    step = max(window - overlap, 0.1)
    wins: List[Tuple[float, float]] = []
    t = 0.0
    while t < duration:
        wins.append((t, min(t + window, duration)))
        if t + window >= duration:
            break
        t += step
    return wins or [(0.0, duration)]


def collect_window_text_midpoint(cues: List[Cue], start: float, end: float) -> str:
    mid = (start + end) / 2.0
    parts: List[str] = []
    for c in cues:
        cm = (c.start + c.end) / 2.0
        if start <= cm < end:
            parts.append(c.text.strip())
    return " ".join(parts).strip()


@dataclass
class ConsensusClipResult:
    cues: List[Cue] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    mean_agreement: float = 0.0


def vote_window_texts(
    texts: List[str],
    *,
    min_ratio: float = 0.5,
) -> Tuple[str, float, List[str]]:
    """Majority-vote words within one time window.

    Uses the longest normalized transcript as ordering scaffold; keeps words
    that appear in at least ``ceil(n_models * min_ratio)`` distinct model outputs.
    Returns (consensus_text, agreement_ratio, flags).
    """
    token_lists = [
        normalize_for_wer(t).split()
        for t in texts
        if t and str(t).strip()
    ]
    if not token_lists:
        return "", 0.0, ["no model text in window"]
    n = len(token_lists)
    threshold = max(1, int(math.ceil(n * min_ratio)))
    anchor = max(token_lists, key=len)
    if not anchor:
        return "", 0.0, ["empty window after normalize"]

    bag: Dict[str, int] = {}
    for tl in token_lists:
        for w in set(tl):
            bag[w] = bag.get(w, 0) + 1

    kept = {w for w, c in bag.items() if c >= threshold}
    result = [w for w in anchor if w in kept]
    dropped = [w for w in anchor if w not in kept]
    agreement = len(result) / max(len(anchor), 1)
    flags: List[str] = []
    if dropped:
        preview = ", ".join(dropped[:6])
        suffix = "…" if len(dropped) > 6 else ""
        flags.append(f"dropped {len(dropped)} minority word(s): {preview}{suffix}")
    return " ".join(result), agreement, flags


def build_consensus_clip(
    duration: float,
    sources: Dict[str, List[Cue]],
    *,
    window: float,
    overlap: float,
    min_agreement: float,
) -> ConsensusClipResult:
    """Window the timeline and vote model texts per window."""
    res = ConsensusClipResult()
    wins = build_windows(duration, window, 0.0)
    agreements: List[float] = []
    for w_start, w_end in wins:
        texts = [
            collect_window_text_midpoint(cues, w_start, w_end)
            for cues in sources.values()
        ]
        text, agree, flags = vote_window_texts(texts, min_ratio=min_agreement)
        agreements.append(agree)
        for f in flags:
            res.flags.append(f"[{w_start:.0f}-{w_end:.0f}s] {f}")
        if agree < min_agreement:
            res.flags.append(
                f"[{w_start:.0f}-{w_end:.0f}s] low agreement {agree * 100:.0f}% "
                f"(threshold {min_agreement * 100:.0f}%)"
            )
        if text.strip():
            res.cues.append(Cue(w_start, w_end, text.strip()))
    res.mean_agreement = sum(agreements) / len(agreements) if agreements else 0.0
    return res


def write_consensus_vtt(audio_path: Path, cues: List[Cue]) -> Path:
    out = audio_path.parent / f"{_fused_base(audio_path)}_Captions_Consensus.vtt"
    lines = ["WEBVTT", ""]
    for i, c in enumerate(cues, start=1):
        lines.append(str(i))
        lines.append(f"{_fmt_vtt_time(c.start)} --> {_fmt_vtt_time(c.end)}")
        lines.append(c.text)
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_disagreement_report(audio_path: Path, flags: List[str], mean_agreement: float) -> Path:
    out = audio_path.parent / f"{_fused_base(audio_path)}_Consensus_Disagreements.md"
    lines = [
        f"# Consensus disagreements — {_fused_base(audio_path)}",
        "",
        f"Mean window agreement: **{mean_agreement * 100:.1f}%**",
        "",
    ]
    if flags:
        lines.append("## Flags")
        lines.append("")
        for f in flags:
            lines.append(f"- {f}")
    else:
        lines.append("_No disagreement flags — models aligned within threshold._")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
