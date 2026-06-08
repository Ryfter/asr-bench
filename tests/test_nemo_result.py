# tests/test_nemo_result.py
import math
from asr_bench import NeMoResult


def test_from_dict_parakeet_shape():
    r = NeMoResult.from_dict({
        "text": "hello world",
        "segments": [{"start": 0.0, "end": 2.0, "text": "hello world"}],
        "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
        "transcribe_sec": 1.5, "vram_peak_bytes": 2_000_000_000, "language": "en",
    })
    assert r.segments[0]["text"] == "hello world"
    assert r.words[0]["word"] == "hello"
    assert r.transcribe_sec == 1.5
    assert r.vram_peak_bytes == 2_000_000_000
    assert r.language == "en"
    assert r.has_timestamps() is True
    assert r.text() == "hello world"          # joined from segments


def test_from_dict_canary_text_only():
    r = NeMoResult.from_dict({"text": "the lecture begins now", "transcribe_sec": 3.2})
    assert r.segments == [] and r.words == []
    assert r.has_timestamps() is False
    assert r.text() == "the lecture begins now"   # falls back to full text
    assert r.vram_peak_bytes is None


def test_text_prefers_segments_over_full_text():
    r = NeMoResult.from_dict({
        "text": "RAW", "segments": [{"start": 0, "end": 1, "text": "clean text"}]})
    assert r.text() == "clean text"


def test_speaker_segments_empty_v04():
    # v0.4 NeMo is transcription-only — no diarization.
    r = NeMoResult.from_dict({"segments": [{"start": 0, "end": 1, "text": "hi"}]})
    assert r.speaker_segments() == []
