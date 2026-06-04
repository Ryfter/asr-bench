import json
import asr_bench
from asr_bench import WhisperXResult


def _result():
    return WhisperXResult.from_dict({
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "hello there", "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "text": "hi back", "speaker": "SPEAKER_01"},
        ],
        "words": [{"word": "hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"}],
        "speakers": ["SPEAKER_00", "SPEAKER_01"], "language": "en",
    })


def test_write_whisperx_vtt_speaker_prefixed(tmp_path):
    audio = tmp_path / "Lec_default.mp4"; audio.write_bytes(b"x")
    out = asr_bench.write_whisperx_vtt(audio, "LargeV3TurboWhisperx", _result())
    body = out.read_text(encoding="utf-8")
    assert "WEBVTT" in body
    assert "SPEAKER_00: hello there" in body
    assert "SPEAKER_01: hi back" in body


def test_write_whisperx_vtt_no_speaker_prefix_when_absent(tmp_path):
    audio = tmp_path / "Lec_default.mp4"; audio.write_bytes(b"x")
    r = WhisperXResult.from_dict({"segments": [{"start": 0, "end": 1, "text": "plain"}], "language": "en"})
    out = asr_bench.write_whisperx_vtt(audio, "M", r)
    body = out.read_text(encoding="utf-8")
    assert "plain" in body and "SPEAKER" not in body


def test_write_words_sidecar(tmp_path):
    audio = tmp_path / "Lec_default.mp4"; audio.write_bytes(b"x")
    out = asr_bench.write_words_sidecar(audio, "M", _result())
    assert out.name == "Lec_Words_M.json"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data[0]["word"] == "hello"
