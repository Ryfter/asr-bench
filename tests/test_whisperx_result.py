import math
import asr_bench


def test_from_dict_full():
    d = {
        "segments": [{"start": 0.0, "end": 2.0, "text": "hello", "speaker": "SPEAKER_00"}],
        "words": [{"word": "hello", "start": 0.0, "end": 0.5, "score": 0.9, "speaker": "SPEAKER_00"}],
        "speakers": ["SPEAKER_00"],
        "der": 0.12,
        "language": "en",
    }
    r = asr_bench.WhisperXResult.from_dict(d)
    assert r.segments[0]["text"] == "hello"
    assert r.speakers == ["SPEAKER_00"]
    assert r.der == 0.12
    assert r.language == "en"
    assert r.text() == "hello"


def test_from_dict_minimal_no_diarization():
    d = {"segments": [{"start": 0, "end": 1, "text": "a"}, {"start": 1, "end": 2, "text": "b"}],
         "language": "en"}
    r = asr_bench.WhisperXResult.from_dict(d)
    assert r.der is None
    assert r.speakers == []
    assert r.words == []
    assert r.text() == "a b"


def test_speaker_segments_helper():
    d = {"segments": [{"start": 0, "end": 1, "text": "a", "speaker": "SPEAKER_00"},
                      {"start": 1, "end": 2, "text": "b", "speaker": "SPEAKER_01"}],
         "speakers": ["SPEAKER_00", "SPEAKER_01"], "language": "en"}
    r = asr_bench.WhisperXResult.from_dict(d)
    assert r.speaker_segments() == [(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")]
