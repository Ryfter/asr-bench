import math
import asr_bench
from asr_bench import RunConfig, Pair, WhisperXResult


def test_whisperx_engine_builds_modelresult(tmp_path, monkeypatch):
    audio = tmp_path / "Lec_default.mp4"; audio.write_bytes(b"x")
    ref = tmp_path / "Lec.txt"; ref.write_text("hello there hi back", encoding="utf-8")

    canned = WhisperXResult.from_dict({
        "segments": [{"start": 0, "end": 2, "text": "hello there", "speaker": "SPEAKER_00"},
                     {"start": 2, "end": 4, "text": "hi back", "speaker": "SPEAKER_01"}],
        "words": [], "speakers": ["SPEAKER_00", "SPEAKER_01"], "der": 0.1, "language": "en",
    })
    monkeypatch.setattr(asr_bench, "make_whisperx_adapter", lambda cfg: asr_bench.FakeWhisperXAdapter(canned))
    monkeypatch.setattr(asr_bench, "find_rttm", lambda p: "dummy.rttm")
    monkeypatch.setattr(asr_bench, "_audio_duration_sec", lambda p: 4.0)

    entry = asr_bench.resolve_model_entry("large-v3-turbo+whisperx")
    cfg = RunConfig(device="cpu", compute_type="int8", diarize=True, hf_token="tok")
    mr = asr_bench.WhisperXEngine().run(entry, [Pair(audio=audio, reference=ref)], cfg)

    assert mr.engine == "whisperx" and len(mr.clips) == 1
    c = mr.clips[0]
    assert c.num_speakers == 2
    assert c.der == 0.1
    assert c.speaker_segments[0] == (0.0, 2.0, "SPEAKER_00")
    assert abs(c.wer) < 1e-9          # hypothesis == reference → 0 WER
    assert any(p.name.endswith(".vtt") for p in tmp_path.iterdir())


def test_whisperx_registered():
    assert "whisperx" in asr_bench.ENGINES
