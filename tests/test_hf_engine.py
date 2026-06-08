import asr_bench
from asr_bench import RunConfig, Pair
from engines.hf import HFResult, FakeHFAdapter, HFTransformersEngine
import engines.hf as hf_mod


def _cfg():
    return RunConfig(device="cpu", compute_type="int8", hf_python="/fake/python")


def test_wav2vec2_writes_vtt_and_scores(tmp_path, monkeypatch):
    audio = tmp_path / "Lec.mp4"; audio.write_bytes(b"x")
    ref = tmp_path / "Lec.txt"; ref.write_text("hello there hi back", encoding="utf-8")
    canned = HFResult.from_dict({
        "text": "hello there hi back",
        "segments": [{"start": 0.0, "end": 2.0, "text": "hello there"},
                     {"start": 2.0, "end": 4.0, "text": "hi back"}],
        "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
        "transcribe_sec": 1.0, "vram_peak_bytes": 1_500_000_000, "language": "en"})
    monkeypatch.setattr(hf_mod, "make_hf_adapter", lambda cfg: FakeHFAdapter(canned))
    monkeypatch.setattr("engines.base._audio_duration_sec", lambda p: 4.0)
    entry = asr_bench.resolve_model_entry("wav2vec2-large-960h")
    mr = HFTransformersEngine().run(entry, [Pair(audio=audio, reference=ref)], _cfg())
    assert mr.engine == "hf" and len(mr.clips) == 1
    c = mr.clips[0]
    assert abs(c.wer) < 1e-9
    assert c.vram_peak_bytes == 1_500_000_000
    assert c.cue_count == 2 and c.vtt_path is not None
    assert any(p.name.endswith(".vtt") for p in tmp_path.iterdir())
    assert any(p.name.endswith(".json") and "_Words_" in p.name for p in tmp_path.iterdir())


def test_all_clips_failed_sets_note(tmp_path, monkeypatch):
    audio = tmp_path / "Lec.mp4"; audio.write_bytes(b"x")
    ref = tmp_path / "Lec.txt"; ref.write_text("hi", encoding="utf-8")

    class Boom(hf_mod.HFAdapter):
        name = "boom"
        def transcribe(self, *a, **k): raise RuntimeError("kaboom")
    monkeypatch.setattr(hf_mod, "make_hf_adapter", lambda cfg: Boom())
    mr = HFTransformersEngine().run(
        asr_bench.resolve_model_entry("wav2vec2-large-960h"),
        [Pair(audio=audio, reference=ref)], _cfg())
    assert mr.clips == [] and "ALL CLIPS FAILED" in mr.notes


def test_hf_registered():
    assert "hf" in asr_bench.ENGINES


def test_main_skips_hf_when_no_venv(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "Lec.mp4"; audio.write_bytes(b"x")
    ref = tmp_path / "Lec.txt"; ref.write_text("hello world", encoding="utf-8")
    monkeypatch.setattr(asr_bench, "_default_hf_python", lambda: None)
    canned = asr_bench.ModelResult(
        model_id="small", display="Whisper Small", fw_name="small", params="244M",
        developer="OpenAI", languages="99", notes="x", disk_bytes=None, load_sec=0.0)
    class StubFW(asr_bench.Engine):
        name = "faster-whisper"
        def run(self, entry, pairs, cfg): return canned
    monkeypatch.setitem(asr_bench.ENGINES, "faster-whisper", StubFW)
    out = tmp_path / "r.md"
    monkeypatch.setattr("sys.argv", [
        "asr_bench.py", "--corpus", str(tmp_path),
        "--models", "small,wav2vec2-large-960h", "--device", "cpu", "--output", str(out)])
    assert asr_bench.main() == 0
    err = capsys.readouterr().err
    assert "wav2vec2-large-960h" in err and "setup_hf_venv" in err
    assert out.is_file()
