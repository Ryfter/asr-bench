# tests/test_nemo_engine.py
import asr_bench
from asr_bench import RunConfig, Pair, NeMoResult


def _cfg():
    return RunConfig(device="cpu", compute_type="int8", nemo_python="/fake/python")


def test_parakeet_shape_writes_vtt_and_scores(tmp_path, monkeypatch):
    audio = tmp_path / "Lec.mp4"; audio.write_bytes(b"x")
    ref = tmp_path / "Lec.txt"; ref.write_text("hello there hi back", encoding="utf-8")
    canned = NeMoResult.from_dict({
        "text": "hello there hi back",
        "segments": [{"start": 0.0, "end": 2.0, "text": "hello there"},
                     {"start": 2.0, "end": 4.0, "text": "hi back"}],
        "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
        "transcribe_sec": 1.0, "vram_peak_bytes": 2_000_000_000, "language": "en",
    })
    monkeypatch.setattr(asr_bench, "make_nemo_adapter", lambda cfg: asr_bench.FakeNeMoAdapter(canned))
    monkeypatch.setattr(asr_bench, "_audio_duration_sec", lambda p: 4.0)

    entry = asr_bench.resolve_model_entry("parakeet-tdt-0.6b-v2")
    mr = asr_bench.NeMoEngine().run(entry, [Pair(audio=audio, reference=ref)], _cfg())

    assert mr.engine == "nemo" and len(mr.clips) == 1
    c = mr.clips[0]
    assert abs(c.wer) < 1e-9                       # hypothesis == reference → 0 WER
    assert c.vram_peak_bytes == 2_000_000_000
    assert c.transcribe_sec == 1.0                 # uses runner-reported time, not wall clock
    assert c.cue_count == 2
    assert c.vtt_path is not None
    assert any(p.name.endswith(".vtt") for p in tmp_path.iterdir())
    assert any(p.name.endswith(".json") and "_Words_" in p.name for p in tmp_path.iterdir())


def test_canary_text_only_no_vtt(tmp_path, monkeypatch):
    audio = tmp_path / "Lec.mp4"; audio.write_bytes(b"x")
    ref = tmp_path / "Lec.txt"; ref.write_text("hello world", encoding="utf-8")
    canned = NeMoResult.from_dict({"text": "hello world", "transcribe_sec": 2.0})
    monkeypatch.setattr(asr_bench, "make_nemo_adapter", lambda cfg: asr_bench.FakeNeMoAdapter(canned))
    monkeypatch.setattr(asr_bench, "_audio_duration_sec", lambda p: 10.0)

    entry = asr_bench.resolve_model_entry("canary-qwen-2.5b")
    mr = asr_bench.NeMoEngine().run(entry, [Pair(audio=audio, reference=ref)], _cfg())

    c = mr.clips[0]
    assert abs(c.wer) < 1e-9
    assert c.vtt_path is None                       # no timestamps → no VTT
    assert c.cue_count == 0
    assert not any(p.name.endswith(".vtt") for p in tmp_path.iterdir())
    assert abs(c.rtfx - 5.0) < 1e-9                 # 10s audio / 2s compute


def test_run_falls_back_to_wall_clock_when_no_transcribe_sec(tmp_path, monkeypatch):
    audio = tmp_path / "Lec.mp4"; audio.write_bytes(b"x")
    ref = tmp_path / "Lec.txt"; ref.write_text("hi", encoding="utf-8")
    canned = NeMoResult.from_dict({"text": "hi"})   # no transcribe_sec
    monkeypatch.setattr(asr_bench, "make_nemo_adapter", lambda cfg: asr_bench.FakeNeMoAdapter(canned))
    monkeypatch.setattr(asr_bench, "_audio_duration_sec", lambda p: 1.0)
    mr = asr_bench.NeMoEngine().run(
        asr_bench.resolve_model_entry("canary-qwen-2.5b"),
        [Pair(audio=audio, reference=ref)], _cfg())
    assert mr.clips[0].transcribe_sec >= 0.0        # wall-clock fallback, did not crash


def test_all_clips_failed_sets_note(tmp_path, monkeypatch):
    audio = tmp_path / "Lec.mp4"; audio.write_bytes(b"x")
    ref = tmp_path / "Lec.txt"; ref.write_text("hi", encoding="utf-8")

    class Boom(asr_bench.NeMoAdapter):
        name = "boom"
        def transcribe(self, *a, **k):
            raise RuntimeError("kaboom")

    monkeypatch.setattr(asr_bench, "make_nemo_adapter", lambda cfg: Boom())
    mr = asr_bench.NeMoEngine().run(
        asr_bench.resolve_model_entry("canary-qwen-2.5b"),
        [Pair(audio=audio, reference=ref)], _cfg())
    assert mr.clips == []
    assert "ALL CLIPS FAILED" in mr.notes


def test_nemo_registered():
    assert "nemo" in asr_bench.ENGINES


def test_config_to_dict_includes_nemo_python_non_secret():
    cfg = asr_bench.RunConfig(device="cpu", compute_type="int8", nemo_python="/x/py")
    d = asr_bench._config_to_dict(cfg)
    assert d["nemo_python"] == "/x/py"


def test_runconfig_defaults_nemo_python_none():
    cfg = asr_bench.RunConfig(device="cpu", compute_type="int8")
    assert cfg.nemo_python is None


def test_main_skips_nemo_when_no_venv(tmp_path, monkeypatch, capsys):
    # NeMo requested but no venv → warn + skip NeMo, still run the whisper model.
    audio = tmp_path / "Lec.mp4"; audio.write_bytes(b"x")
    ref = tmp_path / "Lec.txt"; ref.write_text("hello world", encoding="utf-8")
    monkeypatch.setattr(asr_bench, "_default_nemo_python", lambda: None)

    # Stub the faster-whisper engine so the test needs no model download.
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
        "--models", "small,canary-qwen-2.5b", "--device", "cpu", "--output", str(out)])
    rc = asr_bench.main()
    assert rc == 0
    err = capsys.readouterr().err
    assert "canary-qwen-2.5b" in err and "setup_nemo_venv" in err
    assert out.is_file()
