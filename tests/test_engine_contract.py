import asr_bench

def test_runconfig_defaults():
    cfg = asr_bench.RunConfig(device="cpu", compute_type="int8")
    assert cfg.batch_size == 1
    assert cfg.beam_size == 5
    assert cfg.vad_filter is True
    assert cfg.nim_url == "localhost:50051"
    assert cfg.nim_language == "en-US"
    assert cfg.nim_api_key is None
    assert cfg.nim_ssl is False


def test_faster_whisper_engine_name():
    eng = asr_bench.FasterWhisperEngine()
    assert eng.name == "faster-whisper"
    assert isinstance(eng, asr_bench.Engine)


def test_nim_engine_name():
    eng = asr_bench.NimEngine()
    assert eng.name == "nim"
    assert isinstance(eng, asr_bench.Engine)


def test_engines_registry_has_both():
    assert set(asr_bench.ENGINES.keys()) == {"faster-whisper", "nim", "whisperx"}
    for cls in asr_bench.ENGINES.values():
        assert issubclass(cls, asr_bench.Engine)
