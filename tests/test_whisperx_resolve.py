import asr_bench


def test_resolve_whisperx_model():
    e = asr_bench.resolve_model_entry("large-v3-turbo+whisperx")
    assert e["engine"] == "whisperx"
    assert e["fw_name"] == "large-v3-turbo"
    assert e["id"] == "large-v3-turbo+whisperx"
    assert "WhisperX" in e["display"]


def test_resolve_whisperx_all_sizes():
    for size in ["small", "medium", "large-v3", "large-v3-turbo"]:
        e = asr_bench.resolve_model_entry(f"{size}+whisperx")
        assert e["engine"] == "whisperx" and e["fw_name"] == size


def test_resolve_whisperx_bad_size_errors():
    try:
        asr_bench.resolve_model_entry("bogus+whisperx")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_plain_size_still_faster_whisper():
    assert asr_bench.resolve_model_entry("small")["engine"] == "faster-whisper"
