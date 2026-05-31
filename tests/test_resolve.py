import pytest
import asr_bench

def test_resolve_builtin_whisper():
    entry = asr_bench.resolve_model_entry("small")
    assert entry["id"] == "small"
    assert entry["engine"] == "faster-whisper"
    assert entry["fw_name"] == "small"

def test_resolve_canary_nim():
    entry = asr_bench.resolve_model_entry("canary-nim")
    assert entry["id"] == "canary-nim"
    assert entry["engine"] == "nim"
    assert entry["riva_model"] == ""

def test_resolve_adhoc_nim():
    entry = asr_bench.resolve_model_entry("nim:parakeet-1.1b")
    assert entry["id"] == "nim:parakeet-1.1b"
    assert entry["engine"] == "nim"
    assert entry["riva_model"] == "parakeet-1.1b"
    assert entry["display"] == "NIM (parakeet-1.1b)"

def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        asr_bench.resolve_model_entry("not-a-model")

def test_resolve_empty_adhoc_raises():
    with pytest.raises(ValueError):
        asr_bench.resolve_model_entry("nim:")
