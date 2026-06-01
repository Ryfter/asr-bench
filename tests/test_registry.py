import asr_bench

def test_faster_whisper_entries_have_fw_name():
    """All entries claiming engine=faster-whisper must have a fw_name key."""
    for model_id, entry in asr_bench.MODELS.items():
        if entry.get("engine") == "faster-whisper":
            assert "fw_name" in entry, model_id

def test_canary_nim_entry_present_and_shaped():
    entry = asr_bench.MODELS["canary-nim"]
    assert entry["engine"] == "nim"
    assert entry["developer"] == "NVIDIA"
    assert "riva_model" in entry  # default "" => server default

def test_whisper_ids_have_engine_faster_whisper():
    whisper_ids = ["small", "medium", "large-v3", "large-v3-turbo"]
    for mid in whisper_ids:
        assert asr_bench.MODELS[mid]["engine"] == "faster-whisper", mid
