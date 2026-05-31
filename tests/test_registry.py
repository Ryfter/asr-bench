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
