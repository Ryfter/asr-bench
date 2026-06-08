import asr_bench


def test_distil_v35_registered_as_faster_whisper():
    entry = asr_bench.resolve_model_entry("distil-large-v3.5")
    assert entry["engine"] == "faster-whisper"
    assert entry["fw_name"]            # non-empty CT2 id
    assert entry["id"] == "distil-large-v3.5"


def test_distil_v35_has_vram_cost():
    assert "distil-large-v3.5" in asr_bench._MODEL_VRAM_COST
