import asr_bench

def test_all_builtin_models_are_faster_whisper():
    for model_id, entry in asr_bench.MODELS.items():
        assert entry.get("engine") == "faster-whisper", model_id
