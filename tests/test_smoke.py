def test_import_asr_bench():
    import asr_bench
    assert hasattr(asr_bench, "MODELS")

def test_models_registry_nonempty():
    import asr_bench
    assert "small" in asr_bench.MODELS
    assert "large-v3-turbo" in asr_bench.MODELS
