import asr_bench


def test_wav2vec2_registered():
    e = asr_bench.resolve_model_entry("wav2vec2-large-960h")
    assert e["engine"] == "hf" and e["hf_model"] == "facebook/wav2vec2-large-960h"


def test_conformer_registered():
    e = asr_bench.resolve_model_entry("wav2vec2-conformer-large")
    assert e["engine"] == "hf"
    assert e["hf_model"] == "facebook/wav2vec2-conformer-rope-large-960h-ft"


def test_hf_adhoc_id():
    e = asr_bench.resolve_model_entry("hf:openai/whisper-base")
    assert e["engine"] == "hf" and e["hf_model"] == "openai/whisper-base"


def test_hf_adhoc_empty_rejected():
    import pytest
    with pytest.raises(ValueError):
        asr_bench.resolve_model_entry("hf:")


def test_runconfig_defaults_hf_python_none():
    cfg = asr_bench.RunConfig(device="cpu", compute_type="int8")
    assert cfg.hf_python is None


def test_config_to_dict_includes_hf_python_non_secret():
    cfg = asr_bench.RunConfig(device="cpu", compute_type="int8", hf_python="/x/py")
    assert asr_bench._config_to_dict(cfg)["hf_python"] == "/x/py"
