# tests/test_nemo_resolve.py
import pytest
from asr_bench import resolve_model_entry


def test_registered_parakeet():
    e = resolve_model_entry("parakeet-tdt-0.6b-v2")
    assert e["engine"] == "nemo"
    assert e["nemo_model"] == "nvidia/parakeet-tdt-0.6b-v2"
    assert e["id"] == "parakeet-tdt-0.6b-v2"


def test_registered_canary():
    e = resolve_model_entry("canary-qwen-2.5b")
    assert e["engine"] == "nemo"
    assert e["nemo_model"] == "nvidia/canary-qwen-2.5b"


def test_adhoc_nemo_id():
    e = resolve_model_entry("nemo:nvidia/canary-1b-flash")
    assert e["engine"] == "nemo"
    assert e["nemo_model"] == "nvidia/canary-1b-flash"
    assert e["id"] == "nemo:nvidia/canary-1b-flash"
    assert "canary-1b-flash" in e["display"]


def test_adhoc_nemo_empty_raises():
    with pytest.raises(ValueError, match="empty NeMo model name"):
        resolve_model_entry("nemo:")
