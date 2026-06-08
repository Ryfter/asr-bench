# tests/test_nemo_adapter.py
import json
import pytest
import asr_bench
from asr_bench import RunConfig, NeMoResult


def test_fake_adapter_returns_canned():
    canned = NeMoResult.from_dict({"text": "hi"})
    a = asr_bench.FakeNeMoAdapter(canned)
    assert a.name == "fake"
    assert a.transcribe("x.wav", "nvidia/parakeet-tdt-0.6b-v2",
                        RunConfig(device="cpu", compute_type="int8")) is canned


def test_runner_args_minimal():
    cfg = RunConfig(device="cuda", compute_type="float16")
    args = asr_bench._nemo_runner_args("a.wav", "nvidia/canary-qwen-2.5b", cfg)
    assert args[0] == asr_bench._NEMO_RUNNER_PATH
    assert "--audio" in args and "a.wav" in args
    assert "--model" in args and "nvidia/canary-qwen-2.5b" in args
    assert "--device" in args and "cuda" in args


def test_default_nemo_python_none_when_absent(tmp_path, monkeypatch):
    # Point module dir at an empty tmp dir → no .venv-nemo present.
    monkeypatch.setattr(asr_bench, "__file__", str(tmp_path / "asr_bench.py"))
    assert asr_bench._default_nemo_python() is None


def test_default_nemo_python_finds_venv(tmp_path, monkeypatch):
    venv = tmp_path / ".venv-nemo" / "Scripts"
    venv.mkdir(parents=True)
    (venv / "python.exe").write_text("", encoding="utf-8")
    monkeypatch.setattr(asr_bench, "__file__", str(tmp_path / "asr_bench.py"))
    got = asr_bench._default_nemo_python()
    assert got is not None and got.endswith("python.exe")


def test_make_nemo_adapter_errors_without_venv(monkeypatch):
    monkeypatch.setattr(asr_bench, "_default_nemo_python", lambda: None)
    cfg = RunConfig(device="cpu", compute_type="int8", nemo_python=None)
    with pytest.raises(RuntimeError, match="setup_nemo_venv"):
        asr_bench.make_nemo_adapter(cfg)


def test_make_nemo_adapter_uses_subprocess_with_venv(monkeypatch):
    monkeypatch.setattr(asr_bench, "_default_nemo_python", lambda: "/x/.venv-nemo/bin/python")
    cfg = RunConfig(device="cpu", compute_type="int8")
    a = asr_bench.make_nemo_adapter(cfg)
    assert isinstance(a, asr_bench.SubprocessNeMo)
    assert a.python == "/x/.venv-nemo/bin/python"
