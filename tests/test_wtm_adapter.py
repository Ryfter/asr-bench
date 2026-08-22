# tests/test_wtm_adapter.py
import json
import pytest
import asr_bench
import engines.mlx_wtm
from asr_bench import RunConfig, WtmResult


def test_fake_adapter_returns_canned():
    canned = WtmResult.from_dict({"text": "hi", "segments": [{"start": 0, "end": 1, "text": "hi"}]})
    a = asr_bench.FakeWtmAdapter(canned)
    assert a.name == "fake"
    assert a.transcribe("x.wav", RunConfig(device="cpu", compute_type="int8")) is canned


def test_runner_args_minimal():
    cfg = RunConfig(device="cpu", compute_type="int8", wtm_quick=True)
    args = asr_bench._wtm_runner_args("a.wav", cfg)
    assert args[0] == asr_bench._WTM_RUNNER_PATH
    assert "--audio" in args and "a.wav" in args
    assert "--timestamps" in args
    assert "--quick" in args


def test_default_wtm_python_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(engines.mlx_wtm, "__file__", str(tmp_path / "engines" / "mlx_wtm.py"))
    assert asr_bench._default_wtm_python() is None


def test_make_wtm_adapter_errors_without_venv(monkeypatch):
    monkeypatch.setattr(engines.mlx_wtm, "_default_wtm_python", lambda: None)
    cfg = RunConfig(device="cpu", compute_type="int8", wtm_python=None)
    with pytest.raises(RuntimeError, match="setup_wtm_venv"):
        asr_bench.make_wtm_adapter(cfg)


def test_subprocess_transcribe_parses_json(monkeypatch):
    import subprocess
    from asr_bench import SubprocessWtm, RunConfig

    class _Proc:
        returncode = 0
        stdout = json.dumps({
            "text": "hello world",
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
            "transcribe_sec": 0.5,
        })
        stderr = ""

    def fake_run(cmd, **kw):
        return _Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = SubprocessWtm("python")
    cfg = RunConfig(device="cpu", compute_type="int8")
    result = adapter.transcribe("a.wav", cfg)
    assert result.text() == "hello world"
    assert len(result.cue_tuples()) == 1
